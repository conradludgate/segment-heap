#![feature(alloc_layout_extra)]
#![allow(clippy::never_loop, clippy::while_immutable_condition)]

use std::{
    alloc::{handle_alloc_error, Layout},
    fmt,
    marker::PhantomData,
    num::NonZeroUsize,
    os::fd::RawFd,
    ptr::{addr_of, addr_of_mut, NonNull},
    sync::{
        atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering},
        RwLock,
    },
};

use nix::sys::mman::{mmap, munmap, MapFlags, ProtFlags};

pub struct SegmentHeap<T> {
    inner: [SingleSegment; 31],
    segments: u8,
    _t: PhantomData<T>,
}

impl<T> Default for SegmentHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SegmentHeap<T> {
    const VALID_ALIGNMENT: () = {
        assert!(std::mem::align_of::<T>() <= 4096);
    };

    pub const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const SEGMENT: SingleSegment = SingleSegment::new();
        #[allow(clippy::let_unit_value)]
        let _valid_alignment = Self::VALID_ALIGNMENT;
        Self {
            inner: [SEGMENT; 31],
            segments: 0,
            _t: PhantomData,
        }
    }

    pub fn alloc(&mut self) -> NonNull<T> {
        loop {
            let segments = self.segments;
            let s = match self.inner[segments as usize].alloc(StateSegment::new::<T>(segments)) {
                AllocResult::Alloc(t) => return t.cast(),
                AllocResult::Full => segments + 1,
                AllocResult::Uninit => segments,
            };
            if let Some(x) = self.alloc_segment(segments, s) {
                return x;
            }
        }
    }

    #[cold]
    fn alloc_segment(&mut self, segments: u8, s: u8) -> Option<NonNull<T>> {
        if let segment = &mut self.inner[s as usize] {
            if s > segments {
                self.segments += 1;
            }
            return Some(segment.init(StateSegment::new::<T>(s)).cast());
        }
        None
    }

    /// # Safety
    /// Think about it
    pub unsafe fn dealloc(&mut self, ptr: NonNull<T>) {
        let segments = self.segments;
        for segment in 0..segments {
            let len = self.inner[segment as usize]
                .dealloc(StateSegment::new::<T>(segment), ptr.as_ptr().cast());

            match len {
                usize::MAX => continue,
                0 => {
                    if let seg = &mut self.inner[segment as usize] {
                        // confirm that we can dealloc
                        // if seg.length == 0 {
                        seg.release(StateSegment::new::<T>(segment));
                        // }
                    }
                    return;
                }
                _ => return,
            }
        }
        self.inner[segments as usize]
            .dealloc(StateSegment::new::<T>(segments), ptr.as_ptr().cast());
    }
}

enum AllocResult {
    Alloc(NonNull<()>),
    Full,
    Uninit,
}

struct SingleSegment {
    /// pointer to the allocation this segment allocates into
    slots: *mut u8,
    head: *mut Slot,
    length: usize,
    init: usize,
}

impl fmt::Debug for SingleSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleSegment")
            // .field("slots", &self.slots)
            // .field("head", &self.head)
            // .field("length", &self.length)
            .finish()
    }
}

struct StateSegment {
    layout_t: Layout,
    layout_slot: Layout,
    offset_t: usize,
    segment: u8,
}

impl StateSegment {
    fn new<T>(segment: u8) -> Self {
        let layout_t = Layout::new::<T>();
        let (layout_slot, offset_t) = Layout::new::<Slot>().extend(layout_t).unwrap();
        debug_assert!(layout_slot.align() < 4096);
        Self {
            layout_t,
            layout_slot,
            offset_t,
            segment,
        }
    }
    fn base_segment_size(&self) -> usize {
        // first page should be at least 64kib
        let size = 65536 / self.layout_slot.size().next_power_of_two();
        usize::max(size, 1)
    }
    fn segment_size(&self) -> usize {
        self.base_segment_size() << self.segment
    }
    fn page_layout(&self) -> (Layout, usize) {
        self.layout_slot.repeat(self.segment_size()).unwrap()
    }
}

impl SingleSegment {
    const fn new() -> Self {
        Self {
            slots: core::ptr::null_mut(),
            head: core::ptr::null_mut(),
            length: 0,
            init: 0,
        }
    }

    #[cold]
    fn init(&mut self, state: StateSegment) -> NonNull<()> {
        let (layout_slots, _) = state.page_layout();

        let ptr = unsafe {
            mmap(
                None,
                NonZeroUsize::new(layout_slots.size()).unwrap(),
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_ANONYMOUS | MapFlags::MAP_PRIVATE,
                -1,
                0,
            )
            .unwrap_or_else(|e| {
                dbg!(e);
                handle_alloc_error(layout_slots)
            })
        };

        // let ptr = unsafe { std::alloc::alloc(layout_slots) };
        // let ptr = std::ptr::slice_from_raw_parts_mut(ptr.cast::<Slot<T>>(), len);
        if ptr.is_null() {
            handle_alloc_error(layout_slots)
        }

        *self = Self {
            slots: ptr.cast(),
            head: std::ptr::null_mut(),
            length: 1,
            init: 1,
        };
        unsafe { NonNull::new_unchecked(ptr.add(state.offset_t).cast()) }
    }

    #[cold]
    fn release(&mut self, state: StateSegment) {
        let (layout_slots, _) = state.page_layout();
        unsafe { munmap(self.slots.cast(), layout_slots.size()).unwrap() };
        self.slots = std::ptr::null_mut();
    }

    #[inline]
    fn alloc(&mut self, state: StateSegment) -> AllocResult {
        if self.slots.is_null() {
            return AllocResult::Uninit;
        }

        let len = state.segment_size();
        let mut init = self.init;
        while init < len {
            self.init += 1;

            self.length += 1;
            let (_, offset_slot) = state.page_layout();
            return AllocResult::Alloc(unsafe {
                NonNull::new_unchecked(self.slots.add(offset_slot * init + state.offset_t)).cast()
            });
        }

        let mut head = self.head;
        let slot = loop {
            if head.is_null() {
                return AllocResult::Full;
            }

            let next = unsafe { core::ptr::read(addr_of!((*head).next)) };
            self.head = next;
            break head;
        };

        self.length += 1;

        AllocResult::Alloc(unsafe {
            NonNull::new_unchecked(slot.cast::<u8>().add(state.offset_t)).cast()
        })
    }

    #[inline]
    unsafe fn dealloc(&mut self, state: StateSegment, ptr: *mut u8) -> usize {
        if self.slots.is_null() {
            return usize::MAX;
        }

        let len = state.segment_size();
        let (layout_slots, _) = state.page_layout();
        let start = self.slots;
        let end = unsafe { self.slots.add(layout_slots.size()) };
        if !(start..end).contains(&ptr) {
            return usize::MAX;
        }

        let ptr = unsafe { ptr.sub(state.offset_t).cast::<Slot>() };

        let mut next = self.head;
        loop {
            unsafe {
                addr_of_mut!((*ptr).next).write(next);
            }
            self.head = ptr;
            break;
        }

        self.length -= 1;
        self.length
    }
}

struct Slot {
    /// The next free slot after this one
    next: *mut Slot,
}

#[cfg(test)]
mod tests {
    use rand::{seq::SliceRandom, thread_rng};

    use crate::SegmentHeap;

    #[cfg(miri)]
    const N: usize = (1 << 8) - 1;
    #[cfg(not(miri))]
    const N: usize = (1 << 16) - 1;

    #[test]
    fn happy_path1() {
        let mut heap = SegmentHeap::<usize>::new();
        let mut ptrs = vec![];
        for _ in 0..N {
            ptrs.push(heap.alloc());
        }

        for ptr in ptrs.into_iter().rev() {
            unsafe {
                heap.dealloc(ptr);
            }
        }
    }

    #[test]
    fn happy_path2() {
        let mut heap = SegmentHeap::<usize>::new();
        let mut ptrs = vec![];
        for _ in 0..N {
            ptrs.push(heap.alloc());
        }

        for ptr in ptrs {
            unsafe {
                heap.dealloc(ptr);
            }
        }
    }

    #[test]
    fn happy_path3() {
        let mut heap = SegmentHeap::<usize>::new();
        let mut ptrs = vec![];
        for _ in 0..N {
            ptrs.push(heap.alloc());
        }

        ptrs.shuffle(&mut thread_rng());
        for ptr in ptrs {
            unsafe {
                heap.dealloc(ptr);
            }
        }
    }
}
