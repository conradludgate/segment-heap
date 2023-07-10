#![feature(alloc_layout_extra)]
#![allow(clippy::never_loop, clippy::while_immutable_condition)]

use core::fmt;
use std::{
    alloc::{handle_alloc_error, Layout},
    marker::PhantomData,
    num::NonZeroUsize,
    ptr::{addr_of, addr_of_mut, NonNull},
    sync::{
        atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering},
        OnceLock, RwLock,
    },
};

use nix::{
    sys::mman::{mmap, munmap, MapFlags, ProtFlags},
    unistd::{sysconf, SysconfVar},
};

pub struct SegmentHeap<T> {
    inner: [RwLock<SingleSegment>; 31],
    // bit mask of allocated segments
    segments: AtomicU32,
    page_size: usize,
    _t: PhantomData<T>,
}

impl<T> fmt::Debug for SegmentHeap<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentHeap")
            .field("inner", &self.inner)
            .field("segments", &self.segments)
            .field("_t", &self._t)
            .finish()
    }
}

impl<T> Default for SegmentHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SegmentHeap<T> {
    const fn state(segment: u8) -> StateSegment {
        StateSegment::new::<T>(segment)
    }

    pub fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const SEGMENT: RwLock<SingleSegment> = RwLock::new(SingleSegment::new());
        Self {
            inner: [SEGMENT; 31],
            segments: AtomicU32::new(0),
            page_size: *PAGE_SIZE
                .get_or_init(|| sysconf(SysconfVar::PAGE_SIZE).unwrap().unwrap_or(4096))
                as usize,
            _t: PhantomData,
        }
    }

    pub fn alloc(&self) -> NonNull<T> {
        loop {
            let segments = self.segments.load(Ordering::Acquire);
            if segments == 0 {
                let Some(x) = self.alloc_segment(0) else { continue };
                return x;
            }
            let last_segment = (31 - segments.leading_zeros()) as u8;
            let s = match self.inner[last_segment as usize]
                .read()
                .unwrap()
                .alloc(Self::state(last_segment), self.page_size)
            {
                AllocResult::Alloc(t) => return t.cast(),
                AllocResult::Full => last_segment + 1,
                AllocResult::Uninit => last_segment,
            };
            if let Some(x) = self.alloc_segment(s) {
                return x;
            }
        }
    }

    #[cold]
    fn alloc_segment(&self, s: u8) -> Option<NonNull<T>> {
        if let Ok(mut segment) = self.inner[s as usize].try_write() {
            self.segments.fetch_xor(1 << s, Ordering::Release);
            return Some(segment.init(Self::state(s), self.page_size).cast());
        }
        None
    }

    /// # Safety
    /// Think about it
    pub unsafe fn dealloc(&self, ptr: NonNull<T>) {
        let mut segments = self.segments.load(Ordering::Acquire);
        let mut segment = 0;
        while segments > 1 {
            let len = self.inner[segment as usize].read().unwrap().dealloc(
                Self::state(segment),
                self.page_size,
                ptr.as_ptr().cast(),
            );

            match len {
                None => {
                    segment += 1;
                    segments >>= 1;
                    continue;
                }
                Some(0) => {
                    if let Ok(mut seg) = self.inner[segment as usize].try_write() {
                        self.segments.fetch_xor(1 << segment, Ordering::Release);
                        // confirm that we can dealloc
                        if *seg.length.get_mut() == 0 {
                            seg.release(Self::state(segment), self.page_size);
                        }
                    }
                    return;
                }
                _ => return,
            }
        }
        self.inner[segment as usize].read().unwrap().dealloc(
            Self::state(segment),
            self.page_size,
            ptr.as_ptr().cast(),
        );
    }
}

enum AllocResult {
    Alloc(NonNull<()>),
    Full,
    Uninit,
}

struct StateSegment {
    layout_slot: Layout,
    segment_size: usize,
}

const fn max_usize(x: usize, y: usize) -> usize {
    if x < y {
        y
    } else {
        x
    }
}

static PAGE_SIZE: OnceLock<i64> = OnceLock::new();
impl StateSegment {
    const fn new<T>(segment: u8) -> Self {
        let layout_t = Layout::new::<T>();
        let layout_free = Layout::new::<Slot>();
        let layout_slot = unsafe {
            Layout::from_size_align_unchecked(
                max_usize(layout_t.size(), layout_free.size()),
                max_usize(layout_t.align(), layout_free.align()),
            )
        };

        // first page should be at least 64kib
        let size = 65536 / layout_slot.size().next_power_of_two();
        let segment_size = max_usize(size, 1) << segment;

        Self {
            layout_slot,
            segment_size,
        }
    }

    fn alignment(&self, page_size: usize) -> usize {
        // let page_size =
        //     PAGE_SIZE.get_or_init(|| sysconf(SysconfVar::PAGE_SIZE).unwrap().unwrap_or(4096));
        self.layout_slot.align().saturating_sub(page_size)
    }

    fn page_layout(&self) -> (Layout, usize) {
        self.layout_slot.repeat(self.segment_size).unwrap()
    }
}

struct SingleSegment {
    /// pointer to the allocation this segment allocates into
    slots: *mut u8,
    head: AtomicPtr<Slot>,
    length: AtomicUsize,
    init: AtomicUsize,
}

impl fmt::Debug for SingleSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleSegment")
            .field("slots", &self.slots)
            .field("head", &self.head)
            .field("length", &self.length)
            .field("init", &self.init)
            .finish()
    }
}

impl SingleSegment {
    const fn new() -> Self {
        Self {
            slots: core::ptr::null_mut(),
            head: AtomicPtr::new(core::ptr::null_mut()),
            length: AtomicUsize::new(0),
            init: AtomicUsize::new(0),
        }
    }

    #[cold]
    fn init(&mut self, state: StateSegment, page_size: usize) -> NonNull<()> {
        let (layout_slots, _) = state.page_layout();

        let ptr = unsafe {
            mmap(
                None,
                NonZeroUsize::new(layout_slots.size() + state.alignment(page_size)).unwrap(),
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
            head: AtomicPtr::new(core::ptr::null_mut()),
            length: AtomicUsize::new(1),
            init: AtomicUsize::new(1),
        };
        unsafe { NonNull::new_unchecked(ptr.add(state.alignment(page_size)).cast()) }
    }

    #[cold]
    fn release(&mut self, state: StateSegment, page_size: usize) {
        let (layout_slots, _) = state.page_layout();
        unsafe {
            munmap(
                self.slots.cast(),
                layout_slots.size() + state.alignment(page_size),
            )
            .unwrap()
        };
        self.slots = std::ptr::null_mut();
    }

    #[inline]
    fn alloc(&self, state: StateSegment, page_size: usize) -> AllocResult {
        if self.slots.is_null() {
            return AllocResult::Uninit;
        }

        let len = state.segment_size;
        let mut init = self.init.load(Ordering::Acquire);
        while init < len {
            let (_, offset_slot) = state.page_layout();

            match self.init.compare_exchange_weak(
                init,
                init + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(init) => {
                    self.length.fetch_add(1, Ordering::Acquire);
                    return AllocResult::Alloc(unsafe {
                        NonNull::new_unchecked(
                            self.slots
                                .add(state.alignment(page_size) + offset_slot * init),
                        )
                        .cast()
                    });
                }
                Err(i) => init = i,
            }
        }

        let mut head = self.head.load(Ordering::Acquire);
        let slot = loop {
            if head.is_null() {
                return AllocResult::Full;
            }

            let next = unsafe { core::ptr::read(addr_of!((*head).next)) };

            // try acquire the slot
            match self
                .head
                .compare_exchange_weak(head, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(slot) => break slot,
                Err(slot) => head = slot,
            }
        };

        self.length.fetch_add(1, Ordering::Release);

        AllocResult::Alloc(unsafe { NonNull::new_unchecked(slot).cast() })
    }

    #[inline]
    unsafe fn dealloc(&self, state: StateSegment, page_size: usize, ptr: *mut u8) -> Option<usize> {
        if self.slots.is_null() {
            return None;
        }

        let (layout_slots, _) = state.page_layout();
        let start = unsafe { self.slots.add(state.alignment(page_size)) };
        let end = unsafe { start.add(layout_slots.size()) };
        if !(start..end).contains(&ptr) {
            return None;
        }

        let ptr = ptr.cast::<Slot>();

        let mut next = self.head.load(Ordering::Acquire);
        loop {
            unsafe {
                addr_of_mut!((*ptr).next).write(next);
            }
            match self
                .head
                .compare_exchange_weak(next, ptr, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => break,
                Err(n) => next = n,
            }
        }

        Some(self.length.fetch_sub(1, Ordering::Acquire) - 1)
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
    fn dealloc_lifo() {
        let heap = SegmentHeap::<usize>::new();
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
    fn dealloc_fifo() {
        let heap = SegmentHeap::<usize>::new();
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
    fn dealloc_random() {
        let heap = SegmentHeap::<usize>::new();
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

    #[test]
    fn size() {
        assert_eq!(std::mem::size_of::<SegmentHeap<()>>(), 1496);
    }

    #[repr(align(8192))]
    struct LargeAlign([u8; 8192]);

    #[test]
    fn large_alignment() {
        let heap = SegmentHeap::<LargeAlign>::new();
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
}
