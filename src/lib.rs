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
    fn state(segment: u8) -> StateSegment {
        StateSegment::new::<T>(segment)
    }

    pub const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const SEGMENT: RwLock<SingleSegment> = RwLock::new(SingleSegment::new());
        Self {
            inner: [SEGMENT; 31],
            segments: AtomicU32::new(0),
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
                .alloc(Self::state(last_segment))
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
            return Some(segment.init(Self::state(s)).cast());
        }
        None
    }

    /// # Safety
    /// Think about it
    pub unsafe fn dealloc(&self, ptr: NonNull<T>) {
        let mut segments = self.segments.load(Ordering::Acquire);
        let mut segment = 0;
        while segments > 1 {
            let len = self.inner[segment as usize]
                .read()
                .unwrap()
                .dealloc(Self::state(segment), ptr.as_ptr().cast());

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
                            seg.release(Self::state(segment));
                        }
                    }
                    return;
                }
                _ => return,
            }
        }
        self.inner[segment as usize]
            .read()
            .unwrap()
            .dealloc(Self::state(segment), ptr.as_ptr().cast());
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
    head: AtomicPtr<Slot>,
    length: AtomicUsize,
    init: AtomicUsize,
}

struct StateSegment {
    layout_slot: Layout,
    extra: usize,
    segment: u8,
}

static PAGE_SIZE: OnceLock<Option<i64>> = OnceLock::new();
impl StateSegment {
    fn new<T>(segment: u8) -> Self {
        let layout_t = Layout::new::<T>();
        let layout_free = Layout::new::<Slot>();
        let layout_slot = Layout::from_size_align(
            usize::max(layout_t.size(), layout_free.size()),
            usize::max(layout_t.align(), layout_free.align()),
        )
        .unwrap();

        let page_size = PAGE_SIZE
            .get_or_init(|| sysconf(SysconfVar::PAGE_SIZE).unwrap())
            .unwrap_or(4096);

        let extra = layout_slot.align().saturating_sub(page_size as usize);

        Self {
            layout_slot,
            extra,
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
    fn init(&mut self, state: StateSegment) -> NonNull<()> {
        let (layout_slots, _) = state.page_layout();

        let ptr = unsafe {
            mmap(
                None,
                NonZeroUsize::new(layout_slots.size() + state.extra).unwrap(),
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
        unsafe { NonNull::new_unchecked(ptr.add(state.extra).cast()) }
    }

    #[cold]
    fn release(&mut self, state: StateSegment) {
        let (layout_slots, _) = state.page_layout();
        unsafe { munmap(self.slots.cast(), layout_slots.size() + state.extra).unwrap() };
        self.slots = std::ptr::null_mut();
    }

    #[inline]
    fn alloc(&self, state: StateSegment) -> AllocResult {
        if self.slots.is_null() {
            return AllocResult::Uninit;
        }

        let len = state.segment_size();
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
                        NonNull::new_unchecked(self.slots.add(state.extra + offset_slot * init))
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
    unsafe fn dealloc(&self, state: StateSegment, ptr: *mut u8) -> Option<usize> {
        if self.slots.is_null() {
            return None;
        }

        let (layout_slots, _) = state.page_layout();
        let start = unsafe { self.slots.add(state.extra) };
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
