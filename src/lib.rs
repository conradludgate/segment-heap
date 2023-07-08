#![feature(alloc_layout_extra)]

use std::{
    alloc::{handle_alloc_error, Layout},
    fmt,
    marker::PhantomData,
    ptr::{addr_of, addr_of_mut, NonNull},
    sync::{
        atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering},
        RwLock,
    },
};

pub struct SegmentHeap<T> {
    inner: [RwLock<SingleSegment>; 31],
    segments: AtomicU8,
    _t: PhantomData<T>,
}

impl<T> Default for SegmentHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SegmentHeap<T> {
    pub const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const SEGMENT: RwLock<SingleSegment> = RwLock::new(SingleSegment::new());
        Self {
            inner: [SEGMENT; 31],
            segments: AtomicU8::new(0),
            _t: PhantomData,
        }
    }

    fn slot() -> (Layout, usize) {
        Layout::new::<Slot>().extend(Layout::new::<T>()).unwrap()
    }

    pub fn alloc(&self) -> NonNull<T> {
        loop {
            let segments = self.segments.load(Ordering::Acquire);
            let s = match self.inner[segments as usize]
                .read()
                .unwrap()
                .alloc(segments, Self::slot())
            {
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
    fn alloc_segment(&self, segments: u8, s: u8) -> Option<NonNull<T>> {
        if let Ok(mut segment) = self.inner[s as usize].try_write() {
            if s > segments {
                self.segments.fetch_add(1, Ordering::Release);
            }
            return Some(segment.init(s, Self::slot()).cast());
        }
        None
    }

    /// # Safety
    /// Think about it
    pub unsafe fn dealloc(&self, ptr: NonNull<T>) {
        let segments = self.segments.load(Ordering::Relaxed);
        for segment in (0..=segments).rev() {
            let len = self.inner[segment as usize].read().unwrap().dealloc(
                segment,
                Self::slot(),
                ptr.as_ptr().cast(),
            );

            match len {
                usize::MAX => continue,
                0 => {
                    if let Ok(mut seg) = self.inner[segment as usize].try_write() {
                        // confirm that we can dealloc
                        if *seg.length.get_mut() == 0 {
                            seg.release(segment, Self::slot());
                        }
                    }
                    break;
                }
                _ => break,
            }
        }
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

impl fmt::Debug for SingleSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleSegment")
            // .field("slots", &self.slots)
            // .field("head", &self.head)
            // .field("length", &self.length)
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
    fn init(&mut self, segment: u8, (layout_slot, offset_t): (Layout, usize)) -> NonNull<()> {
        let (layout_slots, _) = layout_slot.repeat(16 << segment).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout_slots) };
        // let ptr = std::ptr::slice_from_raw_parts_mut(ptr.cast::<Slot<T>>(), len);
        if ptr.is_null() {
            handle_alloc_error(layout_slots)
        }

        *self = Self {
            slots: ptr,
            head: AtomicPtr::new(std::ptr::null_mut()),
            length: AtomicUsize::new(1),
            init: AtomicUsize::new(1),
        };
        unsafe { NonNull::new_unchecked(ptr.add(offset_t).cast()) }
    }

    #[cold]
    fn release(&mut self, segment: u8, (layout_slot, _): (Layout, usize)) {
        let (layout_slots, _) = layout_slot.repeat(16 << segment).unwrap();
        unsafe { std::alloc::dealloc(self.slots, layout_slots) };
        self.slots = std::ptr::null_mut();
    }

    #[inline]
    fn alloc(&self, segment: u8, (layout_slot, offset_t): (Layout, usize)) -> AllocResult {
        if self.slots.is_null() {
            return AllocResult::Uninit;
        }

        let len = 16 << segment;
        let mut init = self.init.load(Ordering::Acquire);
        while init < len {
            match self.init.compare_exchange_weak(
                init,
                init + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(init) => {
                    self.length.fetch_add(1, Ordering::Acquire);
                    let (_, offset_slot) = layout_slot.repeat(len).unwrap();
                    return AllocResult::Alloc(unsafe {
                        NonNull::new_unchecked(self.slots.add(offset_slot * init + offset_t)).cast()
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

        self.length.fetch_add(1, Ordering::Acquire);

        AllocResult::Alloc(unsafe {
            NonNull::new_unchecked(slot.cast::<u8>().add(offset_t)).cast()
        })
    }

    unsafe fn dealloc(
        &self,
        segment: u8,
        (layout_slot, offset_t): (Layout, usize),
        ptr: *mut u8,
    ) -> usize {
        if self.slots.is_null() {
            return usize::MAX;
        }

        let len = 16 << segment;
        let (layout_slots, _) = layout_slot.repeat(len).unwrap();
        let start = self.slots;
        let end = unsafe { self.slots.add(layout_slots.size()) };
        if !(start..end).contains(&ptr) {
            return usize::MAX;
        }

        let ptr = unsafe { ptr.sub(offset_t).cast::<Slot>() };

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

        self.length.fetch_sub(1, Ordering::Acquire) - 1
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
    fn happy_path2() {
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
    fn happy_path3() {
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
}
