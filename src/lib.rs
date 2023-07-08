use std::{
    alloc::{handle_alloc_error, Layout},
    mem::MaybeUninit,
    ptr::{addr_of, addr_of_mut, NonNull},
    sync::{
        atomic::{AtomicPtr, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use arc_swap::ArcSwapOption;

pub struct SegmentHeap<T> {
    inner: ArcSwapOption<SegmentHeapInner<T>>,
    alloc_lock: Arc<Mutex<()>>,
}

impl<T> Default for SegmentHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SegmentHeap<T> {
    pub fn new() -> Self {
        Self {
            inner: ArcSwapOption::new(None),
            alloc_lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn alloc(&self) -> NonNull<T> {
        loop {
            let inner = self.inner.load_full();
            match inner.as_ref().and_then(|x| x.alloc()) {
                Some(p) => return p,
                None => {
                    match self.alloc_lock.try_lock() {
                        // we have the alloc lock. we can allocate the segment
                        Ok(_lock) => {
                            let cap = inner.as_ref().map_or(1, |x| x.head.slots.len() * 2);
                            let new = SegmentHeapInner {
                                head: SingleSegment::new(cap),
                                tail: ArcSwapOption::new(inner),
                            };
                            self.inner.swap(Some(Arc::new(new)));
                        }
                        // someone else is allocating the new segment
                        Err(_) => {
                            // wait
                            let _lock = self.alloc_lock.lock().unwrap();
                        }
                    }
                }
            }
        }
    }

    /// # Safety
    /// Think about it
    pub unsafe fn dealloc(&self, ptr: NonNull<T>) {
        let layout_t = Layout::new::<T>();
        let layout_next = Layout::new::<*mut Slot<T>>();
        let (layout_slot, offset) = layout_next
            .extend(layout_t)
            .expect("this layout should match Slot<T>");

        debug_assert_eq!(layout_slot, Layout::new::<Slot<T>>());

        let slot_ptr = ptr.as_ptr().cast::<u8>().sub(offset).cast::<Slot<T>>();
        debug_assert_eq!(addr_of_mut!((*slot_ptr).value).cast(), ptr.as_ptr());
        unsafe { self.inner.load().as_ref().unwrap_unchecked() }.dealloc(slot_ptr);
    }
}

struct SegmentHeapInner<T> {
    head: SingleSegment<T>,
    tail: ArcSwapOption<SegmentHeapInner<T>>,
}

impl<T> SegmentHeapInner<T> {
    fn alloc(&self) -> Option<NonNull<T>> {
        // we don't fall back to tail. we stop using the tail to allocate as it's slower
        // eventually we deallocate it
        let x = self.head.alloc()?;
        Some(x)
    }

    /// # Safety:
    /// pointer must be allocated from this, or the tail
    unsafe fn dealloc(&self, ptr: *mut Slot<T>) -> usize {
        match self.head.dealloc(ptr) {
            usize::MAX => {
                if let Some(tail) = &*self.tail.load() {
                    if tail.dealloc(ptr) == 0 {
                        self.tail.store(tail.tail.load_full());
                    }
                } else if cfg!(debug_assertions) {
                    panic!("pointer not allocated in this segment heap")
                }
                usize::MAX
            }
            len => len,
        }
    }
}

struct SingleSegment<T> {
    slots: NonNull<[Slot<T>]>,
    head: AtomicPtr<Slot<T>>,
    length: AtomicUsize,
}

impl<T> Drop for SingleSegment<T> {
    fn drop(&mut self) {
        debug_assert_eq!(self.length.load(Ordering::SeqCst), 0);

        let layout_slots = Layout::array::<Slot<T>>(self.slots.len()).unwrap();
        unsafe {
            std::alloc::dealloc(self.slots.as_ptr().cast(), layout_slots);
        }
    }
}

impl<T> SingleSegment<T> {
    fn new(len: usize) -> Self {
        let layout_slots = Layout::array::<Slot<T>>(len).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout_slots) };
        let ptr = std::ptr::slice_from_raw_parts_mut(ptr.cast::<Slot<T>>(), len);
        let Some(slots) = NonNull::new(ptr) else { handle_alloc_error(layout_slots) };

        let first_slot = slots.as_ptr().cast::<Slot<T>>();
        unsafe {
            first_slot.write(Slot {
                next: std::ptr::null_mut(),
                value: MaybeUninit::uninit(),
            });
        }
        let mut last_slot = first_slot;
        for i in 1..len {
            unsafe {
                let this_slot = first_slot.add(i);
                this_slot.write(Slot {
                    next: last_slot,
                    value: MaybeUninit::uninit(),
                });
                last_slot = this_slot;
            }
        }

        Self {
            slots,
            head: AtomicPtr::new(last_slot),
            length: AtomicUsize::new(0),
        }
    }

    unsafe fn dealloc(&self, ptr: *mut Slot<T>) -> usize {
        let start = self.slots.as_ptr() as *mut Slot<T>;
        let end = unsafe { start.add(self.slots.len()) };
        if !(start..end).contains(&ptr) {
            return usize::MAX;
        }

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

    fn alloc(&self) -> Option<NonNull<T>> {
        let mut head = self.head.load(Ordering::Acquire);
        let slot = loop {
            if head.is_null() {
                return None;
            }

            debug_assert!(
                {
                    let start = self.slots.as_ptr() as *mut Slot<T>;
                    let end = unsafe { start.add(self.slots.len()) };
                    (start..end).contains(&head)
                },
                "the non-null free-list head should point to inside our slots"
            );

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

        unsafe {
            let slot_value: *mut MaybeUninit<T> = addr_of_mut!((*slot).value);
            Some(NonNull::new_unchecked(slot_value as *mut T))
        }
    }
}

#[repr(C)]
struct Slot<T> {
    /// The next free slot after this one
    next: *mut Slot<T>,
    /// The value
    value: MaybeUninit<T>,
}

#[cfg(test)]
mod tests {
    use rand::{seq::SliceRandom, thread_rng};

    use crate::SegmentHeap;

    #[cfg(miri)]
    const N: usize = (1 << 4) - 1;
    #[cfg(not(miri))]
    const N: usize = (1 << 10) - 1;

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
