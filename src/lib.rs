#![feature(alloc_layout_extra, const_alloc_layout, pointer_is_aligned)]
#![allow(clippy::never_loop, clippy::while_immutable_condition)]

use std::{
    alloc::{handle_alloc_error, Layout},
    cell::Cell,
    convert::identity,
    marker::PhantomData,
    num::NonZeroUsize,
    ptr::{addr_of, addr_of_mut, NonNull},
    sync::{
        atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering},
        OnceLock, RwLock,
    },
    thread::{self, Thread, ThreadId},
};

use nix::{
    sys::mman::{mmap, munmap, MapFlags, ProtFlags},
    unistd::{sysconf, SysconfVar},
};

pub struct SegmentHeap<T> {
    inner: [Option<NonNull<SingleSegment>>; 32],
    // // bit mask of allocated segments
    // segments: AtomicU32,
    // page_size: usize,
    _t: PhantomData<T>,
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
        Self {
            inner: [None; 32],
            _t: PhantomData,
        }
    }

    pub fn alloc(&mut self) -> NonNull<T> {
        let last = self
            .inner
            .iter_mut()
            .enumerate()
            .filter_map(|(i, x)| Some((i, x.as_mut()?)))
            .last();
        let next = if let Some((i, segment)) = last {
            match unsafe { SingleSegment::alloc(*segment, Self::state(i as u8)) } {
                AllocResult::Alloc(p) => return p.cast(),
                AllocResult::Full => i + 1,
                AllocResult::Uninit => todo!(),
            }
        } else {
            0
        };
        self.alloc_segment(next)
    }

    #[cold]
    fn alloc_segment(&mut self, next: usize) -> NonNull<T> {
        let (segment, p) = SingleSegment::init(Self::state(next as u8));
        self.inner[next] = Some(segment);
        p.cast()
    }

    /// # Safety
    /// Think about it
    pub unsafe fn dealloc(ptr: NonNull<T>) {
        SingleSegment::dealloc(ptr.as_ptr().cast())
    }
}

enum AllocResult {
    Alloc(NonNull<()>),
    Full,
    Uninit,
}

struct StateSegment {
    layout_slot: Layout,
    layout_segment: Layout,
    offset_slot: usize,
    offset_slots: usize,
    slots: usize,
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

        let layout_heading = Layout::new::<SingleSegment>();
        let padding = layout_heading.padding_needed_for(layout_slot.align());
        let offset_slots = layout_heading.size() + padding;
        let slot_size = layout_slot.size() + layout_slot.padding_needed_for(layout_slot.align());

        // first page should be at least 64kib
        let size = 65536_usize << segment;
        let remaining = size.saturating_sub(offset_slots);

        let slots = (remaining + slot_size - 1) / slot_size;

        let layout_segment = unsafe {
            Layout::from_size_align_unchecked(
                slots * slot_size + offset_slots,
                max_usize(layout_heading.align(), layout_slot.align()),
            )
        };

        Self {
            layout_slot,
            layout_segment,
            offset_slot: slot_size,
            offset_slots,
            slots,
        }
    }

    // fn alignment(&self, page_size: usize) -> usize {
    //     // let page_size =
    //     //     PAGE_SIZE.get_or_init(|| sysconf(SysconfVar::PAGE_SIZE).unwrap().unwrap_or(4096));
    //     self.layout_slot.align().saturating_sub(page_size)
    // }
}

struct SingleSegment {
    thread_id: ThreadId,
    // padding until the slots start
    padding: usize,
    // slots current initialised
    bump: Cell<usize>,
    // the current free list
    free: Cell<*mut Slot>,
    // the fast thread local free list
    local_free: Cell<*mut Slot>,
    // the slower global free list
    thread_free: AtomicPtr<Slot>,

    // slots currently occupied
    used: Cell<usize>,
    // slots waiting to be freed from other threads
    thread_freed: AtomicUsize,
}
const FOUR_MIB: usize = 4 << 20;

impl SingleSegment {
    // const fn new() -> Self {
    //     Self {
    //         slots: core::ptr::null_mut(),
    //         head: AtomicPtr::new(core::ptr::null_mut()),
    //         length: AtomicUsize::new(0),
    //         init: AtomicUsize::new(0),
    //     }
    // }

    #[cold]
    fn init(state: StateSegment) -> (NonNull<SingleSegment>, NonNull<()>) {
        // 2TiB start
        static PAGE: AtomicUsize = AtomicUsize::new(2 << 40);

        let blocks = (state.layout_segment.size() + FOUR_MIB - 1) / FOUR_MIB;
        let hint = PAGE.fetch_add(blocks * FOUR_MIB, Ordering::Relaxed);

        let ptr = unsafe {
            mmap(
                NonZeroUsize::new(hint),
                NonZeroUsize::new(state.layout_segment.size()).unwrap(),
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_ANONYMOUS | MapFlags::MAP_PRIVATE,
                -1,
                0,
            )
            .unwrap_or_else(|e| {
                dbg!(e);
                handle_alloc_error(state.layout_segment)
            })
        };

        if ptr.is_null() || !ptr.is_aligned_to(FOUR_MIB) {
            handle_alloc_error(state.layout_segment)
        }

        let header = ptr.cast::<Self>();
        unsafe {
            addr_of_mut!((*header).thread_id).write(thread::current().id());
            addr_of_mut!((*header).padding).write(state.offset_slots);
            addr_of_mut!((*header).free).write(Cell::new(core::ptr::null_mut()));
            addr_of_mut!((*header).local_free).write(Cell::new(core::ptr::null_mut()));
            addr_of_mut!((*header).thread_free).write(AtomicPtr::new(core::ptr::null_mut()));
            addr_of_mut!((*header).bump).write(Cell::new(1));
            addr_of_mut!((*header).used).write(Cell::new(1));
            addr_of_mut!((*header).thread_freed).write(AtomicUsize::new(0));
        }

        unsafe {
            (
                NonNull::new_unchecked(header).cast(),
                NonNull::new_unchecked(header.add(state.offset_slots)).cast(),
            )
        }
    }

    #[cold]
    fn release(this: NonNull<Self>, state: StateSegment) {
        unsafe { munmap(this.as_ptr().cast(), state.layout_segment.size()).unwrap() };
    }

    #[inline]
    unsafe fn alloc(this: NonNull<Self>, state: StateSegment) -> AllocResult {
        let padding = *addr_of!((*this.as_ptr()).padding);
        let used_ptr = &*addr_of!((*this.as_ptr()).used);
        let bump_ptr = &*addr_of!((*this.as_ptr()).bump);
        let bump = bump_ptr.get();
        if bump < state.slots {
            used_ptr.set(used_ptr.get() + 1);
            bump_ptr.set(bump + 1);
            return AllocResult::Alloc(NonNull::new_unchecked(
                this.as_ptr()
                    .cast::<u8>()
                    .add(padding + bump * state.offset_slot)
                    .cast(),
            ));
        }

        let free_ptr = &*addr_of!((*this.as_ptr()).free);
        let block = free_ptr.get();
        if block.is_null() {
            return Self::generic_alloc(this);
        }
        free_ptr.set((*block).next);
        used_ptr.set(used_ptr.get() + 1);
        AllocResult::Alloc(unsafe { NonNull::new_unchecked(block.cast()) })
    }

    #[cold]
    unsafe fn generic_alloc(this: NonNull<Self>) -> AllocResult {
        let used_ptr = &*addr_of!((*this.as_ptr()).used);
        let free_ptr = &*addr_of!((*this.as_ptr()).free);
        let local_free_ptr = &*addr_of!((*this.as_ptr()).local_free);
        let thread_free_ptr = &*addr_of!((*this.as_ptr()).thread_free);
        let thread_freed_ptr = &*addr_of!((*this.as_ptr()).thread_freed);
        debug_assert!(free_ptr.get().is_null());

        let thread_free = thread_free_ptr.swap(std::ptr::null_mut(), Ordering::Relaxed);

        let mut current = thread_free;
        let mut last = std::ptr::null_mut();
        let mut count = 0;
        while !current.is_null() {
            last = current;
            count += 1;
            current = (*current).next;
        }
        if thread_free.is_null() {
            free_ptr.set(local_free_ptr.get())
        } else {
            debug_assert!(!last.is_null());
            debug_assert!((*last).next.is_null());
            (*last).next = local_free_ptr.get();
            free_ptr.set(thread_free);
        }
        local_free_ptr.set(std::ptr::null_mut());

        used_ptr.set(used_ptr.get() - count);
        thread_freed_ptr.fetch_sub(count, Ordering::Relaxed);

        let block = free_ptr.get();
        if block.is_null() {
            return AllocResult::Full;
        }
        free_ptr.set((*block).next);
        used_ptr.set(used_ptr.get() + 1);
        AllocResult::Alloc(unsafe { NonNull::new_unchecked(block.cast()) })
    }

    #[inline]
    unsafe fn dealloc(ptr: *mut u8) {
        let offset_down = (ptr as usize) % FOUR_MIB;
        let this = unsafe { ptr.sub(offset_down) }.cast::<Self>();

        let thread_id = *addr_of!((*this).thread_id);
        let used_ptr = &*addr_of!((*this).used);
        let local_free_ptr = &*addr_of!((*this).local_free);
        let thread_free_ptr = &*addr_of!((*this).thread_free);
        let thread_freed_ptr = &*addr_of!((*this).thread_freed);

        let slot = ptr.cast::<Slot>();
        if thread_id == thread::current().id() {
            (*slot).next = local_free_ptr.get();
            local_free_ptr.set(slot);
            used_ptr.set(used_ptr.get() - 1);
        } else {
            let mut next = thread_free_ptr.load(Ordering::Acquire);

            loop {
                unsafe {
                    addr_of_mut!((*slot).next).write(next);
                }
                match thread_free_ptr.compare_exchange_weak(
                    next,
                    slot,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        thread_freed_ptr.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                    Err(n) => next = n,
                }
            }
        }
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
        let mut heap = SegmentHeap::<usize>::new();
        let mut ptrs = vec![];
        for _ in 0..N {
            ptrs.push(heap.alloc());
        }

        for ptr in ptrs.into_iter().rev() {
            unsafe {
                SegmentHeap::dealloc(ptr);
            }
        }
    }

    #[test]
    fn dealloc_fifo() {
        let mut heap = SegmentHeap::<usize>::new();
        let mut ptrs = vec![];
        for _ in 0..N {
            ptrs.push(heap.alloc());
        }

        for ptr in ptrs {
            unsafe {
                SegmentHeap::dealloc(ptr);
            }
        }
    }

    #[test]
    fn dealloc_random() {
        let mut heap = SegmentHeap::<usize>::new();
        let mut ptrs = vec![];
        for _ in 0..N {
            ptrs.push(heap.alloc());
        }

        ptrs.shuffle(&mut thread_rng());
        for ptr in ptrs {
            unsafe {
                SegmentHeap::dealloc(ptr);
            }
        }
    }

    #[test]
    fn size() {
        assert_eq!(std::mem::size_of::<SegmentHeap<()>>(), 256);
    }

    // #[repr(align(8192))]
    // struct LargeAlign([u8; 8192]);

    // #[test]
    // fn large_alignment() {
    //     let mut heap = SegmentHeap::<LargeAlign>::new();
    //     let mut ptrs = vec![];
    //     for _ in 0..N {
    //         ptrs.push(heap.alloc());
    //     }

    //     for ptr in ptrs.into_iter().rev() {
    //         unsafe {
    //             SegmentHeap::dealloc(ptr);
    //         }
    //     }
    // }
}
