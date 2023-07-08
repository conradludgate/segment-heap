use std::alloc::Layout;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use segment_heap::SegmentHeap;

fn alloc(c: &mut Criterion) {
    const SIZES: [usize; 4] = [(1 << 4) - 1, (1 << 8) - 1, (1 << 12) - 1, (1 << 16) - 1];
    for size in SIZES {
        c.bench_with_input(BenchmarkId::new("segment-heap", size), &size, |b, size| {
            let mut store = Vec::with_capacity(*size);
            b.iter(|| {
                let heap = SegmentHeap::<usize>::new();
                for _ in 0..*size {
                    store.push(black_box(heap.alloc()));
                }
                for ptr in store.drain(..) {
                    unsafe { heap.dealloc(black_box(ptr)) }
                }
                heap
            })
        });

        c.bench_with_input(BenchmarkId::new("global", size), &size, |b, size| {
            let mut store = Vec::with_capacity(*size);
            let layout = Layout::new::<usize>();
            b.iter(|| {
                for _ in 0..*size {
                    unsafe { store.push(black_box(std::alloc::alloc(layout))) };
                }
                for ptr in store.drain(..) {
                    unsafe { std::alloc::dealloc(black_box(ptr), layout) }
                }
            })
        });
    }
}

criterion_group!(benches, alloc);

criterion_main!(benches);
