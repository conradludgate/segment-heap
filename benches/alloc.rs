use std::alloc::Layout;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use segment_heap::SegmentHeap;

fn alloc(c: &mut Criterion) {
    let mut g = c.benchmark_group("alloc");
    let sizes: Vec<usize> = (1..=8).map(|x| (1 << (x*2)) - 1).collect();
    for size in sizes {
        g.bench_with_input(BenchmarkId::new("segment", size), &size, |b, size| {
            let mut store = Vec::with_capacity(*size);
            b.iter(|| {
                let heap = SegmentHeap::<usize>::new();
                for i in 0..*size {
                    let ptr = heap.alloc();
                    unsafe { ptr.as_ptr().write(i) };
                    store.push(black_box(ptr));
                }
                for ptr in store.drain(..) {
                    unsafe { heap.dealloc(black_box(ptr)) }
                }
                heap
            })
        });

        g.bench_with_input(BenchmarkId::new("global", size), &size, |b, size| {
            let mut store = Vec::with_capacity(*size);
            let layout = Layout::new::<usize>();
            b.iter(|| {
                for i in 0..*size {
                    let ptr = unsafe { std::alloc::alloc(layout) };
                    unsafe { ptr.cast::<usize>().write(i) };
                    store.push(black_box(ptr));
                }
                for ptr in store.drain(..) {
                    unsafe { std::alloc::dealloc(black_box(ptr), layout) }
                }
            })
        });
    }
    g.finish()
}

criterion_group!(benches, alloc);

criterion_main!(benches);
