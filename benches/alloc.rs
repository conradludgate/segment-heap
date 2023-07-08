use std::alloc::{Layout, System, GlobalAlloc};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use segment_heap::SegmentHeap;

type BigType = [u8; 1024];

fn alloc(c: &mut Criterion) {
    let mut g = c.benchmark_group("alloc");
    let sizes: Vec<usize> = (4..=12).map(|x| (1 << x) - 1).collect();
    for size in &sizes {
        g.bench_with_input(BenchmarkId::new("segment", size), size, |b, size| {
            let mut store = Vec::with_capacity(*size);
            b.iter(|| {
                let heap = SegmentHeap::<BigType>::new();
                for _ in 0..*size {
                    let ptr = heap.alloc();
                    store.push(black_box(ptr));
                }
                for ptr in store.drain(..) {
                    unsafe { heap.dealloc(black_box(ptr)) }
                }
                heap
            })
        });
    }
    for size in &sizes {
        g.bench_with_input(BenchmarkId::new("global", size), size, |b, size| {
            let mut store = Vec::with_capacity(*size);
            let layout = Layout::new::<BigType>();
            b.iter(|| {
                let heap = System;
                for _ in 0..*size {
                    let ptr = unsafe { heap.alloc(layout) };
                    store.push(black_box(ptr));
                }
                for ptr in store.drain(..) {
                    unsafe { heap.dealloc(black_box(ptr), layout) }
                }
                heap
            })
        });
    }
    g.finish()
}

criterion_group!(benches, alloc);

criterion_main!(benches);
