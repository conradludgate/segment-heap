# Segment Heap

This is a typed segment heap, intending to offer much more efficient allocations when types are known

# API

```rust
let heap = SegmentHeap::<MyType>::new();

// ptr is uninit
let ptr = heap.alloc();

unsafe { heap.dealloc(ptr); }
```

