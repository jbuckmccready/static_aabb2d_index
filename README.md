# StaticAABB2DIndex

[![Build Status](https://github.com/jbuckmccready/static_aabb2d_index/actions/workflows/ci.yml/badge.svg)](https://github.com/jbuckmccready/static_aabb2d_index/actions)
[![Crates.io](https://img.shields.io/crates/v/static_aabb2d_index.svg)](https://crates.io/crates/static_aabb2d_index)
[![Docs.rs](https://docs.rs/static_aabb2d_index/badge.svg)](https://docs.rs/static_aabb2d_index)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![Apache](https://img.shields.io/badge/license-Apache-blue.svg)](LICENSE-APACHE)

---

Fast static spatial index data structure for 2D axis aligned bounding boxes utilizing hilbert curve
spatial ordering. This is a rust port (with changes to sorting algorithm and Rust friendly API)
of the excellent [flatbush](https://github.com/mourner/flatbush) javascript library.

By default no unsafe code is used (`#![forbid(unsafe_code)]` is applied). Some unsafe optimizations
can be enabled by toggling on the `unsafe_optimizations` flag. Note the API is still safe when this
flag is enabled, all optimizations are internal to the library. Currently the unsafe code is used
to eliminate slice bounds checking and utilize uninitialized memory to avoid zeroing arrays when
allocated.

## Quick Code Example

```rust
use static_aabb2d_index::*;
// create builder for index containing 4 axis aligned bounding boxes
// index also supports integers and custom types that implement the IndexableNum trait
let mut builder: StaticAABB2DIndexBuilder<f64> = StaticAABB2DIndexBuilder::new(4);
// add bounding boxes to the index
// add takes in (min_x, min_y, max_x, max_y) of the bounding box
builder.add(0.0, 0.0, 2.0, 2.0);
builder.add(-1.0, -1.0, 3.0, 3.0);
builder.add(0.0, 0.0, 1.0, 3.0);
builder.add(4.0, 2.0, 16.0, 8.0);
// note build may return an error if the number of added boxes does not equal the static size
// given at the time the builder was created or the type used fails to cast to a f64
let index: StaticAABB2DIndex<f64> = builder.build().unwrap();
// query the created index (min_x, min_y, max_x, max_y)
let query_results = index.query(-1.0, -1.0, -0.5, -0.5);
// query_results holds the index positions of the boxes that overlap with the box given
// (positions are according to the order boxes were added the index builder)
assert_eq!(query_results, vec![1]);
// the query may also be done with a visiting function that can stop the query early
let mut visited_results: Vec<usize> = Vec::new();
let mut visitor = |box_added_pos: usize| -> Control<()> {
    visited_results.push(box_added_pos);
    // return continue to continue visiting results, break to stop early
    Control::Continue
};

index.visit_query(-1.0, -1.0, -0.5, -0.5, &mut visitor);
assert_eq!(visited_results, vec![1]);
```

## License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license
  ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
