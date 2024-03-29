//! This crate implements a static/fixed size indexing data structure for two dimensional axis
//! aligned bounding boxes. The index allows for fast construction and fast querying but cannot be
//! modified after creation.
//!
//! 2D axis aligned bounding boxes are represented by two extent points (four values):
//! (min_x, min_y), (max_x, max_y).
//!
//! This is a port of the [flatbush](https://github.com/mourner/flatbush) javascript library.
//!
//! By default no unsafe code is used (`#![forbid(unsafe_code)]` is applied). Some unsafe
//! optimizations can be enabled by toggling on the `unsafe_optimizations` flag.
//!
//! # Examples
//! ```
//! use static_aabb2d_index::*;
//! // create builder for index containing 4 axis aligned bounding boxes
//! // index also supports integers and custom types that implement the IndexableNum trait
//! let mut builder: StaticAABB2DIndexBuilder<f64> = StaticAABB2DIndexBuilder::new(4);
//! // add bounding boxes to the index
//! // add takes in (min_x, min_y, max_x, max_y) of the bounding box
//! builder.add(0.0, 0.0, 2.0, 2.0);
//! builder.add(-1.0, -1.0, 3.0, 3.0);
//! builder.add(0.0, 0.0, 1.0, 3.0);
//! builder.add(4.0, 2.0, 16.0, 8.0);
//! // note build() may return an error if the number of added boxes does not equal the static size
//! // given at the time the builder was created or the type used fails to cast to a f64
//! let index: StaticAABB2DIndex<f64> = builder.build().unwrap();
//! // query the created index (min_x, min_y, max_x, max_y)
//! let query_results = index.query(-1.0, -1.0, -0.5, -0.5);
//! // query_results holds the index positions of the boxes that overlap with the box given
//! // (positions are according to the order boxes were added the index builder)
//! assert_eq!(query_results, vec![1]);
//! // the query may also be done with a visiting function that can stop the query early
//! let mut visited_results: Vec<usize> = Vec::new();
//! let mut visitor = |box_added_pos: usize| -> Control<()> {
//!     visited_results.push(box_added_pos);
//!     // return continue to continue visiting results, break to stop early
//!     Control::Continue
//! };
//!
//! index.visit_query(-1.0, -1.0, -0.5, -0.5, &mut visitor);
//! assert_eq!(visited_results, vec![1]);
//! ```

#![cfg_attr(not(feature = "unsafe_optimizations"), forbid(unsafe_code))]

extern crate num_traits;
mod core;
mod static_aabb2d_index;
pub use crate::core::*;
pub use crate::static_aabb2d_index::*;
