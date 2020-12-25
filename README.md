# StaticAABB2DIndex
## Summary
Fast static spatial index data structure for 2D axis aligned bounding boxes utilizing hilbert curve spatial ordering. This is a rust port of the excellent [flatbush](https://github.com/mourner/flatbush) javascript library.

---

## Quick Links
[Crate](https://crates.io/crates/static_aabb2d_index)

[Documentation](https://docs.rs/static_aabb2d_index/latest/static_aabb2d_index/)

[Examples](/examples)

---

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
 // given at the time the builder was created or the type used fails to cast to/from a u16
 let index: StaticAABB2DIndex<f64> = builder.build().unwrap();
 // query the created index (min_x, min_y, max_x, max_y)
 let query_results = index.query(-1.0, -1.0, -0.5, -0.5);
 // query_results holds the index positions of the boxes that overlap with the box given
 // (positions are according to the order boxes were added the index builder)
 assert_eq!(query_results, vec![1]);
 // the query may also be done with a visiting function that can stop the query early
 let mut visited_results: Vec<usize> = Vec::new();
 let mut visitor = |box_added_pos: usize| -> bool {
     visited_results.push(box_added_pos);
     // return true to continue visiting results, false to stop early
     true
 };

 index.visit_query(-1.0, -1.0, -0.5, -0.5, &mut visitor);
 assert_eq!(visited_results, vec![1]);
 ```
