# static_aabb2d_index changelog

All notable changes to the static_aabb2d_index crate will be documented in this file.

## 1.1.0 - 2023-08-31

### Changed 🔧

- Bumped rust edition to 2021.
- Changed internal Vecs to boxed slices (smaller size, and makes clear they don't change in size).
- Internal code improvements for clarity.
- Performance improvement when building an index with large number of item (8-10% measured for
  1_000_000 items).
- Avoid some allocations when building an index by determining the exact length of the
  `level_bounds` array before constructing it (performance improvement).
- Added uninitialized memory optimization when the feature flag `unsafe_optimizations` is on. This
  further improves performance when building an index by avoiding zeroing the array of AABBs
  allocated for the all the boxes. Care was taken to avoid undefined behavior due to reading from or
  creating references to any uninitialized memory.

## 1.0.0 - 2023-03-02

### Added ⭐

- Added `item_indices` method to get a slice over the indices for all item boxes added.

### Changed 🔧

- ⚠️ BREAKING: Index now supports being empty (no longer errors when building the index if item
  count is 0). When the index is empty all queries will yield no results.
- ⚠️ BREAKING: `min_x`, `min_y`, `max_x`, and `max_y` functions on index replaced with single
  `bounds` function which returns the total bounds as an `AABB` or `None` if index item count is 0.
- ⚠️ BREAKING: fixed inconsistency in `visit_query` function to return break result the same as the
  `visit_query_with_stack` function.
- ⚠️ BREAKING: fixed inconsistency in `visit_neighbors` function to return break result the same as
  the `visit_neighbors_with_queue` function.
- ⚠️ BREAKING: renamed `map_all_boxes_index` function to `all_box_indices` and changed signature to
  return a slice rather than indexing into a slice internally.
- Improved doc comments.

## 0.7.1 - 2023-02-22

### Changed 🔧

- Removed unsafe optimization involving uninitialized memory, the code did not strictly uphold the
  invariants required of a `Vec` at all times which could lead to undefined behavior. To properly
  perform this optimization will require more pointer manipulation spread across the code or new
  APIs from the Rust standard library. Index bounds checking is still toggled by the
  `unsafe_optimizations` feature.
- INTERNAL: replaced `get_at_index` and `set_at_index` macros with simple inlined functions and
  simplified some function signatures to use slices rather than `Vec`.

## 0.7.0 - 2023-02-18

### Added ⭐

- ⚠️ BREAKING: Added `total_cmp` method to `IndexableNum` trait to eliminate using `partial_cmp` and
  unwrap which can panic if NaN is present (was used for nearest neighbors query). If implementing
  `IndexableNum` for your own type you must implement `total_cmp` for your type).
- Added `forbid(unsafe_code)` attribute to crate if `unsafe_optimizations` feature is not enabled.
  This ensures there is no unsafe code used in this crate unless `unsafe_optimizations` feature is
  turned on.
- Added CHANGELOG.md to the project for tracking changes.

### Changed 🔧

- BREAKING: renamed feature `allow_unsafe` to `unsafe_optimizations`.
- Use `std::cmp::min` and `std::cmp::max` in implementing `IndexableNum` for integers (ensures use
  of any available compiler intrinsic for optimizations).
- Use `f32::min`, `f32::max`, `f64::min`, and `f64::max` in implementing `IndexableNum` (ensures
  use of any available compiler intrinsic for optimizations).
