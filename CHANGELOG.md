# static_aabb2d_index changelog

All notable changes to the static_aabb2d_index crate will be documented in this file.

## Unreleased

### Added ⭐

- BREAKING: Added `total_cmp` method to `IndexableNum` trait to eliminate using `partial_cmp` and
  unwrap which can panic if NaN is present (was used for nearest neighbors query). If implementing
  `IndexableNum` for your own type you must implement `total_cmp` for your type).
- Added CHANGELOG.md to the project for tracking changes.

### Changed 🔧

- Use `std::cmp::min` and `std::cmp::max` in implementing `IndexableNum` for integers (ensures use
  of any available compiler intrinsic for optimizations).
- Use `f32::min`, `f32::max`, `f64::min`, and `f64::max` in implementing `IndexableNum` (ensures
  use of any available compiler intrinsic for optimizations).
