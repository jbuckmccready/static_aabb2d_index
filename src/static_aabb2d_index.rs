use fmt::Debug;
use num_traits::ToPrimitive;
use std::fmt;
use std::{cmp::min, collections::BinaryHeap};

use crate::{AABB, ControlFlow, IndexableNum, NeighborVisitor, QueryVisitor, try_control};

/// Error type for errors that may be returned in attempting to build the index.
#[derive(Debug, PartialEq)]
pub enum StaticAABB2DIndexBuildError {
    /// Error for the case when the number of items added does not match the size given at
    /// construction.
    ItemCountError {
        /// The number of items that were added.
        added: usize,
        /// The number of items that were expected (set at construction).
        expected: usize,
    },
    /// Error for the case when the numeric type T used for the index fails to cast to f64.
    NumericCastError,
}

impl std::error::Error for StaticAABB2DIndexBuildError {}

impl fmt::Display for StaticAABB2DIndexBuildError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StaticAABB2DIndexBuildError::ItemCountError { added, expected } => write!(
                f,
                "added item count should equal static size given to builder \
                (added: {}, expected: {})",
                added, expected
            ),
            StaticAABB2DIndexBuildError::NumericCastError => {
                write!(f, "numeric type T used for index failed to cast to f64")
            }
        }
    }
}

/// Used to build a [StaticAABB2DIndex].
#[derive(Debug, Clone)]
pub struct StaticAABB2DIndexBuilder<T = f64>
where
    T: IndexableNum,
{
    node_size: usize,
    num_items: usize,
    level_bounds: Box<[usize]>,
    #[cfg(feature = "unsafe_optimizations")]
    boxes: Box<[std::mem::MaybeUninit<AABB<T>>]>,
    #[cfg(not(feature = "unsafe_optimizations"))]
    boxes: Box<[AABB<T>]>,
    indices: Box<[usize]>,
    pos: usize,
}

/// Static/fixed size indexing data structure for two dimensional axis aligned bounding boxes.
///
/// The index allows for fast construction and fast querying but cannot be modified after creation.
/// This type is constructed from a [`StaticAABB2DIndexBuilder`].
///
/// 2D axis aligned bounding boxes are represented by two extent points (four values):
/// (min_x, min_y), (max_x, max_y).
///
/// # Examples
/// ```
/// use static_aabb2d_index::*;
/// // create builder for index containing 4 axis aligned bounding boxes
/// // index also supports integers and custom types that implement the IndexableNum trait
/// let mut builder: StaticAABB2DIndexBuilder<f64> = StaticAABB2DIndexBuilder::new(4);
/// // add bounding boxes to the index
/// // add takes in (min_x, min_y, max_x, max_y) of the bounding box
/// builder.add(0.0, 0.0, 2.0, 2.0);
/// builder.add(-1.0, -1.0, 3.0, 3.0);
/// builder.add(0.0, 0.0, 1.0, 3.0);
/// builder.add(4.0, 2.0, 16.0, 8.0);
/// // note build may return an error if the number of added boxes does not equal the static size
/// // given at the time the builder was created or the type used fails to cast to a f64
/// let index: StaticAABB2DIndex<f64> = builder.build().unwrap();
/// // query the created index (min_x, min_y, max_x, max_y)
/// let query_results = index.query(-1.0, -1.0, -0.5, -0.5);
/// // query_results holds the index positions of the boxes that overlap with the box given
/// // (positions are according to the order boxes were added the index builder)
/// assert_eq!(query_results, vec![1]);
/// // the query may also be done with a visiting function that can stop the query early
/// let mut visited_results: Vec<usize> = Vec::new();
/// let mut visitor = |box_added_pos: usize| -> Control<()> {
///     visited_results.push(box_added_pos);
///     // return continue to continue visiting results, break to stop early
///     Control::Continue
/// };
///
/// index.visit_query(-1.0, -1.0, -0.5, -0.5, &mut visitor);
/// assert_eq!(visited_results, vec![1]);
/// ```
#[derive(Debug, Clone)]
pub struct StaticAABB2DIndex<T = f64>
where
    T: IndexableNum,
{
    node_size: usize,
    num_items: usize,
    level_bounds: Box<[usize]>,
    boxes: Box<[AABB<T>]>,
    indices: Box<[usize]>,
}

// Helper functions to toggle bounds checking/uninitialized memory handling. NOTE: the functions are
// not marked unsafe to facilitate easy global usages since we never rely on these functions
// throwing a panic (so with unsafe_optimizations feature on we assume correct bounds and
// initialization).
#[cfg(not(feature = "unsafe_optimizations"))]
#[inline(always)]
fn get_at_index<T>(container: &[T], index: usize) -> &T {
    &container[index]
}

#[cfg(feature = "unsafe_optimizations")]
#[inline(always)]
fn get_at_index<T>(container: &[T], index: usize) -> &T {
    unsafe { container.get_unchecked(index) }
}

#[cfg(feature = "unsafe_optimizations")]
#[inline(always)]
fn get_uninit_at_index<T>(container: &[std::mem::MaybeUninit<T>], index: usize) -> T {
    unsafe { container.get_unchecked(index).assume_init_read() }
}

#[cfg(not(feature = "unsafe_optimizations"))]
#[inline(always)]
fn set_at_index<T>(container: &mut [T], index: usize, value: T) {
    container[index] = value;
}

#[cfg(feature = "unsafe_optimizations")]
#[inline(always)]
fn set_at_index<T>(container: &mut [T], index: usize, value: T) {
    unsafe {
        *container.get_unchecked_mut(index) = value;
    }
}

#[cfg(feature = "unsafe_optimizations")]
fn write_uninit_at_index<T>(container: &mut [std::mem::MaybeUninit<T>], index: usize, value: T) {
    unsafe {
        container.get_unchecked_mut(index).write(value);
    }
}

#[cfg(not(feature = "unsafe_optimizations"))]
fn write_uninit_at_index<T>(container: &mut [T], index: usize, value: T) {
    container[index] = value;
}

impl<T> StaticAABB2DIndexBuilder<T>
where
    T: IndexableNum,
{
    fn init(num_items: usize, node_size: usize) -> Self {
        if num_items == 0 {
            // just return early, with no items added
            return StaticAABB2DIndexBuilder {
                node_size,
                num_items,
                level_bounds: Box::new([]),
                boxes: Box::new([]),
                indices: Box::new([]),
                pos: 0,
            };
        }

        let node_size = node_size.clamp(2, 65535);

        let mut n = num_items;
        let level_bounds_len = {
            // keep subdividing num_items by node_size to get length of level bounds array to
            // represent the R-tree (doing this now to get exact allocation required)
            let mut len = 1;
            loop {
                n = (n as f64 / node_size as f64).ceil() as usize;
                len += 1;
                if n == 1 {
                    break;
                }
            }
            len
        };

        // allocate the exact length required for the level bounds and add the level bound index
        // positions and build up total num_nodes for the tree
        n = num_items;
        let mut num_nodes = num_items;
        let mut level_bounds: Vec<usize> = Vec::with_capacity(level_bounds_len);
        level_bounds.push(n);
        loop {
            n = (n as f64 / node_size as f64).ceil() as usize;
            num_nodes += n;
            level_bounds.push(num_nodes);
            if n == 1 {
                break;
            }
        }

        debug_assert_eq!(
            level_bounds.capacity(),
            level_bounds.len(),
            "ensure exact allocation"
        );

        #[cfg(not(feature = "unsafe_optimizations"))]
        let boxes = std::iter::repeat_with(AABB::default)
            .take(num_nodes)
            .collect();

        #[cfg(feature = "unsafe_optimizations")]
        let boxes = std::iter::repeat_with(std::mem::MaybeUninit::uninit)
            .take(num_nodes)
            .collect();

        StaticAABB2DIndexBuilder {
            node_size,
            num_items,
            level_bounds: level_bounds.into_boxed_slice(),
            boxes,
            indices: (0..num_nodes).collect(),
            pos: 0,
        }
    }

    /// Construct a new [`StaticAABB2DIndexBuilder`] to fit exactly the specified `count` number of
    /// items.
    #[inline]
    pub fn new(count: usize) -> Self {
        StaticAABB2DIndexBuilder::init(count, 16)
    }

    /// Construct a new [`StaticAABB2DIndexBuilder`] to fit exactly the specified `count` number of
    /// items and use `node_size` for the index tree shape.
    ///
    /// Each node in the index tree has a maximum size which may be adjusted by `node_size` for
    /// performance reasons, however the default value of 16 when calling
    /// [`StaticAABB2DIndexBuilder::new`] is tested to be optimal in most cases.
    ///
    /// If `node_size` is less than 2 then 2 is used, if `node_size` is greater than 65535 then
    /// 65535 is used.
    #[inline]
    pub fn new_with_node_size(count: usize, node_size: usize) -> Self {
        StaticAABB2DIndexBuilder::init(count, node_size)
    }

    /// Add an axis aligned bounding box with the extent points (`min_x`, `min_y`),
    /// (`max_x`, `max_y`) to the index.
    ///
    /// For performance reasons the sanity checks of `min_x <= max_x` and `min_y <= max_y` are only
    /// debug asserted. If an invalid box is added it may lead to a panic or unexpected behavior
    /// from the constructed [`StaticAABB2DIndex`].
    #[inline]
    pub fn add(&mut self, min_x: T, min_y: T, max_x: T, max_y: T) -> &mut Self {
        // catch adding past num_items (error will be returned when build is called)
        if self.pos >= self.num_items {
            self.pos += 1;
            return self;
        }
        debug_assert!(min_x <= max_x);
        debug_assert!(min_y <= max_y);

        #[cfg(not(feature = "unsafe_optimizations"))]
        set_at_index(
            &mut self.boxes,
            self.pos,
            AABB::new(min_x, min_y, max_x, max_y),
        );

        #[cfg(feature = "unsafe_optimizations")]
        // SAFETY: we checked the index bounds by comparing self.pos with self.num_items already.
        unsafe {
            self.boxes
                .get_unchecked_mut(self.pos)
                .write(AABB::new(min_x, min_y, max_x, max_y));
        }

        self.pos += 1;
        self
    }

    /// Build the [`StaticAABB2DIndex`] with the boxes that have been added.
    ///
    /// If the number of added items does not match the count given at the time the builder was
    /// created then a [`StaticAABB2DIndexBuildError::ItemCountError`] will be returned.
    ///
    /// If the numeric type T fails to cast to a f64 for any reason then a
    /// [`StaticAABB2DIndexBuildError::NumericCastError`] will be returned.
    pub fn build(mut self) -> Result<StaticAABB2DIndex<T>, StaticAABB2DIndexBuildError> {
        if self.pos != self.num_items {
            return Err(StaticAABB2DIndexBuildError::ItemCountError {
                added: self.pos,
                expected: self.num_items,
            });
        }

        if self.num_items == 0 {
            return Ok(StaticAABB2DIndex {
                node_size: self.node_size,
                num_items: self.num_items,
                level_bounds: self.level_bounds,
                boxes: Box::new([]),
                indices: self.indices,
            });
        }

        #[cfg(feature = "unsafe_optimizations")]
        // SAFETY: All the item boxes are initialized (all elements from index 0 to num_items - 1).
        let item_boxes: &mut [AABB<T>] =
            unsafe { std::mem::transmute(&mut self.boxes[0..self.num_items]) };

        #[cfg(not(feature = "unsafe_optimizations"))]
        let item_boxes = &mut self.boxes[0..self.num_items];

        // calculate total bounds
        let mut item_boxes_iter = item_boxes.iter();
        // initialize values with first box
        let first_box = item_boxes_iter.next().unwrap();
        let mut min_x = first_box.min_x;
        let mut min_y = first_box.min_y;
        let mut max_x = first_box.max_x;
        let mut max_y = first_box.max_y;
        // using for_each method on iterator yields noticeable performance improvement (8-10%) for
        // large number of items (1_000_000+ items) instead of using a for loop on the iterator
        item_boxes_iter.for_each(|item| {
            min_x = min_x.min(item.min_x);
            min_y = min_y.min(item.min_y);
            max_x = max_x.max(item.max_x);
            max_y = max_y.max(item.max_y);
        });

        // if number of items is less than node size then skip sorting since each node of boxes must
        // be fully scanned regardless and there is only one node
        if self.num_items <= self.node_size {
            set_at_index(&mut self.indices, self.pos, 0);
            // fill root box with total extents
            write_uninit_at_index(
                &mut self.boxes,
                self.pos,
                AABB::new(min_x, min_y, max_x, max_y),
            );

            #[cfg(feature = "unsafe_optimizations")]
            // SAFETY: All boxes are initialized.
            let boxes: Box<[AABB<T>]> = unsafe { std::mem::transmute(self.boxes) };

            #[cfg(not(feature = "unsafe_optimizations"))]
            let boxes = self.boxes;

            return Ok(StaticAABB2DIndex {
                node_size: self.node_size,
                num_items: self.num_items,
                level_bounds: self.level_bounds,
                boxes,
                indices: self.indices,
            });
        }

        // helper function to cast T to f64
        let cast_to_f64 = |x: T| -> Result<f64, StaticAABB2DIndexBuildError> {
            x.to_f64()
                .ok_or(StaticAABB2DIndexBuildError::NumericCastError)
        };

        let width = cast_to_f64(max_x - min_x)?;
        let height = cast_to_f64(max_y - min_y)?;
        let extent_min_x = cast_to_f64(min_x)?;
        let extent_min_y = cast_to_f64(min_y)?;

        // hilbert max input value for x and y
        let hilbert_max = u16::MAX as f64;
        let scaled_width = hilbert_max / width;
        let scaled_height = hilbert_max / height;

        // helper function to build hilbert coordinate value from AABB
        fn hilbert_coord(scaled_extent: f64, aabb_min: f64, aabb_max: f64, extent_min: f64) -> u16 {
            let value = scaled_extent * (0.5 * (aabb_min + aabb_max) - extent_min);
            // this should successfully convert to u16 since scaled_extent should be between 0 and
            // u16::MAX and the coefficient should be between 0.0 and 1.0, but in the case of
            // positive/negative infinity (width or height is 0.0) or NAN (inputs contain NAN) we
            // want to continue
            value.to_u16().unwrap_or(
                // saturate
                if value > u16::MAX as f64 {
                    u16::MAX
                } else if value < u16::MIN as f64 {
                    u16::MIN
                } else {
                    // NAN
                    0
                },
            )
        }

        // mapping the x and y coordinates of the center of the item boxes to values in the range
        // [0 -> n - 1] such that the min of the entire set of bounding boxes maps to 0 and the max
        // of the entire set of bounding boxes maps to n - 1 our 2d space is x: [0 -> n-1] and
        // y: [0 -> n-1], our 1d hilbert curve value space is d: [0 -> n^2 - 1]
        let mut hilbert_values: Vec<u32> = Vec::with_capacity(self.num_items);
        for aabb in item_boxes.iter() {
            let aabb_min_x = cast_to_f64(aabb.min_x)?;
            let aabb_min_y = cast_to_f64(aabb.min_y)?;
            let aabb_max_x = cast_to_f64(aabb.max_x)?;
            let aabb_max_y = cast_to_f64(aabb.max_y)?;

            let x = hilbert_coord(scaled_width, aabb_min_x, aabb_max_x, extent_min_x);
            let y = hilbert_coord(scaled_height, aabb_min_y, aabb_max_y, extent_min_y);
            hilbert_values.push(hilbert_xy_to_index(x, y));
        }

        // sort items by their Hilbert value for constructing the tree
        sort(
            &mut hilbert_values,
            item_boxes,
            &mut self.indices,
            0,
            self.num_items - 1,
            self.node_size,
        );

        // generate nodes at each tree level, bottom-up
        let mut pos = 0;
        for &level_end in self.level_bounds[0..self.level_bounds.len() - 1].iter() {
            // generate a parent node for each block of consecutive node_size nodes
            while pos < level_end {
                let mut node_min_x = T::max_value();
                let mut node_min_y = T::max_value();
                let mut node_max_x = T::min_value();
                let mut node_max_y = T::min_value();
                let node_index = pos;

                // calculate bounding box for the new node
                let mut j = 0;
                while j < self.node_size && pos < level_end {
                    #[cfg(not(feature = "unsafe_optimizations"))]
                    let aabb = get_at_index(&self.boxes, pos);
                    #[cfg(feature = "unsafe_optimizations")]
                    let aabb = get_uninit_at_index(&self.boxes, pos);
                    pos += 1;
                    node_min_x = T::min(node_min_x, aabb.min_x);
                    node_min_y = T::min(node_min_y, aabb.min_y);
                    node_max_x = T::max(node_max_x, aabb.max_x);
                    node_max_y = T::max(node_max_y, aabb.max_y);
                    j += 1;
                }

                // add the new node to the tree
                set_at_index(&mut self.indices, self.pos, node_index);
                write_uninit_at_index(
                    &mut self.boxes,
                    self.pos,
                    AABB::new(node_min_x, node_min_y, node_max_x, node_max_y),
                );
                self.pos += 1;
            }
        }

        #[cfg(feature = "unsafe_optimizations")]
        // SAFETY: All boxes are initialized.
        let boxes: Box<[AABB<T>]> = unsafe { std::mem::transmute(self.boxes) };

        #[cfg(not(feature = "unsafe_optimizations"))]
        let boxes = self.boxes;

        Ok(StaticAABB2DIndex {
            node_size: self.node_size,
            num_items: self.num_items,
            level_bounds: self.level_bounds,
            boxes,
            indices: self.indices,
        })
    }
}

/// Maps 2d space to 1d hilbert curve space.
///
/// 2d space is `x: [0 -> n-1]` and `y: [0 -> n-1]`, 1d hilbert curve value space is
/// `d: [0 -> n^2 - 1]`, where n = 2^16, so `x` and `y` must be between 0 and [`u16::MAX`]
/// (65535 or 2^16 - 1).
pub fn hilbert_xy_to_index(x: u16, y: u16) -> u32 {
    let x = x as u32;
    let y = y as u32;

    // Fast Hilbert curve algorithm by http://threadlocalmutex.com/
    // Ported from C++ https://github.com/rawrunprotected/hilbert_curves (public domain)
    let mut a_1 = x ^ y;
    let mut b_1 = 0xFFFF ^ a_1;
    let mut c_1 = 0xFFFF ^ (x | y);
    let mut d_1 = x & (y ^ 0xFFFF);

    let mut a_2 = a_1 | (b_1 >> 1);
    let mut b_2 = (a_1 >> 1) ^ a_1;
    let mut c_2 = ((c_1 >> 1) ^ (b_1 & (d_1 >> 1))) ^ c_1;
    let mut d_2 = ((a_1 & (c_1 >> 1)) ^ (d_1 >> 1)) ^ d_1;

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    a_2 = (a_1 & (a_1 >> 2)) ^ (b_1 & (b_1 >> 2));
    b_2 = (a_1 & (b_1 >> 2)) ^ (b_1 & ((a_1 ^ b_1) >> 2));
    c_2 ^= (a_1 & (c_1 >> 2)) ^ (b_1 & (d_1 >> 2));
    d_2 ^= (b_1 & (c_1 >> 2)) ^ ((a_1 ^ b_1) & (d_1 >> 2));

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    a_2 = (a_1 & (a_1 >> 4)) ^ (b_1 & (b_1 >> 4));
    b_2 = (a_1 & (b_1 >> 4)) ^ (b_1 & ((a_1 ^ b_1) >> 4));
    c_2 ^= (a_1 & (c_1 >> 4)) ^ (b_1 & (d_1 >> 4));
    d_2 ^= (b_1 & (c_1 >> 4)) ^ ((a_1 ^ b_1) & (d_1 >> 4));

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    c_2 ^= (a_1 & (c_1 >> 8)) ^ (b_1 & (d_1 >> 8));
    d_2 ^= (b_1 & (c_1 >> 8)) ^ ((a_1 ^ b_1) & (d_1 >> 8));

    a_1 = c_2 ^ (c_2 >> 1);
    b_1 = d_2 ^ (d_2 >> 1);

    let mut i0 = x ^ y;
    let mut i1 = b_1 | (0xFFFF ^ (i0 | a_1));

    i0 = (i0 | (i0 << 8)) & 0x00FF00FF;
    i0 = (i0 | (i0 << 4)) & 0x0F0F0F0F;
    i0 = (i0 | (i0 << 2)) & 0x33333333;
    i0 = (i0 | (i0 << 1)) & 0x55555555;

    i1 = (i1 | (i1 << 8)) & 0x00FF00FF;
    i1 = (i1 | (i1 << 4)) & 0x0F0F0F0F;
    i1 = (i1 | (i1 << 2)) & 0x33333333;
    i1 = (i1 | (i1 << 1)) & 0x55555555;

    (i1 << 1) | i0
}

// modified quick sort that skips sorting boxes within the same node
fn sort<T>(
    values: &mut [u32],
    boxes: &mut [AABB<T>],
    indices: &mut [usize],
    left: usize,
    right: usize,
    node_size: usize,
) where
    T: IndexableNum,
{
    debug_assert!(left <= right);

    if left / node_size >= right / node_size {
        // remaining to be sorted fits within the the same node, skip sorting further
        // since all boxes within a node must be visited when querying regardless
        return;
    }

    let mid = (left + right) / 2;
    let pivot = *get_at_index(values, mid);
    let mut i = left.wrapping_sub(1);
    let mut j = right.wrapping_add(1);

    loop {
        loop {
            i = i.wrapping_add(1);
            if *get_at_index(values, i) >= pivot {
                break;
            }
        }

        loop {
            j = j.wrapping_sub(1);
            if *get_at_index(values, j) <= pivot {
                break;
            }
        }

        if i >= j {
            break;
        }

        swap(values, boxes, indices, i, j);
    }

    sort(values, boxes, indices, left, j, node_size);
    sort(values, boxes, indices, j.wrapping_add(1), right, node_size);
}

#[inline]
fn swap<T>(values: &mut [u32], boxes: &mut [AABB<T>], indices: &mut [usize], i: usize, j: usize)
where
    T: IndexableNum,
{
    values.swap(i, j);
    boxes.swap(i, j);
    indices.swap(i, j);
}

struct QueryIterator<'a, T>
where
    T: IndexableNum,
{
    aabb_index: &'a StaticAABB2DIndex<T>,
    stack: Vec<usize>,
    min_x: T,
    min_y: T,
    max_x: T,
    max_y: T,
    node_index: usize,
    level: usize,
    pos: usize,
    end: usize,
}

impl<'a, T> QueryIterator<'a, T>
where
    T: IndexableNum,
{
    #[inline]
    fn new(
        aabb_index: &'a StaticAABB2DIndex<T>,
        min_x: T,
        min_y: T,
        max_x: T,
        max_y: T,
    ) -> QueryIterator<'a, T> {
        if aabb_index.num_items == 0 {
            // empty index
            return Self {
                aabb_index,
                stack: Vec::new(),
                min_x,
                min_y,
                max_x,
                max_y,
                node_index: 0,
                level: 0,
                pos: 0,
                end: 0,
            };
        }

        let node_index = aabb_index.boxes.len() - 1;
        let pos = node_index;
        let level = aabb_index.level_bounds.len() - 1;
        let end = min(
            node_index + aabb_index.node_size,
            *get_at_index(&aabb_index.level_bounds, level),
        );

        Self {
            aabb_index,
            stack: Vec::with_capacity(16),
            min_x,
            min_y,
            max_x,
            max_y,
            node_index,
            level,
            pos,
            end,
        }
    }
}

impl<T> Iterator for QueryIterator<'_, T>
where
    T: IndexableNum,
{
    type Item = usize;

    // NOTE: The inline attribute here shows significant performance improvements in benchmarks.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.aabb_index.num_items == 0 {
            return None;
        }

        loop {
            while self.pos < self.end {
                let current_pos = self.pos;
                self.pos += 1;

                let aabb = get_at_index(&self.aabb_index.boxes, current_pos);
                if !aabb.overlaps(self.min_x, self.min_y, self.max_x, self.max_y) {
                    // no overlap
                    continue;
                }

                let index = *get_at_index(&self.aabb_index.indices, current_pos);
                if self.node_index < self.aabb_index.num_items {
                    return Some(index);
                }

                self.stack.push(index);
                self.stack.push(self.level - 1);
            }

            if self.stack.len() > 1 {
                self.level = self.stack.pop().unwrap();
                self.node_index = self.stack.pop().unwrap();
                self.pos = self.node_index;
                self.end = min(
                    self.node_index + self.aabb_index.node_size,
                    *get_at_index(&self.aabb_index.level_bounds, self.level),
                );
            } else {
                break;
            }
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.pos >= self.end && self.stack.len() < 2 {
            // iterator exhausted
            (0, Some(0))
        } else {
            // never yields more than the number of items in the index
            (0, Some(self.aabb_index.num_items))
        }
    }
}

struct QueryIteratorStackRef<'a, T>
where
    T: IndexableNum,
{
    aabb_index: &'a StaticAABB2DIndex<T>,
    stack: &'a mut Vec<usize>,
    min_x: T,
    min_y: T,
    max_x: T,
    max_y: T,
    node_index: usize,
    level: usize,
    pos: usize,
    end: usize,
}

impl<'a, T> QueryIteratorStackRef<'a, T>
where
    T: IndexableNum,
{
    #[inline]
    fn new(
        aabb_index: &'a StaticAABB2DIndex<T>,
        stack: &'a mut Vec<usize>,
        min_x: T,
        min_y: T,
        max_x: T,
        max_y: T,
    ) -> QueryIteratorStackRef<'a, T> {
        if aabb_index.num_items == 0 {
            // empty index
            return Self {
                aabb_index,
                stack,
                min_x,
                min_y,
                max_x,
                max_y,
                node_index: 0,
                level: 0,
                pos: 0,
                end: 0,
            };
        }

        let node_index = aabb_index.boxes.len() - 1;
        let pos = node_index;
        let level = aabb_index.level_bounds.len() - 1;
        let end = min(
            node_index + aabb_index.node_size,
            *get_at_index(&aabb_index.level_bounds, level),
        );

        // ensure the stack is empty for use
        stack.clear();

        Self {
            aabb_index,
            stack,
            min_x,
            min_y,
            max_x,
            max_y,
            node_index,
            level,
            pos,
            end,
        }
    }
}

impl<T> Iterator for QueryIteratorStackRef<'_, T>
where
    T: IndexableNum,
{
    type Item = usize;

    // NOTE: The inline attribute here shows significant performance improvements in benchmarks.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.aabb_index.num_items == 0 {
            return None;
        }

        loop {
            while self.pos < self.end {
                let current_pos = self.pos;
                self.pos += 1;

                let aabb = get_at_index(&self.aabb_index.boxes, current_pos);
                if !aabb.overlaps(self.min_x, self.min_y, self.max_x, self.max_y) {
                    // no overlap
                    continue;
                }

                let index = *get_at_index(&self.aabb_index.indices, current_pos);
                if self.node_index < self.aabb_index.num_items {
                    return Some(index);
                }

                self.stack.push(index);
                self.stack.push(self.level - 1);
            }

            if self.stack.len() > 1 {
                self.level = self.stack.pop().unwrap();
                self.node_index = self.stack.pop().unwrap();
                self.pos = self.node_index;
                self.end = min(
                    self.node_index + self.aabb_index.node_size,
                    *get_at_index(&self.aabb_index.level_bounds, self.level),
                );
            } else {
                break;
            }
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.pos >= self.end && self.stack.len() < 2 {
            // iterator exhausted
            (0, Some(0))
        } else {
            // never yields more than the number of items in the index
            (0, Some(self.aabb_index.num_items))
        }
    }
}

/// Type alias for priority queue used for nearest neighbor searches.
///
/// See: [`StaticAABB2DIndex::visit_neighbors_with_queue`].
pub type NeighborPriorityQueue<T> = BinaryHeap<NeighborsState<T>>;

/// Holds state for priority queue used in nearest neighbors query.
///
/// Note this type is public for use in passing in an existing priority queue but
/// all fields and constructor are private for internal use only.
///
/// See also: [`StaticAABB2DIndex::visit_neighbors_with_queue`].
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct NeighborsState<T>
where
    T: IndexableNum,
{
    index: usize,
    is_leaf_node: bool,
    dist: T,
}

impl<T> NeighborsState<T>
where
    T: IndexableNum,
{
    #[inline]
    fn new(index: usize, is_leaf_node: bool, dist: T) -> Self {
        NeighborsState {
            index,
            is_leaf_node,
            dist,
        }
    }
}

impl<T> Eq for NeighborsState<T> where T: IndexableNum {}

impl<T> Ord for NeighborsState<T>
where
    T: IndexableNum,
{
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // flip ordering (compare other to self rather than self to other) to prioritize minimum
        // dist in priority queue
        other.dist.total_cmp(&self.dist)
    }
}

impl<T> PartialOrd for NeighborsState<T>
where
    T: IndexableNum,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> StaticAABB2DIndex<T>
where
    T: IndexableNum,
{
    /// Gets the total bounds of all the items that were added to the index or `None` if the index
    /// had no items added in construction (item count is 0).
    #[inline]
    pub fn bounds(&self) -> Option<AABB<T>> {
        self.boxes.last().copied()
    }

    /// Gets the total count of items that were added to the index during construction.
    #[inline]
    pub fn count(&self) -> usize {
        self.num_items
    }

    /// Queries the index, returning a collection of indices to items that overlap with the bounding
    /// box given.
    ///
    /// `min_x`, `min_y`, `max_x`, and `max_y` represent the bounding box to use for the query.
    /// Indexes returned match with the order items were added to the index using
    /// [`StaticAABB2DIndexBuilder::add`].
    #[inline]
    pub fn query(&self, min_x: T, min_y: T, max_x: T, max_y: T) -> Vec<usize> {
        let mut results = Vec::new();
        let mut visitor = |i| {
            results.push(i);
        };
        self.visit_query(min_x, min_y, max_x, max_y, &mut visitor);
        results
    }

    /// The same as [`StaticAABB2DIndex::query`] but instead of returning a [`Vec`] of results a
    /// iterator is returned which yields the results by lazily querying the index.
    ///
    /// # Examples
    /// ```
    /// use static_aabb2d_index::*;
    /// let mut builder = StaticAABB2DIndexBuilder::new(4);
    /// builder
    ///     .add(0.0, 0.0, 2.0, 2.0)
    ///     .add(-1.0, -1.0, 3.0, 3.0)
    ///     .add(0.0, 0.0, 1.0, 3.0)
    ///     .add(4.0, 2.0, 16.0, 8.0);
    /// let index = builder.build().unwrap();
    /// let query_results = index.query_iter(-1.0, -1.0, -0.5, -0.5).collect::<Vec<usize>>();
    /// assert_eq!(query_results, vec![1]);
    /// ```
    #[inline]
    pub fn query_iter<'a>(
        &'a self,
        min_x: T,
        min_y: T,
        max_x: T,
        max_y: T,
    ) -> impl Iterator<Item = usize> + 'a {
        QueryIterator::<'a, T>::new(self, min_x, min_y, max_x, max_y)
    }

    /// The same as [`StaticAABB2DIndex::query_iter`] but allows using an existing buffer for stack
    /// traversal. This is useful for performance when many queries will be done repeatedly to avoid
    /// allocating a new stack for each query (this is for performance benefit only).
    #[inline]
    pub fn query_iter_with_stack<'a>(
        &'a self,
        min_x: T,
        min_y: T,
        max_x: T,
        max_y: T,
        stack: &'a mut Vec<usize>,
    ) -> impl Iterator<Item = usize> + 'a {
        QueryIteratorStackRef::<'a, T>::new(self, stack, min_x, min_y, max_x, max_y)
    }

    /// Same as [`StaticAABB2DIndex::query`] but instead of returning a collection of indices a
    /// `visitor` function is called for each index that would be returned.  The `visitor` returns a
    /// control flow indicating whether to continue visiting or break.
    ///
    /// The [`ControlFlow`] and [`QueryVisitor`] traits are implemented to allow passing in a
    /// function [`FnMut`] visitor that returns no value (all results will be visited ) or a
    /// [`ControlFlow`] to break early.
    #[inline]
    pub fn visit_query<V, C>(&self, min_x: T, min_y: T, max_x: T, max_y: T, visitor: &mut V) -> C
    where
        C: ControlFlow,
        V: QueryVisitor<T, C>,
    {
        if self.num_items == 0 {
            // empty index, return early since no results to visit (avoid allocating for stack)
            return C::continuing();
        }
        let mut stack: Vec<usize> = Vec::with_capacity(16);
        self.visit_query_with_stack_impl(min_x, min_y, max_x, max_y, visitor, &mut stack)
    }

    /// Returns all the item [`AABB`] that were added to the index by
    /// [`StaticAABB2DIndexBuilder::add`] during construction.
    ///
    /// Use [`StaticAABB2DIndex::item_indices`] or [`StaticAABB2DIndex::all_box_indices`] to map a
    /// box's positional index to the original index position the item was added.
    #[inline]
    pub fn item_boxes(&self) -> &[AABB<T>] {
        &self.boxes[0..self.num_items]
    }

    /// Used to map an item box index position from [`StaticAABB2DIndex::item_boxes`] back to the
    /// original index position the item was added.
    #[inline]
    pub fn item_indices(&self) -> &[usize] {
        &self.indices[0..self.num_items]
    }

    /// Gets the node size used for the index.
    ///
    /// The node size is the maximum number of boxes stored as children of each node in the index
    /// tree.
    #[inline]
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Gets the level bounds for all the boxes in the index.
    ///
    /// The level bounds are the index positions in [`StaticAABB2DIndex::all_boxes`] where a change
    /// in the level of the index tree occurs.
    #[inline]
    pub fn level_bounds(&self) -> &[usize] {
        &self.level_bounds
    }

    /// Gets all the bounding boxes for the index.
    ///
    /// The boxes are ordered from the bottom of the tree up, so from 0 to
    /// [`StaticAABB2DIndex::count`] are all the item bounding boxes. Use
    /// [`StaticAABB2DIndex::all_box_indices`] to map a box back to the original index position it
    /// was added or find the start position for the children of a node box.
    #[inline]
    pub fn all_boxes(&self) -> &[AABB<T>] {
        &self.boxes
    }

    /// Used to map an item box index position from [`StaticAABB2DIndex::all_boxes`] back to the
    /// original index position the item was added. Or if indexing past [`StaticAABB2DIndex::count`]
    /// it will yield the [`StaticAABB2DIndex::all_boxes`] starting index of the node's children
    /// boxes. See the `index_tree_structure.rs` example for more information.
    #[inline]
    pub fn all_box_indices(&self) -> &[usize] {
        &self.indices
    }

    /// Same as [`StaticAABB2DIndex::query`] but allows using an existing buffer for stack
    /// traversal. This is useful for performance when many queries will be done repeatedly to avoid
    /// allocating a new stack for each query (this is for performance benefit only).
    #[inline]
    pub fn query_with_stack(
        &self,
        min_x: T,
        min_y: T,
        max_x: T,
        max_y: T,
        stack: &mut Vec<usize>,
    ) -> Vec<usize> {
        let mut results = Vec::new();
        let mut visitor = |i| {
            results.push(i);
        };
        self.visit_query_with_stack(min_x, min_y, max_x, max_y, &mut visitor, stack);
        results
    }

    /// Same as [`StaticAABB2DIndex::visit_query`] but allows using an existing buffer for stack
    /// traversal. This is useful for performance when many queries will be done repeatedly to avoid
    /// allocating a new stack for each query (this is for performance benefit only).
    #[inline]
    pub fn visit_query_with_stack<V, C>(
        &self,
        min_x: T,
        min_y: T,
        max_x: T,
        max_y: T,
        visitor: &mut V,
        stack: &mut Vec<usize>,
    ) -> C
    where
        C: ControlFlow,
        V: QueryVisitor<T, C>,
    {
        if self.num_items == 0 {
            // empty index, return early since no results to visit
            return C::continuing();
        }
        self.visit_query_with_stack_impl(min_x, min_y, max_x, max_y, visitor, stack)
    }

    // Implementation function which assumes self.num_items > 0 (for performance it helped to move
    // the self.num_items == 0 check outside of this function).
    fn visit_query_with_stack_impl<V, C>(
        &self,
        min_x: T,
        min_y: T,
        max_x: T,
        max_y: T,
        visitor: &mut V,
        stack: &mut Vec<usize>,
    ) -> C
    where
        C: ControlFlow,
        V: QueryVisitor<T, C>,
    {
        let mut node_index = self.boxes.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        // ensure the stack is empty for use
        stack.clear();

        loop {
            let end = min(
                node_index + self.node_size,
                *get_at_index(&self.level_bounds, level),
            );

            for pos in node_index..end {
                let aabb = get_at_index(&self.boxes, pos);
                if !aabb.overlaps(min_x, min_y, max_x, max_y) {
                    // no overlap
                    continue;
                }

                let index = *get_at_index(&self.indices, pos);
                if node_index < self.num_items {
                    try_control!(visitor.visit(index))
                } else {
                    stack.push(index);
                    stack.push(level - 1);
                }
            }

            if stack.len() > 1 {
                level = stack.pop().unwrap();
                node_index = stack.pop().unwrap();
            } else {
                return C::continuing();
            }
        }
    }

    /// Visit all neighboring items in order of minimum euclidean distance to the point defined by
    /// `x` and `y` until `visitor` breaks or all items have been visited.
    ///
    /// ## Notes
    /// * The visitor function must break to stop visiting items or all items will be visited.
    /// * The visitor function receives the index of the item being visited and the squared
    ///   euclidean distance to that item from the point given.
    /// * Because distances are squared (`dx * dx + dy * dy`) be cautious of smaller numeric types
    ///   overflowing (e.g. it's easy to overflow an i32 with squared distances).
    /// * If the point is inside of an item's bounding box then the euclidean distance is 0.
    /// * If repeatedly calling this method then [`StaticAABB2DIndex::visit_neighbors_with_queue`]
    ///   can be used to avoid repeated allocations for the priority queue used internally.
    #[inline]
    pub fn visit_neighbors<V, C>(&self, x: T, y: T, visitor: &mut V) -> C
    where
        C: ControlFlow,
        V: NeighborVisitor<T, C>,
    {
        if self.num_items == 0 {
            // empty index, return early since no results to visit
            return C::continuing();
        }
        let mut queue = NeighborPriorityQueue::with_capacity(8);
        self.visit_neighbors_with_queue_impl(x, y, visitor, &mut queue)
    }

    /// Works the same as [`StaticAABB2DIndex::visit_neighbors`] but accepts an existing binary heap
    /// to be used as a priority queue to avoid allocations.
    #[inline]
    pub fn visit_neighbors_with_queue<V, C>(
        &self,
        x: T,
        y: T,
        visitor: &mut V,
        queue: &mut NeighborPriorityQueue<T>,
    ) -> C
    where
        C: ControlFlow,
        V: NeighborVisitor<T, C>,
    {
        if self.num_items == 0 {
            // empty index, return early since no results to visit
            return C::continuing();
        }

        self.visit_neighbors_with_queue_impl(x, y, visitor, queue)
    }

    // Implementation function which assumes self.num_items > 0 (for performance it helped to move
    // the self.num_items == 0 check outside of this function).
    fn visit_neighbors_with_queue_impl<V, C>(
        &self,
        x: T,
        y: T,
        visitor: &mut V,
        queue: &mut NeighborPriorityQueue<T>,
    ) -> C
    where
        C: ControlFlow,
        V: NeighborVisitor<T, C>,
    {
        // small helper function to compute axis distance between point and bounding box axis
        #[inline]
        fn axis_dist<U>(k: U, min: U, max: U) -> U
        where
            U: IndexableNum,
        {
            if k < min {
                min - k
            } else if k > max {
                k - max
            } else {
                U::zero()
            }
        }

        let mut node_index = self.boxes.len() - 1;
        queue.clear();

        loop {
            let upper_bound_level_index = match self.level_bounds.binary_search(&node_index) {
                // level bound found, add one to get upper bound
                Ok(i) => i + 1,
                // level bound not found (node_index is between bounds, do not need to add one to
                // get upper bound)
                Err(i) => i,
            };

            // end index of the node
            let end = min(
                node_index + self.node_size,
                self.level_bounds[upper_bound_level_index],
            );

            // add nodes to queue
            for pos in node_index..end {
                let aabb = get_at_index(&self.boxes, pos);
                let dx = axis_dist(x, aabb.min_x, aabb.max_x);
                let dy = axis_dist(y, aabb.min_y, aabb.max_y);
                let dist = dx * dx + dy * dy;
                let index = *get_at_index(&self.indices, pos);
                let is_leaf_node = node_index < self.num_items;
                queue.push(NeighborsState::new(index, is_leaf_node, dist));
            }

            let mut continue_search = false;
            // pop and visit items in queue
            while let Some(state) = queue.pop() {
                if state.is_leaf_node {
                    // visit leaf node
                    try_control!(visitor.visit(state.index, state.dist))
                } else {
                    // update node index for next iteration
                    node_index = state.index;
                    // set flag to continue search
                    continue_search = true;
                    break;
                }
            }

            if !continue_search {
                return C::continuing();
            }
        }
    }
}
