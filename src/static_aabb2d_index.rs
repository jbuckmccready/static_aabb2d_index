use fmt::Debug;
use std::fmt;
use std::{
    cmp::{max, min},
    collections::BinaryHeap,
};

use crate::{try_control, ControlFlow, IndexableNum, NeighborVisitor, QueryVisitor, AABB};

/// Error type for errors that may be returned in attempting to build the index.
#[derive(Debug, PartialEq)]
pub enum StaticAABB2DIndexBuildError {
    /// Error for the case when the item count given is 0.
    ZeroItemsError,
    /// Error for the case when the number of items added does not match the size given at
    /// construction.
    ItemCountError {
        /// The number of items that were added.
        added: usize,
        /// The number of items that were expected (set at construction).
        expected: usize,
    },
    /// Error for the case when the numeric type T used for the index fails to cast to/from u16.
    NumericCastError,
}

impl std::error::Error for StaticAABB2DIndexBuildError {}

impl fmt::Display for StaticAABB2DIndexBuildError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StaticAABB2DIndexBuildError::ZeroItemsError => {
                write!(f, "item count given cannot be zero")
            }
            StaticAABB2DIndexBuildError::ItemCountError { added, expected } => write!(
                f,
                "added item count should equal static size given to builder \
                (added: {}, expected: {})",
                added, expected
            ),
            StaticAABB2DIndexBuildError::NumericCastError => write!(
                f,
                "numeric cast to/from type T to u16 failed (may be due to overflow/underflow)"
            ),
        }
    }
}

/// Used to build a [StaticAABB2DIndex].
#[derive(Debug, Clone)]
pub struct StaticAABB2DIndexBuilder<T = f64>
where
    T: IndexableNum,
{
    min_x: T,
    min_y: T,
    max_x: T,
    max_y: T,
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    /// boxes holds the tree data (all nodes and items)
    boxes: Vec<AABB<T>>,
    /// indices is used to map from sorted indices to indices ordered according to the order items
    /// were added
    indices: Vec<usize>,
    // used to keep track of the current position for boxes added
    pos: usize,
}

/// Static/fixed size indexing data structure for two dimensional axis aligned bounding boxes.
///
/// The index allows for fast construction and fast querying but cannot be modified after creation.
/// This type is constructed from a [StaticAABB2DIndexBuilder].
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
/// // given at the time the builder was created or the type used fails to cast to/from a u16
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
    min_x: T,
    min_y: T,
    max_x: T,
    max_y: T,
    node_size: usize,
    num_items: usize,
    level_bounds: Vec<usize>,
    /// boxes holds the tree data (all nodes and items)
    boxes: Vec<AABB<T>>,
    /// indices is used to map from sorted indices to indices ordered according to the order items
    /// were added
    indices: Vec<usize>,
}

// get_at_index! and set_at_index! macros to toggle bounds checking at compile time
#[cfg(not(feature = "allow_unsafe"))]
macro_rules! get_at_index {
    ($container:expr, $index:expr) => {
        &$container[$index]
    };
}

#[cfg(feature = "allow_unsafe")]
macro_rules! get_at_index {
    ($container:expr, $index:expr) => {
        unsafe { $container.get_unchecked($index) }
    };
}

#[cfg(not(feature = "allow_unsafe"))]
macro_rules! set_at_index {
    ($container:expr, $index:expr, $value:expr) => {
        $container[$index] = $value
    };
}

#[cfg(feature = "allow_unsafe")]
macro_rules! set_at_index {
    ($container:expr, $index:expr, $value:expr) => {
        unsafe { *$container.get_unchecked_mut($index) = $value }
    };
}

impl<T> StaticAABB2DIndexBuilder<T>
where
    T: IndexableNum,
{
    fn init(num_items: usize, node_size: usize) -> Self {
        if num_items == 0 {
            // just return early, build() method will return error result
            return StaticAABB2DIndexBuilder {
                min_x: T::max_value(),
                min_y: T::max_value(),
                max_x: T::min_value(),
                max_y: T::min_value(),
                node_size,
                num_items,
                level_bounds: Vec::new(),
                boxes: Vec::new(),
                indices: Vec::new(),
                pos: 0,
            };
        }

        let node_size = min(max(node_size, 2), 65535);

        let mut n = num_items;
        let mut num_nodes = num_items;
        let mut level_bounds: Vec<usize> = vec![n];

        // calculate the total number of nodes in the R-tree to allocate space for
        // and the index of each tree level (level_bounds, used in search later)
        loop {
            n = (n as f64 / node_size as f64).ceil() as usize;
            num_nodes += n;
            level_bounds.push(num_nodes);
            if n == 1 {
                break;
            }
        }

        // unsafe alternative for performance (uninitialized memory rather than initialize to zero)
        // since it is all initialized later before use
        #[cfg(feature = "allow_unsafe")]
        let init_boxes = || {
            let mut boxes = Vec::with_capacity(num_nodes);
            unsafe {
                boxes.set_len(num_nodes);
            }
            boxes
        };

        #[cfg(not(feature = "allow_unsafe"))]
        let init_boxes = || vec![AABB::default(); num_nodes];

        let boxes = init_boxes();

        StaticAABB2DIndexBuilder {
            min_x: T::max_value(),
            min_y: T::max_value(),
            max_x: T::min_value(),
            max_y: T::min_value(),
            node_size,
            num_items,
            level_bounds,
            boxes,
            indices: (0..num_nodes).collect(),
            pos: 0,
        }
    }

    /// Construct a new [StaticAABB2DIndexBuilder] to fit exactly the specified `count` number of
    /// items.
    #[inline]
    pub fn new(count: usize) -> Self {
        StaticAABB2DIndexBuilder::init(count, 16)
    }

    /// Construct a new [StaticAABB2DIndexBuilder] to fit exactly the specified `count` number of
    /// items and use `node_size` for the index tree shape.
    ///
    /// Each node in the index tree has a maximum size which may be adjusted by `node_size` for
    /// performance reasons, however the default value of 16 when calling
    /// `StaticAABB2DIndexBuilder::new` is tested to be optimal in most cases.
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
    /// from the constructed [StaticAABB2DIndex].
    pub fn add(&mut self, min_x: T, min_y: T, max_x: T, max_y: T) -> &mut Self {
        // catch adding past num_items (error will be returned when build is called)
        if self.pos >= self.num_items {
            self.pos += 1;
            return self;
        }
        debug_assert!(min_x <= max_x);
        debug_assert!(min_y <= max_y);

        set_at_index!(self.boxes, self.pos, AABB::new(min_x, min_y, max_x, max_y));
        self.pos += 1;

        self.min_x = T::min(self.min_x, min_x);
        self.min_y = T::min(self.min_y, min_y);
        self.max_x = T::max(self.max_x, max_x);
        self.max_y = T::max(self.max_y, max_y);
        self
    }

    /// Build the [StaticAABB2DIndex] with the boxes that have been added.
    ///
    /// If the number of added items does not match the count given at the time the builder was
    /// created then a [StaticAABB2DIndexBuildError::ItemCountError] will be returned.
    ///
    /// If the numeric type T fails to cast to/from a u16 for any reason then a
    /// [StaticAABB2DIndexBuildError::NumericCastError] will be returned.
    pub fn build(mut self) -> Result<StaticAABB2DIndex<T>, StaticAABB2DIndexBuildError> {
        if self.pos != self.num_items {
            return Err(StaticAABB2DIndexBuildError::ItemCountError {
                added: self.pos,
                expected: self.num_items,
            });
        }

        if self.num_items == 0 {
            return Err(StaticAABB2DIndexBuildError::ZeroItemsError);
        }

        // if number of items is less than node size then skip sorting since each node of boxes must
        // be fully scanned regardless and there is only one node
        if self.num_items <= self.node_size {
            set_at_index!(self.indices, self.pos, 0);
            // fill root box with total extents
            set_at_index!(
                self.boxes,
                self.pos,
                AABB::new(self.min_x, self.min_y, self.max_x, self.max_y)
            );
            return Ok(StaticAABB2DIndex {
                min_x: self.min_x,
                min_y: self.min_y,
                max_x: self.max_x,
                max_y: self.max_y,
                node_size: self.node_size,
                num_items: self.num_items,
                level_bounds: self.level_bounds,
                boxes: self.boxes,
                indices: self.indices,
            });
        }

        let width = self.max_x - self.min_x;
        let height = self.max_y - self.min_y;

        // hilbert max input value for x and y
        let hilbert_max = T::from(u16::MAX).ok_or(StaticAABB2DIndexBuildError::NumericCastError)?;
        let two = T::from(2u16).ok_or(StaticAABB2DIndexBuildError::NumericCastError)?;

        // mapping the x and y coordinates of the center of the item boxes to values in the range
        // [0 -> n - 1] such that the min of the entire set of bounding boxes maps to 0 and the max
        // of the entire set of bounding boxes maps to n - 1 our 2d space is x: [0 -> n-1] and
        // y: [0 -> n-1], our 1d hilbert curve value space is d: [0 -> n^2 - 1]
        let mut hilbert_values: Vec<u32> = Vec::with_capacity(self.num_items);
        for aabb in self.boxes.iter().take(self.num_items) {
            let x = if width == T::zero() {
                0
            } else {
                (hilbert_max * ((aabb.min_x + aabb.max_x) / two - self.min_x) / width)
                    .to_u16()
                    .ok_or(StaticAABB2DIndexBuildError::NumericCastError)?
            };
            let y = if height == T::zero() {
                0
            } else {
                (hilbert_max * ((aabb.min_y + aabb.max_y) / two - self.min_y) / height)
                    .to_u16()
                    .ok_or(StaticAABB2DIndexBuildError::NumericCastError)?
            };
            hilbert_values.push(hilbert_xy_to_index(x, y));
        }

        // sort items by their Hilbert value for constructing the tree
        sort(
            &mut hilbert_values,
            &mut self.boxes,
            &mut self.indices,
            0,
            self.num_items - 1,
            self.node_size,
        );

        // generate nodes at each tree level, bottom-up
        let mut pos = 0;
        for i in 0..self.level_bounds.len() - 1 {
            let end = *get_at_index!(self.level_bounds, i);

            // generate a parent node for each block of consecutive node_size nodes
            while pos < end {
                let mut node_min_x = T::max_value();
                let mut node_min_y = T::max_value();
                let mut node_max_x = T::min_value();
                let mut node_max_y = T::min_value();
                let node_index = pos;

                // calculate bounding box for the new node
                let mut j = 0;
                while j < self.node_size && pos < end {
                    let aabb = get_at_index!(self.boxes, pos);
                    pos += 1;
                    node_min_x = T::min(node_min_x, aabb.min_x);
                    node_min_y = T::min(node_min_y, aabb.min_y);
                    node_max_x = T::max(node_max_x, aabb.max_x);
                    node_max_y = T::max(node_max_y, aabb.max_y);
                    j += 1;
                }

                // add the new node to the tree
                set_at_index!(self.indices, self.pos, node_index);
                set_at_index!(
                    self.boxes,
                    self.pos,
                    AABB::new(node_min_x, node_min_y, node_max_x, node_max_y)
                );
                self.pos += 1;
            }
        }

        Ok(StaticAABB2DIndex {
            min_x: self.min_x,
            min_y: self.min_y,
            max_x: self.max_x,
            max_y: self.max_y,
            node_size: self.node_size,
            num_items: self.num_items,
            level_bounds: self.level_bounds,
            boxes: self.boxes,
            indices: self.indices,
        })
    }
}

/// Maps 2d space to 1d hilbert curve space.
///
/// 2d space is `x: [0 -> n-1]` and `y: [0 -> n-1]`, 1d hilbert curve value space is
/// `d: [0 -> n^2 - 1]`, where n = 2^16, so `x` and `y` must be between 0 and [u16::MAX]
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
    values: &mut Vec<u32>,
    boxes: &mut Vec<AABB<T>>,
    indices: &mut Vec<usize>,
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

    let pivot = *get_at_index!(values, (left + right) >> 1);
    let mut i = left.wrapping_sub(1);
    let mut j = right.wrapping_add(1);

    loop {
        loop {
            i = i.wrapping_add(1);
            if *get_at_index!(values, i) >= pivot {
                break;
            }
        }

        loop {
            j = j.wrapping_sub(1);
            if *get_at_index!(values, j) <= pivot {
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
fn swap<T>(
    values: &mut Vec<u32>,
    boxes: &mut Vec<AABB<T>>,
    indices: &mut Vec<usize>,
    i: usize,
    j: usize,
) where
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
        let node_index = aabb_index.boxes.len() - 1;
        let pos = node_index;
        let level = aabb_index.level_bounds.len() - 1;
        let end = min(
            node_index + aabb_index.node_size,
            *get_at_index!(aabb_index.level_bounds, level),
        );
        QueryIterator {
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

impl<'a, T> Iterator for QueryIterator<'a, T>
where
    T: IndexableNum,
{
    type Item = usize;

    // NOTE: The inline attribute here shows significant performance improvements in benchmarks.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            while self.pos < self.end {
                let current_pos = self.pos;
                self.pos += 1;

                let aabb = get_at_index!(self.aabb_index.boxes, current_pos);
                if !aabb.overlaps(self.min_x, self.min_y, self.max_x, self.max_y) {
                    // no overlap
                    continue;
                }

                let index = *get_at_index!(self.aabb_index.indices, current_pos);
                if self.node_index < self.aabb_index.num_items {
                    return Some(index);
                } else {
                    self.stack.push(index);
                    self.stack.push(self.level - 1);
                }
            }

            if self.stack.len() > 1 {
                self.level = self.stack.pop().unwrap();
                self.node_index = self.stack.pop().unwrap();
                self.pos = self.node_index;
                self.end = min(
                    self.node_index + self.aabb_index.node_size,
                    *get_at_index!(self.aabb_index.level_bounds, self.level),
                );
            } else {
                return None;
            }
        }
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
        let node_index = aabb_index.boxes.len() - 1;
        let pos = node_index;
        let level = aabb_index.level_bounds.len() - 1;
        let end = min(
            node_index + aabb_index.node_size,
            *get_at_index!(aabb_index.level_bounds, level),
        );

        // ensure the stack is empty for use
        stack.clear();

        QueryIteratorStackRef {
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

impl<'a, T> Iterator for QueryIteratorStackRef<'a, T>
where
    T: IndexableNum,
{
    type Item = usize;

    // NOTE: The inline attribute here shows significant performance improvements in benchmarks.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            while self.pos < self.end {
                let current_pos = self.pos;
                self.pos += 1;

                let aabb = get_at_index!(self.aabb_index.boxes, current_pos);
                if !aabb.overlaps(self.min_x, self.min_y, self.max_x, self.max_y) {
                    // no overlap
                    continue;
                }

                let index = *get_at_index!(self.aabb_index.indices, current_pos);
                if self.node_index < self.aabb_index.num_items {
                    return Some(index);
                } else {
                    self.stack.push(index);
                    self.stack.push(self.level - 1);
                }
            }

            if self.stack.len() > 1 {
                self.level = self.stack.pop().unwrap();
                self.node_index = self.stack.pop().unwrap();
                self.pos = self.node_index;
                self.end = min(
                    self.node_index + self.aabb_index.node_size,
                    *get_at_index!(self.aabb_index.level_bounds, self.level),
                );
            } else {
                return None;
            }
        }
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
/// See: [StaticAABB2DIndex::visit_neighbors_with_queue].
pub type NeighborPriorityQueue<T> = BinaryHeap<NeighborsState<T>>;

/// Holds state for priority queue used in nearest neighbors query.
///
/// Note this type is public for use in passing in an existing priority queue but
/// all fields and constructor are private for internal use only.
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
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if let Some(ord) = self.partial_cmp(other) {
            ord
        } else {
            // if ordering not possible (due to NAN) then just consider equal
            std::cmp::Ordering::Equal
        }
    }
}

impl<T> PartialOrd for NeighborsState<T>
where
    T: IndexableNum,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // flip ordering (compare other to self rather than self to other) to prioritize minimum
        // dist in priority queue
        other.dist.partial_cmp(&self.dist)
    }
}

impl<T> StaticAABB2DIndex<T>
where
    T: IndexableNum,
{
    /// Gets the min_x extent value of the all the bounding boxes in the index.
    #[inline]
    pub fn min_x(&self) -> T {
        self.min_x
    }

    /// Gets the min_y extent value of the all the bounding boxes in the index.
    #[inline]
    pub fn min_y(&self) -> T {
        self.min_y
    }

    /// Gets the max_x extent value of the all the bounding boxes in the index.
    #[inline]
    pub fn max_x(&self) -> T {
        self.max_x
    }

    /// Gets the max_y extent value of the all the bounding boxes in the index.
    #[inline]
    pub fn max_y(&self) -> T {
        self.max_y
    }

    /// Gets the total count of items that were added to the index.
    #[inline]
    pub fn count(&self) -> usize {
        self.num_items
    }

    /// Queries the index, returning a collection of indexes to items that overlap with the bounding
    /// box given.
    ///
    /// `min_x`, `min_y`, `max_x`, and `max_y` represent the bounding box to use for the query.
    /// Indexes returned match with the order items were added to the index using
    /// [StaticAABB2DIndexBuilder::add].
    #[inline]
    pub fn query(&self, min_x: T, min_y: T, max_x: T, max_y: T) -> Vec<usize> {
        let mut results = Vec::new();
        let mut visitor = |i| {
            results.push(i);
        };
        self.visit_query(min_x, min_y, max_x, max_y, &mut visitor);
        results
    }

    /// The same as [StaticAABB2DIndex::query] but instead of returning a [Vec] of results a lazy
    /// iterator is returned which yields the results.
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
        QueryIterator::<'a, T>::new(&self, min_x, min_y, max_x, max_y)
    }

    /// The same as [StaticAABB2DIndex::query_iter] but allows using an existing buffer for stack
    /// traversal.
    #[inline]
    pub fn query_iter_with_stack<'a>(
        &'a self,
        min_x: T,
        min_y: T,
        max_x: T,
        max_y: T,
        stack: &'a mut Vec<usize>,
    ) -> impl Iterator<Item = usize> + 'a {
        QueryIteratorStackRef::<'a, T>::new(&self, stack, min_x, min_y, max_x, max_y)
    }

    /// Same as [StaticAABB2DIndex::query] but instead of returning a collection of indexes a
    /// `visitor` function is called for each index that would be returned.  The `visitor` returns a
    /// control flow indicating whether to continue visiting or break.
    #[inline]
    pub fn visit_query<V, C>(&self, min_x: T, min_y: T, max_x: T, max_y: T, visitor: &mut V)
    where
        C: ControlFlow,
        V: QueryVisitor<T, C>,
    {
        let mut stack: Vec<usize> = Vec::with_capacity(16);
        self.visit_query_with_stack(min_x, min_y, max_x, max_y, visitor, &mut stack);
    }

    /// Returns all the item [AABB] that were added to the index by [StaticAABB2DIndexBuilder::add].
    ///
    /// Use [StaticAABB2DIndex::map_all_boxes_index] to map a box back to the original index
    /// position it was added.
    #[inline]
    pub fn item_boxes(&self) -> &[AABB<T>] {
        &self.boxes[0..self.num_items]
    }

    /// Gets the node size used for the [StaticAABB2DIndex].
    ///
    /// The node size is the maximum number of boxes stored as children of each node in the index
    /// tree.
    #[inline]
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// Gets the level bounds for all the boxes in the [StaticAABB2DIndex].
    ///
    /// The level bounds are the index positions in [StaticAABB2DIndex::all_boxes] where a change in
    /// the level of the index tree occurs.
    #[inline]
    pub fn level_bounds(&self) -> &[usize] {
        &self.level_bounds
    }

    /// Gets all the bounding boxes for the [StaticAABB2DIndex].
    ///
    /// The boxes are ordered from the bottom of the tree up, so from 0 to
    /// [StaticAABB2DIndex::count] are all the item bounding boxes. Use
    /// [StaticAABB2DIndex::map_all_boxes_index] to map a box back to the original index position it
    /// was added or find the start position for the children of a node box.
    #[inline]
    pub fn all_boxes(&self) -> &[AABB<T>] {
        &self.boxes
    }

    /// Gets the original item index position (from the time it was added) from a
    /// [StaticAABB2DIndex::all_boxes] slice index position.
    ///
    /// If `all_boxes_index` is greater than [StaticAABB2DIndex::count] then it will return the
    /// [StaticAABB2DIndex::all_boxes] starting index of the node's children boxes.
    /// See the index_tree_structure.rs example for more information.
    #[inline]
    pub fn map_all_boxes_index(&self, all_boxes_index: usize) -> usize {
        self.indices[all_boxes_index]
    }

    /// Same as [StaticAABB2DIndex::query] but accepts an existing [Vec] to be used as a stack
    /// buffer when performing the query to avoid the need for allocation (this is for performance
    /// benefit only).
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

    /// Same as [StaticAABB2DIndex::visit_query] but accepts an existing [Vec] to be used as a stack
    /// buffer when performing the query to avoid the need for allocation (this is for performance
    /// benefit only).
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
        let mut node_index = self.boxes.len() - 1;
        let mut level = self.level_bounds.len() - 1;
        // ensure the stack is empty for use
        stack.clear();

        loop {
            let end = min(
                node_index + self.node_size,
                *get_at_index!(self.level_bounds, level),
            );

            for pos in node_index..end {
                let aabb = get_at_index!(self.boxes, pos);
                if !aabb.overlaps(min_x, min_y, max_x, max_y) {
                    // no overlap
                    continue;
                }

                let index = *get_at_index!(self.indices, pos);
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
    /// * If repeatedly calling this method then [StaticAABB2DIndex::visit_neighbors_with_queue] can
    ///   be used to avoid repeated allocations for the priority queue used internally.
    #[inline]
    pub fn visit_neighbors<V, C>(&self, x: T, y: T, visitor: &mut V)
    where
        C: ControlFlow,
        V: NeighborVisitor<T, C>,
    {
        let mut queue = NeighborPriorityQueue::new();
        self.visit_neighbors_with_queue(x, y, visitor, &mut queue);
    }

    /// Works the same as [StaticAABB2DIndex::visit_neighbors] but accepts an existing binary heap
    /// to be used as a priority queue to avoid allocations.
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
        // small helper function to compute axis distance between point and bounding box axis
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
                let aabb = get_at_index!(self.boxes, pos);
                let dx = axis_dist(x, aabb.min_x, aabb.max_x);
                let dy = axis_dist(y, aabb.min_y, aabb.max_y);
                let dist = dx * dx + dy * dy;
                let index = *get_at_index!(self.indices, pos);
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
