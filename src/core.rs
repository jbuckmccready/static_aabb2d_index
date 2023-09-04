use std::cmp::Ordering;

use num_traits::{Bounded, Num, NumCast};

/// Trait used by the [StaticAABB2DIndex](crate::StaticAABB2DIndex) that is required to be
/// implemented for type T. It is blanket implemented for all supported primitive numeric types.
pub trait IndexableNum: Copy + Num + PartialOrd + Default + Bounded + NumCast {
    /// Simple default min implementation for [PartialOrd] types.
    #[inline]
    fn min(self, other: Self) -> Self {
        if self < other {
            return self;
        }

        other
    }

    /// Simple default max implementation for [PartialOrd] types.
    #[inline]
    fn max(self, other: Self) -> Self {
        if self > other {
            return self;
        }

        other
    }

    /// Total comparison between numbers. For types which implement [Ord] this should just use the
    /// [Ord::cmp] method. For [f32] and [f64] types the `f32::total_cmp` and `f64::total_cmp`
    /// methods are used.
    fn total_cmp(&self, other: &Self) -> Ordering;
}

macro_rules! impl_indexable_num_for_ord_type {
    ($t:ty) => {
        impl IndexableNum for $t {
            #[inline]
            fn min(self, other: Self) -> Self {
                std::cmp::min(self, other)
            }
            #[inline]
            fn max(self, other: Self) -> Self {
                std::cmp::max(self, other)
            }
            #[inline]
            fn total_cmp(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
        }
    };
}

// impl for all supported built in types
impl_indexable_num_for_ord_type!(i8);
impl_indexable_num_for_ord_type!(u8);
impl_indexable_num_for_ord_type!(i16);
impl_indexable_num_for_ord_type!(u16);
impl_indexable_num_for_ord_type!(i32);
impl_indexable_num_for_ord_type!(u32);
impl_indexable_num_for_ord_type!(i64);
impl_indexable_num_for_ord_type!(u64);
impl_indexable_num_for_ord_type!(i128);
impl_indexable_num_for_ord_type!(u128);
impl IndexableNum for f32 {
    #[inline]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    #[inline]
    fn total_cmp(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}
impl IndexableNum for f64 {
    #[inline]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    #[inline]
    fn total_cmp(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

/// Simple 2D axis aligned bounding box which holds the extents of a 2D box.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct AABB<T = f64> {
    /// Min x extent of the axis aligned bounding box.
    pub min_x: T,
    /// Min y extent of the axis aligned bounding box.
    pub min_y: T,
    /// Max x extent of the axis aligned bounding box.
    pub max_x: T,
    /// Max y extent of the axis aligned bounding box.
    pub max_y: T,
}

impl<T> Default for AABB<T>
where
    T: IndexableNum,
{
    #[inline]
    fn default() -> Self {
        AABB {
            min_x: T::zero(),
            min_y: T::zero(),
            max_x: T::zero(),
            max_y: T::zero(),
        }
    }
}

impl<T> AABB<T>
where
    T: IndexableNum,
{
    #[inline]
    pub fn new(min_x: T, min_y: T, max_x: T, max_y: T) -> AABB<T> {
        AABB {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Tests if this AABB overlaps another AABB (inclusive).
    ///
    /// # Examples
    /// ```
    /// use static_aabb2d_index::AABB;
    /// let box_a = AABB::new(0, 0, 2, 2);
    /// let box_b = AABB::new(1, 1, 3, 3);
    /// assert!(box_a.overlaps_aabb(&box_b));
    /// assert!(box_b.overlaps_aabb(&box_a));
    ///
    /// let box_c = AABB::new(-1, -1, 0, 0);
    /// assert!(!box_c.overlaps_aabb(&box_b));
    /// // note: overlap check is inclusive of edges/corners touching
    /// assert!(box_c.overlaps_aabb(&box_a));
    /// ```
    #[inline]
    pub fn overlaps_aabb(&self, other: &AABB<T>) -> bool {
        self.overlaps(other.min_x, other.min_y, other.max_x, other.max_y)
    }

    /// Tests if this AABB overlaps another AABB.
    /// Same as [AABB::overlaps_aabb] but accepts AABB extent parameters directly.
    #[inline]
    pub fn overlaps(&self, min_x: T, min_y: T, max_x: T, max_y: T) -> bool {
        if self.max_x < min_x || self.max_y < min_y || self.min_x > max_x || self.min_y > max_y {
            return false;
        }

        true
    }

    /// Tests if this AABB fully contains another AABB (inclusive).
    ///
    /// # Examples
    /// ```
    /// use static_aabb2d_index::AABB;
    /// let box_a = AABB::new(0, 0, 3, 3);
    /// let box_b = AABB::new(1, 1, 2, 2);
    /// assert!(box_a.contains_aabb(&box_b));
    /// assert!(!box_b.contains_aabb(&box_a));
    /// ```
    #[inline]
    pub fn contains_aabb(&self, other: &AABB<T>) -> bool {
        self.contains(other.min_x, other.min_y, other.max_x, other.max_y)
    }

    /// Tests if this AABB fully contains another AABB.
    /// Same as [AABB::contains] but accepts AABB extent parameters directly.
    #[inline]
    pub fn contains(&self, min_x: T, min_y: T, max_x: T, max_y: T) -> bool {
        self.min_x <= min_x && self.min_y <= min_y && self.max_x >= max_x && self.max_y >= max_y
    }
}

/// Basic control flow enum that can be used when visiting query results.
#[derive(Debug, Default)]
pub enum Control<B> {
    /// Indicates to the query function to continue visiting results.
    #[default]
    Continue,
    /// Indicates to the query function to stop visiting results and return a value.
    Break(B),
}

/// Trait for control flow inside query functions.
pub trait ControlFlow {
    /// Constructs state indicating to continue.
    fn continuing() -> Self;
    /// Should return true if control flow should break.
    fn should_break(&self) -> bool;
}

impl<B> ControlFlow for Control<B> {
    #[inline]
    fn continuing() -> Self {
        Control::Continue
    }

    #[inline]
    fn should_break(&self) -> bool {
        matches!(*self, Control::Break(_))
    }
}

impl ControlFlow for () {
    #[inline]
    fn continuing() -> Self {}

    #[inline]
    fn should_break(&self) -> bool {
        false
    }
}

impl<C, E> ControlFlow for Result<C, E>
where
    C: ControlFlow,
{
    fn continuing() -> Self {
        Ok(C::continuing())
    }

    fn should_break(&self) -> bool {
        self.is_err()
    }
}

/// Visitor trait used to visit the results of a StaticAABB2DIndex query.
///
/// This trait is blanket implemented for FnMut(usize) -> impl ControlFlow.
pub trait QueryVisitor<T, C>
where
    T: IndexableNum,
    C: ControlFlow,
{
    /// Visit the index position of AABB returned by query.
    fn visit(&mut self, index_pos: usize) -> C;
}

impl<T, C, F> QueryVisitor<T, C> for F
where
    T: IndexableNum,
    C: ControlFlow,
    F: FnMut(usize) -> C,
{
    #[inline]
    fn visit(&mut self, index_pos: usize) -> C {
        self(index_pos)
    }
}

/// Visitor trait used to visit the results of a StaticAABB2DIndex nearest neighbors query.
pub trait NeighborVisitor<T, C>
where
    T: IndexableNum,
    C: ControlFlow,
{
    /// Visits the result containing the index position of the AABB neighbor and its euclidean
    /// distance squared to the nearest neighbor input.
    fn visit(&mut self, index_pos: usize, dist_squared: T) -> C;
}

impl<T, C, F> NeighborVisitor<T, C> for F
where
    T: IndexableNum,
    C: ControlFlow,
    F: FnMut(usize, T) -> C,
{
    #[inline]
    fn visit(&mut self, index_pos: usize, dist_squared: T) -> C {
        self(index_pos, dist_squared)
    }
}

#[macro_export]
macro_rules! try_control {
    ($e:expr) => {
        match $e {
            x => {
                if x.should_break() {
                    return x;
                }
            }
        }
    };
}
