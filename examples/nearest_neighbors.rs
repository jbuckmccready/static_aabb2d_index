use static_aabb2d_index::{Control, StaticAABB2DIndex, StaticAABB2DIndexBuilder};

fn main() {
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

    // perform nearest neighbor query, limited to visiting the nearest 2 neighbors
    let mut neighbors_result: Vec<(usize, f64)> = Vec::new();
    let max_neighbors = 2;
    let mut neighbor_visitor = |box_added_pos: usize, dist_squared: f64| {
        neighbors_result.push((box_added_pos, dist_squared));
        if neighbors_result.len() == max_neighbors {
            // stop visiting when reaching the max neighbors limit
            return Control::Break(());
        }
        Control::Continue
    };
    index.visit_neighbors(100.0, 100.0, &mut neighbor_visitor);
    // calculate expected distance squared values
    // distances for box at index position 1
    let x_dist1 = 100.0 - 3.0;
    let y_dist1 = 100.0 - 3.0;
    // distances for box at index position 3
    let x_dist3 = 100.0 - 16.0;
    let y_dist3 = 100.0 - 8.0;
    // distances squared
    let dist1_expected = x_dist1 * x_dist1 + y_dist1 * y_dist1;
    let dist3_expected = x_dist3 * x_dist3 + y_dist3 * y_dist3;

    // expect box at 3 before box at 1 due to shorter distance
    assert_eq!(
        neighbors_result,
        vec![(3, dist3_expected), (1, dist1_expected)]
    );
}
