use static_aabb2d_index::*;
use std::collections::HashSet;

fn create_test_data() -> Vec<i32> {
    vec![
        8, 62, 11, 66, 57, 17, 57, 19, 76, 26, 79, 29, 36, 56, 38, 56, 92, 77, 96, 80, 87, 70, 90,
        74, 43, 41, 47, 43, 0, 58, 2, 62, 76, 86, 80, 89, 27, 13, 27, 15, 71, 63, 75, 67, 25, 2,
        27, 2, 87, 6, 88, 6, 22, 90, 23, 93, 22, 89, 22, 93, 57, 11, 61, 13, 61, 55, 63, 56, 17,
        85, 21, 87, 33, 43, 37, 43, 6, 1, 7, 3, 80, 87, 80, 87, 23, 50, 26, 52, 58, 89, 58, 89, 12,
        30, 15, 34, 32, 58, 36, 61, 41, 84, 44, 87, 44, 18, 44, 19, 13, 63, 15, 67, 52, 70, 54, 74,
        57, 59, 58, 59, 17, 90, 20, 92, 48, 53, 52, 56, 92, 68, 92, 72, 26, 52, 30, 52, 56, 23, 57,
        26, 88, 48, 88, 48, 66, 13, 67, 15, 7, 82, 8, 86, 46, 68, 50, 68, 37, 33, 38, 36, 6, 15, 8,
        18, 85, 36, 89, 38, 82, 45, 84, 48, 12, 2, 16, 3, 26, 15, 26, 16, 55, 23, 59, 26, 76, 37,
        79, 39, 86, 74, 90, 77, 16, 75, 18, 78, 44, 18, 45, 21, 52, 67, 54, 71, 59, 78, 62, 78, 24,
        5, 24, 8, 64, 80, 64, 83, 66, 55, 70, 55, 0, 17, 2, 19, 15, 71, 18, 74, 87, 57, 87, 59, 6,
        34, 7, 37, 34, 30, 37, 32, 51, 19, 53, 19, 72, 51, 73, 55, 29, 45, 30, 45, 94, 94, 96, 95,
        7, 22, 11, 24, 86, 45, 87, 48, 33, 62, 34, 65, 18, 10, 21, 14, 64, 66, 67, 67, 64, 25, 65,
        28, 27, 4, 31, 6, 84, 4, 85, 5, 48, 80, 50, 81, 1, 61, 3, 61, 71, 89, 74, 92, 40, 42, 43,
        43, 27, 64, 28, 66, 46, 26, 50, 26, 53, 83, 57, 87, 14, 75, 15, 79, 31, 45, 34, 45, 89, 84,
        92, 88, 84, 51, 85, 53, 67, 87, 67, 89, 39, 26, 43, 27, 47, 61, 47, 63, 23, 49, 25, 53, 12,
        3, 14, 5, 16, 50, 19, 53, 63, 80, 64, 84, 22, 63, 22, 64, 26, 66, 29, 66, 2, 15, 3, 15, 74,
        77, 77, 79, 64, 11, 68, 11, 38, 4, 39, 8, 83, 73, 87, 77, 85, 52, 89, 56, 74, 60, 76, 63,
        62, 66, 65, 67,
    ]
}

fn aabb_from_data<T: IndexableNum>(data: &[T]) -> Vec<AABB<T>> {
    (0..data.len())
        .step_by(4)
        .map(|i| AABB::new(data[i], data[i + 1], data[i + 2], data[i + 3]))
        .collect()
}

fn create_index<T: IndexableNum>(data: &[T]) -> StaticAABB2DIndex<T> {
    let mut builder = StaticAABB2DIndexBuilder::new_with_node_size(data.len() / 4, 16);
    for pos in (0..data.len()).step_by(4) {
        builder.add(data[pos], data[pos + 1], data[pos + 2], data[pos + 3]);
    }

    builder.build().unwrap()
}

fn create_index_with_node_size<T: IndexableNum>(
    data: &[T],
    node_size: usize,
) -> StaticAABB2DIndex<T> {
    let mut builder = StaticAABB2DIndexBuilder::new_with_node_size(data.len() / 4, node_size);
    for pos in (0..data.len()).step_by(4) {
        builder.add(data[pos], data[pos + 1], data[pos + 2], data[pos + 3]);
    }

    builder.build().unwrap()
}

fn create_test_index() -> StaticAABB2DIndex<i32> {
    create_index(&create_test_data())
}

fn create_small_test_index() -> StaticAABB2DIndex<i32> {
    let data = create_test_data();
    let item_count = 14;
    let small_data: Vec<i32> = data.into_iter().take(item_count * 4).collect();
    create_index(&small_data)
}

#[test]
fn building_from_zeroes_is_ok() {
    {
        // f64 boxes
        let item_count = 50;
        let data = vec![0.0; item_count * 4];
        let index = create_index(&data);

        let query_results: HashSet<usize> = index.query(-1.0, -1.0, 1.0, 1.0).into_iter().collect();
        let expected_results: HashSet<usize> = (0..data.len() / 4).collect();
        assert_eq!(query_results, expected_results);

        let query_results: HashSet<usize> = index.query(1.0, 1.0, 2.0, 2.0).into_iter().collect();
        assert!(query_results.is_empty());
    }
    {
        // i32 boxes
        let item_count = 50;
        let data = vec![0; item_count * 4];
        let index = create_index(&data);

        let query_results: HashSet<usize> = index.query(-1, -1, 1, 1).into_iter().collect();
        let expected_results: HashSet<usize> = (0..data.len() / 4).collect();
        assert_eq!(query_results, expected_results);

        let query_results: HashSet<usize> = index.query(1, 1, 2, 2).into_iter().collect();
        assert!(query_results.is_empty());
    }
}

#[test]
fn zero_item_index_works() {
    let builder = StaticAABB2DIndexBuilder::<f64>::new(0);
    let index = builder.build().expect("should build OK");
    assert_eq!(index.count(), 0);
    assert_eq!(index.bounds(), None);
    let results: Vec<usize> = index.query(f64::MIN, f64::MIN, f64::MAX, f64::MAX);
    assert_eq!(results, &[]);
    let mut stack = Vec::new();
    let results: Vec<usize> =
        index.query_with_stack(f64::MIN, f64::MIN, f64::MAX, f64::MAX, &mut stack);
    assert_eq!(results, &[]);

    let iter_results: Vec<usize> = index
        .query_iter(f64::MIN, f64::MIN, f64::MAX, f64::MAX)
        .collect();
    assert_eq!(iter_results, &[]);

    let iter_results: Vec<usize> = index
        .query_iter_with_stack(f64::MIN, f64::MIN, f64::MAX, f64::MAX, &mut stack)
        .collect();
    assert_eq!(iter_results, &[]);

    let mut found_item = false;
    index.visit_neighbors(0.0, 0.0, &mut |_, _| {
        found_item = true;
        Control::Break(())
    });
    assert!(!found_item);
}

#[test]
fn building_from_too_few_items_errors() {
    let data = create_test_data();
    let mut builder = StaticAABB2DIndexBuilder::new(10);
    for pos in (0..9 * 4).step_by(4) {
        builder.add(data[pos], data[pos + 1], data[pos + 2], data[pos + 3]);
    }

    assert!(matches!(
        builder.build(),
        Err(StaticAABB2DIndexBuildError::ItemCountError {
            added: 9,
            expected: 10
        })
    ));
}

#[test]
fn building_from_too_many_items_errors() {
    let data = create_test_data();
    let mut builder = StaticAABB2DIndexBuilder::new(10);
    for pos in (0..20 * 4).step_by(4) {
        builder.add(data[pos], data[pos + 1], data[pos + 2], data[pos + 3]);
    }

    assert!(matches!(
        builder.build(),
        Err(StaticAABB2DIndexBuildError::ItemCountError {
            added: 20,
            expected: 10
        })
    ));
}

#[test]
fn skip_sorting_small_index() {
    let index = create_small_test_index();
    let total_bounds = index
        .bounds()
        .expect("items were added, bounds should exist");

    assert_eq!(total_bounds.min_x, 0);
    assert_eq!(total_bounds.min_y, 2);
    assert_eq!(total_bounds.max_x, 96);
    assert_eq!(total_bounds.max_y, 93);

    assert_eq!(index.level_bounds().len(), 2);
    assert_eq!(index.level_bounds(), vec![14, 15]);
    assert_eq!(index.all_boxes().len(), 15);

    let expected_item_boxes = [
        AABB::new(8, 62, 11, 66),
        AABB::new(57, 17, 57, 19),
        AABB::new(76, 26, 79, 29),
        AABB::new(36, 56, 38, 56),
        AABB::new(92, 77, 96, 80),
        AABB::new(87, 70, 90, 74),
        AABB::new(43, 41, 47, 43),
        AABB::new(0, 58, 2, 62),
        AABB::new(76, 86, 80, 89),
        AABB::new(27, 13, 27, 15),
        AABB::new(71, 63, 75, 67),
        AABB::new(25, 2, 27, 2),
        AABB::new(87, 6, 88, 6),
        AABB::new(22, 90, 23, 93),
    ];

    let actual_item_boxes = index.item_boxes();
    // note order should always match (should not be sorted differently from order added since
    // num_items < node_size)
    assert_eq!(actual_item_boxes, &expected_item_boxes);
}

#[test]
fn many_tree_levels() {
    let test_data = create_test_data();
    let input_boxes = aabb_from_data(&test_data);
    let index = create_index_with_node_size(&test_data, 4);

    assert_eq!(index.level_bounds(), vec![100, 125, 132, 134, 135]);
    assert_eq!(index.count(), test_data.len() / 4);
    assert_eq!(
        index.all_boxes().len(),
        *index.level_bounds().last().unwrap()
    );

    let all_boxes = index.all_boxes();

    // all_box_indices should map back to original aabb index
    for (i, curr_box) in all_boxes.iter().enumerate().take(index.count()) {
        let added_item_index = index.all_box_indices()[i];
        assert_eq!(&input_boxes[added_item_index], curr_box);
    }

    // all_box_indices should get child start index
    for parent_node_index in index.count()..all_boxes.len() - 1 {
        let children_start_index = index.all_box_indices()[parent_node_index];
        let children_end_index = if parent_node_index == all_boxes.len() - 1 {
            all_boxes.len()
        } else {
            index.all_box_indices()[parent_node_index + 1]
        };

        // all child boxes should be contained by their parent
        for i in children_start_index..children_end_index {
            assert!(all_boxes[parent_node_index].contains_aabb(&all_boxes[i]));
        }
    }
}

#[test]
fn total_extents() {
    let index = create_test_index();
    let total_bounds = index
        .bounds()
        .expect("items were added, bounds should exist");
    assert_eq!(total_bounds.min_x, 0);
    assert_eq!(total_bounds.min_y, 1);
    assert_eq!(total_bounds.max_x, 96);
    assert_eq!(total_bounds.max_y, 95);
}

#[test]
fn expected_indices_order() {
    let index = create_test_index();
    let expected_indices = &[
        95, 92, 87, 70, 67, 64, 55, 52, 49, 43, 40, 11, 26, 19, 44, 9, 59, 84, 77, 39, 6, 75, 80,
        18, 23, 62, 58, 88, 86, 27, 90, 0, 73, 7, 37, 30, 13, 14, 48, 17, 56, 79, 25, 38, 85, 76,
        91, 66, 24, 33, 21, 3, 99, 16, 54, 28, 29, 68, 50, 31, 22, 72, 78, 83, 53, 89, 51, 93, 81,
        20, 8, 96, 4, 63, 74, 5, 47, 32, 10, 98, 61, 82, 57, 97, 65, 35, 41, 2, 45, 46, 36, 42, 69,
        34, 1, 60, 15, 94, 12, 71, 0, 16, 32, 48, 64, 80, 96, 100,
    ];
    assert_eq!(index.all_box_indices(), expected_indices);
}

#[test]
fn query() {
    let index = create_test_index();
    let mut results: Vec<usize> = index.query(40, 40, 60, 60);
    results.sort();
    let expected_indexes = vec![6, 29, 31, 75];
    assert_eq!(results, expected_indexes);
}

#[test]
fn query_with_many_levels() {
    let index = create_index_with_node_size(&create_test_data(), 4);
    let mut results: Vec<usize> = index.query(40, 40, 60, 60);
    results.sort();
    let expected_indexes = vec![6, 29, 31, 75];
    assert_eq!(results, expected_indexes);
}

#[test]
fn query_iter() {
    let index = create_test_index();
    let mut results: Vec<usize> = index.query_iter(40, 40, 60, 60).collect();
    results.sort();
    let expected_indexes = vec![6, 29, 31, 75];
    assert_eq!(results, expected_indexes);
}

#[test]
fn query_iter_with_many_levels() {
    let index = create_index_with_node_size(&create_test_data(), 4);
    let mut results: Vec<usize> = index.query_iter(40, 40, 60, 60).collect();
    results.sort();
    let expected_indexes = vec![6, 29, 31, 75];
    assert_eq!(results, expected_indexes);
}

#[test]
fn query_iter_with_stack() {
    let index = create_test_index();
    // start stack with some garbage to test it gets cleared before use
    let mut stack = vec![7, 7, 7];
    let mut results: Vec<usize> = index
        .query_iter_with_stack(40, 40, 60, 60, &mut stack)
        .collect();
    results.sort();
    let expected_indexes = vec![6, 29, 31, 75];
    assert_eq!(results, expected_indexes);
}

#[test]
fn query_iter_with_stack_with_many_levels() {
    let index = create_index_with_node_size(&create_test_data(), 4);
    // start stack with some garbage to test it gets cleared before use
    let mut stack = vec![7, 7, 7];
    let mut results: Vec<usize> = index
        .query_iter_with_stack(40, 40, 60, 60, &mut stack)
        .collect();
    results.sort();
    let expected_indexes = vec![6, 29, 31, 75];
    assert_eq!(results, expected_indexes);
}

#[test]
fn visit_query() {
    let index = create_test_index();
    let mut results = Vec::new();
    let mut visitor = |i| {
        results.push(i);
        Control::<()>::Continue
    };

    let break_value_result = index.visit_query(40, 40, 60, 60, &mut visitor);
    assert!(matches!(break_value_result, Control::Continue));

    results.sort();
    let expected_indexes = vec![6, 29, 31, 75];
    assert_eq!(results, expected_indexes);
}

#[test]
fn visit_query_with_many_levels() {
    let index = create_index_with_node_size(&create_test_data(), 4);
    let mut results = Vec::new();
    let mut visitor = |i| {
        results.push(i);
    };

    index.visit_query(40, 40, 60, 60, &mut visitor);

    results.sort();
    let expected_indexes = vec![6, 29, 31, 75];
    assert_eq!(results, expected_indexes);
}

#[test]
fn visit_query_with_stack() {
    let index = create_test_index();
    // start stack with some garbage to test it gets cleared before use
    let mut stack = vec![7, 7, 7];
    let mut results = Vec::new();
    let mut visitor = |i| {
        results.push(i);
        Control::<()>::Continue
    };

    let break_value_result = index.visit_query_with_stack(40, 40, 60, 60, &mut visitor, &mut stack);
    assert!(matches!(break_value_result, Control::Continue));

    results.sort();
    let expected_indexes = vec![6, 29, 31, 75];
    assert_eq!(results, expected_indexes);
}

#[test]
fn visit_query_with_stack_with_many_levels() {
    let index = create_index_with_node_size(&create_test_data(), 4);
    // start stack with some garbage to test it gets cleared before use
    let mut stack = vec![7, 7, 7];
    let mut results = Vec::new();
    let mut visitor = |i| {
        results.push(i);
    };

    index.visit_query_with_stack(40, 40, 60, 60, &mut visitor, &mut stack);

    results.sort();
    let expected_indexes = vec![6, 29, 31, 75];
    assert_eq!(results, expected_indexes);
}

#[test]
fn visit_query_stops_early() {
    let index = create_test_index();
    let mut results = HashSet::new();
    let mut visitor = |i| {
        results.insert(i);
        if results.len() != 2 {
            Control::Continue
        } else {
            Control::Break("test")
        }
    };

    let break_value_result = index.visit_query(40, 40, 60, 60, &mut visitor);
    assert!(matches!(break_value_result, Control::Break("test")));
    assert_eq!(results.len(), 2);
    let expected_superset_indexes: HashSet<usize> = [6, 29, 31, 75].iter().cloned().collect();
    assert!(results.is_subset(&expected_superset_indexes));
}

#[test]
fn visit_neighbors_max_results() {
    let index = create_test_index();
    let mut results = Vec::new();
    let max_results = 3;
    let mut visitor = |i, _| {
        results.push(i);
        if results.len() < max_results {
            Control::Continue
        } else {
            Control::Break("hello")
        }
    };

    let break_value_result = index.visit_neighbors(50, 50, &mut visitor);
    assert!(matches!(break_value_result, Control::Break("hello")));
    results.sort();
    let expected_indexes = vec![6, 31, 75];
    assert_eq!(results, expected_indexes);
}

#[test]
fn visit_neighbors_max_distance() {
    let index = create_test_index();
    let mut results = Vec::new();
    let max_distance = 12.0;
    let max_distance_squared = max_distance * max_distance;
    let mut visitor = |i, d| {
        if (d as f64) < max_distance_squared {
            results.push(i);
            return Control::Continue;
        }
        Control::Break(42)
    };

    let break_value_result = index.visit_neighbors(50, 50, &mut visitor);
    assert!(matches!(break_value_result, Control::Break(42)));
    results.sort();
    let expected_indexes = vec![6, 29, 31, 75, 85];
    assert_eq!(results, expected_indexes);
}

#[test]
fn visit_neighbors_max_results_filtered() {
    let index = create_test_index();
    let mut results = Vec::new();
    let max_results = 6;
    let mut visitor = |i, _| {
        // filtering by only collecting indexes which are even
        if i % 2 == 0 {
            results.push(i);
            if results.len() < max_results {
                return Control::Continue;
            }

            return Control::Break(88);
        }
        Control::Continue
    };

    let break_value_result = index.visit_neighbors(50, 50, &mut visitor);
    assert!(matches!(break_value_result, Control::Break(88)));
    results.sort();
    let expected_indexes = vec![6, 16, 18, 24, 54, 80];
    assert_eq!(results, expected_indexes);
}

#[test]
fn visit_neighbors_all_items() {
    let index = create_test_index();
    let mut results = Vec::new();
    let mut visitor = |i, _| {
        results.push(i);
    };

    index.visit_neighbors(50, 50, &mut visitor);
    results.sort();
    let expected_indexes = (0..index.count()).collect::<Vec<_>>();
    assert_eq!(results, expected_indexes);
}
