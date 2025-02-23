use static_aabb2d_index::{AABB, StaticAABB2DIndexBuilder};

fn main() {
    // this is an example demonstrating how to extract the index tree from an index
    let data = create_test_data();
    let mut input_boxes = Vec::new();
    // building index with node size of 4 to create more levels in the index tree
    let mut builder = StaticAABB2DIndexBuilder::new_with_node_size(data.len() / 4, 4);
    for pos in (0..data.len()).step_by(4) {
        builder.add(data[pos], data[pos + 1], data[pos + 2], data[pos + 3]);
        input_boxes.push(AABB::new(
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
        ));
    }

    let index = builder.build().unwrap();

    assert_eq!(index.level_bounds(), vec![100, 125, 132, 134, 135]);
    assert_eq!(index.count(), data.len() / 4);
    assert_eq!(
        index.all_boxes().len(),
        *index.level_bounds().last().unwrap()
    );

    // if only interested in the boxes and their associated level in the tree then just level_bounds
    // can be used while iterating all_boxes
    let level_bounds = index.level_bounds();
    let all_boxes = index.all_boxes();
    let mut level_bound_index = 0;
    let mut next_level_bound = level_bounds[level_bound_index];
    // boxes_in_levels holds a vector of AABB at each level in the index tree
    // with level 0 being associated with the items added, level 1 the parent nodes of those items
    // level 2 the parents of the level 1 nodes, etc.
    let mut boxes_in_levels = Vec::<Vec<AABB<i32>>>::new();
    boxes_in_levels.push(Vec::new());
    for (i, aabb) in all_boxes.iter().enumerate() {
        if i >= next_level_bound {
            level_bound_index += 1;
            next_level_bound = level_bounds[level_bound_index];
            boxes_in_levels.push(Vec::new());
        }

        boxes_in_levels.last_mut().unwrap().push(*aabb);
    }

    assert_eq!(boxes_in_levels[0].len(), level_bounds[0]);
    for i in 1..level_bounds.len() {
        assert_eq!(
            boxes_in_levels[i].len(),
            level_bounds[i] - level_bounds[i - 1]
        );
    }

    // dbg!(index.indices_map());
    // dbg!(index.indices_map().len());
    // dbg!(index.indices_map()[125]);

    // in order to access the actual tree node relations the all_box_indices method can be used
    // the indices_map will return the start index of the children for a given node index
    // or map back to the added item index in the case that the node has no children

    // going from all_boxes index to the index the item was originally added
    for (i, b) in all_boxes.iter().enumerate().take(index.count()) {
        let added_item_index = index.all_box_indices()[i];
        assert_eq!(input_boxes[added_item_index], *b);
    }

    // finding children node start index for a node
    // note: accessing index past the item count so it must be a node with children
    let parent_node_index = index.count() + 5;
    let children_start_index = index.all_box_indices()[parent_node_index];
    let children_end_index = index.all_box_indices()[parent_node_index + 1];

    let child_indices: Vec<usize> = (children_start_index..children_end_index).collect();
    // all child boxes should be contained by their parent
    for i in child_indices {
        assert!(all_boxes[parent_node_index].contains_aabb(&all_boxes[i]));
    }

    // in the case of the root index the end cannot be found using the indices_map
    // and must be checked for (the end is just all_boxes.len())
    // here we loop through all the parent nodes and check that their child boxes are within
    // the parent nodes extents and handle the case of the root node
    for parent_node_index in index.count()..all_boxes.len() - 1 {
        let children_start_index = index.all_box_indices()[parent_node_index];
        let children_end_index = if parent_node_index == all_boxes.len() - 1 {
            // root node, all_boxes length is the end
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
