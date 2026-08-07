use criterion::{Bencher, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use static_aabb2d_index::{
    Control, NeighborPriorityQueue, StaticAABB2DIndex, StaticAABB2DIndexBuilder,
};
use std::hint::black_box;

const QUERY_COUNT: usize = 100;
const MAX_NEIGHBORS: usize = 10;

#[derive(Clone, Copy, Debug)]
struct BoundingBox(f64, f64, f64, f64);

fn grid_columns(count: usize) -> usize {
    // ceil(sqrt(count)), with one column for an empty grid
    count.saturating_sub(1).isqrt() + 1
}

fn next_random(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn sample_index(state: &mut u64, upper_bound: usize) -> usize {
    usize::try_from(next_random(state) % u64::try_from(upper_bound).unwrap()).unwrap()
}

fn sample_unit(state: &mut u64) -> f64 {
    let sample = u32::try_from(next_random(state) >> 32).unwrap();
    f64::from(sample) / (f64::from(u32::MAX) + 1.0)
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap())
}

#[allow(
    clippy::cast_precision_loss,
    reason = "Precision loss does not materially affect benchmark box placement."
)]
fn create_grid_boxes(count: usize) -> Vec<BoundingBox> {
    let columns = grid_columns(count);

    let mut boxes = Vec::with_capacity(count);
    for i in 0..count {
        let x = (i % columns) as f64 * 2.0;
        let y = (i / columns) as f64 * 2.0;
        boxes.push(BoundingBox(x, y, x + 1.0, y + 1.0));
    }
    boxes
}

fn shuffle_boxes(boxes: &mut [BoundingBox]) {
    let mut state = 0x8EBC_6AF0_9C88_C6E3_u64;
    for i in (1..boxes.len()).rev() {
        boxes.swap(i, sample_index(&mut state, i + 1));
    }
}

fn create_circle_boxes(count: usize) -> Vec<BoundingBox> {
    let radius = 10_000.0;
    let angle_step = std::f64::consts::TAU / usize_to_f64(count);
    (0..count)
        .map(|i| {
            let angle = usize_to_f64(i) * angle_step;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            BoundingBox(x, y, x + 1.0, y + 1.0)
        })
        .collect()
}

fn create_figure_eight_boxes(count: usize) -> Vec<BoundingBox> {
    let radius = 10_000.0;
    let angle_step = std::f64::consts::TAU / usize_to_f64(count);
    (0..count)
        .map(|i| {
            let angle = usize_to_f64(i) * angle_step;
            let sin = angle.sin();
            let x = radius * sin;
            let y = radius * sin * angle.cos();
            BoundingBox(x, y, x + 1.0, y + 1.0)
        })
        .collect()
}

fn create_triangle_boxes(count: usize) -> Vec<BoundingBox> {
    let row_count = count.saturating_mul(2).isqrt() + 1;
    let mut boxes = Vec::with_capacity(count);
    'rows: for row in 0..row_count {
        let y = usize_to_f64(row) * 3.0_f64.sqrt();
        let x_offset = -usize_to_f64(row);
        for column in 0..=row {
            let x = usize_to_f64(column) * 2.0 + x_offset;
            boxes.push(BoundingBox(x, y, x + 1.0, y + 1.0));
            if boxes.len() == count {
                break 'rows;
            }
        }
    }
    boxes
}

fn create_clustered_boxes(count: usize) -> Vec<BoundingBox> {
    const CENTERS: [(f64, f64); 8] = [
        (-300.0, -300.0),
        (0.0, -300.0),
        (300.0, -300.0),
        (-300.0, 0.0),
        (300.0, 0.0),
        (-300.0, 300.0),
        (0.0, 300.0),
        (300.0, 300.0),
    ];
    let mut state = 0xA076_1D64_78BD_642F_u64;
    (0..count)
        .map(|_| {
            let (center_x, center_y) = CENTERS[sample_index(&mut state, CENTERS.len())];
            let x = center_x + (sample_unit(&mut state) - 0.5) * 80.0;
            let y = center_y + (sample_unit(&mut state) - 0.5) * 80.0;
            BoundingBox(x, y, x + 1.0, y + 1.0)
        })
        .collect()
}

fn hilbert_index_to_xy(mut index: u32) -> (u16, u16) {
    let original_index = index;
    let mut x = 0;
    let mut y = 0;
    let mut scale = 1;

    while scale < 65_536 {
        let rotate_x = (index >> 1) & 1;
        let rotate_y = (index ^ rotate_x) & 1;
        if rotate_y == 0 {
            if rotate_x == 1 {
                x = scale - 1 - x;
                y = scale - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        x += scale * rotate_x;
        y += scale * rotate_y;
        index >>= 2;
        scale <<= 1;
    }

    let xy = (u16::try_from(x).unwrap(), u16::try_from(y).unwrap());
    debug_assert_eq!(
        static_aabb2d_index::hilbert_xy_to_index(xy.0, xy.1),
        original_index
    );
    xy
}

fn create_two_value_boxes(count: usize) -> Vec<BoundingBox> {
    let first_hilbert_value = static_aabb2d_index::hilbert_xy_to_index(32767, 32767);
    let second_hilbert_value = first_hilbert_value + 1;
    let (first_x, first_y) = hilbert_index_to_xy(first_hilbert_value);
    let (second_x, second_y) = hilbert_index_to_xy(second_hilbert_value);

    (0..count)
        .map(|i| {
            if i == 0 {
                BoundingBox(0.0, 0.0, 65535.0, 65535.0)
            } else if i % 2 == 0 {
                let x = f64::from(first_x);
                let y = f64::from(first_y);
                BoundingBox(x, y, x, y)
            } else {
                let x = f64::from(second_x);
                let y = f64::from(second_y);
                BoundingBox(x, y, x, y)
            }
        })
        .collect()
}

fn create_adversarial_boxes(count: usize) -> Vec<BoundingBox> {
    assert!(count > 1);
    let mut sorted_hilbert_values = (0..count)
        .map(|rank| {
            u32::try_from(
                u64::try_from(rank).unwrap() * u64::from(u32::MAX)
                    / u64::try_from(count - 1).unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    // This box sets both construction extents to exactly 0..=u16::MAX while its center still maps
    // to the chosen Hilbert value. That keeps all other generated coordinates unchanged by scaling.
    let extent_box_hilbert_value = static_aabb2d_index::hilbert_xy_to_index(32767, 32767);
    match sorted_hilbert_values.binary_search(&extent_box_hilbert_value) {
        Ok(_) => {}
        Err(position) => sorted_hilbert_values[position] = extent_box_hilbert_value,
    }

    // Assign decreasing values to the element that each recursive midpoint partition will choose.
    // Each pivot is the range maximum, so the large left partition grows the quicksort call stack.
    let mut positions = (0..count).collect::<Vec<_>>();
    let mut ranks = vec![usize::MAX; count];
    let left = 0;
    let mut right = count - 1;
    let mut rank = count;
    while left / 16 < right / 16 {
        let mid = left.midpoint(right);
        rank -= 1;
        ranks[positions[mid]] = rank;
        positions.swap(mid, right);
        right -= 1;
    }
    for &position in &positions[left..=right] {
        rank -= 1;
        ranks[position] = rank;
    }
    debug_assert_eq!(rank, 0);

    ranks
        .into_iter()
        .map(|rank| {
            let hilbert_value = sorted_hilbert_values[rank];
            if hilbert_value == extent_box_hilbert_value {
                return BoundingBox(0.0, 0.0, 65535.0, 65535.0);
            }
            let (x, y) = hilbert_index_to_xy(hilbert_value);
            let x = f64::from(x);
            let y = f64::from(y);
            BoundingBox(x, y, x, y)
        })
        .collect()
}

fn create_single_hit_queries(boxes: &[BoundingBox]) -> Vec<BoundingBox> {
    let mut state = 0xD1B5_4A32_D192_ED03_u64;
    (0..QUERY_COUNT)
        .map(|_| boxes[sample_index(&mut state, boxes.len())])
        .collect()
}

fn create_hundred_hit_queries(boxes: &[BoundingBox]) -> Vec<BoundingBox> {
    let columns = grid_columns(boxes.len());
    let complete_rows = boxes.len() / columns;
    let start_column_count = columns - 9;
    let start_row_count = complete_rows - 9;
    let mut state = 0x94D0_49BB_1331_11EB_u64;

    (0..QUERY_COUNT)
        .map(|_| {
            let column = sample_index(&mut state, start_column_count);
            let row = sample_index(&mut state, start_row_count);
            let top_left = boxes[row * columns + column];
            BoundingBox(top_left.0, top_left.1, top_left.0 + 19.0, top_left.1 + 19.0)
        })
        .collect()
}

fn create_miss_queries() -> Vec<BoundingBox> {
    vec![BoundingBox(-2.0, -2.0, -1.0, -1.0); QUERY_COUNT]
}

fn assert_query_result_count(
    index: &StaticAABB2DIndex<f64>,
    queries: &[BoundingBox],
    expected: usize,
) {
    for query in queries {
        assert_eq!(
            index.query(query.0, query.1, query.2, query.3).len(),
            expected
        );
    }
}

fn index_from_boxes(boxes: &[BoundingBox]) -> StaticAABB2DIndex<f64> {
    let mut builder = StaticAABB2DIndexBuilder::new(boxes.len());
    for b in boxes {
        builder.add(b.0, b.1, b.2, b.3);
    }

    builder.build().unwrap()
}

fn bench_create_index(b: &mut Bencher, boxes: &[BoundingBox]) {
    b.iter(|| index_from_boxes(black_box(boxes)));
}

fn create_index_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_index");
    let item_counts = [100, 10_000, 1_000_000];
    for count in item_counts {
        let mut boxes = create_grid_boxes(count);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &boxes, |b, boxes| {
            bench_create_index(b, boxes);
        });
        if count == 1_000_000 {
            shuffle_boxes(&mut boxes);
            group.bench_with_input(
                BenchmarkId::new("shuffled_grid", count),
                &boxes,
                |b, boxes| bench_create_index(b, boxes),
            );
        }
    }

    let count = 1_000_000;
    group.throughput(Throughput::Elements(count as u64));
    for (name, create_boxes) in [
        (
            "circle",
            create_circle_boxes as fn(usize) -> Vec<BoundingBox>,
        ),
        ("figure_eight", create_figure_eight_boxes),
        ("triangle", create_triangle_boxes),
        ("clusters", create_clustered_boxes),
    ] {
        let mut boxes = create_boxes(count);
        group.bench_with_input(BenchmarkId::new(name, count), &boxes, |b, boxes| {
            bench_create_index(b, boxes);
        });
        shuffle_boxes(&mut boxes);
        group.bench_with_input(
            BenchmarkId::new(format!("shuffled_{name}"), count),
            &boxes,
            |b, boxes| bench_create_index(b, boxes),
        );
    }

    let identical_boxes = vec![BoundingBox(0.0, 0.0, 1.0, 1.0); count];
    group.bench_with_input(
        BenchmarkId::new("identical", count),
        &identical_boxes,
        |b, boxes| bench_create_index(b, boxes),
    );

    let two_value_boxes = create_two_value_boxes(count);
    group.bench_with_input(
        BenchmarkId::new("two_adjacent_hilbert_values", count),
        &two_value_boxes,
        |b, boxes| bench_create_index(b, boxes),
    );

    let count = 10_000;
    let adversarial_boxes = create_adversarial_boxes(count);
    group.throughput(Throughput::Elements(count as u64));
    group.bench_with_input(
        BenchmarkId::new("adversarial_midpoint_pivot", count),
        &adversarial_boxes,
        |b, boxes| bench_create_index(b, boxes),
    );

    group.finish();
}

fn bench_brute_force_query(b: &mut Bencher, boxes: &[BoundingBox], queries: &[BoundingBox]) {
    let mut query_results = Vec::with_capacity(16);
    b.iter(|| {
        let mut result_count = 0;
        let boxes = black_box(boxes);
        for query in black_box(queries) {
            query_results.clear();
            for (index, bbox) in boxes.iter().enumerate() {
                if bbox.2 < query.0 || bbox.3 < query.1 || bbox.0 > query.2 || bbox.1 > query.3 {
                    continue;
                }
                query_results.push(index);
            }
            result_count += query_results.len();
            black_box(query_results.as_slice());
        }
        result_count
    });
}

fn bench_query(b: &mut Bencher, index: &StaticAABB2DIndex<f64>, queries: &[BoundingBox]) {
    b.iter(|| {
        let mut result_count = 0;
        for query in black_box(queries) {
            let results = index.query(query.0, query.1, query.2, query.3);
            result_count += results.len();
            black_box(results.as_slice());
        }
        result_count
    });
}

fn bench_query_reuse_stack(
    b: &mut Bencher,
    index: &StaticAABB2DIndex<f64>,
    queries: &[BoundingBox],
) {
    let mut stack = Vec::with_capacity(16);
    if let Some(query) = queries.first() {
        drop(index.query_with_stack(query.0, query.1, query.2, query.3, &mut stack));
    }

    b.iter(|| {
        let mut result_count = 0;
        for query in black_box(queries) {
            let results = index.query_with_stack(query.0, query.1, query.2, query.3, &mut stack);
            result_count += results.len();
            black_box(results.as_slice());
        }
        result_count
    });
}

fn bench_visit_query(b: &mut Bencher, index: &StaticAABB2DIndex<f64>, queries: &[BoundingBox]) {
    let mut query_results = Vec::with_capacity(16);
    b.iter(|| {
        let mut result_count = 0;
        for query in black_box(queries) {
            query_results.clear();
            index.visit_query(query.0, query.1, query.2, query.3, &mut |index: usize| {
                query_results.push(index);
            });
            result_count += query_results.len();
            black_box(query_results.as_slice());
        }
        result_count
    });
}

fn bench_visit_query_reuse_stack(
    b: &mut Bencher,
    index: &StaticAABB2DIndex<f64>,
    queries: &[BoundingBox],
) {
    let mut query_results = Vec::with_capacity(16);
    let mut stack = Vec::with_capacity(16);
    if let Some(query) = queries.first() {
        index.visit_query_with_stack(
            query.0,
            query.1,
            query.2,
            query.3,
            &mut |index: usize| query_results.push(index),
            &mut stack,
        );
    }

    b.iter(|| {
        let mut result_count = 0;
        for query in black_box(queries) {
            query_results.clear();
            index.visit_query_with_stack(
                query.0,
                query.1,
                query.2,
                query.3,
                &mut |index: usize| query_results.push(index),
                &mut stack,
            );
            result_count += query_results.len();
            black_box(query_results.as_slice());
        }
        result_count
    });
}

fn bench_query_iter(b: &mut Bencher, index: &StaticAABB2DIndex<f64>, queries: &[BoundingBox]) {
    let mut query_results = Vec::with_capacity(16);
    b.iter(|| {
        let mut result_count = 0;
        for query in black_box(queries) {
            query_results.clear();
            query_results.extend(index.query_iter(query.0, query.1, query.2, query.3));
            result_count += query_results.len();
            black_box(query_results.as_slice());
        }
        result_count
    });
}

fn bench_query_iter_reuse_stack(
    b: &mut Bencher,
    index: &StaticAABB2DIndex<f64>,
    queries: &[BoundingBox],
) {
    let mut query_results = Vec::with_capacity(16);
    let mut stack = Vec::with_capacity(16);
    if let Some(query) = queries.first() {
        let _ = index
            .query_iter_with_stack(query.0, query.1, query.2, query.3, &mut stack)
            .count();
    }

    b.iter(|| {
        let mut result_count = 0;
        for query in black_box(queries) {
            query_results.clear();
            query_results.extend(
                index.query_iter_with_stack(query.0, query.1, query.2, query.3, &mut stack),
            );
            result_count += query_results.len();
            black_box(query_results.as_slice());
        }
        result_count
    });
}

fn query_scale_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_scale");
    let item_counts = [1_000, 100_000, 1_000_000];
    group.throughput(Throughput::Elements(QUERY_COUNT as u64));

    for count in item_counts {
        let boxes = create_grid_boxes(count);
        let index = index_from_boxes(&boxes);
        let miss_queries = create_miss_queries();
        let single_hit_queries = create_single_hit_queries(&boxes);
        let hundred_hit_queries = create_hundred_hit_queries(&boxes);
        assert_query_result_count(&index, &miss_queries, 0);
        assert_query_result_count(&index, &single_hit_queries, 1);
        assert_query_result_count(&index, &hundred_hit_queries, 100);

        group.bench_function(BenchmarkId::new("brute_force_single_hit", count), |b| {
            bench_brute_force_query(b, &boxes, &single_hit_queries);
        });
        let scenarios = [
            ("miss", miss_queries),
            ("single_hit", single_hit_queries),
            ("hundred_hits", hundred_hit_queries),
        ];
        for (scenario, queries) in scenarios {
            group.bench_function(BenchmarkId::new(scenario, count), |b| {
                bench_visit_query_reuse_stack(b, &index, &queries);
            });
        }
    }

    group.finish();
}

fn query_api_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_api");
    let count = 100_000;
    let boxes = create_grid_boxes(count);
    let index = index_from_boxes(&boxes);
    let queries = create_single_hit_queries(&boxes);
    assert_query_result_count(&index, &queries, 1);
    group.throughput(Throughput::Elements(QUERY_COUNT as u64));

    group.bench_function(BenchmarkId::new("query", count), |b| {
        bench_query(b, &index, &queries);
    });
    group.bench_function(BenchmarkId::new("query_reuse_stack", count), |b| {
        bench_query_reuse_stack(b, &index, &queries);
    });
    group.bench_function(BenchmarkId::new("visit_query", count), |b| {
        bench_visit_query(b, &index, &queries);
    });
    group.bench_function(BenchmarkId::new("visit_query_reuse_stack", count), |b| {
        bench_visit_query_reuse_stack(b, &index, &queries);
    });
    group.bench_function(BenchmarkId::new("query_iter", count), |b| {
        bench_query_iter(b, &index, &queries);
    });
    group.bench_function(BenchmarkId::new("query_iter_reuse_stack", count), |b| {
        bench_query_iter_reuse_stack(b, &index, &queries);
    });

    group.finish();
}

fn create_neighbor_query_points(boxes: &[BoundingBox]) -> Vec<(f64, f64)> {
    create_single_hit_queries(boxes)
        .into_iter()
        .map(|b| ((b.0 + b.2) * 0.5, (b.1 + b.3) * 0.5))
        .collect()
}

fn bench_neighbors(b: &mut Bencher, index: &StaticAABB2DIndex<f64>, query_points: &[(f64, f64)]) {
    let mut query_results = Vec::with_capacity(MAX_NEIGHBORS);
    b.iter(|| {
        let mut result_count = 0;
        for &(x, y) in black_box(query_points) {
            query_results.clear();
            index.visit_neighbors(x, y, &mut |index: usize, _| {
                query_results.push(index);
                if query_results.len() == MAX_NEIGHBORS {
                    Control::Break(())
                } else {
                    Control::Continue
                }
            });
            result_count += query_results.len();
            black_box(query_results.as_slice());
        }
        result_count
    });
}

fn bench_neighbors_reuse_queue(
    b: &mut Bencher,
    index: &StaticAABB2DIndex<f64>,
    query_points: &[(f64, f64)],
) {
    let mut query_results = Vec::with_capacity(MAX_NEIGHBORS);
    let mut queue = NeighborPriorityQueue::new();
    if let Some(&(x, y)) = query_points.first() {
        let mut count = 0;
        index.visit_neighbors_with_queue(
            x,
            y,
            &mut |_: usize, _| {
                count += 1;
                if count == MAX_NEIGHBORS {
                    Control::Break(())
                } else {
                    Control::Continue
                }
            },
            &mut queue,
        );
    }

    b.iter(|| {
        let mut result_count = 0;
        for &(x, y) in black_box(query_points) {
            query_results.clear();
            index.visit_neighbors_with_queue(
                x,
                y,
                &mut |index: usize, _| {
                    query_results.push(index);
                    if query_results.len() == MAX_NEIGHBORS {
                        Control::Break(())
                    } else {
                        Control::Continue
                    }
                },
                &mut queue,
            );
            result_count += query_results.len();
            black_box(query_results.as_slice());
        }
        result_count
    });
}

fn nearest_neighbors_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_neighbors");
    let item_counts = [1_000, 100_000, 1_000_000];
    group.throughput(Throughput::Elements(QUERY_COUNT as u64));

    for count in item_counts {
        let boxes = create_grid_boxes(count);
        let index = index_from_boxes(&boxes);
        let query_points = create_neighbor_query_points(&boxes);

        if count == 100_000 {
            group.bench_function(BenchmarkId::new("visit_neighbors", count), |b| {
                bench_neighbors(b, &index, &query_points);
            });
        }
        group.bench_function(
            BenchmarkId::new("visit_neighbors_reuse_queue", count),
            |b| bench_neighbors_reuse_queue(b, &index, &query_points),
        );
    }

    group.finish();
}

criterion_group!(create_index, create_index_group);
criterion_group!(query_scale, query_scale_group);
criterion_group!(query_api, query_api_group);
criterion_group!(nearest_neighbors, nearest_neighbors_group);
criterion_main!(create_index, query_scale, query_api, nearest_neighbors);
