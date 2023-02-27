use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};

use static_aabb2d_index::*;

use std::f64::consts::TAU;
const RADIUS: f64 = 100.0;
const QUERY_COUNT: usize = 100;

struct Point(f64, f64);

fn create_points_on_circle(count: usize) -> Vec<Point> {
    let mut result: Vec<Point> = Vec::with_capacity(count);

    for i in 0..count {
        let angle = (i as f64 / count as f64) * TAU;
        let x = RADIUS * angle.cos();
        let y = RADIUS * angle.sin();
        result.push(Point(x, y));
    }

    result
}

#[derive(Debug)]
struct BoundingBox(f64, f64, f64, f64);

fn create_boxes_from_point_pairs(points: &Vec<Point>) -> Vec<BoundingBox> {
    let mut result: Vec<BoundingBox> = Vec::new();
    let mut add_box = |pt1: &Point, pt2: &Point| {
        let (min_x, max_x) = {
            if pt1.0 < pt2.0 {
                (pt1.0, pt2.0)
            } else {
                (pt2.0, pt1.0)
            }
        };

        let (min_y, max_y) = {
            if pt1.1 < pt2.1 {
                (pt1.1, pt2.1)
            } else {
                (pt2.1, pt1.1)
            }
        };

        result.push(BoundingBox(min_x, min_y, max_x, max_y));
    };
    for pts in points.windows(2) {
        add_box(&pts[0], &pts[1]);
    }
    // push last pair
    add_box(&points[points.len() - 1], &points[0]);
    result
}

fn index_from_boxes(boxes: &[BoundingBox]) -> StaticAABB2DIndex<f64> {
    let mut builder = StaticAABB2DIndexBuilder::new(boxes.len());
    for b in boxes {
        builder.add(b.0, b.1, b.2, b.3);
    }

    builder.build().unwrap()
}

fn bench_create_index(b: &mut Bencher, boxes: &[BoundingBox]) {
    b.iter(|| index_from_boxes(boxes))
}

fn create_index_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_index");
    let item_counts = [100, 1_000, 10_000, 100_000, 1_000_000];
    for i in item_counts {
        group.bench_with_input(BenchmarkId::new("create_index", i), &i, |b, i| {
            bench_create_index(
                b,
                &create_boxes_from_point_pairs(&create_points_on_circle(*i)),
            )
        });
    }

    group.finish();
}

fn bench_visit_query(b: &mut Bencher, index: &StaticAABB2DIndex<f64>) {
    let mut query_results: Vec<usize> = Vec::new();
    let delta = 1.0;
    let step = index.count() / QUERY_COUNT;
    b.iter(|| {
        for b in index.item_boxes().iter().step_by(step) {
            query_results.clear();
            index.visit_query(
                b.min_x - delta,
                b.min_y - delta,
                b.max_x + delta,
                b.max_y + delta,
                &mut |index: usize| {
                    query_results.push(index);
                },
            );
        }
    })
}

fn bench_query_iter(b: &mut Bencher, index: &StaticAABB2DIndex<f64>) {
    let mut query_results: Vec<usize> = Vec::new();
    let delta = 1.0;
    let step = index.count() / QUERY_COUNT;
    b.iter(|| {
        for b in index.item_boxes().iter().step_by(step) {
            query_results.clear();
            for i in index.query_iter(
                b.min_x - delta,
                b.min_y - delta,
                b.max_x + delta,
                b.max_y + delta,
            ) {
                query_results.push(i);
            }
        }
    })
}

fn bench_query_iter_reuse_stack(b: &mut Bencher, index: &StaticAABB2DIndex<f64>) {
    let mut query_results: Vec<usize> = Vec::new();
    let mut stack = Vec::with_capacity(16);
    let delta = 1.0;
    let step = index.count() / QUERY_COUNT;
    b.iter(|| {
        for b in index.item_boxes().iter().step_by(step) {
            query_results.clear();
            for i in index.query_iter_with_stack(
                b.min_x - delta,
                b.min_y - delta,
                b.max_x + delta,
                b.max_y + delta,
                &mut stack,
            ) {
                query_results.push(i);
            }
        }
    })
}

fn bench_visit_query_reuse_stack(b: &mut Bencher, index: &StaticAABB2DIndex<f64>) {
    let mut query_results: Vec<usize> = Vec::new();
    let mut stack = Vec::with_capacity(16);
    let delta = 1.0;
    let step = index.count() / QUERY_COUNT;
    b.iter(|| {
        for b in index.item_boxes().iter().step_by(step) {
            query_results.clear();
            index.visit_query_with_stack(
                b.min_x - delta,
                b.min_y - delta,
                b.max_x + delta,
                b.max_y + delta,
                &mut |index: usize| {
                    query_results.push(index);
                },
                &mut stack,
            );
        }
    })
}

fn query_index_group(c: &mut Criterion) {
    fn create_index_with_count(i: usize) -> StaticAABB2DIndex<f64> {
        index_from_boxes(&create_boxes_from_point_pairs(&create_points_on_circle(i)))
    }
    let mut group = c.benchmark_group("query_index");
    let item_counts = [100, 1_000, 10_000, 100_000, 1_000_000];
    for i in item_counts {
        group.bench_with_input(BenchmarkId::new("visit_query", i), &i, |b, i| {
            bench_visit_query(b, &create_index_with_count(*i))
        });
        group.bench_with_input(BenchmarkId::new("query_iter", i), &i, |b, i| {
            bench_query_iter(b, &create_index_with_count(*i))
        });
        group.bench_with_input(BenchmarkId::new("query_iter_reuse_stack", i), &i, |b, i| {
            bench_query_iter_reuse_stack(b, &create_index_with_count(*i))
        });
        group.bench_with_input(
            BenchmarkId::new("visit_query_reuse_stack", i),
            &i,
            |b, i| bench_visit_query_reuse_stack(b, &create_index_with_count(*i)),
        );
    }

    group.finish();
}

fn bench_neighbors(b: &mut Bencher, index: &StaticAABB2DIndex<f64>) {
    let mut query_results = Vec::new();
    let max_neighbors = 10;
    b.iter(|| {
        for b in index.item_boxes() {
            query_results.clear();
            index.visit_neighbors(b.min_x, b.min_y, &mut |index: usize, _| {
                query_results.push(index);
                if query_results.len() == max_neighbors {
                    Control::Break(())
                } else {
                    Control::Continue
                }
            });
        }
    })
}

fn bench_neighbors_reuse_queue(b: &mut Bencher, index: &StaticAABB2DIndex<f64>) {
    let mut query_results = Vec::new();
    let mut queue = NeighborPriorityQueue::new();
    let max_neighbors = 10;
    b.iter(|| {
        for b in index.item_boxes() {
            query_results.clear();
            index.visit_neighbors_with_queue(
                b.min_x,
                b.min_y,
                &mut |index: usize, _| {
                    query_results.push(index);
                    if query_results.len() == max_neighbors {
                        Control::Break(())
                    } else {
                        Control::Continue
                    }
                },
                &mut queue,
            );
        }
    })
}

fn nearest_neighbors_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_neighbors");
    let item_counts = [100, 1_000, 10_000, 100_000];
    fn create_index_with_count(i: usize) -> StaticAABB2DIndex<f64> {
        index_from_boxes(&create_boxes_from_point_pairs(&create_points_on_circle(i)))
    }
    for i in item_counts {
        group.bench_with_input(BenchmarkId::new("visit_neighbors", i), &i, |b, i| {
            bench_neighbors(b, &create_index_with_count(*i))
        });
        group.bench_with_input(
            BenchmarkId::new("visit_neighbors_reuse_queue", i),
            &i,
            |b, i| bench_neighbors_reuse_queue(b, &create_index_with_count(*i)),
        );
    }

    group.finish();
}

criterion_group!(create_index, create_index_group,);
criterion_group!(query_index, query_index_group);
criterion_group!(nearest_neighbors, nearest_neighbors_group);
criterion_main!(create_index, query_index, nearest_neighbors);
