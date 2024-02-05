use criterion::{criterion_group, criterion_main, Criterion};
use fastrand;
use ndarray::Array2;
use photoncube2video::transforms::interpolate_where_mask;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

pub fn benchmark_interpolate_where_mask(c: &mut Criterion) {
    let mut rng = fastrand::Rng::with_seed(0x4d595df4d0f33173);

    let data = Array2::from_shape_fn((255, 255), |(i, j)| i.max(j) as u8);
    let mask = Array2::from_shape_simple_fn((255, 255), || rng.f32() < 0.01);

    c.bench_function("interpolate_where_mask", |b| {
        b.iter(|| {
            let _ = interpolate_where_mask(&data, &mask, false);
        })
    });
}

#[cfg(target_os = "linux")]
criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_profiler(
            PProfProfiler::new(100, Output::Flamegraph(None))
        );
    targets =
        benchmark_interpolate_where_mask,
}

#[cfg(not(target_os = "linux"))]
criterion_group! {
    benches,
    benchmark_interpolate_where_mask
}

criterion_main!(benches);
