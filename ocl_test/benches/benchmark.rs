use ocl_test::*;

use rand::{thread_rng, Rng};
use criterion::{black_box, Bencher};
use criterion::{criterion_group, criterion_main, Criterion, Fun};


fn share_bench(bench: &mut Bencher, _i: &()) {
/*
    let p = 4610415792919412737u64;
    let r2 = 1266473570726112470u64;
    let r3 = 2230453091198852918u64;

    let mut pss = OclContext::<u64>::new(p, r2, r3, 
        512, 729, 51200, 512, 700).unwrap();

	let mut rng = thread_rng();
    let mut secrets = vec![0u64; 51200];
    for i in 0..51200 {
        secrets[i] = rng.gen_range(0, u64::MAX);
    }
*/

    let p = 3073700804129980417u64;
    let r2 = 414345200490731620u64;
    let r3 = 1697820560572790570u64;

    let mut pss = OclContext::<u64>::new(p, r2, r3, 
        8, 27, 200, 5, 25).unwrap();

    let mut rng = thread_rng();
    let mut secrets = vec![0u64; 200];
    for i in 0..200 {
        secrets[i] = rng.gen_range(0, u64::MAX);
    }

    bench.iter(|| black_box(
    	pss.share(black_box(&secrets))
    ));
}

fn criterion_benchmark(c: &mut Criterion) {
	c.bench_functions(
		"pss",
        vec![
            Fun::new("share", share_bench),
        ],
        (),
	);
}

criterion_group!{
	name = benches;
	config = Criterion::default().sample_size(50);
	targets = criterion_benchmark
}
criterion_main!(benches);