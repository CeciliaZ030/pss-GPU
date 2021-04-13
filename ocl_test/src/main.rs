extern crate ocl;
use ocl_test::*;
use ocl_test::util::ModPow;


/*
use this prime : 4610415792919412737

512th root ot unity: 1266473570726112470

729th root of unity: 2230453091198852918

3073700804129980417, 414345200490731620, 1697820560572790570, 8, 27, 2, 10

*/

fn main() {
    //trivial_exploded().unwrap();

    println!("Hello, world!");

    let p = 4610415792919412737u64;
    let r2 = 1266473570726112470u64;
    let r3 = 2230453091198852918u64;

    let mut pss = OclContext::<u64>::new(p, (r2 as u128).modpow(4u128, p as u128) as u64, r3, 
        512/4, 729, 60000, 100, 700).unwrap();
    //prime: u64, root2: u64, root3:u64, degree2: usize, degree3: usize, 
    //total_len: usize, packing_len: usize, num_shares: usize
    
    let mut secrets = vec![0u64; 60000];
    for i in 0..60000 {
        secrets[i] = (i * i + 1) as u64;
    }

    pss.share(&secrets);
}
