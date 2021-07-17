extern crate ocl;
use ocl_test::*;
use ocl_test::util::ModPow;

use std::env;


/*
use this prime : 4610415792919412737

512th root ot unity: 1266473570726112470

729th root of unity: 2230453091198852918

3073700804129980417, 414345200490731620, 1697820560572790570, 8, 27, 2, 10

*/

fn main() {
    //trivial_exploded().unwrap();

    println!("Hello, world!");

    let p = 3224862721u64;
    let r2 = 889378889u64;
    let r3 = 388768380u64;

    let args: Vec<String> = env::args().collect();
    let r2_divisor = args[1].parse::<usize>().unwrap();
    let total_len = args[2].parse::<usize>().unwrap();
    let packing_len = args[3].parse::<usize>().unwrap();

    let mut pss = OclContext::new(p, 
        (r2 as u128).modpow(r2_divisor as u128, p as u128) as u64, 
        (r3 as u128).modpow(9u128, p as u128) as u64, 
        512/r2_divisor, 729/9, total_len, packing_len, 729/9).unwrap();
    //prime: u64, root2: u64, root3:u64, degree2: usize, degree3: usize, 
    //total_len: usize, packing_len: usize, num_shares: usize
    
    let mut secrets = vec![0u64; total_len];
    for i in 0..total_len {
        secrets[i] = (1) as u64;
    }
    println!("{:?}", secrets);

    let shares = pss.share(&secrets);
    println!("Result {:?}", shares);
    let reconstruction = pss.reconstruct(&shares);
    println!("Result {:?}", &reconstruction);
}
