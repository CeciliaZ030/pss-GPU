use pss_compare::*;
use pss_compare::util::ModPow;

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

    let p = 4610415792919412737u128;
    let r2 = 1266473570726112470u128;
    let r3 = 2230453091198852918u128;

    let args: Vec<String> = env::args().collect();
    let r2_divisor = args[1].parse::<usize>().unwrap();
    let total_len = args[2].parse::<usize>().unwrap();
    let packing_len = args[3].parse::<usize>().unwrap();
    let new_r2 = (r2 as u128).modpow(r2_divisor as u128, p as u128) as u128;
    let mut pss = PackedSecretSharing::new(p, new_r2, r3, 
        512/r2_divisor, 729, total_len, packing_len, 700);
    //prime: u128, root2: u128, root3:u128, degree2: usize, degree3: usize, 
    //total_len: usize, packing_len: usize, num_shares: usize
    let mut secrets = vec![0u128; total_len];
    for i in 0..total_len {
        secrets[i] = (i * i + 1) as u128;
    }

    pss.share_seperate(&secrets);
}