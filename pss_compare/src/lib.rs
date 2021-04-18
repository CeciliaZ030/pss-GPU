#![allow(non_snake_case)]

use std::convert::*;
use time;

use core::fmt::Debug;

use rand::{thread_rng, Rng};
use rand::distributions::uniform::SampleUniform;

use num_traits::{One, Zero};
use num::traits::Unsigned;


mod ntt;
pub mod util;
pub use util::*;

#[derive(Clone, Debug)]
pub struct PackedSecretSharing<T> {

	prime: T,
	root2: T,
	root3: T,
	pub rootTable2: Vec<T>,
	pub rootTable3: Vec<T>,
	// degree of the sharing poly
	degree2: usize,
	degree3: usize,

	V: usize,
	L: usize,
	N: usize,
}

impl<T> PackedSecretSharing<T>
where T: ModPow + Unsigned + Copy + Debug + From<u64> + SampleUniform + PartialOrd,
{

	pub fn new(prime: T, root2: T, root3:T, 
			   degree2: usize, degree3: usize, total_len: usize, packing_len: usize, num_shares: usize) -> PackedSecretSharing<T> {
		assert!(total_len % packing_len == 0);
		assert!(packing_len <= degree2);
		assert!(degree2 <= num_shares);
		assert!(num_shares+1 <= degree3);
		let mut rootTable2: Vec<T> = Vec::new();
		for i in 0..degree2 {
			rootTable2.push(root2.modpow((i as u64).into(), prime));	
		}

  		let mut rootTable3: Vec<T> = Vec::new();
		for i in 0..degree3 as u64 {
			rootTable3.push(root3.modpow((i as u64).into(), prime));
		}
		PackedSecretSharing {

			prime: prime,
			root2: root2,
			root3: root3,
			rootTable2: rootTable2,
			rootTable3: rootTable3,

			degree2: degree2,
			degree3: degree3,
			V: total_len,
			L: packing_len,
			N: num_shares,
		}
	}

	pub fn share<U>(&mut self, secrets: &[U]) -> Vec<Vec<U>>
	where U: TryFrom<T> + Into<T> + Copy + HasMax + SampleUniform + Unsigned + Debug,
		  <U as TryFrom<T>>::Error: Debug
	{	
		/* Input Format
		   [x0, ..., xv]
		*/
		//assert!(secrets.len() == self.V);	
		let L2 = self.degree2;
		let L3 = self.degree3;
		let B = secrets.len() / self.L;
		let zero = U::zero();
		println!("V = {:?}, B = {}, L = {}", self.V, B, self.L);

		/* Convert U into T
		T has to be a larger integer type than U to prevent overflow
		*/
		let mut secret_blocks: Vec<Vec<T>> = vec![Vec::new(); B];
		let mut rng = thread_rng();
		for i in 0..B {
			for j in 0..self.L {
				secret_blocks[i].push(secrets[i*self.L + j].into());
			}
			/* Pack randomness for unused transform points
			randomness is no greater than max of U to prevent overflow
			*/
			for _ in self.L..L2 {
				secret_blocks[i].push(rng.gen_range(zero, &U::max()).into());
			}
		}

		println!("Start ");
        let kern_start = time::get_time();
		let mut ret: Vec<Vec<U>> = vec![vec![U::zero(); B]; self.N];
		for (i, block) in secret_blocks.iter().enumerate() {
			/* use radix2_DFT to from the poly
			*/
			let mut poly = ntt::inverse2(block.to_vec(), self.prime, &self.rootTable2);
			for _ in L2 ..L3 {
				poly.push(T::zero());
			}
			/* share with radix3_DFT
			*/
			let mut shares = ntt::transform3(poly, self.prime, &self.rootTable3);
			for j in 0..self.N {
				ret[j][i] = shares[j + 1].try_into().unwrap();
			}
		}
		print_elapsed("total elapsed", kern_start);
		/* Return Format:
		   [[s00, s01, ..., s0b],	//shares of party 0
		    [s10, s11, ..., s1b],	//shares of party 1
		    ...
		    [sm0, sm1, ..., smb]]	//shares of party m
		*/
		ret
	}

	pub fn share_seperate<U>(&mut self, secrets: &[U]) -> Vec<Vec<U>>
	where U: TryFrom<T> + Into<T> + Copy + HasMax + SampleUniform + Unsigned + Debug,
		  <U as TryFrom<T>>::Error: Debug
	{	
		/* Input Format
		   [x0, ..., xv]
		*/
		//assert!(secrets.len() == self.V);	
		let L2 = self.degree2;
		let L3 = self.degree3;
		let B = secrets.len() / self.L;
		let zero = U::zero();
		println!("V = {:?}, B = {}, L = {}", self.V, B, self.L);

		/* Convert U into T
		T has to be a larger integer type than U to prevent overflow
		*/
		let mut secret_blocks: Vec<Vec<T>> = vec![Vec::new(); B];
		let mut rng = thread_rng();
		for i in 0..B {
			for j in 0..self.L {
				secret_blocks[i].push(secrets[i*self.L + j].into());
			}
			/* Pack randomness for unused transform points
			randomness is no greater than max of U to prevent overflow
			*/
			for _ in self.L..L2 {
				secret_blocks[i].push(rng.gen_range(zero, &U::max()).into());
			}
		}

		println!("Start radix2");
        let kern_start = time::get_time();
        let mut polys = Vec::new();
		for (i, block) in secret_blocks.iter().enumerate() {
			/* use radix2_DFT to from the poly
			*/
			let mut poly = ntt::inverse2(block.to_vec(), self.prime, &self.rootTable2);
			polys.push(poly);
		}
		println!("{:?}, {}", polys.len(), polys[0].len());
		print_elapsed("total elapsed", kern_start);

		println!("Start radix3");
        let kern_start = time::get_time();
		let mut ret: Vec<Vec<U>> = vec![vec![U::zero(); B]; self.N];
		for (i, mut poly) in polys.into_iter().enumerate() {
			for _ in L2 ..L3 {
				poly.push(T::zero());
			}
			/* share with radix3_DFT
			*/
			let mut shares = ntt::transform3(poly, self.prime, &self.rootTable3);
			for j in 0..self.N {
				ret[j][i] = shares[j + 1].try_into().unwrap();
			}
		}
		print_elapsed("total elapsed", kern_start);

		/* Return Format:
		   [[s00, s01, ..., s0b],	//shares of party 0
		    [s10, s11, ..., s1b],	//shares of party 1
		    ...
		    [sm0, sm1, ..., smb]]	//shares of party m
		*/
		ret
	}



	pub fn reconstruct<U>(&self, shares: &[Vec<U>], shares_point: &[U]) -> Vec<U> 
	where  U: TryFrom<T> + Into<T> + Copy + HasMax + SampleUniform + Unsigned,
	   	   <U as TryFrom<T>>::Error: Debug
	{
		/* Input Format:
		   [[s00, s01, ..., s0b],	//shares of party 0
		    [s10, s11, ..., s1b],	//shares of party 1
		    ...
		    [sm0, sm1, ..., smb]]	//shares of party 

		Number of shares collected > than threshold
		but smaller than initially distributed number
		*/
		let B = self.V / self.L;
		let M = shares_point.len();
		assert!(shares.len() == shares_point.len());
		assert!(M >= self.degree2);
		assert!(M <= self.degree3);

		/* Convert U into T
		For shares, transpose into polys
		*/
		let mut converted_ponts = Vec::<T>::new();
		for p in shares_point {
			converted_ponts.push((*p).into());
		}
		let mut converted_shares = vec![vec![T::zero(); M]; B];
		for i in 0..M {
			for (j, s) in shares[i].iter().enumerate() {
				converted_shares[j][i] = (*s).into();
			}
		}
		assert!(converted_ponts.len() == converted_shares[0].len());
		/* Evaluate up till the secrets, split to disard randomness
		Reconstruct each poly
		*/

		let mut ret: Vec<U> = Vec::new();
		for i in 0..B {
			let mut secrets_block: Vec<T> = ntt::lagrange_interpolation(
				&converted_ponts, &converted_shares[i], &self.rootTable2, self.prime
			);
			secrets_block.split_off(self.L);
			for j in 0..self.L {
				ret.push(secrets_block[j].try_into().unwrap());
			}
		}
		/* Output Format
		   [s0, ..., sv]
		*/
		ret		
	}
}

fn print_elapsed(title: &str, start: time::Timespec) {
    let time_elapsed = time::get_time() - start;
    let elapsed_ms = time_elapsed.num_milliseconds();
    let separator = if title.len() > 0 { ": " } else { "" };
    println!("    {}{}: {}.{:03}", title, separator, time_elapsed.num_seconds(), elapsed_ms);
}