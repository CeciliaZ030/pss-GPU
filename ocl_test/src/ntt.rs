use core::fmt::Debug;
use std::cmp::PartialOrd;
use num_traits::*;
use num_traits::{One, Zero};

use crate::util::*;


//out-of-place transform
//input reference and perform in-place DFT on copy of input
pub fn transform2<T>(mut a: Vec<T>, P: T, rootTable: &Vec<T>) -> Vec<T> 
where T: Unsigned + Copy + Debug + From<u64> + PartialOrd
{
	bit_reverse2(&mut a);
	DFT_radix2(&mut a, P, rootTable);
	a
}

pub fn transform3<T>(mut a: Vec<T>, P: T, rootTable: &Vec<T>) -> Vec<T> 
where T: Unsigned + Copy + Debug + From<u64> + PartialOrd
{
	bit_reverse3(&mut a);
	DFT_radix3(&mut a, P, rootTable);
	a
}


pub fn inverse2<T: ModPow>(mut b: Vec<T>, P: T, rootTable: &Vec<T>) -> Vec<T> 
where T: Unsigned + Copy + Debug + From<u64> + PartialOrd
{


	let L = b.len();
	let w = rootTable[1].modpow(P - 2.into(), P);
	let zero = T::zero();

	//calculating inverse omegas
	let mut inverseTable = vec![zero; L];
	for i in 0..L {
		inverseTable[i] = w.modpow((i as u64).into(), P);
	}

	bit_reverse2(&mut b);
	DFT_radix2(&mut b, P, &inverseTable);

	// F^-1(Y) = nX
	// Thus divide output by n or multiply n^-1
	let L_into: T = (L as u64).into();
	let L_inverse: T = L_into.modpow(P - 2.into(), P);
	for i in 0..L {
		b[i] = b[i] * L_inverse % P;
	}

	b
}

pub fn inverse3<T: ModPow>(mut b: Vec<T>, P: T, rootTable: &Vec<T>) -> Vec<T> 
where T: Unsigned + Copy + Debug + From<u64> + PartialOrd
{

	let L = b.len();
	let w = rootTable[1].modpow(P - 2.into(), P);
	let zero = T::zero();

	//calculating inverse omegas
	let mut inverseTable = vec![zero; L];
	for i in 0..L {
		inverseTable[i] = w.modpow((i as u64).into(), P);
	}

	bit_reverse3(&mut b);
	DFT_radix3(&mut b, P, &inverseTable);

	// F^-1(Y) = nX
	// Thus divide output by n or multiply n^-1
	let L_into: T = (L as u64).into();
	let L_inverse = L_into.modpow(P - 2.into(), P);
	for i in 0..L {
		b[i] = b[i] * L_inverse % P;
	}

	b
}

//in-place, use mutable reference
pub fn DFT_radix2<T>(a: &mut Vec<T>, P: T, rootTable: &Vec<T>)
where T: Unsigned + Copy + Debug + From<u64> + PartialOrd
{
	let L = a.len();
	let L_bitNum = (L as f64).log2().trunc() as usize;

    //Cooley-Tukey DFT
	for s in 1..(L_bitNum + 1) {
		let m = pow(2, s) as usize;
		let mut i = 0;
		while i < L {
			let mut j = 0;
			while j < m/2 {
				let t = rootTable[j*(L/m as usize)] * a[i + j + m/2 ] % P;
				let u = a[i + j] % P;
				a[i + j] = (u + t) % P;
				if u <= t {
					a[i + j + m/2] = ((P + u) - t) % P;
				} else {
					a[i + j + m/2] = (u - t) % P;
				}
				j+= 1;
			}
			i += m;
		}
	}
}

pub fn DFT_radix3<T>(a: &mut Vec<T>, P: T, rootTable: &Vec<T>) 
where T: Unsigned + Copy + Debug + From<u64> + PartialOrd
{	
	let L = a.len();
	let w = rootTable[L/3];
	let w_sqr = rootTable[L/3*2];

	let mut i = 1;
	while i < L {
		let jump = 3 * i;
		let stride = L/jump;
		for j in 0..i {
			let mut pair = j;
			while pair < L {
				let (x, y, z) = (a[pair],
								a[pair + i] * rootTable[j * stride] % P,
								a[pair + 2 * i] * rootTable[2 * j * stride] %P);
				a[pair] 	  	= (x + y + z) % P;
				a[pair + i]   	= (x % P + w * y % P + w_sqr * z % P) % P;
                a[pair + 2 * i] = (x % P + w_sqr * y % P + w * z % P) % P;
				
				pair += jump;
			}
		}
		i = jump;
	}
}

pub fn bit_reverse2<T: Debug>(a: &mut Vec<T>) {

	let L = a.len();

    let mut j = 0;
    for i in 0..L {
        if j > i {
            a.swap(i, j);
        }
        let mut mask = L >> 1;
        while j & mask != 0 {
            j &= !mask;
            mask >>= 1;
        }
        j |= mask;
    }
}

pub fn bit_reverse3<T>(a: &mut Vec<T>) {

    let L = a.len();
    let tri_L = trigits_len(L - 1);
    let mut trigits = vec![0; tri_L];

    let mut t = 0usize;
    for i in 0..L {
        if t > i {
            a.swap(t, i);
        }
        for j in 0..tri_L {
            if trigits[j] < 2 {
                trigits[j] += 1;
                t += 3usize.pow((tri_L-j-1)as u32);
                break;
            } else {
                trigits[j] = 0;
                t -= 2 * 3usize.pow((tri_L-j-1)as u32);
            }

        }
    }
}

pub fn trigits_len(n: usize) -> usize {
    let mut result = 1;
    let mut value = 3;
    while value < n + 1 {
        result += 1;
        value *= 3;
    }
    result
}


pub fn lagrange_interpolation<T: ModPow>(points: &Vec<T>, values: &Vec<T>, roots: &Vec<T>, P: T) -> Vec<T> 
where T: Unsigned + Copy + Debug + From<u64> + PartialOrd
{
	assert!(points.len() == values.len());
	let L = points.len();
	let mut denominators: Vec<T> = Vec::new();

	for i in 0..L {
		let mut d = T::one();
		for j in 0..L {
			if i != j {
				if points[i] >= points[j]{
					d = d * (points[i] - points[j]);
				} else {
					d = d * ((points[i] + P - points[j]) % P);
				}
				d = d % P;
			}
		}
		d = d.modpow(P - 2.into(), P);
		denominators.push(d);
	}

	let mut evals: Vec<T> = Vec::new();
	for r in roots {
		let mut eval = T::zero();
		for i in 0..L {
			let mut li = T::one();
			for j in 0..L {
				if i != j {
					if *r >= points[j] {
						li = li * (*r - points[j]);
					} else {
						li = li * ((*r + P - points[j]) % P);
					}
					li = li % P;
				}
			}
			li = li * denominators[i] % P;
			eval = eval + (li * values[i] % P);
		}
		evals.push(eval % P);
	}
	
	evals
}




