#![allow(non_snake_case)]

use std::convert::*;
use std::fmt::Debug;
use time;

use rand::{thread_rng, Rng};
use rand::distributions::uniform::SampleUniform;
use num::traits::Unsigned;

use ocl::{Platform, Device, ProQue, Buffer, SpatialDims};
use ocl::core;
use ocl::core::{DeviceInfo};
use ocl::traits::OclPrm;

pub mod util;
mod kernel;
use util::*;
use kernel::*;

pub struct OclContext<T> {

    pub compute_units: u128,
    pro_que: ocl::ProQue,

    prime: T,
    roots2: Vec<T>,
    roots3: Vec<T>,
    degree2: usize,
    degree3: usize,

    V: usize,
    L: usize,
    N: usize,

}



impl<T> OclContext<T> 
where T: Unsigned + Copy + Debug + TryFrom<u128> + Into<u128> + 
        SampleUniform + PartialOrd + OclPrm + HasMax + ModPow,
      <T as TryFrom<u128>>::Error:Debug
{
    /* Use u128 because any T implements Into<u128>
    */

    pub fn new(prime: T, root2: T, root3:T, 
               degree2: usize, degree3: usize, total_len: usize, packing_len: usize, num_shares: usize) -> Option<OclContext<T>> {
        
        println!("{:?} {:?} {:?} {:?} {:?}", degree2, degree3, total_len, packing_len, num_shares);
        // wich Device should we choose?
        // the one with the most Compute units!
        let mut compute_units = 0;
        let mut ocl_device = None;
        let platforms = Platform::list();
        for p_idx in 0..platforms.len() {
            let platform = &platforms[p_idx];
            let devices = Device::list_all(platform).unwrap();
            for d_idx in 0..devices.len() {
                let device = devices[d_idx];
                let deviceinfodest = core::get_device_info(
                    &device, 
                    DeviceInfo::MaxComputeUnits
                );
                let units = deviceinfodest
                            .unwrap()
                            .to_string()
                            .parse()
                            .unwrap();
                if units > compute_units {
                    ocl_device = Some(device);
                    compute_units = units;
                }
                println!("{:?} {:?}", core::get_device_info(&device, DeviceInfo::Name),
                    core::get_device_info(&device, DeviceInfo::MaxComputeUnits));
            }

        }
        // something went wrong, opencl not installed
        if compute_units == 0 {
            return None
        }

        let L3_trigits_len = trigits_len(degree3);
        let kernel_code = format!("{}{}{}{}{}{}", 
            NTT_INVERSE2,
            NTT_TRANSFORM3_PART1,
            L3_trigits_len,
            NTT_TRANSFORM3_PART2,
            L3_trigits_len,
            NTT_TRANSFORM3_PART3);

        let que = ProQue::builder()
                  .device(ocl_device.unwrap())
                  .src(kernel_code)
                  .build().expect("Build ProQue");

        
        assert!(total_len % packing_len == 0);
        assert!(packing_len <= degree2);
        assert!(degree2 <= num_shares);
        assert!(num_shares <= degree3);

        let P: u128 = prime.into();
        let mut inv_roots2: Vec<T> = Vec::new();
        let inv_root2: u128 = root2.into()
                                   .modpow(P - 2u128, P);
        for i in 0..degree2 {
            let wi = inv_root2.modpow(i as u128, P);
            inv_roots2.push(wi.try_into().unwrap());    
        }
        //Constant memory __constant

        let mut roots3: Vec<T> = Vec::new();
        for i in 0..degree3 {
            let wi: u128 = root3.into()
                                .modpow(i as u128, P);
            roots3.push(wi.try_into().unwrap());
        }

        Some(OclContext {
            compute_units: compute_units,
            pro_que : que,

            prime: prime,
            roots2: inv_roots2,
            roots3: roots3,
            degree2: degree2,
            degree3: degree3,

            V: total_len,
            L: packing_len,
            N: num_shares,
        })
    }

    pub fn share(&mut self, secrets: &[T]) -> Vec<Vec<T>> {   
        /* Input Format
           [x0, ..., xv]
        */
        assert!(secrets.len() == self.V);
        let L2 = self.degree2;
        let L2_bit_mum: T = ((L2 as f64).log2().trunc() as u128)
                                    .try_into().unwrap();
        let L3 = self.degree3;
        let B = self.V / self.L;
        println!("V = {:?}, B = {}, L = {}", self.V, B, self.L);
        
        let ref mut ocl_pq = self.pro_que;
        let mut secret_blocks: Vec<T> = Vec::new();
        let mut rng = thread_rng();
        for i in 0..B {
            for j in 0..self.L {
                secret_blocks.push(secrets[i*self.L+j]);
            }
            /* Pack randomness for unused transform points
            randomness is no greater than max of T to prevent overflow
            */
            for _ in self.L..L2 {
                secret_blocks.push(rng.gen_range(T::zero(), self.prime));
            }
        }
        println!("secret_blocks {:?}\n roots2 {:?}", secret_blocks, self.roots2);

        // set work dimension
        ocl_pq.set_dims(B);
        // copy matrix to device
        let mut source: Buffer<T>;
        let roots2: Buffer<T>;
        let roots3: Buffer<T>;
        unsafe {
            source = Buffer::new(
                &ocl_pq.queue().clone(),
                core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR, 
                SpatialDims::One(B * L2), 
                Some(&secret_blocks)
            ).unwrap();
            roots2 = Buffer::new(
                &ocl_pq.queue().clone(),
                core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR, 
                SpatialDims::One(L2), 
                Some(&self.roots2)
            ).unwrap();
            roots3 = Buffer::new(
                &ocl_pq.queue().clone(),
                core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR, 
                SpatialDims::One(L3), 
                Some(&self.roots3)
            ).unwrap();

        }

        println!("Enqueuing ntt_inverse2 kernel");
        let kern_start = time::get_time();
        let mut kernel = ocl_pq.kernel_builder("ntt_inverse2")
            .arg(&source)
            .arg(&roots2)
            .arg(L2 as u64)
            .arg(self.roots2[1].clone())
            .arg(L2_bit_mum)
            .arg(self.prime)
            .build()
            .unwrap();
        // Enqueue kernel: send to device and run it
        unsafe {
            kernel
            // 1 poly / local work goup
            //.set_default_local_work_size((1, L2).into())
            .enq()
            .unwrap();
        }
        println!("{:?}", kernel);
        ocl_pq.queue().finish();
        print_elapsed("total elapsed", kern_start);

        println!("Buffer reads [B*L2]");
        let buff_start = time::get_time();
        // Read dests from the device into dest_buffer's local vector:
        let mut poly = vec![T::zero(); (L2*B) as usize];
        source.read(&mut poly).enq().unwrap();
        print_elapsed("queue unfinished", buff_start);
        ocl_pq.queue().finish();
        print_elapsed("queue finished", buff_start);
        println!("source {:?}", source);
  
        //__________________________________________________
   
        for i in 0..B {
            for _ in L2 ..L3 {
                poly.insert(i*L3, T::zero());
            }
        }
        assert!(poly.len() == B * L3);
        //println!("poly inserted to L3 {:?}", poly);


        unsafe {
            source = Buffer::new(
                &ocl_pq.queue().clone(),
                core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR, 
                SpatialDims::One(B * L3), 
                Some(&poly)
            ).unwrap();
        }

        println!("Enqueuing ntt_transform3 kernel");
        let kern_start = time::get_time();
        let mut kernel = ocl_pq.kernel_builder("ntt_transform3")
            .arg(&source)
            .arg(&roots3)
            .arg(L3 as u64)
            .arg(self.prime)
            .build()
            .unwrap();
        // Enqueue kernel: send to device and run it
        unsafe {
            kernel
            .enq()
            .unwrap();
        }
        println!("{:?}", kernel);
        ocl_pq.queue().finish();
        print_elapsed("total elapsed", kern_start);

        println!("Buffer reads [B*L3]");
        let buff_start = time::get_time();
        // Read dests from the device into dest_buffer's local vector:
        let mut res = vec![T::zero(); B * L3];
        source.read(&mut res).enq().unwrap();
        print_elapsed("queue unfinished", buff_start);
        ocl_pq.queue().finish();
        print_elapsed("queue finished", buff_start);
        
        let mut ret = vec![Vec::with_capacity(B); self.N];
        for i in 0..self.N {
            ret[i].extend(&res[i*B..(i+1)*B]);
        }
        ret
    }

    // pub fn share2(&mut self, secrets: &[T]) -> Vec<Vec<T>> {   
    //     /* Input Format
    //        [x0, ..., xv]
    //     */
    //     assert!(secrets.len() == self.V);
    //     let L2 = self.degree2;
    //     let L2_bit_mum: T = ((L2 as f64).log2().trunc() as u128)
    //                                 .try_into().unwrap();
    //     let L3 = self.degree3;
    //     let B = self.V / self.L;
    //     println!("V = {:?}, B = {}, L = {}", self.V, B, self.L);
        
    //     let ref mut ocl_pq = self.pro_que;
    //     let mut secret_blocks: Vec<T> = Vec::new();
    //     let mut rng = thread_rng();
    //     for i in 0..B {
    //         for j in 0..self.L {
    //             secret_blocks.push(secrets[i*self.L+j]);
    //         }
    //          Pack randomness for unused transform points
    //         randomness is no greater than max of T to prevent overflow
            
    //         for _ in self.L..L2 {
    //             secret_blocks.push(rng.gen_range(T::zero(), self.prime));
    //         }
    //     }
    //     let mut ret: Vec<Vec<T>> = vec![Vec::with_capacity(B); self.N];
    //     for (i, block) in secret_blocks.iter().enumerate() {
    //         /* use radix2_DFT to from the poly
    //         */
    //         let mut poly = ntt::inverse2(block.to_vec(), self.prime, &self.roots2);
    //         for _ in L2 ..L3 {
    //             poly.push(T::zero());
    //         }
    //         /* share with radix3_DFT
    //         */
    //         let mut shares = ntt::transform3(poly, self.prime, &self.roots3);
    //         ret[i] = shares[1..self.N+1].to_vec();
    //     }
    //     ret
    // }

    // Comput rootThrees
    // Shares should in order
    pub fn reconstruct(&mut self, shares: &Vec<Vec<T>>) -> Vec<T> {
        assert!(shares.len() >= self.roots3.len());
        let B = self.V/self.L;
        let mut transposed = vec![Vec::with_capacity(B); self.N];
        for i in 0..B {
            transposed[i] = shares.iter().map(|s: &Vec<T>| s[i]).collect::<Vec<T>>();
        }
        let mut ret = Vec::new();
        for t in transposed {
            ret.extend(lagrange_interpolation(&self.roots3, &t, &self.roots2, self.prime));
        }
        ret
    }

}

pub fn lagrange_interpolation<T>(points: &Vec<T>, values: &Vec<T>, roots: &Vec<T>, P: T) -> Vec<T> 
    where T: Unsigned + Copy + Debug + PartialOrd + ModPow + TryFrom<u128>,
        <T as TryFrom<u128>>::Error:Debug
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
        d = d.modpow(P - 2u128.try_into().unwrap(), P);
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

fn print_elapsed(title: &str, start: time::Timespec) {
    let time_elapsed = time::get_time() - start;
    let elapsed_ms = time_elapsed.num_microseconds();
    let separator = if title.len() > 0 { ": " } else { "" };
    println!("    {}{}: {} us", title, separator, elapsed_ms.unwrap());
}