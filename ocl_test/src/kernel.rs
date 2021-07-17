pub static NTT_INVERSE2: &str = r#"

void radix2_bitreverse (
	__global ulong* source,
  	const ulong base,
  	const ulong L)
{
	int j = 0;
	for (int i = 0; i < L; i++) {
		if (j > i) {
			ulong temp = source[base+i];
			source[base+i] = source[base+j];
			source[base+j] = temp;
      	}
		int mask = L >> 1;
		while ((j & mask) != 0) {
			j &= !mask;
			mask >>= 1;
		}
		j |= mask;
    }
}


void radix2_dft (
	__global ulong* source,
	__global const ulong* roots2,
	const ulong base,
  	const ulong L,
	const ulong L_inv,
  	const ulong L_bit_num,
  	const ulong P)
{
	for (int s = 0; s < L_bit_num + 1; s++) {
		int m = (int) pown((float) 2, s);
		int i = 0;

		while (i < L) {
			int j = 0;
			while (j < m/2) {
				ulong t = roots2[base+j*(L/m)] * source[base+i+j+m/2] % P;
				ulong u = source[base+i+j] % P;
				source[i+j] = (u + t) % P;
				if (u <= t) {
					source[base+i+j+m/2] = (P + u - t) % P;
				} else {
					source[base+i+j+m/2] = (u - t) % P;
				}
				printf("base # %d, changing %u, (%u, %u, %u, %u): %llu, %llu\n", 
						get_global_id(0), i+j+m/2, base, i, j, m, u, t);
				j ++;
			}
			i += m;
		}
	}

	for (int i = 0; i < L; i ++){
		source[base+i] = source[base+i] * L_inv % P;
	}
}

//alias
//typedef
//c preprocessor

__kernel void ntt_inverse2 (
	__global ulong* source,
	__global ulong* roots2,		//inversed
  	const ulong L,
  	const ulong L_inv,
  	const ulong L_bit_num,
  	const ulong P)
{
	ulong const base = get_global_id(0) * L;
	printf("%d", get_local_id(0));

	for (int i = 0; i < L; i++) {
		printf("%d, source %d roots2 %d\n", i, source[i], roots2[i]);
	}

	radix2_bitreverse(source, base, L);
	radix2_dft(source, roots2, base, L, L_inv, L_bit_num, P);

}
"#;


pub static NTT_TRANSFORM3_PART1: &str = r#"

void radix3_bitreverse (
	__global ulong* source, 
	const ulong base,
  	const ulong L)
{
	int L_trigits_num ="#;

pub static NTT_TRANSFORM3_PART2: &str = r#";
	__private int trigits["#;
pub static NTT_TRANSFORM3_PART3: &str = r#"] = {0};
	int t = 0;
	for (int i = 0; i < L; i++) {
		if (t > i) {
			ulong temp = source[base+i];
			source[base+i] = source[base+t];
			source[base+t] = temp;
      	}
      	for (int j = 0; j < L_trigits_num; j++){
      		if (trigits[j] < 2) {
      			trigits[j] += 1;
      			t += (int)pown((float)3, L_trigits_num-j-1);
      			break;
      		} else {
      			trigits[j] = 0;
      			t -= 2 * (int)pown((float)3, L_trigits_num-j-1);
      		}
      	}
    }
}


void radix3_dft (
	__global ulong* source,
	__global ulong* roots3,
	const ulong base,
	const ulong L,
	const ulong P)
{
	ulong w = roots3[L/3];
	ulong w_sqr = roots3[L/3*2];

	int i = 1;
	while (i < L) {
		int jump = 3 * i;
		int stride = L/jump;
		for (int j = 0; j < i; j++) {
			int pair = j;
			while (pair < L) {
				ulong x = source[base+pair];
				ulong y = source[base+pair+1] * roots3[j*stride] % P;
				ulong z = source[base+pair+2*i] * roots3[2*j*stride] % P;

				source[base+pair] 	  = (x + y + z) % P;
				source[base+pair+i]   = (x % P + w * y % P + w_sqr * z % P) % P;
                source[base+pair+2*i] = (x % P + w_sqr * y % P + w * z % P) % P;

                pair += jump;
  			}
		}
		i = jump;
	}
}

__kernel void ntt_transform3 (
	__global ulong* source, 
	__global ulong* roots3,
  	const ulong L,
  	const ulong P)
{
	ulong const base = get_global_id(0) * L;
	// printf("%d", get_local_id(0));

	radix3_bitreverse(source, base, L);
	radix3_dft(source, roots3, base, L, P);

}
"#;

