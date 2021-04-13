
pub trait ModPow {
    fn modpow(&self, exponent: Self, modulus: Self) -> Self;
}

macro_rules! modpow {
    ($numType: ty) => (
        fn modpow(&self, exponent: Self, modulus: Self) -> Self {
                assert!(modulus != (0 as $numType), "divide by zero!");
                if exponent == (0 as $numType) {
                    return (1 as $numType)
                }

                let mut base = self % modulus;
                let mut exp = exponent.clone();
                let mut res = (1 as $numType);

                while exp > 0 {
                    if exp % (2 as $numType) == 1 {
                        res = res * base % modulus;
                    }
                    exp >>= 1;
                    base = base * base % modulus;
                }
                return res
        }
    )
}


impl ModPow for u128 {
    modpow!(u128);
}

impl ModPow for u64 {
    modpow!(u64);
}

impl ModPow for u32 {
    modpow!(u32);
}

pub trait HasMax {
    fn max() -> Self;
}

impl HasMax for u32 {
    fn max() -> u32 {
        std::u32::MAX
    }
}

impl HasMax for u64 {
    fn max() -> u64 {
        std::u64::MAX
    }
}

impl HasMax for u128 {
    fn max() -> u128 {
        std::u128::MAX
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
