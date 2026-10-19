pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

struct Lookup {
    salt: u64,
    mult: Vec<u64>,
}

pub struct NgramHasher {
    rows: u64,
    lookups: Vec<Lookup>,
}

impl NgramHasher {
    pub fn new(seed: u64, rows: u64, orders: &[usize], n_hash: usize) -> Option<Self> {
        if rows == 0 || n_hash == 0 || orders.is_empty() || orders.contains(&0) {
            return None;
        }
        let mut rng = SplitMix64::new(seed);
        let mut lookups = Vec::with_capacity(orders.len() * n_hash);
        for &order in orders {
            for _ in 0..n_hash {
                let salt = rng.next_u64() >> 33;
                let mult = (0..order).map(|_| (rng.next_u64() >> 33) | 1).collect();
                lookups.push(Lookup { salt, mult });
            }
        }
        Some(Self { rows, lookups })
    }

    pub fn lookup_count(&self) -> usize {
        self.lookups.len()
    }

    pub fn row(&self, lookup: usize, context: &[u32]) -> Option<u64> {
        let entry = self.lookups.get(lookup)?;
        let mut acc = entry.salt;
        let mut back = context.iter().rev();
        for &m in &entry.mult {
            let token = back.next().map_or(0, |&t| u64::from(t) + 1);
            acc = acc.wrapping_add(token.wrapping_mul(m));
        }
        Some(acc % self.rows)
    }
}

#[cfg(test)]
mod tests {
    use super::{NgramHasher, SplitMix64};

    #[test]
    fn splitmix64_matches_reference_values() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(rng.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert!(NgramHasher::new(1, 0, &[2], 1).is_none());
        assert!(NgramHasher::new(1, 8, &[], 1).is_none());
        assert!(NgramHasher::new(1, 8, &[0], 1).is_none());
        assert!(NgramHasher::new(1, 8, &[2], 0).is_none());
    }

    #[test]
    fn counts_lookups_per_order_and_hash() {
        let hasher = NgramHasher::new(1, 1000, &[2, 3], 4).unwrap();
        assert_eq!(hasher.lookup_count(), 8);
        assert!(hasher.row(8, &[1, 2, 3]).is_none());
    }

    #[test]
    fn rows_are_in_range_and_deterministic() {
        let hasher = NgramHasher::new(20_261_016, 4093, &[2, 3], 2).unwrap();
        let again = NgramHasher::new(20_261_016, 4093, &[2, 3], 2).unwrap();
        for lookup in 0..hasher.lookup_count() {
            let row = hasher.row(lookup, &[5, 17, 99]).unwrap();
            assert!(row < 4093);
            assert_eq!(Some(row), again.row(lookup, &[5, 17, 99]));
        }
    }

    #[test]
    fn only_the_last_n_tokens_matter() {
        let hasher = NgramHasher::new(7, 65_537, &[2], 1).unwrap();
        assert_eq!(hasher.row(0, &[9, 9, 4, 8]), hasher.row(0, &[1, 2, 4, 8]));
        assert_ne!(hasher.row(0, &[9, 9, 4, 8]), hasher.row(0, &[9, 9, 5, 8]));
    }

    #[test]
    fn start_of_text_differs_from_token_zero() {
        let hasher = NgramHasher::new(7, 65_537, &[2], 1).unwrap();
        assert_ne!(hasher.row(0, &[3]), hasher.row(0, &[0, 3]));
    }
}