use std::error::Error;
use std::fmt;

use crate::hash::{NgramHasher, SplitMix64};

const MAX_PART_DIM: usize = 4096;
const PROJECTION_SALT: u64 = 0xA5A5_5A5A_C3C3_3C3C;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SketchSpec {
    pub seed: u64,
    pub rows: u64,
    pub orders: Vec<usize>,
    pub n_hash: usize,
    pub part_dim: usize,
    pub vocab_size: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum SketchError {
    InvalidSpec(&'static str),
    TooLarge,
    TokenOutOfRange { token: u32, vocab_size: usize },
}

impl fmt::Display for SketchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpec(reason) => write!(f, "invalid sketch configuration: {reason}"),
            Self::TooLarge => write!(f, "the sketch does not fit in memory"),
            Self::TokenOutOfRange { token, vocab_size } => {
                write!(
                    f,
                    "token id {token} is outside the vocabulary of {vocab_size}"
                )
            }
        }
    }
}

impl Error for SketchError {}

#[derive(Clone, Debug, PartialEq)]
pub struct Sketch {
    pub spec: SketchSpec,
    pub lookups: usize,
    pub pairs: u64,
    pub projection: Vec<f32>,
    pub sums: Vec<f32>,
    pub counts: Vec<u32>,
}

pub struct SketchBuilder {
    spec: SketchSpec,
    hasher: NgramHasher,
    max_order: usize,
    projection: Vec<f32>,
    sums: Vec<f64>,
    counts: Vec<u32>,
    carry: Vec<u32>,
    pairs: u64,
}

fn zeroed<T: Clone + Default>(length: usize) -> Result<Vec<T>, SketchError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| SketchError::TooLarge)?;
    values.resize(length, T::default());
    Ok(values)
}

#[allow(clippy::cast_precision_loss)]
fn open_unit(rng: &mut SplitMix64) -> f64 {
    ((rng.next_u64() >> 11) as f64 + 1.0) / 9_007_199_254_740_993.0
}

#[allow(clippy::cast_possible_truncation)]
fn projection(spec: &SketchSpec) -> Result<Vec<f32>, SketchError> {
    let length = spec
        .vocab_size
        .checked_mul(spec.part_dim)
        .ok_or(SketchError::TooLarge)?;
    let mut values: Vec<f32> = Vec::new();
    values
        .try_reserve_exact(length + 1)
        .map_err(|_| SketchError::TooLarge)?;
    let width = u16::try_from(spec.part_dim)
        .map_err(|_| SketchError::InvalidSpec("part_dim must be in 1..=4096"))?;
    let scale = 1.0 / f64::from(width).sqrt();
    let mut rng = SplitMix64::new(spec.seed ^ PROJECTION_SALT);
    while values.len() < length {
        let radius = (-2.0 * open_unit(&mut rng).ln()).sqrt();
        let angle = std::f64::consts::TAU * open_unit(&mut rng);
        values.push((radius * angle.cos() * scale) as f32);
        values.push((radius * angle.sin() * scale) as f32);
    }
    values.truncate(length);
    Ok(values)
}

impl SketchBuilder {
    pub fn new(spec: SketchSpec) -> Result<Self, SketchError> {
        if spec.part_dim == 0 || spec.part_dim > MAX_PART_DIM {
            return Err(SketchError::InvalidSpec("part_dim must be in 1..=4096"));
        }
        if spec.vocab_size == 0 {
            return Err(SketchError::InvalidSpec("vocab_size must be positive"));
        }
        let hasher = NgramHasher::new(spec.seed, spec.rows, &spec.orders, spec.n_hash).ok_or(
            SketchError::InvalidSpec("rows, orders and hashes must be positive"),
        )?;
        let rows = usize::try_from(spec.rows).map_err(|_| SketchError::TooLarge)?;
        let cells = rows
            .checked_mul(spec.part_dim)
            .ok_or(SketchError::TooLarge)?;
        let max_order = spec.orders.iter().copied().max().unwrap_or(1);
        Ok(Self {
            projection: projection(&spec)?,
            sums: zeroed(cells)?,
            counts: zeroed(rows)?,
            carry: Vec::new(),
            pairs: 0,
            max_order,
            hasher,
            spec,
        })
    }

    pub fn add(&mut self, chunk: &[u32]) -> Result<(), SketchError> {
        let vocab_size = self.spec.vocab_size;
        if let Some(&token) = chunk
            .iter()
            .find(|&&token| usize::try_from(token).map_or(true, |id| id >= vocab_size))
        {
            return Err(SketchError::TokenOutOfRange { token, vocab_size });
        }
        let first = self.carry.len().saturating_sub(1);
        let mut buffer = std::mem::take(&mut self.carry);
        buffer.extend_from_slice(chunk);
        for position in first..buffer.len().saturating_sub(1) {
            let start = (position + 1).saturating_sub(self.max_order);
            if let (Some(context), Some(&next)) =
                (buffer.get(start..=position), buffer.get(position + 1))
            {
                self.record(context, next);
            }
        }
        let keep_from = buffer.len().saturating_sub(self.max_order);
        self.carry = buffer.split_off(keep_from);
        Ok(())
    }

    fn record(&mut self, context: &[u32], next: u32) {
        let width = self.spec.part_dim;
        let Some(vector) = usize::try_from(next)
            .ok()
            .and_then(|id| id.checked_mul(width))
            .and_then(|from| self.projection.get(from..from + width))
        else {
            return;
        };
        for lookup in 0..self.hasher.lookup_count() {
            let Some(row) = self
                .hasher
                .row(lookup, context)
                .and_then(|row| usize::try_from(row).ok())
            else {
                continue;
            };
            if let Some(cells) = self.sums.get_mut(row * width..(row + 1) * width) {
                for (cell, &value) in cells.iter_mut().zip(vector) {
                    *cell += f64::from(value);
                }
            }
            if let Some(count) = self.counts.get_mut(row) {
                *count = count.saturating_add(1);
            }
        }
        self.pairs += 1;
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn finish(self) -> Sketch {
        Sketch {
            lookups: self.hasher.lookup_count(),
            pairs: self.pairs,
            projection: self.projection,
            sums: self.sums.into_iter().map(|sum| sum as f32).collect(),
            counts: self.counts,
            spec: self.spec,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SketchBuilder, SketchError, SketchSpec};
    use crate::hash::{NgramHasher, SplitMix64};

    fn spec(orders: &[usize], n_hash: usize) -> SketchSpec {
        SketchSpec {
            seed: 11,
            rows: 1_000_003,
            orders: orders.to_vec(),
            n_hash,
            part_dim: 4,
            vocab_size: 50,
        }
    }

    fn stream(length: usize) -> Vec<u32> {
        let mut rng = SplitMix64::new(5);
        (0..length)
            .map(|_| u32::try_from(rng.next_u64() % 50).unwrap())
            .collect()
    }

    fn build(spec: SketchSpec, tokens: &[u32], chunk: usize) -> super::Sketch {
        let mut builder = SketchBuilder::new(spec).unwrap();
        for part in tokens.chunks(chunk) {
            builder.add(part).unwrap();
        }
        builder.finish()
    }

    #[test]
    fn rejects_invalid_configurations() {
        let bad = |change: fn(&mut SketchSpec)| {
            let mut value = spec(&[2], 1);
            change(&mut value);
            SketchBuilder::new(value).err()
        };
        assert!(matches!(
            bad(|s| s.part_dim = 0),
            Some(SketchError::InvalidSpec(_))
        ));
        assert!(matches!(
            bad(|s| s.part_dim = 5000),
            Some(SketchError::InvalidSpec(_))
        ));
        assert!(matches!(
            bad(|s| s.vocab_size = 0),
            Some(SketchError::InvalidSpec(_))
        ));
        assert!(matches!(
            bad(|s| s.rows = 0),
            Some(SketchError::InvalidSpec(_))
        ));
        assert!(matches!(
            bad(|s| s.orders = vec![]),
            Some(SketchError::InvalidSpec(_))
        ));
        assert!(matches!(
            bad(|s| s.orders = vec![0]),
            Some(SketchError::InvalidSpec(_))
        ));
        assert!(matches!(
            bad(|s| s.n_hash = 0),
            Some(SketchError::InvalidSpec(_))
        ));
    }

    #[test]
    fn absurd_sizes_fail_instead_of_aborting() {
        let mut huge = spec(&[2], 1);
        huge.rows = u64::MAX / 2;
        assert_eq!(SketchBuilder::new(huge).err(), Some(SketchError::TooLarge));
    }

    #[test]
    fn rows_hold_the_sum_of_the_next_token_vectors_seen_after_a_context() {
        let tokens = [1, 2, 1, 3];
        let sketch = build(spec(&[1], 1), &tokens, 4);
        let hasher = NgramHasher::new(11, 1_000_003, &[1], 1).unwrap();
        let after_one = usize::try_from(hasher.row(0, &[1]).unwrap()).unwrap();
        let after_two = usize::try_from(hasher.row(0, &[2]).unwrap()).unwrap();
        assert_ne!(after_one, after_two);
        let vector = |token: usize| &sketch.projection[token * 4..token * 4 + 4];
        let row = |index: usize| &sketch.sums[index * 4..index * 4 + 4];
        for dim in 0..4 {
            assert!((row(after_one)[dim] - (vector(2)[dim] + vector(3)[dim])).abs() < 1e-6);
            assert!((row(after_two)[dim] - vector(1)[dim]).abs() < 1e-6);
        }
        assert_eq!(sketch.counts[after_one], 2);
        assert_eq!(sketch.counts[after_two], 1);
        assert_eq!(sketch.pairs, 3);
    }

    #[test]
    fn the_first_positions_use_the_start_of_text_padding() {
        let sketch = build(spec(&[2], 1), &[7, 8, 9], 3);
        let hasher = NgramHasher::new(11, 1_000_003, &[2], 1).unwrap();
        let first = usize::try_from(hasher.row(0, &[7]).unwrap()).unwrap();
        let second = usize::try_from(hasher.row(0, &[7, 8]).unwrap()).unwrap();
        assert_eq!(sketch.counts[first], 1);
        assert_eq!(sketch.counts[second], 1);
        assert_eq!(sketch.pairs, 2);
    }

    #[test]
    fn chunking_never_changes_the_result() {
        let tokens = stream(500);
        let whole = build(spec(&[2, 3], 2), &tokens, tokens.len());
        for chunk in [1, 2, 3, 4, 5, 7, 64, 499] {
            let parts = build(spec(&[2, 3], 2), &tokens, chunk);
            assert_eq!(parts, whole, "chunk size {chunk}");
        }
    }

    #[test]
    fn every_pair_is_counted_once_per_lookup() {
        let tokens = stream(300);
        let sketch = build(spec(&[2, 3], 2), &tokens, 50);
        let total: u64 = sketch.counts.iter().map(|&count| u64::from(count)).sum();
        assert_eq!(sketch.pairs, 299);
        assert_eq!(sketch.lookups, 4);
        assert_eq!(total, 299 * 4);
    }

    #[test]
    fn short_and_empty_streams_make_no_pairs() {
        let mut builder = SketchBuilder::new(spec(&[2], 1)).unwrap();
        builder.add(&[]).unwrap();
        builder.add(&[3]).unwrap();
        builder.add(&[]).unwrap();
        assert_eq!(builder.finish().pairs, 0);
    }

    #[test]
    fn tokens_outside_the_vocabulary_are_rejected() {
        let mut builder = SketchBuilder::new(spec(&[2], 1)).unwrap();
        assert_eq!(
            builder.add(&[1, 50]),
            Err(SketchError::TokenOutOfRange {
                token: 50,
                vocab_size: 50
            })
        );
        assert_eq!(builder.finish().pairs, 0);
    }

    #[test]
    fn the_projection_is_deterministic_and_has_unit_norm_on_average() {
        let mut big = spec(&[2], 1);
        big.vocab_size = 2000;
        big.part_dim = 16;
        let first = SketchBuilder::new(big.clone()).unwrap().finish();
        let second = SketchBuilder::new(big).unwrap().finish();
        assert_eq!(first.projection, second.projection);
        assert_eq!(first.projection.len(), 2000 * 16);
        let squares: f64 = first.projection.iter().map(|&v| f64::from(v).powi(2)).sum();
        let mean_norm = squares / 2000.0;
        assert!(
            (mean_norm - 1.0).abs() < 0.1,
            "mean squared norm {mean_norm}"
        );
    }

    #[test]
    fn different_seeds_give_different_projections() {
        let a = SketchBuilder::new(spec(&[2], 1)).unwrap().finish();
        let mut other = spec(&[2], 1);
        other.seed = 12;
        let b = SketchBuilder::new(other).unwrap().finish();
        assert_ne!(a.projection, b.projection);
    }
}
