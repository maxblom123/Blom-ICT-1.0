use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

use blom_core::sketch::{Sketch, SketchBuilder, SketchSpec};
use serde_json::{Value, json};

use crate::Error;

pub const FORMAT: &str = "sketch-v1";
const CHUNK_TOKENS: usize = 1 << 20;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SketchParams {
    pub seed: u64,
    pub rows: u64,
    pub orders: Vec<usize>,
    pub n_hash: usize,
    pub part_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SketchReport {
    pub pairs: u64,
    pub lookups: usize,
    pub rows: u64,
    pub rows_used: u64,
    pub max_count: u32,
}

struct DataMeta {
    vocab_size: usize,
    train_tokens: u64,
}

fn read_meta(data_dir: &Path) -> Result<DataMeta, Error> {
    let path = data_dir.join("meta.json");
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let meta: Value = serde_json::from_str(&text)
        .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
    if meta.get("format").and_then(Value::as_str) != Some("u16-le") {
        return Err(format!("{} does not describe u16-le shards", path.display()).into());
    }
    let vocab_size = meta
        .get("vocab_size")
        .and_then(Value::as_u64)
        .ok_or("meta.json has no integer vocab_size")?;
    let train_tokens = meta
        .get("train")
        .and_then(|train| train.get("tokens"))
        .and_then(Value::as_u64)
        .ok_or("meta.json has no integer train.tokens")?;
    Ok(DataMeta {
        vocab_size: usize::try_from(vocab_size)?,
        train_tokens,
    })
}

fn fill(reader: &mut impl Read, buffer: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buffer.len() {
        let Some(rest) = buffer.get_mut(filled..) else {
            break;
        };
        match reader.read(rest) {
            Ok(0) => break,
            Ok(count) => filled += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(filled)
}

fn feed(builder: &mut SketchBuilder, shard: &Path, chunk_tokens: usize) -> Result<u64, Error> {
    let mut file =
        File::open(shard).map_err(|error| format!("cannot open {}: {error}", shard.display()))?;
    let mut bytes = vec![0_u8; chunk_tokens * 2];
    let mut tokens = 0_u64;
    loop {
        let filled = fill(&mut file, &mut bytes)?;
        if filled == 0 {
            return Ok(tokens);
        }
        let used = bytes.get(..filled).unwrap_or_default();
        let (pairs, rest) = used.as_chunks::<2>();
        if !rest.is_empty() {
            return Err(
                format!("{} is not a whole number of 16-bit tokens", shard.display()).into(),
            );
        }
        let chunk: Vec<u32> = pairs
            .iter()
            .map(|pair| u32::from(u16::from_le_bytes(*pair)))
            .collect();
        builder.add(&chunk)?;
        tokens += u64::try_from(chunk.len())?;
    }
}

fn write_atomically(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>,
) -> Result<(), Error> {
    let temporary = path.with_extension("tmp");
    let result = File::create(&temporary)
        .map(BufWriter::new)
        .and_then(|mut writer| {
            write(&mut writer)?;
            writer.flush()
        });
    match result {
        Ok(()) => Ok(fs::rename(&temporary, path)?),
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(error.into())
        }
    }
}

fn write_f32(path: &Path, values: &[f32]) -> Result<(), Error> {
    write_atomically(path, |writer| {
        values
            .iter()
            .try_for_each(|value| writer.write_all(&value.to_le_bytes()))
    })
}

fn write_u32(path: &Path, values: &[u32]) -> Result<(), Error> {
    write_atomically(path, |writer| {
        values
            .iter()
            .try_for_each(|value| writer.write_all(&value.to_le_bytes()))
    })
}

fn write_sketch(
    out_dir: &Path,
    sketch: &Sketch,
    source_tokens: u64,
) -> Result<SketchReport, Error> {
    fs::create_dir_all(out_dir)?;
    let manifest = out_dir.join("sketch.json");
    match fs::remove_file(&manifest) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    write_f32(&out_dir.join("projection.f32"), &sketch.projection)?;
    write_f32(&out_dir.join("sums.f32"), &sketch.sums)?;
    write_u32(&out_dir.join("counts.u32"), &sketch.counts)?;

    let rows_used = u64::try_from(sketch.counts.iter().filter(|&&count| count > 0).count())?;
    let max_count = sketch.counts.iter().copied().max().unwrap_or(0);
    let description = json!({
        "format": FORMAT,
        "seed": sketch.spec.seed,
        "rows": sketch.spec.rows,
        "orders": sketch.spec.orders,
        "hashes": sketch.spec.n_hash,
        "part_dim": sketch.spec.part_dim,
        "vocab_size": sketch.spec.vocab_size,
        "lookups": sketch.lookups,
        "pairs": sketch.pairs,
        "source_tokens": source_tokens,
        "rows_used": rows_used,
        "max_count": max_count,
        "files": {
            "projection": "projection.f32",
            "sums": "sums.f32",
            "counts": "counts.u32",
        },
    });
    let text = serde_json::to_string_pretty(&description)? + "\n";
    write_atomically(&manifest, |writer| writer.write_all(text.as_bytes()))?;
    Ok(SketchReport {
        pairs: sketch.pairs,
        lookups: sketch.lookups,
        rows: sketch.spec.rows,
        rows_used,
        max_count,
    })
}

pub fn build_sketch(
    data_dir: &Path,
    out_dir: &Path,
    params: &SketchParams,
) -> Result<SketchReport, Error> {
    build_with_chunks(data_dir, out_dir, params, CHUNK_TOKENS)
}

fn build_with_chunks(
    data_dir: &Path,
    out_dir: &Path,
    params: &SketchParams,
    chunk_tokens: usize,
) -> Result<SketchReport, Error> {
    let meta = read_meta(data_dir)?;
    let spec = SketchSpec {
        seed: params.seed,
        rows: params.rows,
        orders: params.orders.clone(),
        n_hash: params.n_hash,
        part_dim: params.part_dim,
        vocab_size: meta.vocab_size,
    };
    let mut builder = SketchBuilder::new(spec)?;
    let tokens = feed(&mut builder, &data_dir.join("train.bin"), chunk_tokens)?;
    if tokens != meta.train_tokens {
        return Err(format!(
            "train.bin has {tokens} tokens but meta.json says {}",
            meta.train_tokens
        )
        .into());
    }
    let sketch = builder.finish();
    if sketch.pairs == 0 {
        return Err("the training shard needs at least two tokens".into());
    }
    write_sketch(out_dir, &sketch, tokens)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use blom_core::hash::SplitMix64;
    use blom_core::sketch::{SketchBuilder, SketchSpec};
    use serde_json::Value;

    use super::{SketchParams, build_sketch, build_with_chunks};

    const VOCAB: usize = 50;

    fn params() -> SketchParams {
        SketchParams {
            seed: 3,
            rows: 10_007,
            orders: vec![2, 3],
            n_hash: 2,
            part_dim: 4,
        }
    }

    fn tokens(length: usize) -> Vec<u16> {
        let mut rng = SplitMix64::new(9);
        (0..length)
            .map(|_| u16::try_from(rng.next_u64() % 50).unwrap())
            .collect()
    }

    fn write_data(dir: &Path, tokens: &[u16], claimed: u64) {
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        fs::write(dir.join("train.bin"), bytes).unwrap();
        let meta = format!(
            "{{\"format\":\"u16-le\",\"vocab_size\":{VOCAB},\"end_of_text_id\":0,\"train\":{{\"tokens\":{claimed}}},\"val\":{{\"tokens\":1}}}}"
        );
        fs::write(dir.join("meta.json"), meta).unwrap();
    }

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = fs::read(path).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    }

    fn read_u32(path: &Path) -> Vec<u32> {
        let bytes = fs::read(path).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect()
    }

    fn setup(length: usize) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        let out = dir.path().join("out");
        fs::create_dir_all(&data).unwrap();
        write_data(&data, &tokens(length), length as u64);
        (dir, data, out)
    }

    #[test]
    fn writes_the_files_and_the_manifest() {
        let (_dir, data, out) = setup(400);
        let report = build_sketch(&data, &out, &params()).unwrap();
        assert_eq!(report.pairs, 399);
        assert_eq!(report.lookups, 4);
        assert_eq!(report.rows, 10_007);
        assert!(report.rows_used > 0 && report.rows_used <= 399 * 4);

        let manifest: Value =
            serde_json::from_str(&fs::read_to_string(out.join("sketch.json")).unwrap()).unwrap();
        assert_eq!(manifest["format"], "sketch-v1");
        assert_eq!(manifest["seed"], 3);
        assert_eq!(manifest["rows"], 10_007);
        assert_eq!(manifest["orders"], serde_json::json!([2, 3]));
        assert_eq!(manifest["hashes"], 2);
        assert_eq!(manifest["part_dim"], 4);
        assert_eq!(manifest["vocab_size"], 50);
        assert_eq!(manifest["lookups"], 4);
        assert_eq!(manifest["pairs"], 399);
        assert_eq!(manifest["source_tokens"], 400);
        assert_eq!(manifest["files"]["sums"], "sums.f32");

        assert_eq!(read_f32(&out.join("projection.f32")).len(), 50 * 4);
        assert_eq!(read_f32(&out.join("sums.f32")).len(), 10_007 * 4);
        let counts = read_u32(&out.join("counts.u32"));
        assert_eq!(counts.len(), 10_007);
        assert_eq!(counts.iter().map(|&c| u64::from(c)).sum::<u64>(), 399 * 4);
        assert_eq!(
            u64::from(*counts.iter().max().unwrap()),
            u64::from(report.max_count)
        );
        assert!(
            fs::read_dir(&out)
                .unwrap()
                .all(|e| { e.unwrap().path().extension().is_none_or(|ext| ext != "tmp") })
        );
    }

    #[test]
    fn matches_building_in_memory_whatever_the_read_chunk_size() {
        let stream = tokens(1000);
        let wide: Vec<u32> = stream.iter().map(|&t| u32::from(t)).collect();
        let spec = SketchSpec {
            seed: 3,
            rows: 10_007,
            orders: vec![2, 3],
            n_hash: 2,
            part_dim: 4,
            vocab_size: VOCAB,
        };
        let mut builder = SketchBuilder::new(spec).unwrap();
        builder.add(&wide).unwrap();
        let expected = builder.finish();

        for chunk in [1, 2, 3, 7, 100, 1000, 5000] {
            let dir = tempfile::tempdir().unwrap();
            let data = dir.path().join("data");
            let out = dir.path().join("out");
            fs::create_dir_all(&data).unwrap();
            write_data(&data, &stream, 1000);
            build_with_chunks(&data, &out, &params(), chunk).unwrap();
            assert_eq!(
                read_f32(&out.join("sums.f32")),
                expected.sums,
                "chunk {chunk}"
            );
            assert_eq!(
                read_u32(&out.join("counts.u32")),
                expected.counts,
                "chunk {chunk}"
            );
            assert_eq!(read_f32(&out.join("projection.f32")), expected.projection);
        }
    }

    #[test]
    fn rebuilding_replaces_the_previous_sketch() {
        let (_dir, data, out) = setup(300);
        build_sketch(&data, &out, &params()).unwrap();
        let mut other = params();
        other.part_dim = 8;
        build_sketch(&data, &out, &other).unwrap();
        assert_eq!(read_f32(&out.join("sums.f32")).len(), 10_007 * 8);
        let manifest: Value =
            serde_json::from_str(&fs::read_to_string(out.join("sketch.json")).unwrap()).unwrap();
        assert_eq!(manifest["part_dim"], 8);
    }

    #[test]
    fn a_failed_build_leaves_the_previous_sketch_alone() {
        let (_dir, data, out) = setup(300);
        build_sketch(&data, &out, &params()).unwrap();
        let before = fs::read_to_string(out.join("sketch.json")).unwrap();
        fs::write(data.join("train.bin"), [1_u8, 2, 3]).unwrap();
        assert!(build_sketch(&data, &out, &params()).is_err());
        assert_eq!(fs::read_to_string(out.join("sketch.json")).unwrap(), before);
    }

    #[test]
    fn rejects_bad_inputs_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        let out = dir.path().join("out");
        fs::create_dir_all(&data).unwrap();

        let error = build_sketch(&data, &out, &params())
            .unwrap_err()
            .to_string();
        assert!(error.contains("cannot read"), "{error}");

        fs::write(data.join("meta.json"), "not json").unwrap();
        let error = build_sketch(&data, &out, &params())
            .unwrap_err()
            .to_string();
        assert!(error.contains("cannot parse"), "{error}");

        fs::write(data.join("meta.json"), "{\"format\":\"u32\"}").unwrap();
        let error = build_sketch(&data, &out, &params())
            .unwrap_err()
            .to_string();
        assert!(error.contains("u16-le"), "{error}");

        write_data(&data, &tokens(100), 99);
        let error = build_sketch(&data, &out, &params())
            .unwrap_err()
            .to_string();
        assert!(error.contains("says 99"), "{error}");

        write_data(&data, &[1], 1);
        let error = build_sketch(&data, &out, &params())
            .unwrap_err()
            .to_string();
        assert!(error.contains("at least two"), "{error}");

        write_data(&data, &[1, 2, 60, 3], 4);
        let error = build_sketch(&data, &out, &params())
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside the vocabulary"), "{error}");

        fs::write(data.join("train.bin"), [1_u8, 0, 2]).unwrap();
        let error = build_sketch(&data, &out, &params())
            .unwrap_err()
            .to_string();
        assert!(error.contains("16-bit"), "{error}");

        fs::remove_file(data.join("train.bin")).unwrap();
        let error = build_sketch(&data, &out, &params())
            .unwrap_err()
            .to_string();
        assert!(error.contains("cannot open"), "{error}");

        let mut bad = params();
        bad.orders = vec![];
        write_data(&data, &tokens(100), 100);
        let error = build_sketch(&data, &out, &bad).unwrap_err().to_string();
        assert!(error.contains("invalid sketch configuration"), "{error}");
        assert!(!out.exists());
    }
}
