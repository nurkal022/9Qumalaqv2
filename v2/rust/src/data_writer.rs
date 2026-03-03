/// Binary data writer for self-play training samples.
///
/// Format per sample (70*4 + 9*4 + 4 = 316 bytes):
///   features: [f32; 70]  - 280 bytes (little-endian)
///   policy:   [f32; 9]   - 36 bytes (little-endian)
///   value:    f32         - 4 bytes (little-endian)

use std::fs::File;
use std::io::{BufWriter, Write};

use byteorder::{LittleEndian, WriteBytesExt};

use crate::features::FEATURE_SIZE;
use crate::network::NUM_ACTIONS;

/// Size of one training sample in bytes
pub const SAMPLE_SIZE: usize = (FEATURE_SIZE + NUM_ACTIONS + 1) * 4; // 316 bytes

/// One training sample from self-play
pub struct TrainingSample {
    pub features: [f32; FEATURE_SIZE],
    pub policy: [f32; NUM_ACTIONS],
    pub value: f32,
}

/// Buffered binary writer for training data
pub struct DataWriter {
    writer: BufWriter<File>,
    count: u64,
}

impl DataWriter {
    /// Create a new data writer for the given output path
    pub fn new(path: &str) -> Self {
        let file = File::create(path).expect("Failed to create data file");
        DataWriter {
            writer: BufWriter::with_capacity(1024 * 1024, file), // 1MB buffer
            count: 0,
        }
    }

    /// Write one training sample in binary format
    pub fn write_sample(&mut self, sample: &TrainingSample) {
        for &f in &sample.features {
            self.writer.write_f32::<LittleEndian>(f).unwrap();
        }
        for &p in &sample.policy {
            self.writer.write_f32::<LittleEndian>(p).unwrap();
        }
        self.writer.write_f32::<LittleEndian>(sample.value).unwrap();
        self.count += 1;
    }

    /// Flush and return the number of samples written
    pub fn finish(&mut self) -> u64 {
        self.writer.flush().unwrap();
        self.count
    }

    pub fn sample_count(&self) -> u64 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn test_write_and_read_sample() {
        let path = "/tmp/test_v2_data.bin";

        let sample = TrainingSample {
            features: {
                let mut f = [0.0f32; FEATURE_SIZE];
                for i in 0..FEATURE_SIZE {
                    f[i] = i as f32 / FEATURE_SIZE as f32;
                }
                f
            },
            policy: [0.1, 0.15, 0.05, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
            value: 0.75,
        };

        {
            let mut writer = DataWriter::new(path);
            writer.write_sample(&sample);
            let count = writer.finish();
            assert_eq!(count, 1);
        }

        // Verify file size
        let metadata = std::fs::metadata(path).unwrap();
        assert_eq!(metadata.len(), SAMPLE_SIZE as u64);

        // Read back and verify
        let mut file = File::open(path).unwrap();
        let mut buf = vec![0u8; SAMPLE_SIZE];
        file.read_exact(&mut buf).unwrap();

        // Check first feature value
        let first_feature = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        assert!((first_feature - 0.0).abs() < 0.001);

        // Check value (last 4 bytes)
        let value_offset = (FEATURE_SIZE + NUM_ACTIONS) * 4;
        let value = f32::from_le_bytes([
            buf[value_offset],
            buf[value_offset + 1],
            buf[value_offset + 2],
            buf[value_offset + 3],
        ]);
        assert!((value - 0.75).abs() < 0.001);

        std::fs::remove_file(path).ok();
    }
}
