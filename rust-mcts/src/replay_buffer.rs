/// Compact binary replay buffer format.
///
/// Each record (63 bytes):
///   pits[0][0..9]:    9 bytes  (u8)
///   pits[1][0..9]:    9 bytes  (u8)
///   kazan[0]:         1 byte   (u8)
///   kazan[1]:         1 byte   (u8)
///   tuzdyk[0]:        1 byte   (i8)
///   tuzdyk[1]:        1 byte   (i8)
///   side_to_move:     1 byte   (u8, 0=White, 1=Black)
///   policy[0..9]:     36 bytes (9 x f32 little-endian)
///   value:            4 bytes  (f32 little-endian)
///   Total: 63 bytes

use crate::board::Board;
use std::io::{self, Write, BufWriter};
use std::fs::File;

pub const RECORD_SIZE: usize = 23 + 36 + 4; // 63 bytes

pub struct TrainingRecord {
    pub board: Board,
    pub policy: [f32; 9],
    pub value: f32,
}

impl TrainingRecord {
    /// Serialize to bytes (63 bytes)
    pub fn to_bytes(&self) -> [u8; RECORD_SIZE] {
        let mut buf = [0u8; RECORD_SIZE];
        let mut offset = 0;

        // pits[0][0..9]
        for i in 0..9 {
            buf[offset] = self.board.pits[0][i];
            offset += 1;
        }
        // pits[1][0..9]
        for i in 0..9 {
            buf[offset] = self.board.pits[1][i];
            offset += 1;
        }
        // kazan[0], kazan[1]
        buf[offset] = self.board.kazan[0]; offset += 1;
        buf[offset] = self.board.kazan[1]; offset += 1;
        // tuzdyk[0], tuzdyk[1] (as i8 -> u8)
        buf[offset] = self.board.tuzdyk[0] as u8; offset += 1;
        buf[offset] = self.board.tuzdyk[1] as u8; offset += 1;
        // side_to_move
        buf[offset] = self.board.side_to_move as u8; offset += 1;

        // policy (9 x f32 LE)
        for i in 0..9 {
            let bytes = self.policy[i].to_le_bytes();
            buf[offset..offset + 4].copy_from_slice(&bytes);
            offset += 4;
        }

        // value (f32 LE)
        let bytes = self.value.to_le_bytes();
        buf[offset..offset + 4].copy_from_slice(&bytes);

        buf
    }
}

/// Write a collection of records to a binary file
pub fn write_records(path: &str, records: &[TrainingRecord]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for record in records {
        writer.write_all(&record.to_bytes())?;
    }

    writer.flush()?;
    eprintln!("Wrote {} records ({:.1} MB) to {}",
        records.len(),
        (records.len() * RECORD_SIZE) as f64 / 1e6,
        path,
    );

    Ok(())
}

/// Append records to an existing file
pub fn append_records(path: &str, records: &[TrainingRecord]) -> io::Result<()> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let mut writer = BufWriter::new(file);

    for record in records {
        writer.write_all(&record.to_bytes())?;
    }

    writer.flush()?;
    Ok(())
}
