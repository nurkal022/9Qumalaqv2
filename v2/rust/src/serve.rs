/// Persistent engine protocol for web integration.
///
/// Reads JSON commands from stdin, writes JSON responses to stdout.
/// Compatible with the V1 serve protocol but uses Gumbel AlphaZero search.

use std::io::{self, BufRead, Write};
use std::sync::Arc;

use crate::board::{Board, Side, NUM_PITS};
use crate::gumbel::gumbel_alphazero_search;
use crate::network::Network;

/// Run the serve loop: read commands from stdin, respond on stdout
pub fn serve_loop(
    network: Arc<Network>,
    simulations: u32,
    candidates: usize,
    sigma_scale: f32,
) {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l.trim().to_string(),
            Err(_) => break,
        };

        if line.is_empty() {
            continue;
        }

        let response = handle_command(&line, &network, simulations, candidates, sigma_scale);

        writeln!(stdout, "{}", response).unwrap();
        stdout.flush().unwrap();
    }
}

fn handle_command(
    command: &str,
    network: &Network,
    simulations: u32,
    candidates: usize,
    sigma_scale: f32,
) -> String {
    // Parse JSON command
    // Expected format: {"command": "analyze", "board": {...}} or {"command": "best_move", ...}

    // Simple parsing (avoiding serde_json dependency for now)
    if command.contains("\"ping\"") {
        return r#"{"status": "ok", "engine": "v2-gumbel-alphazero"}"#.to_string();
    }

    if command.contains("\"analyze\"") || command.contains("\"best_move\"") {
        // Parse board state from JSON
        match parse_board_from_json(command) {
            Some(board) => {
                let results = gumbel_alphazero_search(
                    &board, network, simulations, candidates, sigma_scale,
                );

                if results.is_empty() {
                    return r#"{"error": "no legal moves"}"#.to_string();
                }

                let best = results
                    .iter()
                    .max_by(|a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();

                // Format policy as JSON array
                let mut policy = [0.0f32; NUM_PITS];
                let max_score = results.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = results.iter().map(|(_, s)| (s - max_score).exp()).sum();
                for (mov, score) in &results {
                    policy[*mov] = (score - max_score).exp() / sum;
                }

                let policy_str: Vec<String> =
                    policy.iter().map(|p| format!("{:.4}", p)).collect();

                format!(
                    r#"{{"best_move": {}, "score": {:.4}, "policy": [{}]}}"#,
                    best.0 + 1, // 1-indexed for user display
                    best.1,
                    policy_str.join(", ")
                )
            }
            None => r#"{"error": "invalid board state"}"#.to_string(),
        }
    } else {
        r#"{"error": "unknown command"}"#.to_string()
    }
}

/// Parse a Board from a JSON command string.
/// Expected board fields: pits_white, pits_black, kazan_white, kazan_black,
/// tuzdyk_white, tuzdyk_black, side_to_move
fn parse_board_from_json(json: &str) -> Option<Board> {
    // Simple JSON number extraction (avoiding serde dependency)
    let mut board = Board::new();

    // Extract pits
    if let Some(pw) = extract_array(json, "pits_white") {
        if pw.len() == NUM_PITS {
            for (i, &v) in pw.iter().enumerate() {
                board.pits[0][i] = v as u8;
            }
        }
    }

    if let Some(pb) = extract_array(json, "pits_black") {
        if pb.len() == NUM_PITS {
            for (i, &v) in pb.iter().enumerate() {
                board.pits[1][i] = v as u8;
            }
        }
    }

    if let Some(v) = extract_number(json, "kazan_white") {
        board.kazan[0] = v as u8;
    }
    if let Some(v) = extract_number(json, "kazan_black") {
        board.kazan[1] = v as u8;
    }
    if let Some(v) = extract_number(json, "tuzdyk_white") {
        board.tuzdyk[0] = v as i8;
    }
    if let Some(v) = extract_number(json, "tuzdyk_black") {
        board.tuzdyk[1] = v as i8;
    }
    if let Some(v) = extract_number(json, "side_to_move") {
        board.side_to_move = if v == 0 { Side::White } else { Side::Black };
    }

    Some(board)
}

fn extract_number(json: &str, key: &str) -> Option<i32> {
    let pattern = format!("\"{}\"", key);
    let pos = json.find(&pattern)?;
    let after_key = &json[pos + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();

    let end = after_colon
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(after_colon.len());
    after_colon[..end].parse().ok()
}

fn extract_array(json: &str, key: &str) -> Option<Vec<i32>> {
    let pattern = format!("\"{}\"", key);
    let pos = json.find(&pattern)?;
    let after_key = &json[pos + pattern.len()..];
    let bracket_start = after_key.find('[')?;
    let bracket_end = after_key.find(']')?;
    let array_str = &after_key[bracket_start + 1..bracket_end];

    let values: Vec<i32> = array_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    Some(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_board_json() {
        let json = r#"{"command": "analyze", "pits_white": [9,9,9,9,9,9,9,9,9], "pits_black": [9,9,9,9,9,9,9,9,9], "kazan_white": 0, "kazan_black": 0, "tuzdyk_white": -1, "tuzdyk_black": -1, "side_to_move": 0}"#;
        let board = parse_board_from_json(json).unwrap();
        assert_eq!(board.pits[0], [9; 9]);
        assert_eq!(board.pits[1], [9; 9]);
        assert_eq!(board.kazan, [0, 0]);
        assert_eq!(board.side_to_move, Side::White);
    }

    #[test]
    #[ignore] // Requires ONNX model for integration testing
    fn test_ping_command() {
        // This test needs a real ONNX model loaded via Network::load()
        // Run with: cargo test -- --ignored
    }
}
