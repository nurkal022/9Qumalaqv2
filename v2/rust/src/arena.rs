/// Arena: match play between models or V2 vs V1
///
/// Plays matches between two networks and reports win/draw/loss statistics.

use std::sync::Arc;

use crate::board::{Board, GameResult, Side};
use crate::gumbel::gumbel_alphazero_search;
use crate::network::Network;

/// Result of a single match game
#[derive(Debug, Clone, Copy)]
pub enum MatchResult {
    WhiteWin,
    BlackWin,
    Draw,
}

/// Play one game between two networks.
/// white_net plays as White, black_net plays as Black.
pub fn play_match_game(
    white_net: &Network,
    black_net: &Network,
    simulations: u32,
    candidates: usize,
    sigma_scale: f32,
) -> MatchResult {
    let mut board = Board::new();
    let mut move_count = 0;

    while !board.is_terminal() && move_count < 300 {
        let network = match board.side_to_move {
            Side::White => white_net,
            Side::Black => black_net,
        };

        let results =
            gumbel_alphazero_search(&board, network, simulations, candidates, sigma_scale);

        if results.is_empty() {
            break;
        }

        // Greedy move selection (no temperature in arena)
        let best_move = results
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0;

        board.make_move(best_move);
        move_count += 1;
    }

    match board.game_result() {
        Some(GameResult::Win(Side::White)) => MatchResult::WhiteWin,
        Some(GameResult::Win(Side::Black)) => MatchResult::BlackWin,
        Some(GameResult::Draw) | None => MatchResult::Draw,
    }
}

/// Run a match between two networks (alternating colors).
/// Returns (new_wins, old_wins, draws).
pub fn run_arena(
    new_net: Arc<Network>,
    old_net: Arc<Network>,
    num_games: u32,
    simulations: u32,
    candidates: usize,
    sigma_scale: f32,
) -> (u32, u32, u32) {
    let mut new_wins = 0u32;
    let mut old_wins = 0u32;
    let mut draws = 0u32;

    for game_idx in 0..num_games {
        // Alternate colors: even games new=White, odd games new=Black
        let result = if game_idx % 2 == 0 {
            play_match_game(&new_net, &old_net, simulations, candidates, sigma_scale)
        } else {
            play_match_game(&old_net, &new_net, simulations, candidates, sigma_scale)
        };

        match (game_idx % 2, result) {
            (0, MatchResult::WhiteWin) => new_wins += 1,
            (0, MatchResult::BlackWin) => old_wins += 1,
            (1, MatchResult::WhiteWin) => old_wins += 1,
            (1, MatchResult::BlackWin) => new_wins += 1,
            (_, MatchResult::Draw) => draws += 1,
            _ => unreachable!(),
        }

        if (game_idx + 1) % 50 == 0 || game_idx + 1 == num_games {
            let total = new_wins + old_wins + draws;
            let winrate = if total > 0 {
                (new_wins as f64 + 0.5 * draws as f64) / total as f64
            } else {
                0.5
            };
            eprintln!(
                "  Arena [{}/{}]: New {} - {} Old ({} draws) | Winrate: {:.1}%",
                game_idx + 1,
                num_games,
                new_wins,
                old_wins,
                draws,
                winrate * 100.0
            );
        }
    }

    (new_wins, old_wins, draws)
}

/// Calculate approximate Elo difference from win rate
pub fn winrate_to_elo(winrate: f64) -> f64 {
    if winrate <= 0.0 || winrate >= 1.0 {
        return if winrate >= 1.0 { 800.0 } else { -800.0 };
    }
    -400.0 * (1.0 / winrate - 1.0).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_winrate_to_elo() {
        // 50% winrate -> 0 Elo difference
        assert!((winrate_to_elo(0.5) - 0.0).abs() < 0.1);

        // Higher winrate -> positive Elo
        assert!(winrate_to_elo(0.75) > 0.0);
        assert!(winrate_to_elo(0.9) > winrate_to_elo(0.75));

        // Lower winrate -> negative Elo
        assert!(winrate_to_elo(0.25) < 0.0);
    }
}
