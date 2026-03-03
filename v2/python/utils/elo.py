"""
Elo rating calculation from match results.
"""

import math


def winrate_to_elo(winrate):
    """Convert win rate to Elo difference."""
    if winrate <= 0.0:
        return -800.0
    if winrate >= 1.0:
        return 800.0
    return -400.0 * math.log10(1.0 / winrate - 1.0)


def elo_to_winrate(elo_diff):
    """Convert Elo difference to expected win rate."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def compute_elo_from_results(wins, losses, draws):
    """Compute Elo difference from match results."""
    total = wins + losses + draws
    if total == 0:
        return 0.0
    winrate = (wins + 0.5 * draws) / total
    return winrate_to_elo(winrate)


def elo_confidence_interval(wins, losses, draws, confidence=0.95):
    """Approximate 95% confidence interval for Elo difference."""
    total = wins + losses + draws
    if total == 0:
        return 0.0, (-800.0, 800.0)

    winrate = (wins + 0.5 * draws) / total
    elo = winrate_to_elo(winrate)

    # Standard error of win rate
    se = math.sqrt(winrate * (1.0 - winrate) / total)

    # Z-score for confidence level
    z = 1.96 if confidence == 0.95 else 2.576

    lo = max(winrate - z * se, 0.001)
    hi = min(winrate + z * se, 0.999)

    return elo, (winrate_to_elo(lo), winrate_to_elo(hi))


if __name__ == "__main__":
    # Example
    wins, losses, draws = 120, 60, 20
    elo, (lo, hi) = elo_confidence_interval(wins, losses, draws)
    wr = (wins + 0.5 * draws) / (wins + losses + draws)
    print(f"Results: +{wins} ={draws} -{losses} (WR: {wr:.1%})")
    print(f"Elo: {elo:+.0f} ({lo:+.0f}, {hi:+.0f})")
