"""
Betting market calculations derived from the Dixon-Coles score matrix.

IMPORTANT:
This module does NOT modify the underlying prediction model.
It converts the existing score probability matrix into betting-market
probabilities.
"""

from __future__ import annotations

import numpy as np


def total_goals_probability(matrix, threshold: float, over: bool = True) -> float:
    """
    Probability of total goals being over/under a threshold.
    """
    size = matrix.shape[0]

    probability = 0.0

    for home_goals in range(size):
        for away_goals in range(size):
            total = home_goals + away_goals

            if over and total > threshold:
                probability += matrix[home_goals, away_goals]

            elif not over and total < threshold:
                probability += matrix[home_goals, away_goals]

    return float(probability)


def btts_probability(matrix, yes: bool = True) -> float:
    """
    Probability of Both Teams To Score.
    """
    probability = 0.0

    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):

            btts = home_goals >= 1 and away_goals >= 1

            if btts == yes:
                probability += matrix[home_goals, away_goals]

    return float(probability)


def team_goals_probability(
    matrix,
    team: str,
    threshold: float,
    over: bool = True,
) -> float:
    """
    Probability of a team scoring over/under a threshold.

    team:
        "home" or "away"
    """

    probability = 0.0

    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):

            goals = home_goals if team == "home" else away_goals

            if over and goals > threshold:
                probability += matrix[home_goals, away_goals]

            elif not over and goals < threshold:
                probability += matrix[home_goals, away_goals]

    return float(probability)


def double_chance_probabilities(matrix):
    """
    Returns:
        1X, X2, 12
    """

    home = np.tril(matrix, k=-1).sum()
    draw = np.trace(matrix)
    away = np.triu(matrix, k=1).sum()

    return {
        "1X": float(home + draw),
        "X2": float(draw + away),
        "12": float(home + away),
    }


def fair_odds(probability: float) -> float | None:
    """
    Convert probability into fair decimal odds.
    """

    if probability <= 0:
        return None

    return 1.0 / probability


def implied_probability(decimal_odds: float) -> float | None:
    """
    Convert decimal bookmaker odds into raw implied probability.
    """

    if decimal_odds <= 1:
        return None

    return 1.0 / decimal_odds


def edge(model_probability: float, bookmaker_odds: float) -> float | None:
    """
    Percentage-point probability edge.

    Example:
        Model = 0.57
        Bookmaker implied probability = 0.50

        Edge = +0.07
    """

    implied = implied_probability(bookmaker_odds)

    if implied is None:
        return None

    return model_probability - implied


def expected_value(
    model_probability: float,
    bookmaker_odds: float,
) -> float | None:
    """
    Expected value per unit stake.

    EV = (model probability × decimal odds) - 1
    """

    if bookmaker_odds <= 1:
        return None

    return (model_probability * bookmaker_odds) - 1.0


def most_likely_scores(matrix, n: int = 10):
    """
    Return the n most probable exact scores.
    """

    scores = []

    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):

            scores.append(
                (
                    home_goals,
                    away_goals,
                    float(matrix[home_goals, away_goals]),
                )
            )

    scores.sort(key=lambda x: x[2], reverse=True)

    return scores[:n]