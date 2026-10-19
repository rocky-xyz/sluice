import math

import pytest

from sluice.evaluation.metrics import sharpe_difference


def test_by_hand_when_the_two_move_together():
    # a: mean 2, sd 2, SR 1; b: mean 2, sd 1, SR 2; rho 1
    # Var = [0 + ½(1 + 4 − 2·1·2·1)] / 3 = 1/6, z = (1 − 2) / √(1/6) = −√6
    result = sharpe_difference([0.0, 2.0, 4.0], [1.0, 2.0, 3.0])
    assert result.rho == pytest.approx(1.0)
    assert result.z == pytest.approx(-math.sqrt(6))
    assert result.days == 3


def test_by_hand_when_they_do_not():
    # b: mean 3, sd 2, SR 1.5; deviations (−2, 0, 2) and (0, 2, −2) give rho −0.5
    # Var = [2·1.5 + ½(1 + 2.25 − 2·1·1.5·0.25)] / 3 = 4.25 / 3
    result = sharpe_difference([0.0, 2.0, 4.0], [3.0, 5.0, 1.0])
    assert result.rho == pytest.approx(-0.5)
    assert result.z == pytest.approx(-0.5 / math.sqrt(4.25 / 3))


def test_it_is_antisymmetric_and_ignores_leverage():
    a, b = [0.01, -0.02, 0.03, 0.0, 0.015], [0.012, -0.01, 0.02, -0.005, 0.01]
    assert sharpe_difference(b, a).z == pytest.approx(-sharpe_difference(a, b).z)
    assert sharpe_difference([2 * x for x in a], b).z == pytest.approx(sharpe_difference(a, b).z)


def test_a_steady_edge_on_the_same_days_is_seen():
    base = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01] * 25  # 150 days
    better = [r + 0.0005 for r in base]  # the same days, a little more each
    # rho = 1, so Var = ½(SR_a − SR_b)² / T and z = √(2T)
    assert sharpe_difference(better, base).z == pytest.approx(math.sqrt(300))


def test_degenerate_inputs():
    assert sharpe_difference([0.01, 0.02], [0.0, 0.01]) is None  # too short
    assert sharpe_difference([0.01] * 5, [0.0, 0.01, 0.02, 0.0, 0.01]) is None  # flat
    with pytest.raises(ValueError, match="same days"):
        sharpe_difference([0.01, 0.02, 0.03], [0.0, 0.01])
