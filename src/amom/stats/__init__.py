"""Statistical significance battery for the Artemis momentum factor (Stage 2).

Ported from the author's earlier ``factor_eval.stats`` and ``cmom.overfitting`` modules:
Newey-West HAC t-stats, Bonferroni / Harvey-Liu-Zhu multiple-testing, Sharpe /
drawdown / Calmar, the Deflated Sharpe Ratio and PBO/CSCV. All functions are
pure and perform no I/O.

The Stage-2-native additions (no earlier counterpart) live in ``sharpe_se``:
the HAC bandwidth rule (``maxlags_for``), the autocorrelation-aware Lo (2002)
Sharpe SE (``lo_sharpe_se``), and effective-n / power labelling
(``effective_n_and_power``).
"""

from amom.stats.sharpe_se import (
    effective_n_and_power,
    lo_sharpe_se,
    maxlags_for,
)
from amom.stats.subsample import sign_stability

__all__ = [
    "effective_n_and_power",
    "lo_sharpe_se",
    "maxlags_for",
    "sign_stability",
]
