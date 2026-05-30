"""Statistical significance battery for the Artemis momentum factor (Stage 2).

Ported from the Project 1 ``factor_eval.stats`` and ``cmom.overfitting`` modules:
Newey-West HAC t-stats, Bonferroni / Harvey-Liu-Zhu multiple-testing, Sharpe /
drawdown / Calmar, the Deflated Sharpe Ratio and PBO/CSCV. All functions are
pure and perform no I/O.
"""
