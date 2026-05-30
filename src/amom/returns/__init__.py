"""Holding-return layer (spec §4 Stage 1.2).

Turns the survivorship-preserving universe panel (which keeps collapsed/delisted
coins and carries a point-in-time ``delisted_asof`` death signal) into realized
**spot** holding returns. The payoff: a coin that crashes ~95% then stops
reporting contributes its realized terminal crash return — it is never dropped.
"""
