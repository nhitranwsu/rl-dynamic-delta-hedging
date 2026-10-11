"""Held out comparison: the agent against the best tuned rule, on a grid of scenarios, with standard errors.

Run: python evaluate.py [paths_per_scenario]
Every row uses fresh price paths (not the ones used in training or in the app). The tuned rule is the best of periodic
every 1 to 5 steps, thresholds 0.5% to 8%, and Whalley-Wilmott at kappa and at half and double kappa, chosen on the very
paths it is scored on, so the comparison favours the rules.
"""
import sys

import numpy as np

import hedging as H
from policy import Policy, agent

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
import os
pol = Policy.load(os.environ.get("POLICY", "policy.npz"))

SC = [
    ("Short straddle, plain costs", "Short ATM straddle", H.Market(), H.Costs(3, 0.005, 0), H.Horizon(30, 30, 1), 0.1),
    ("Short straddle, market impact", "Short ATM straddle", H.Market(), H.Costs(3, 0.005, 25), H.Horizon(30, 30, 1), 0.1),
    ("Short straddle, high risk aversion", "Short ATM straddle", H.Market(), H.Costs(3, 0.005, 5), H.Horizon(30, 30, 1), 1.0),
    ("Short straddle, wide spread", "Short ATM straddle", H.Market(), H.Costs(10, 0.0, 5), H.Horizon(30, 30, 1), 0.1),
    ("Short strangle, jumps", "Short OTM strangle", H.Market(jump_lambda=12.0, jump_mean=-0.03, jump_sd=0.04), H.Costs(4, 0.0, 10), H.Horizon(30, 30, 1), 0.2),
    ("Long straddle, 2 hedges per day", "Long ATM straddle", H.Market(sigma_imp=0.30, sigma_real=0.33), H.Costs(3, 0.0, 10), H.Horizon(20, 45, 2), 0.1),
    ("Risk reversal, quiet vol", "Risk reversal (short put, long call)", H.Market(sigma_imp=0.18, sigma_real=0.16), H.Costs(5, 0.0, 5), H.Horizon(40, 40, 1), 0.3),
    ("Customer flow book, wide costs", "Customer flow (random book)", H.Market(sigma_imp=0.35, sigma_real=0.35), H.Costs(8, 0.0, 15), H.Horizon(20, 20, 1), 0.1),
    ("Short straddle, last 5 days to expiry", "Short ATM straddle", H.Market(), H.Costs(3, 0.005, 10), H.Horizon(5, 5, 2), 0.1),
]

print(f"{N:,} fresh paths per scenario, objective in dollars, difference is agent minus best tuned rule\n")
print(f"{'scenario':40s} {'agent':>9s} {'best rule':>9s} {'rule':>24s} {'diff':>8s} {'± se':>6s} {'diff %':>7s}")
rows = []
for i, (name, bk, mkt, cs, hz, kap) in enumerate(SC):
    book = H.make_book(bk, 100.0, 50, 100, seed=7)
    P = H.build_paths(mkt, cs, hz, book, N, seed=9000 + i, kappa=kap)
    u = float(P.u[0])
    res_a = H.run_strategy("agent", agent(pol), P)
    cands = [(f"Periodic {k}", H.periodic(k)) for k in (1, 2, 3, 4, 5) if k <= P.T]
    cands += [(f"Threshold {b * 100:.1f}%", H.threshold(b)) for b in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08)]
    cands += [(f"WW x{m}", H.whalley_wilmott(kap * m)) for m in (0.5, 1.0, 2.0)]
    best = None
    for lab, f in cands:
        r = H.run_strategy(lab, f, P)
        j = H.summarize(r, P)["objective"]
        if best is None or j > best[2]:
            best = (lab, r, j)
    ja = H.summarize(res_a, P)["objective"]
    d, se = H.objective_diff(res_a.pnl, best[1].pnl, kap / u)
    pct = 100.0 * d / abs(best[2])
    rows.append((name, ja, best[2], best[0], d, se, pct))
    print(f"{name:40s} {ja:9.0f} {best[2]:9.0f} {best[0]:>24s} {d:8.0f} {se:6.0f} {pct:6.1f}%", flush=True)
