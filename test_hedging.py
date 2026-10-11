import numpy as np
import pytest

import hedging as H
from black_scholes import bs_np
from policy import Policy, agent, features

MKT = H.Market()
BOOK = H.make_book("Short ATM straddle", 100.0)


def paths(spd=1, costs=H.Costs(2.0, 0.0, 0.0), n=3000, seed=1, book=BOOK, mkt=MKT, days=30, kappa=0.1):
    return H.build_paths(mkt, costs, H.Horizon(days, days, spd), book, n, seed=seed, kappa=kappa)


def test_pnl_bridge_adds_up_for_every_rule():
    P = paths(costs=H.Costs(3.0, 0.005, 10.0), n=500)
    for rule in (H.unhedged, H.periodic(1), H.periodic(3), H.threshold(0.03), H.whalley_wilmott(0.1)):
        r = H.run_strategy("x", rule, P)
        assert np.abs(sum(r.parts.values()) - r.pnl).max() < 1e-6


def test_unhedged_has_no_trades_and_no_costs():
    P = paths(n=200)
    r = H.run_strategy("u", H.unhedged, P)
    assert np.all(r.trades == 0) and np.all(r.parts["Stock hedge"] == 0) and np.all(r.parts["Costs"] == 0)


def test_book_matches_black_scholes_toolkit():
    S = np.array([[85.0, 100.0, 118.0]])
    tau = np.array([0.3, 0.2, 0.1])
    for is_call in (True, False):
        bk = H.Book(np.array([100.0]), np.array([1.0]), np.array([is_call]))
        V, D, G = H.book_state(S, bk, tau, 0.03, 0.27)
        for j in range(3):
            ref = bs_np(S[0, j], 100.0, tau[j], 0.03, 0.27, is_call)
            assert abs(V[0, j] - ref["price"]) < 1e-10
            assert abs(D[0, j] - ref["delta"]) < 1e-10
            assert abs(G[0, j] - ref["gamma"]) < 1e-10


def test_put_call_parity_of_the_book():
    bk = H.Book(np.array([100.0, 100.0]), np.array([1.0, -1.0]), np.array([True, False]))
    V, _, _ = H.book_state(np.array([[80.0, 130.0]]), bk, np.array([0.25, 0.25]), 0.03, 0.3)
    assert np.allclose(V[0], np.array([80.0, 130.0]) - 100.0 * np.exp(-0.03 * 0.25))


def test_expiry_value_is_intrinsic_with_no_greeks():
    bk = H.Book(np.array([100.0]), np.array([1.0]), np.array([True]))
    V, D, G = H.book_state(np.array([[110.0, 90.0]]), bk, np.array([0.0, 0.0]), 0.03, 0.3)
    assert np.allclose(V[0], [10.0, 0.0]) and np.all(D == 0) and np.all(G == 0)


def test_hedging_error_without_cost_falls_like_one_over_root_n():
    steps, stds = [], []
    for spd in (1, 4, 16):
        P = paths(spd=spd, costs=H.Costs(0.0, 0.0, 0.0), n=3000)
        stds.append(H.run_strategy("p", H.periodic(1), P).pnl.std())
        steps.append(P.T)
    slope = np.polyfit(np.log(steps), np.log(stds), 1)[0]
    assert abs(slope + 0.5) < 0.06


def test_gross_pnl_is_a_martingale_under_fair_marks():
    """With realised vol equal to marked vol and drift r, no hedging rule can make money before costs."""
    P = paths(costs=H.Costs(5.0, 0.0, 20.0), n=20000, seed=4)
    for rule in (H.threshold(0.05), H.whalley_wilmott(0.1)):
        r = H.run_strategy("x", rule, P)
        gross = r.pnl + (-r.parts["Costs"])
        assert abs(gross.mean()) < 4 * gross.std() / np.sqrt(len(gross))


def test_trade_cost_is_linear_plus_quadratic_impact():
    c1 = H.trade_cost(np.array([100.0]), 50.0, 1000.0, 4.0, 0.01, 0.0)
    assert abs(c1[0] - (100 * (50 * 4 / 1e4 + 0.01))) < 1e-12
    full = H.trade_cost(np.array([1000.0]), 50.0, 1000.0, 0.0, 0.0, 10.0)      # a trade as big as the book
    assert abs(full[0] - 1000 * 50 * 10 / 1e4) < 1e-9
    half = H.trade_cost(np.array([500.0]), 50.0, 1000.0, 0.0, 0.0, 10.0)
    assert abs(half[0] - full[0] / 4) < 1e-9


def test_whalley_wilmott_band_scales_with_cost_to_the_one_third():
    a = H.ww_band(100.0, 40.0, 0.1, 0.03, 2.0, 0.1, 1000.0)
    b = H.ww_band(100.0, 40.0, 0.1, 0.03, 16.0, 0.1, 1000.0)
    assert abs(b / a - 2.0) < 1e-9


def test_ww_never_trades_inside_the_band_and_lands_on_its_edge():
    P = paths(n=300)
    rule = H.whalley_wilmott(0.1)
    ctx = H.Ctx(5, P.S[:, 5], P.D[:, 5], P.G[:, 5], np.zeros(P.n), P.D[:, 5], P)
    d = rule(ctx)
    Hw = H.ww_halfwidth(ctx, 0.1)
    after = ctx.net + d
    assert np.all(np.abs(after) <= Hw + 1e-9)
    assert np.all(d[np.abs(ctx.net) <= Hw] == 0)


def test_same_paths_for_every_strategy_and_reproducible():
    a, b = paths(seed=9, n=200), paths(seed=9, n=200)
    assert np.array_equal(a.S, b.S)
    r1 = H.run_strategy("p", H.periodic(1), a)
    r2 = H.run_strategy("p", H.periodic(1), b)
    assert np.array_equal(r1.pnl, r2.pnl)


def test_objective_diff_of_identical_results_is_zero():
    x = np.random.default_rng(0).normal(size=500)
    d, se = H.objective_diff(x, x, 0.01)
    assert d == 0.0 and se == 0.0


def test_jump_paths_are_wider_than_diffusion_only():
    a = H.simulate_prices(H.Market(), H.Horizon(30, 30, 1), 4000, seed=2)[0]
    b = H.simulate_prices(H.Market(jump_lambda=20.0, jump_mean=-0.03, jump_sd=0.05), H.Horizon(30, 30, 1), 4000, seed=2)[0]
    assert np.std(np.log(b[:, -1])) > np.std(np.log(a[:, -1]))


def test_policy_features_agree_between_numpy_and_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n = 6
    args = dict(net=rng.normal(size=n) * 200, G=-rng.uniform(1, 400, n), S=rng.uniform(50, 150, n),
                sigma=rng.uniform(0.15, 0.5, n), Q=np.full(n, 8000.0), cost_bps=rng.uniform(0, 12, n),
                impact_bps=rng.uniform(0, 20, n), kappa=rng.uniform(0.05, 1, n))
    x_np = features(np, args["net"], args["G"], args["S"], args["sigma"], args["Q"], 1 / 252, args["cost_bps"],
                    args["impact_bps"], args["kappa"], 20.0, 25.0)
    t = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in args.items()}
    x_t = features(torch, t["net"], t["G"], t["S"], t["sigma"], t["Q"], 1 / 252, t["cost_bps"], t["impact_bps"], t["kappa"], 20.0, 25.0)
    assert np.allclose(x_np, x_t.numpy())


def test_agent_trades_a_share_of_net_delta_between_zero_and_all():
    import os
    path = os.path.join(os.path.dirname(__file__), "policy.npz")
    if not os.path.exists(path):
        pytest.skip("policy.npz not trained yet")
    P = paths(n=400)
    rule = agent(Policy.load(path), deadzone=0.0)
    for t in (0, 10, 25):
        ctx = H.Ctx(t, P.S[:, t], P.D[:, t], P.G[:, t], np.zeros(P.n), P.D[:, t], P)
        d = rule(ctx)
        frac = -d / ctx.net
        assert np.all(frac >= -1e-12) and np.all(frac <= 1.0 + 1e-12)
