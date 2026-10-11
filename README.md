# Delta hedging desk

An options market maker holds a book of customer options and has to hedge the delta with the underlying. Hedging more often
cuts risk and costs more. This app runs that trade off on thousands of simulated price paths and compares a learned hedging
agent with the rules a desk actually uses.

Live app: https://rl-dynamic-delta-hedging.streamlit.app/

## What is in it

| View | What it shows |
|---|---|
| Monte Carlo | Final P&L distribution, cost against risk frontier, and a results table for every strategy on the same paths, with an error bar on "agent against the best tuned rule" |
| Path replay | One path: spot with the agent's trades, net delta, cumulative P&L and cost for every strategy, a P&L bridge, and a trade blotter |
| Agent | What the learned policy does at each state, next to the Whalley-Wilmott band, plus the learning curve |
| Model | Validation checks that run live, the accounting, assumptions and what a desk would push on |

Strategies: unhedged, periodic (flatten every k steps), threshold (flatten when the net delta passes a band), Whalley-Wilmott
(trade back to the edge of the optimal no trade band), and the agent. A "best tuned rule" row picks the best of the rules on the
same paths, which favours the rules.

## How the desk is modelled

* Book of up to 5 European options, marked and hedged with Black-Scholes at the desk's implied vol. The price process can use a
  different realised vol, Merton jumps and vol dispersion across paths.
* P&L is `Pi = V + B + h*S` (book mark, cash, stock hedge), financed at r. It splits exactly into option book, stock hedge, cash
  carry and costs, and the app checks that.
* Costs: half spread and per share fee are linear in the trade, impact is quadratic and equals `impact_bps` for a trade as big as
  the whole book. The stock hedge is unwound at the end at the same cost.
* Objective: mean P&L minus half kappa times its variance. Risk aversion is a pure number: kappa = kappa_tilde / u, with u the
  20% of the book's one day 1 sigma move.

## The agent

A small network (2 x 128 ReLU) maps nine scale free features (net delta, gamma times the one step move, costs, impact, risk
aversion, time left, time to expiry, step length, gamma times spot over book size) to the share of the net delta to trade, 0 to 1.

It is trained with the reward of Kolm and Ritter (2019), `r = dPnL/u - kappa/2 (dPnL/u)^2` per step, by pathwise policy gradient
through the simulator (deep hedging, Buehler et al. 2019). Each batch draws random books, vols, costs, jumps, horizons and risk
aversion, so one network covers them all and the app runs it in numpy only. About 6.8 million random scenarios in total.

The first version of this project used a DQN with a 5 number state. That was replaced because Q learning has to average the price
noise away, which is slow here since the action does not change the price, and because the old reward and P&L left out the value of
the option book, so "No Hedge" showed exactly zero P&L.

## Held out results

`python evaluate.py 10000`: 10,000 fresh price paths per scenario, agent against the best of periodic every 1 to 5 steps,
thresholds from 0.5% to 8%, and Whalley-Wilmott at kappa, half and double kappa, chosen on the same paths it is scored on.

```
scenario                                     agent best rule                     rule     diff   ± se  diff %
Short straddle, plain costs                  -1066     -1116           Threshold 5.0%       50     26    4.5%
Short straddle, market impact                -1792     -1878                  WW x2.0       86     22    4.6%
Short straddle, high risk aversion           -6026     -5945           Threshold 0.5%      -81     28   -1.4%
Short straddle, wide spread                  -2242     -2304                  WW x2.0       62     31    2.7%
Short strangle, jumps                        -9795     -9988                  WW x2.0      193     26    1.9%
Long straddle, 2 hedges per day               1672      1663                  WW x2.0        8     11    0.5%
Risk reversal, quiet vol                     -1124     -1133                  WW x2.0       10      8    0.8%
Customer flow book, wide costs                -230      -214               Periodic 5      -15     10   -7.2%
Short straddle, last 5 days to expiry         -943      -961                  WW x2.0       18     19    1.9%
```

Objective in dollars, higher is better. Reading it honestly: with proportional costs on a diffusion a tuned band rule is already close to
optimal, so the agent matches it, and its edge is a few percent. It is ahead in seven of nine scenarios (by about two standard errors or more in four),
behind in two, and it never needs tuning per scenario, which the rules do. The scenarios are of the same types as the six used
to monitor training, with fresh paths.

## Run it

```
pip install -r requirements.txt
streamlit run app.py
pytest test_hedging.py
```

Retraining needs torch (`pip install torch`), then `python train.py 3000`. A second pass with weight averaging was
`INIT=policy_last.npz BATCH=2048 LR0=6e-4 LR1=2e-5 EMA=0.995 OUT=ft python train.py 1800`.

## Limits

One underlying, constant vol with no skew dynamics, delta only hedging (the book keeps its gamma, vega and theta), simple cost
model, and an agent that has only been tested inside the ranges of the sliders. See the Model tab.

---
Author: Nhi Q Tran. Copyright (c) 2026 Nhi Q Tran. All rights reserved. See LICENSE.
