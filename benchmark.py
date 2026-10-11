import numpy as np
import torch
from environment import DeltaHedgingEnv
from agent import DQNAgent

class BenchmarkRunner:
    def __init__(self, model_path="dqn_hedger.pth"):
        self.agent = DQNAgent(state_dim=5)
        # Load the pre-trained weights
        self.agent.q_net.load_state_dict(torch.load(model_path))
        self.agent.q_net.eval()

    def run_comparison(self, env_seed=None, threshold=25.0):
        """
        Runs 4 strategies on identical market & options flow:
        1. RL Agent (DQN)
        2. No Hedge
        3. Periodic (Hedge every tick)
        4. Threshold-Based (Hedge only when |net delta| > threshold)
        """
        if env_seed is not None:
            np.random.seed(env_seed)

        # Base environment to generate paths
        base_env = DeltaHedgingEnv()
        base_env.reset()
        
        # Snapshot the initial state & portfolio
        options_book = base_env.options_book
        spot_init = base_env.spot_price

        # Strategy records
        strategies = ["RL Agent (DQN)", "No Hedge", "Periodic (Daily)", f"Threshold (±{int(threshold)})"]
        results = {s: {"pnl": [], "delta_path": [], "trades": 0, "costs": 0.0, "final_pnl": 0.0} for s in strategies}

        # Pre-generate the exact price path
        dt = base_env.dt
        steps = base_env.episode_length
        shocks = np.random.normal(0, np.sqrt(dt), size=steps)

        for strat in strategies:
            env = DeltaHedgingEnv()
            env.reset()
            env.options_book = options_book
            env.spot_price = spot_init
            env._update_portfolio_greeks()

            cum_pnl = 0.0
            pnl_history = [0.0]
            delta_history = [env.inventory + env.portfolio_delta]
            total_cost = 0.0
            trade_count = 0

            for t in range(steps):
                net_delta = env.inventory + env.portfolio_delta
                action = 0

                if strat == "RL Agent (DQN)":
                    state = env._get_state()
                    action, _ = self.agent.act(state, evaluate=True)
                elif strat == "No Hedge":
                    action = 0
                elif strat == "Periodic (Daily)":
                    # Fully flatten net delta to zero
                    action = int(-round(net_delta))
                elif strat.startswith("Threshold"):
                    if abs(net_delta) > threshold:
                        action = int(-round(net_delta))

                # Step manually along the locked shock path
                trade_cost = abs(action) * env.trading_fee
                env.inventory += action
                env.cash -= (action * env.spot_price) + trade_cost
                total_cost += trade_cost
                if action != 0:
                    trade_count += 1

                # Advance price using locked shock
                env.current_step += 1
                env.time_to_expiry -= env.dt
                dW = shocks[t]
                drift = (env.risk_free_rate - 0.5 * env.volatility**2) * env.dt
                env.spot_price *= np.exp(drift + env.volatility * dW)
                env._update_portfolio_greeks()

                # Mark to market P&L
                mtm_pnl = env.cash + (env.inventory * env.spot_price)
                pnl_history.append(mtm_pnl)
                delta_history.append(env.inventory + env.portfolio_delta)

            results[strat]["pnl"] = pnl_history
            results[strat]["delta_path"] = delta_history
            results[strat]["trades"] = trade_count
            results[strat]["costs"] = total_cost
            results[strat]["final_pnl"] = pnl_history[-1]

        return results