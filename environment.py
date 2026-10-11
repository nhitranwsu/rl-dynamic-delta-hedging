import numpy as np
import random
from black_scholes import bs_greeks

class DeltaHedgingEnv:
    def __init__(self):
        # 1. Market & Engine Parameters
        self.spot_price = 100.0
        self.volatility = 0.20
        self.risk_free_rate = 0.05
        self.trading_fee = 0.01          # Transaction cost per share
        self.dt = 1.0 / 252.0            # One trading day
        self.episode_length = 60         # 60 trading days to expiry
        
        # Options Portfolio limits
        self.max_contracts = 100
        self.multiplier = 100
        
    def reset(self):
        # 2. Initialize a fresh trading session
        self.current_step = 0
        self.time_to_expiry = self.episode_length * self.dt
        self.spot_price = 100.0
        self.inventory = 0.0
        self.cash = 0.0
        
        # Generate a random initial customer options portfolio
        # Positive qty = we bought options, Negative = we sold to customers
        self.options_book = self._generate_random_portfolio()
        
        self._update_portfolio_greeks()
        return self._get_state()

    def step(self, action_shares):
        # 3. Execute the AI's hedging action
        trade_cost = abs(action_shares) * self.trading_fee
        self.inventory += action_shares
        self.cash -= (action_shares * self.spot_price) + trade_cost
        
        # 4. Advance time and simulate market movement
        self.current_step += 1
        self.time_to_expiry -= self.dt
        self._simulate_next_price()
        
        # 5. Calculate new risk profile
        self._update_portfolio_greeks()
        
        # 6. Calculate Reward (Objective: minimize delta exposure and costs)
        reward = self._calculate_reward(trade_cost)
        
        # 7. Check if expiration is reached
        done = self.current_step >= self.episode_length
        
        return self._get_state(), reward, done

    def _get_state(self):
        # The exact variables the neural network will "see"
        return np.array([
            self.spot_price,
            self.time_to_expiry,
            self.inventory,
            self.portfolio_delta,
            self.portfolio_gamma
        ], dtype=np.float32)

    def _simulate_next_price(self):
        # Geometric Brownian Motion (GBM)
        dW = np.random.normal(0, np.sqrt(self.dt))
        drift = (self.risk_free_rate - 0.5 * self.volatility**2) * self.dt
        diffusion = self.volatility * dW
        self.spot_price *= np.exp(drift + diffusion)

    def _generate_random_portfolio(self):
        # Create a book of 1-5 random option positions with the same expiry
        portfolio = []
        num_positions = random.randint(1, 5)
        for _ in range(num_positions):
            strike = random.choice([90, 95, 100, 105, 110])
            qty = random.randint(-10, 10) * self.multiplier 
            opt_type = random.choice(['call', 'put'])
            if qty != 0:
                portfolio.append({'strike': strike, 'qty': qty, 'type': opt_type})
        return portfolio

    def _update_portfolio_greeks(self):
        # Aggregate Greeks across all active customer contracts
        total_delta = 0.0
        total_gamma = 0.0
        
        for opt in self.options_book:
            greeks = bs_greeks(
                S=self.spot_price,
                K=opt['strike'],
                T=max(self.time_to_expiry, 1e-5), # Prevent division by zero
                r=self.risk_free_rate,
                sigma=self.volatility,
                option_type=opt['type']
            )
            total_delta += greeks['delta'] * opt['qty']
            total_gamma += greeks['gamma'] * opt['qty']
            
        self.portfolio_delta = total_delta
        self.portfolio_gamma = total_gamma

    def _calculate_reward(self, trade_cost):
        # The AI's Score: Unhedged risk + transaction fees
        net_delta_exposure = self.inventory + self.portfolio_delta
        
        # Penalize large unhedged delta (squared to punish larger deviations more)
        risk_penalty = 0.5 * (net_delta_exposure ** 2) 
        
        # The reward is negative because the AI wants to minimize these penalties
        return -(risk_penalty + trade_cost)