"""
Gambling Task Environment for Dopamine-driven Reinforcement Learning Simulation
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box


class GamblingTaskEnv(gym.Env):
    """2-Armed Bandit Environment with Reversal Learning (API Gym Ancienne)"""
    
    def __init__(self, p_left=0.4, p_right=0.6, reversal_trial=500, seed=None):
        super().__init__()
        
        # Reward probabilities
        self.p_left = float(p_left)
        self.p_right = float(p_right)
        self.reversal_trial = int(reversal_trial)
        self.current_trial = 0
        
        # State configuration
        self.state_size = 10
        
        # Gymnasium spaces
        self.action_space = Discrete(2)  # 0=Left, 1=Right
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(self.state_size,), dtype=np.float32
        )
        
        # Reward history buffer
        self.reward_history_buffer = np.zeros(self.state_size, dtype=np.float32)
        
        # 🔥 CRITIQUE: Stocker le seed initial
        self.initial_seed = seed
        self.rng = np.random.default_rng(seed)
    
    def _update_state(self, action, reward):
        """Update sliding window buffer"""
        self.reward_history_buffer = np.roll(self.reward_history_buffer, -1)
        self.reward_history_buffer[-1] = float(reward)
        return self.reward_history_buffer.copy()
    
    def step(self, action):
        """Execute action (retourne 4 valeurs pour API Gym ancienne)"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        
        # Determine current probabilities
        if self.current_trial >= self.reversal_trial:
            p_left_curr, p_right_curr = self.p_right, self.p_left
        else:
            p_left_curr, p_right_curr = self.p_left, self.p_right
        
        # Generate reward
        if action == 0:  # Left
            reward = float(self.rng.binomial(1, p_left_curr))
        else:  # Right
            reward = float(self.rng.binomial(1, p_right_curr))
        
        # Update state
        observation = self._update_state(action, reward)
        self.current_trial += 1
        
        # Check termination
        terminated = self.current_trial >= 1000
        truncated = False
        
        info = {
            'trial': self.current_trial,
            'p_left_current': p_left_curr,
            'p_right_current': p_right_curr,
            'reversal_occurred': self.current_trial > self.reversal_trial
        }
        
        # ANCIENNE API: 4 valeurs
        return observation, reward, terminated, info
    
    def reset(self, seed=None, options=None):
        """Reset environment avec seeding DÉTERMINISTE TOTAL"""
        self.current_trial = 0
        self.reward_history_buffer = np.zeros(self.state_size, dtype=np.float32)
        
        # 🔥 CRITIQUE: Seed BOTH rng and action_space
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space.seed(seed)  # 🔥 POUR action_space.sample()
        else:
            self.rng = np.random.default_rng(self.initial_seed)
            if self.initial_seed is not None:
                self.action_space.seed(self.initial_seed)
        
        # ANCIENNE API: retourne seulement l'array
        return self.reward_history_buffer.copy()
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass


# Test rapide
if __name__ == "__main__":
    env = GamblingTaskEnv(seed=42)
    obs = env.reset()
    print(f"✅ Reset OK: shape={obs.shape}, values={obs[:3]}...")
    
    for i in range(3):
        obs, reward, done, info = env.step(0)
        print(f"✅ Step {i+1}: reward={reward}, p_left={info['p_left_current']:.1f}")