"""
Actor-Critic Agent with Explicit Reward Prediction Error (RPE)
"""

import numpy as np


def softmax(x, temperature=1.0):
    """Convert logits to probabilities"""
    x_stable = x - np.max(x)
    exp_x = np.exp(x_stable / temperature)
    return exp_x / np.sum(exp_x)


class ActorCriticAgent:
    """Manual Actor-Critic with explicit RPE computation"""
    
    def __init__(self, state_size=10, action_size=2, learning_rate=0.01, gamma=0.99, temperature=1.0, seed=None):
        # 🔥 P1: LE PARAMÈTRE SEED EST BIEN PRÉSENT ICI
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.alpha = float(learning_rate)
        self.gamma = float(gamma)
        self.temperature = float(temperature)
        
        # 🔥 P2: CRÉATION DU RNG AVEC LE SEED
        self.rng = np.random.default_rng(seed)
        
        # Actor network (policy) - utiliser self.rng
        self.actor_weights = self.rng.standard_normal((self.state_size, self.action_size)) * 0.01
        self.actor_bias = np.zeros(self.action_size)
        
        # Critic network (value) - utiliser self.rng
        self.critic_weights = self.rng.standard_normal((self.state_size, 1)) * 0.01
        self.critic_bias = np.zeros(1)
        
        # History
        self.rpe_history = []
        self.reward_history = []
        self.value_history = []
        self.action_history = []
        self.policy_history = []
    
    def _extract_state(self, state):
        """GÈRE LES TUPLES : Si state est (obs, info), retourne obs"""
        if isinstance(state, tuple):
            return state[0]
        return state
    
    def forward_actor(self, state):
        """Compute action logits (compatible avec tuple)"""
        state = self._extract_state(state)
        state = np.asarray(state).flatten()
        return np.dot(state, self.actor_weights) + self.actor_bias
    
    def forward_critic(self, state):
        """Compute value estimate (compatible avec tuple)"""
        state = self._extract_state(state)
        state = np.asarray(state).flatten()
        value = np.dot(state, self.critic_weights) + self.critic_bias
        return float(value[0])
    
    def get_action(self, state, deterministic=False):
        """Select action from policy"""
        logits = self.forward_actor(state)
        probs = softmax(logits, self.temperature)
        self.policy_history.append(probs.copy())
        
        if deterministic:
            return int(np.argmax(probs))
        else:
            # 🔥 P3: UTILISER self.rng au lieu de np.random
            return int(self.rng.choice(self.action_size, p=probs))
    
    def compute_rpe(self, state_t, action_t, reward_t, state_next, done):
        """Compute Reward Prediction Error (δ)"""
        v_t = self.forward_critic(state_t)
        v_next = self.forward_critic(state_next) if not done else 0.0
        delta = reward_t + self.gamma * v_next - v_t
        return float(delta)
    
    def update(self, state_t, action_t, reward_t, state_next, done):
        """Update Actor and Critic using RPE"""
        delta = self.compute_rpe(state_t, action_t, reward_t, state_next, done)
        
        state_t_clean = self._extract_state(state_t)
        state_t_clean = np.asarray(state_t_clean).flatten()
        self.critic_weights += self.alpha * delta * state_t_clean[:, np.newaxis]
        self.critic_bias += self.alpha * delta
        
        logits = self.forward_actor(state_t)
        probs = softmax(logits, self.temperature)
        grad_log_pi = -probs
        grad_log_pi[action_t] += 1.0
        
        grad_actor = np.outer(state_t_clean, grad_log_pi)
        self.actor_weights += self.alpha * delta * grad_actor
        self.actor_bias += self.alpha * delta * grad_log_pi
        
        self.rpe_history.append(delta)
        self.reward_history.append(reward_t)
        self.value_history.append(self.forward_critic(state_t))
        self.action_history.append(action_t)
        
        return delta
    
    def get_statistics(self):
        """Return training statistics"""
        return {
            'rpe_history': np.array(self.rpe_history),
            'reward_history': np.array(self.reward_history),
            'value_history': np.array(self.value_history),
            'action_history': np.array(self.action_history),
            'policy_history': np.array(self.policy_history),
            'actor_weights': self.actor_weights.copy(),
            'critic_weights': self.critic_weights.copy(),
        }
    
    def reset_history(self):
        """Clear history buffers"""
        self.rpe_history.clear()
        self.reward_history.clear()
        self.value_history.clear()
        self.action_history.clear()
        self.policy_history.clear()


# Test rapide
if __name__ == "__main__":
    agent = ActorCriticAgent(seed=42)
    state = np.random.randn(10)
    action = agent.get_action(state)
    delta = agent.compute_rpe(state, action, 1.0, state, False)
    print(f"✅ Initial RPE: {delta:.4f}")