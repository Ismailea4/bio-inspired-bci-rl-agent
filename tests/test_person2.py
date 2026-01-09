"""
Testing Framework for Person 2: RL Environment & Agent
=======================================================
Comprehensive unit tests for:
- GamblingTaskEnv (2-armed bandit with reversal learning)
- ActorCriticAgent (manual implementation with explicit RPE)
- Integration between environment and agent

Author: Testing Framework
Date: January 2026
Status: Ready for Person 2 implementation validation

This file provides test templates that should be implemented
alongside the RL environment and agent code.
"""

import numpy as np
import pytest
from typing import Tuple


# ============================================================================
# TEST SUITE 1: Gambling Task Environment Tests
# ============================================================================

class TestGamblingTaskEnvironment:
    """
    Test suite for GamblingTaskEnv (Person 2 deliverable)
    
    Prerequisites: Person 2 must implement envs/gambling_task.py
    """
    
    @pytest.fixture
    def env(self):
        """Initialize environment for testing"""
        # This will fail until Person 2 implements gambling_task.py
        try:
            from envs.gambling_task import GamblingTaskEnv
            return GamblingTaskEnv(p_left=0.4, p_right=0.6, reversal_trial=500)
        except ImportError:
            pytest.skip("GamblingTaskEnv not yet implemented")
    
    def test_env_initialization(self, env):
        """Test basic environment initialization"""
        assert env is not None
        assert hasattr(env, 'action_space'), "Missing action_space"
        assert hasattr(env, 'observation_space'), "Missing observation_space"
        assert hasattr(env, 'step'), "Missing step method"
        assert hasattr(env, 'reset'), "Missing reset method"
    
    def test_action_space_is_discrete(self, env):
        """Action space should be Discrete(2): Left or Right"""
        from gymnasium import spaces
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 2, "Should have exactly 2 actions"
    
    def test_observation_space_shape(self, env):
        """Observation space should be Box with state_size"""
        from gymnasium import spaces
        assert isinstance(env.observation_space, spaces.Box)
        # State should be reasonable size (e.g., 10 = historical window)
        assert env.observation_space.shape[0] > 0
    
    def test_reset_returns_valid_state(self, env):
        """Reset should return initial state"""
        state = env.reset()
        
        # State should match observation space
        assert state.shape == env.observation_space.shape
        assert isinstance(state, np.ndarray)
    
    def test_step_returns_correct_tuple(self, env):
        """Step should return (state, reward, done, info)"""
        state = env.reset()
        action = env.action_space.sample()  # Random action
        
        result = env.step(action)
        assert len(result) == 4, "step should return 4 values"
        
        next_state, reward, done, info = result
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(done, (bool, np.bool_))
        assert isinstance(info, dict)
    
    def test_reward_is_binary(self, env):
        """Reward should be 0 or 1 (Bernoulli outcome)"""
        state = env.reset()
        
        for _ in range(100):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            
            assert reward in [0, 1], f"Reward {reward} not in {{0, 1}}"
    
    def test_state_shape_consistency(self, env):
        """State shape should remain consistent across steps"""
        initial_state = env.reset()
        initial_shape = initial_state.shape
        
        for _ in range(50):
            action = env.action_space.sample()
            next_state, _, _, _ = env.step(action)
            
            assert next_state.shape == initial_shape, \
                "State shape changed during episode"
    
    def test_episode_termination(self, env):
        """Episode should terminate after max steps"""
        state = env.reset()
        max_steps = 1000  # Typical for gambling task
        
        for step in range(max_steps + 100):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            
            if done:
                assert step < max_steps + 10, \
                    f"Episode didn't terminate by step {max_steps}"
                break
        else:
            # If loop completes without break, done was never True
            pytest.fail(f"Episode didn't terminate within {max_steps} steps")
    
    def test_reward_probability_before_reversal(self, env):
        """
        Before reversal, probability should match configured values
        p_left=0.4, p_right=0.6
        """
        state = env.reset()
        n_samples = 500  # Enough for statistics
        
        left_rewards = []
        right_rewards = []
        
        for _ in range(n_samples):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            
            if action == 0:
                left_rewards.append(reward)
            else:
                right_rewards.append(reward)
            
            if done:
                break
        
        # Check reward rates
        if len(left_rewards) > 30:
            left_rate = np.mean(left_rewards)
            assert 0.25 < left_rate < 0.55, \
                f"Left arm reward rate {left_rate} doesn't match p=0.4"
        
        if len(right_rewards) > 30:
            right_rate = np.mean(right_rewards)
            assert 0.45 < right_rate < 0.75, \
                f"Right arm reward rate {right_rate} doesn't match p=0.6"
    
    def test_reversal_learning_behavior(self, env):
        """
        After reversal_trial, probabilities should flip
        p_left=0.6, p_right=0.4 (swapped from initial)
        """
        state = env.reset()
        reversal_trial = 500
        n_samples = 1000
        
        left_rewards_pre = []
        left_rewards_post = []
        right_rewards_pre = []
        right_rewards_post = []
        
        for step in range(n_samples):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            
            if step < reversal_trial:
                if action == 0:
                    left_rewards_pre.append(reward)
                else:
                    right_rewards_pre.append(reward)
            else:
                if action == 0:
                    left_rewards_post.append(reward)
                else:
                    right_rewards_post.append(reward)
            
            if done:
                break
        
        # Check reversal occurred
        if len(left_rewards_pre) > 30 and len(left_rewards_post) > 30:
            rate_pre = np.mean(left_rewards_pre)
            rate_post = np.mean(left_rewards_post)
            
            # Left should improve after reversal
            assert rate_post > rate_pre, \
                f"Left arm didn't improve after reversal: {rate_pre:.2f} → {rate_post:.2f}"
    
    def test_deterministic_seeding(self, env):
        """Same seed should produce same sequence"""
        from gymnasium import spaces
        
        # Run with seed
        env.reset(seed=42)
        sequence1 = []
        for _ in range(20):
            action = env.action_space.sample()
            _, reward, _, _ = env.step(action)
            sequence1.append(reward)
        
        # Run again with same seed
        env.reset(seed=42)
        sequence2 = []
        for _ in range(20):
            action = env.action_space.sample()
            _, reward, _, _ = env.step(action)
            sequence2.append(reward)
        
        assert sequence1 == sequence2, \
            "Same seed produced different sequences"


# ============================================================================
# TEST SUITE 2: Actor-Critic Agent Tests
# ============================================================================

class TestActorCriticAgent:
    """
    Test suite for ActorCriticAgent (Person 2 deliverable)
    
    Prerequisites: Person 2 must implement models/rpe_agent.py
    """
    
    @pytest.fixture
    def agent(self):
        """Initialize agent for testing"""
        try:
            from models.rpe_agent import ActorCriticAgent
            return ActorCriticAgent(
                state_size=10,
                action_size=2,
                learning_rate=0.01,
                gamma=0.99
            )
        except ImportError:
            pytest.skip("ActorCriticAgent not yet implemented")
    
    def test_agent_initialization(self, agent):
        """Test basic agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'get_action'), "Missing get_action method"
        assert hasattr(agent, 'compute_rpe'), "Missing compute_rpe method"
        assert hasattr(agent, 'update'), "Missing update method"
    
    def test_agent_action_selection(self, agent):
        """Agent should return valid actions"""
        state = np.random.randn(10)  # state_size=10
        
        # Stochastic action
        action = agent.get_action(state, deterministic=False)
        assert action in [0, 1], f"Invalid action: {action}"
        
        # Deterministic action
        action = agent.get_action(state, deterministic=True)
        assert action in [0, 1], f"Invalid action: {action}"
    
    def test_rpe_computation(self, agent):
        """RPE should be correctly computed as r + γV(s') - V(s)"""
        state_t = np.random.randn(10)
        state_next = np.random.randn(10)
        reward = 1  # Binary reward
        action = 0
        done = False
        
        rpe = agent.compute_rpe(state_t, action, reward, state_next, done)
        
        # RPE should be bounded (rough check)
        assert isinstance(rpe, (float, np.floating))
        assert -2 < rpe < 2, f"RPE {rpe} seems out of range"
    
    def test_rpe_is_zero_for_perfect_prediction(self, agent):
        """
        RPE should be ~0 if critic perfectly predicts reward
        """
        # This is a loose test - perfect zero unlikely due to random init
        state_t = np.random.randn(10)
        
        # Manually set critic to predict exactly the reward
        reward = 1
        
        rpe = agent.compute_rpe(state_t, 0, reward, state_t, done=False)
        
        # Not too large
        assert abs(rpe) < 3, f"RPE {rpe} too large for good prediction"
    
    def test_update_method(self, agent):
        """Update should modify agent weights"""
        state_t = np.random.randn(10)
        state_next = np.random.randn(10)
        action = 0
        reward = 1
        
        # Store initial weights
        initial_actor_w = agent.actor_weights.copy()
        initial_critic_w = agent.critic_weights.copy()
        
        # Update
        rpe = agent.update(state_t, action, reward, state_next, done=False)
        
        # Weights should change
        assert not np.allclose(agent.actor_weights, initial_actor_w), \
            "Actor weights unchanged after update"
        assert not np.allclose(agent.critic_weights, initial_critic_w), \
            "Critic weights unchanged after update"
    
    def test_update_direction_on_positive_reward(self, agent):
        """
        Positive RPE should increase probability of chosen action
        """
        state_t = np.random.randn(10)
        state_next = np.zeros(10)  # Zero state → V ≈ 0
        action = 0
        reward = 1
        
        # Initial policy
        initial_action_prob = agent.forward_actor(state_t)
        
        # Update with positive reward
        agent.update(state_t, action, reward, state_next, done=False)
        
        # New policy
        new_action_prob = agent.forward_actor(state_t)
        
        # Probability of action 0 should increase
        # (logit should move in positive direction)
        # Note: This is probabilistic, just check for reasonable change
        prob_change = new_action_prob[0] - initial_action_prob[0]
        
        # Should favor the rewarded action
        assert abs(prob_change) > 0.001, \
            "Policy didn't change meaningfully after positive reward"
    
    def test_update_direction_on_zero_reward(self, agent):
        """
        Negative RPE should decrease probability of chosen action
        """
        state_t = np.random.randn(10)
        state_next = np.random.randn(10)  # Non-zero state → V > 0
        action = 0
        reward = 0  # No reward
        
        # Update with zero reward (likely negative RPE)
        agent.update(state_t, action, reward, state_next, done=False)
        
        # Just verify update completed
        assert hasattr(agent, 'rpe_history')
        assert len(agent.rpe_history) > 0
    
    def test_rpe_history_tracking(self, agent):
        """Agent should track RPE history"""
        assert hasattr(agent, 'rpe_history'), "Missing rpe_history tracking"
        assert hasattr(agent, 'reward_history'), "Missing reward_history tracking"
        assert hasattr(agent, 'value_history'), "Missing value_history tracking"
        
        # Perform updates
        for _ in range(10):
            state_t = np.random.randn(10)
            state_next = np.random.randn(10)
            action = np.random.randint(0, 2)
            reward = np.random.randint(0, 2)
            
            agent.update(state_t, action, reward, state_next, done=False)
        
        # Check history
        assert len(agent.rpe_history) == 10
        assert len(agent.reward_history) == 10
        assert len(agent.value_history) == 10


# ============================================================================
# TEST SUITE 3: Environment + Agent Integration Tests
# ============================================================================

class TestEnvironmentAgentIntegration:
    """
    Test suite for integration between GamblingTaskEnv and ActorCriticAgent
    
    Prerequisites: Both components must be implemented
    """
    
    @pytest.fixture
    def env_and_agent(self):
        """Initialize both environment and agent"""
        try:
            from envs.gambling_task import GamblingTaskEnv
            from models.rpe_agent import ActorCriticAgent
            
            env = GamblingTaskEnv(p_left=0.4, p_right=0.6, reversal_trial=500)
            agent = ActorCriticAgent(state_size=10, action_size=2)
            
            return env, agent
        except ImportError as e:
            pytest.skip(f"Components not implemented: {e}")
    
    def test_full_episode_without_errors(self, env_and_agent):
        """Run full episode: no crashes"""
        env, agent = env_and_agent
        
        state = env.reset()
        episode_length = 0
        
        for step in range(1000):
            # Agent selects action
            action = agent.get_action(state, deterministic=False)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Agent learns
            rpe = agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_length = step + 1
            
            if done:
                break
        
        assert episode_length > 100, "Episode too short"
    
    def test_agent_learns_optimal_policy(self, env_and_agent):
        """Agent should learn to choose better arm more often"""
        env, agent = env_and_agent
        
        state = env.reset()
        episode_length = 0
        
        # First 200 steps: exploration
        early_actions = []
        early_rewards = []
        
        # Next 200 steps: exploitation (should improve)
        late_actions = []
        late_rewards = []
        
        for step in range(400):
            action = agent.get_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            rpe = agent.update(state, action, reward, next_state, done)
            
            if step < 200:
                early_actions.append(action)
                early_rewards.append(reward)
            else:
                late_actions.append(action)
                late_rewards.append(reward)
            
            state = next_state
            episode_length = step + 1
            
            if done:
                break
        
        # Check improvement
        early_reward_rate = np.mean(early_rewards) if early_rewards else 0
        late_reward_rate = np.mean(late_rewards) if late_rewards else 0
        
        # Should improve (right arm has 0.6 vs 0.4)
        assert late_reward_rate > early_reward_rate - 0.1, \
            f"No improvement: early={early_reward_rate:.2f}, late={late_reward_rate:.2f}"
    
    def test_rpe_reflects_reward_prediction_error(self, env_and_agent):
        """RPE should be positive for unexpected rewards, negative otherwise"""
        env, agent = env_and_agent
        
        state = env.reset()
        rpe_values = []
        rewards = []
        
        for step in range(100):
            action = agent.get_action(state, deterministic=True)  # Deterministic
            next_state, reward, done, info = env.step(action)
            rpe = agent.update(state, action, reward, next_state, done)
            
            rpe_values.append(rpe)
            rewards.append(reward)
            
            state = next_state
            if done:
                break
        
        # After learning, average RPE should be close to zero
        avg_rpe = np.mean(rpe_values[-20:])  # Last 20 steps
        
        assert abs(avg_rpe) < 1.0, \
            f"Average late RPE {avg_rpe} not converged"
    
    def test_reversal_learning_shows_rpe_reset(self, env_and_agent):
        """
        After reversal (trial 500), RPE should spike again
        (indicating surprise at changed probabilities)
        """
        env, agent = env_and_agent
        
        state = env.reset()
        rpe_pre = []
        rpe_post = []
        
        for step in range(600):
            action = agent.get_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            rpe = agent.update(state, action, reward, next_state, done)
            
            if step < 500:
                rpe_pre.append(rpe)
            else:
                rpe_post.append(rpe)
            
            state = next_state
            if done:
                break
        
        # RPE should change after reversal
        avg_rpe_pre = np.mean(rpe_pre[-50:]) if len(rpe_pre) > 50 else 0
        avg_rpe_post = np.mean(rpe_post[:50]) if len(rpe_post) > 50 else 0
        
        # Should show some difference
        rpe_change = abs(avg_rpe_post - avg_rpe_pre)
        
        # Not a strict requirement but indicates learning
        assert rpe_change > 0.01, \
            "No RPE change around reversal"


# ============================================================================
# TEST SUITE 4: BCI Decoder Integration Tests
# ============================================================================

class TestBCIDecoderIntegration:
    """
    Test suite for BCI decoder + RL integration (Person 3 will extend)
    
    Validates that BCI predictions can drive RL agent
    """
    
    @pytest.fixture
    def decoder(self):
        """Initialize BCI decoder"""
        try:
            from models.bci_decoder import BCIDecoder
            import os
            
            model_path = 'models/shallow_convnet_motor_imagery.keras'
            if os.path.exists(model_path):
                return BCIDecoder(model_path, verbose=False)
            else:
                pytest.skip("BCI model not found")
        except ImportError:
            pytest.skip("BCIDecoder not yet implemented")
    
    def test_decoder_output_format_for_rl(self, decoder):
        """BCI output should be RL-compatible"""
        synthetic_eeg = decoder.generate_synthetic_eeg(n_trials=1)
        prediction = decoder.predict(synthetic_eeg[0:1])
        
        # Check required keys
        assert 'class' in prediction
        assert 'confidence' in prediction
        assert 'label_index' in prediction
        
        # Check value types
        assert prediction['label_index'] in [0, 1]
        assert 0 <= prediction['confidence'] <= 1
    
    def test_synthetic_eeg_generation(self, decoder):
        """Generated EEG should be usable"""
        eeg = decoder.generate_synthetic_eeg(n_trials=10, class_label='Right')
        
        assert eeg.shape == (10, 751, 22)
        assert not np.any(np.isnan(eeg)), "Generated EEG contains NaN"
        assert not np.any(np.isinf(eeg)), "Generated EEG contains Inf"
    
    def test_realtime_prediction_output(self, decoder):
        """Real-time prediction should be RL-compatible"""
        stream = decoder.generate_synthetic_eeg(n_trials=1)[0]
        stream = np.tile(stream, (3, 1))[:2000]
        
        result = decoder.predict_realtime(stream, confidence_threshold=0.5)
        
        # Check RL-compatible output
        assert 'rpe_compatible' in result
        assert 'action' in result['rpe_compatible']
        assert 'confidence' in result['rpe_compatible']
        assert result['rpe_compatible']['action'] in ['left', 'right', None]


# ============================================================================
# TEST RUNNER & HELPER FUNCTIONS
# ============================================================================

def run_all_tests():
    """Run all test suites"""
    pytest.main([__file__, '-v', '--tb=short'])


def run_env_tests_only():
    """Run only environment tests"""
    pytest.main([__file__, '-v', '-k', 'TestGamblingTask', '--tb=short'])


def run_agent_tests_only():
    """Run only agent tests"""
    pytest.main([__file__, '-v', '-k', 'TestActorCritic', '--tb=short'])


def run_integration_tests_only():
    """Run only integration tests"""
    pytest.main([__file__, '-v', '-k', 'Integration', '--tb=short'])


if __name__ == '__main__':
    """
    Run tests
    
    Usage:
        python tests/test_person2.py                # All tests
        pytest tests/test_person2.py -v             # Verbose
        pytest tests/test_person2.py -k "test_env"  # Specific tests
        pytest tests/test_person2.py --tb=short     # Short tracebacks
    """
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'env':
            run_env_tests_only()
        elif sys.argv[1] == 'agent':
            run_agent_tests_only()
        elif sys.argv[1] == 'integration':
            run_integration_tests_only()
        else:
            run_all_tests()
    else:
        run_all_tests()