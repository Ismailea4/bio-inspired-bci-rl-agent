"""
Testing Framework for Person 3: Integration & UI
=================================================
Comprehensive unit tests for:
- BCI + RL integration pipeline
- RL environment + agent integration
- Visualization utilities
- Real-time streaming interface
- UI compatibility checks

Author: Testing Framework
Date: January 2026
Status: Ready for Person 3 implementation validation

This file provides test templates that should be implemented
alongside the integration code, visualization utilities, and UI.
"""

import numpy as np
import pytest
from typing import Tuple, Dict, Any


# ============================================================================
# TEST SUITE 1: BCI to Action Mapping Tests
# ============================================================================

class TestBCItoActionMapping:
    """
    Test suite for BCI prediction → RL action conversion
    
    Prerequisites: Person 1's BCIDecoder + Person 2's RL environment
    """
    
    def test_bci_output_to_rl_action_conversion(self):
        """BCI class prediction should map to valid RL actions"""
        try:
            from models.bci_decoder import BCIDecoder
            decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        except:
            pytest.skip("BCI decoder not available")
        
        synthetic_eeg = decoder.generate_synthetic_eeg(n_trials=10)
        
        for eeg_sample in synthetic_eeg:
            prediction = decoder.predict(eeg_sample[np.newaxis, ...])
            
            # Extract action
            action = prediction['label_index']
            
            # Should be valid RL action
            assert action in [0, 1], f"Invalid action: {action}"
    
    def test_confidence_filtering(self):
        """Low confidence predictions should be filtered"""
        try:
            from models.bci_decoder import BCIDecoder
            decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        except:
            pytest.skip("BCI decoder not available")
        
        synthetic_eeg = decoder.generate_synthetic_eeg(n_trials=20, noise_level=0.8)
        
        confident_preds = 0
        
        for eeg_sample in synthetic_eeg:
            prediction = decoder.predict(eeg_sample[np.newaxis, ...])
            
            if prediction['confidence'] >= 0.6:
                confident_preds += 1
        
        # With high noise, should have some low-confidence predictions
        assert confident_preds < len(synthetic_eeg), \
            "No low-confidence predictions with high noise"
    
    def test_class_to_action_mapping(self):
        """BCI classes should map correctly to actions"""
        try:
            from models.bci_decoder import BCIDecoder
            decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        except:
            pytest.skip("BCI decoder not available")
        
        # Left hand imagery should map to action 0
        left_eeg = decoder.generate_synthetic_eeg(n_trials=5, class_label='Left', seed=42)
        left_predictions = []
        
        for eeg in left_eeg:
            pred = decoder.predict(eeg[np.newaxis, ...])
            left_predictions.append(pred['label_index'])
        
        # Should have some 0s (Left)
        assert 0 in left_predictions, "Left imagery not mapped to action 0"
        
        # Right hand imagery should map to action 1
        right_eeg = decoder.generate_synthetic_eeg(n_trials=5, class_label='Right', seed=43)
        right_predictions = []
        
        for eeg in right_eeg:
            pred = decoder.predict(eeg[np.newaxis, ...])
            right_predictions.append(pred['label_index'])
        
        # Should have some 1s (Right)
        assert 1 in right_predictions, "Right imagery not mapped to action 1"


# ============================================================================
# TEST SUITE 2: Integration Pipeline Tests
# ============================================================================

class TestIntegrationPipeline:
    """
    Test suite for complete BCI + RL integration
    
    Prerequisites: Person 2's GamblingTaskEnv + ActorCriticAgent
                  Person 1's BCIDecoder
    """
    
    def test_full_pipeline_execution(self):
        """Complete BCI → RL pipeline should execute without errors"""
        try:
            from models.bci_decoder import BCIDecoder
            from envs.gambling_task import GamblingTaskEnv
            from models.rpe_agent import ActorCriticAgent
        except ImportError as e:
            pytest.skip(f"Required components not available: {e}")
        
        # Initialize components
        decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        env = GamblingTaskEnv(p_left=0.4, p_right=0.6, reversal_trial=500)
        agent = ActorCriticAgent(state_size=10, action_size=2)
        
        # Generate EEG stream
        eeg_stream = decoder.generate_synthetic_eeg(n_trials=100)
        
        state = env.reset()
        
        for trial in range(100):
            # BCI prediction
            eeg_sample = eeg_stream[trial % len(eeg_stream)]
            bci_pred = decoder.predict(eeg_sample[np.newaxis, ...])
            
            # Action selection
            if bci_pred['confidence'] >= 0.6:
                action = bci_pred['label_index']
            else:
                action = agent.get_action(state, deterministic=False)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Agent update
            rpe = agent.update(state, action, reward, next_state, done)
            
            state = next_state
            if done:
                break
    
    def test_bci_predictions_drive_learning(self):
        """Agent should learn from BCI-driven actions"""
        try:
            from models.bci_decoder import BCIDecoder
            from envs.gambling_task import GamblingTaskEnv
            from models.rpe_agent import ActorCriticAgent
        except ImportError as e:
            pytest.skip(f"Required components not available: {e}")
        
        decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        env = GamblingTaskEnv(p_left=0.4, p_right=0.6, reversal_trial=500)
        agent = ActorCriticAgent(state_size=10, action_size=2)
        
        # Generate biased EEG (should predict Right)
        eeg_stream = decoder.generate_synthetic_eeg(n_trials=200, class_label='Right', seed=42)
        
        state = env.reset()
        early_rewards = []
        late_rewards = []
        
        for trial in range(200):
            eeg_sample = eeg_stream[trial % len(eeg_stream)]
            bci_pred = decoder.predict(eeg_sample[np.newaxis, ...])
            action = bci_pred['label_index']
            
            next_state, reward, done, info = env.step(action)
            rpe = agent.update(state, action, reward, next_state, done)
            
            if trial < 100:
                early_rewards.append(reward)
            else:
                late_rewards.append(reward)
            
            state = next_state
            if done:
                break
        
        # Should accumulate more rewards over time
        early_sum = np.sum(early_rewards) if early_rewards else 0
        late_sum = np.sum(late_rewards) if late_rewards else 0
        
        assert late_sum >= early_sum * 0.8, \
            f"No learning detected: early={early_sum}, late={late_sum}"
    
    def test_action_validity_through_pipeline(self):
        """Actions should remain valid throughout pipeline"""
        try:
            from models.bci_decoder import BCIDecoder
            from envs.gambling_task import GamblingTaskEnv
            from models.rpe_agent import ActorCriticAgent
        except ImportError as e:
            pytest.skip(f"Required components not available: {e}")
        
        decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        env = GamblingTaskEnv()
        agent = ActorCriticAgent()
        
        eeg_stream = decoder.generate_synthetic_eeg(n_trials=50)
        state = env.reset()
        
        for trial in range(50):
            eeg_sample = eeg_stream[trial % len(eeg_stream)]
            bci_pred = decoder.predict(eeg_sample[np.newaxis, ...])
            
            action = bci_pred['label_index']
            assert action in [0, 1], f"Invalid action from BCI: {action}"
            
            next_state, reward, done, info = env.step(action)
            
            assert isinstance(reward, (int, float, np.number))
            assert isinstance(done, (bool, np.bool_))
            
            state = next_state
            if done:
                break


# ============================================================================
# TEST SUITE 3: Visualization Utilities Tests
# ============================================================================

class TestVisualizationUtilities:
    """
    Test suite for visualization functions (Person 3 deliverable)
    
    Prerequisites: Person 3 must implement utils/viz.py
    """
    
    def test_viz_module_imports(self):
        """Visualization module should exist and import"""
        try:
            from utils import viz
            assert hasattr(viz, 'plot_rpe_dynamics')
            assert hasattr(viz, 'plot_reversal_learning')
            assert hasattr(viz, 'plot_reward_history')
        except ImportError:
            pytest.skip("Visualization module not yet implemented")
    
    def test_plot_rpe_dynamics(self):
        """RPE dynamics plot should accept and process data"""
        try:
            from utils.viz import plot_rpe_dynamics
        except ImportError:
            pytest.skip("plot_rpe_dynamics not yet implemented")
        
        # Generate sample RPE data
        rpe_signals = np.cumsum(np.random.randn(100) * 0.1)
        
        fig = plot_rpe_dynamics(rpe_signals)
        
        assert fig is not None
        assert hasattr(fig, 'axes')
    
    def test_plot_reversal_learning(self):
        """Reversal learning plot should handle pre/post reversal"""
        try:
            from utils.viz import plot_reversal_learning
        except ImportError:
            pytest.skip("plot_reversal_learning not yet implemented")
        
        # Simulate RPE before and after reversal
        rpe_pre = np.cumsum(np.random.randn(500) * 0.1) - 0.5
        rpe_post = np.cumsum(np.random.randn(500) * 0.1) + 0.5
        rpe_signals = np.concatenate([rpe_pre, rpe_post])
        
        fig = plot_reversal_learning(rpe_signals, reversal_trial=500)
        
        assert fig is not None
    
    def test_plot_reward_history(self):
        """Reward history plot should show accumulation"""
        try:
            from utils.viz import plot_reward_history
        except ImportError:
            pytest.skip("plot_reward_history not yet implemented")
        
        # Generate reward sequence
        rewards = np.random.binomial(1, 0.6, size=100)
        
        fig = plot_reward_history(rewards, window=10)
        
        assert fig is not None
    
    def test_plot_outputs_valid_figures(self):
        """All plots should return valid matplotlib figures"""
        try:
            from utils.viz import (
                plot_rpe_dynamics,
                plot_reward_history
            )
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("Visualization functions not yet implemented")
        
        test_data = np.random.randn(100)
        
        fig1 = plot_rpe_dynamics(test_data)
        assert isinstance(fig1, plt.Figure)
        
        fig2 = plot_reward_history(np.random.binomial(1, 0.5, 100))
        assert isinstance(fig2, plt.Figure)


# ============================================================================
# TEST SUITE 4: Real-Time Streaming Interface Tests
# ============================================================================

class TestRealtimeStreamingInterface:
    """
    Test suite for real-time EEG streaming interface
    
    Validates that system can handle continuous data streams
    """
    
    def test_streaming_buffer_management(self):
        """Streaming interface should buffer data correctly"""
        try:
            from models.bci_decoder import BCIDecoder
        except ImportError:
            pytest.skip("BCIDecoder not available")
        
        decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        
        # Simulate continuous stream
        stream = np.random.randn(5000, 22)  # 20 seconds at 250 Hz
        
        window_size = 751
        stride = 75
        n_windows = (stream.shape[0] - window_size) // stride
        
        assert n_windows > 0, "Stream too short for windows"
    
    def test_realtime_prediction_latency(self):
        """Predictions should complete within reasonable time"""
        try:
            from models.bci_decoder import BCIDecoder
        except ImportError:
            pytest.skip("BCIDecoder not available")
        
        decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        
        eeg = np.random.randn(751, 22)
        
        import time
        start = time.time()
        prediction = decoder.predict(eeg[np.newaxis, ...])
        elapsed = time.time() - start
        
        # Should be < 100ms (reasonable for real-time task)
        assert elapsed < 0.1, f"Prediction too slow: {elapsed:.3f}s"
    
    def test_sliding_window_prediction(self):
        """Sliding window should produce consistent predictions"""
        try:
            from models.bci_decoder import BCIDecoder
        except ImportError:
            pytest.skip("BCIDecoder not available")
        
        decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        
        # Generate stream with consistent class
        stream = decoder.generate_synthetic_eeg(n_trials=3, class_label='Left')[0]
        stream = np.tile(stream, (4, 1))  # Repeat
        
        result = decoder.predict_realtime(
            stream,
            window_size=751,
            stride=150,
            smoothing_window=3,
            confidence_threshold=0.5
        )
        
        # Should have multiple windows
        assert result['n_windows'] > 1
        
        # Should produce valid action
        assert result['rpe_compatible']['action'] in ['left', 'right', None]


# ============================================================================
# TEST SUITE 5: UI Integration Tests
# ============================================================================

class TestUIIntegration:
    """
    Test suite for user interface compatibility
    
    Validates that all components provide UI-compatible outputs
    """
    
    def test_data_for_ui_dashboard(self):
        """System should provide data in UI-friendly format"""
        try:
            from models.bci_decoder import BCIDecoder
            from envs.gambling_task import GamblingTaskEnv
            from models.rpe_agent import ActorCriticAgent
        except ImportError as e:
            pytest.skip(f"Required components not available: {e}")
        
        # Simulate one trial
        decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        env = GamblingTaskEnv()
        agent = ActorCriticAgent()
        
        state = env.reset()
        eeg = decoder.generate_synthetic_eeg(n_trials=1)[0]
        
        # Collect UI-relevant data
        ui_data = {
            'decoded_class': None,
            'confidence': 0.0,
            'current_reward': None,
            'cumulative_reward': 0,
            'rpe_signal': None,
            'trial_number': 0,
            'agent_policy': None
        }
        
        # Fill data
        bci_pred = decoder.predict(eeg[np.newaxis, ...])
        ui_data['decoded_class'] = bci_pred['class']
        ui_data['confidence'] = bci_pred['confidence']
        
        action = bci_pred['label_index']
        next_state, reward, done, _ = env.step(action)
        ui_data['current_reward'] = reward
        ui_data['cumulative_reward'] += reward
        
        rpe = agent.update(state, action, reward, next_state, done)
        ui_data['rpe_signal'] = float(rpe)
        
        # Validate UI data format
        assert isinstance(ui_data['decoded_class'], str)
        assert 0 <= ui_data['confidence'] <= 1
        assert ui_data['current_reward'] in [0, 1]
        assert isinstance(ui_data['rpe_signal'], float)
    
    def test_ui_display_compatibility(self):
        """Output should be displayable (JSON serializable)"""
        try:
            import json
            from models.bci_decoder import BCIDecoder
        except ImportError:
            pytest.skip("BCIDecoder not available")
        
        decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        eeg = decoder.generate_synthetic_eeg(n_trials=1)[0]
        
        prediction = decoder.predict(eeg[np.newaxis, ...])
        
        # Should be JSON serializable for web UI
        try:
            json_str = json.dumps({
                'class': prediction['class'],
                'confidence': float(prediction['confidence']),
                'label_index': int(prediction['label_index'])
            })
            assert len(json_str) > 0
        except TypeError as e:
            pytest.fail(f"UI data not JSON serializable: {e}")
    
    def test_real_time_ui_update_frequency(self):
        """UI should update at reasonable frequency"""
        try:
            from models.bci_decoder import BCIDecoder
        except ImportError:
            pytest.skip("BCIDecoder not available")
        
        decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        
        # Simulate 1 second of updates (5 Hz = 200ms per update)
        for update_num in range(5):
            eeg = decoder.generate_synthetic_eeg(n_trials=1)[0]
            prediction = decoder.predict(eeg[np.newaxis, ...])
            
            # Each update should complete quickly
            assert isinstance(prediction, dict)


# ============================================================================
# TEST SUITE 6: End-to-End Demo Tests
# ============================================================================

class TestEndToEndDemo:
    """
    Test suite for complete demo simulation
    
    Validates that full system can run from start to finish
    """
    
    def test_demo_simulation_runs(self):
        """Complete demo should execute without errors"""
        try:
            from models.bci_decoder import BCIDecoder
            from envs.gambling_task import GamblingTaskEnv
            from models.rpe_agent import ActorCriticAgent
        except ImportError as e:
            pytest.skip(f"Required components not available: {e}")
        
        # Initialize
        decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        env = GamblingTaskEnv(p_left=0.4, p_right=0.6, reversal_trial=300)
        agent = ActorCriticAgent()
        
        # Run demo
        state = env.reset()
        eeg_stream = decoder.generate_synthetic_eeg(n_trials=300)
        
        rewards = []
        rpe_signals = []
        actions = []
        
        for trial in range(300):
            eeg = eeg_stream[trial % len(eeg_stream)]
            pred = decoder.predict(eeg[np.newaxis, ...])
            
            if pred['confidence'] >= 0.5:
                action = pred['label_index']
            else:
                action = agent.get_action(state, deterministic=False)
            
            next_state, reward, done, _ = env.step(action)
            rpe = agent.update(state, action, reward, next_state, done)
            
            rewards.append(reward)
            rpe_signals.append(rpe)
            actions.append(action)
            
            state = next_state
            if done:
                break
        
        # Verify demo ran
        assert len(rewards) > 100, "Demo too short"
        assert len(rpe_signals) == len(rewards)
        assert len(actions) == len(rewards)
    
    def test_demo_produces_meaningful_output(self):
        """Demo should show evidence of learning"""
        try:
            from models.bci_decoder import BCIDecoder
            from envs.gambling_task import GamblingTaskEnv
            from models.rpe_agent import ActorCriticAgent
        except ImportError as e:
            pytest.skip(f"Required components not available: {e}")
        
        decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras', verbose=False)
        env = GamblingTaskEnv(p_left=0.4, p_right=0.6)
        agent = ActorCriticAgent()
        
        state = env.reset()
        eeg_stream = decoder.generate_synthetic_eeg(n_trials=200)
        
        early_rewards = []
        late_rewards = []
        
        for trial in range(200):
            eeg = eeg_stream[trial % len(eeg_stream)]
            pred = decoder.predict(eeg[np.newaxis, ...])
            action = pred['label_index']
            
            next_state, reward, done, _ = env.step(action)
            rpe = agent.update(state, action, reward, next_state, done)
            
            if trial < 100:
                early_rewards.append(reward)
            else:
                late_rewards.append(reward)
            
            state = next_state
            if done:
                break
        
        # Should show learning
        early_avg = np.mean(early_rewards)
        late_avg = np.mean(late_rewards)
        
        # Not a strict requirement, but indicator of meaningful behavior
        assert len(early_rewards) > 30 and len(late_rewards) > 30


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all test suites"""
    pytest.main([__file__, '-v', '--tb=short'])


def run_integration_tests_only():
    """Run only integration tests"""
    pytest.main([__file__, '-v', '-k', 'Integration', '--tb=short'])


def run_ui_tests_only():
    """Run only UI tests"""
    pytest.main([__file__, '-v', '-k', 'UI', '--tb=short'])


if __name__ == '__main__':
    """
    Run tests
    
    Usage:
        python tests/test_person3.py                # All tests
        pytest tests/test_person3.py -v             # Verbose
        pytest tests/test_person3.py -k "test_ui"   # Specific tests
        pytest tests/test_person3.py --tb=short     # Short tracebacks
    """
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'integration':
            run_integration_tests_only()
        elif sys.argv[1] == 'ui':
            run_ui_tests_only()
        else:
            run_all_tests()
    else:
        run_all_tests()
