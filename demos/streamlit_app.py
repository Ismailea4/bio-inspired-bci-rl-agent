"""
Brain-Behavior Mapping: Enhanced Real-Time Streamlit Dashboard
==============================================================
Person 3 Deliverable: Interactive UI with XAI, interpretability & watch mode

Features:
- Real-time simulation with "Watch Mode" (1-2 min playback)
- XAI: EEG channel importance visualization
- Clear interpretations and explanations
- Enhanced UX with tooltips and guides

Author: Person 3 - Integration, UX & Explainability
Date: January 2026

Usage:
    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Import project modules
from envs.gambling_task import GamblingTaskEnv
from models.rpe_agent import ActorCriticAgent

# Import XAI utilities with SHAP/MNE/MOABB integration
try:
    from utils.xai import (
        SHAP_AVAILABLE, MNE_AVAILABLE, MOABB_AVAILABLE,
        compute_shap_values, plot_shap_summary,
        create_mne_info, plot_mne_topomap,
        get_dataset_info, load_sample_eeg_data,
        EEG_CHANNELS as XAI_CHANNELS
    )
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    SHAP_AVAILABLE = False
    MNE_AVAILABLE = False
    MOABB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🧠 Brain-Behavior Mapping",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UX
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 10px 0;
    }
    .explanation-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .metric-positive {
        color: #27ae60;
        font-size: 28px;
        font-weight: bold;
    }
    .metric-negative {
        color: #e74c3c;
        font-size: 28px;
        font-weight: bold;
    }
    .insight-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .phase-pre {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
    }
    .phase-post {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# EEG Channel info for XAI visualization
EEG_CHANNELS = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 
                'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
MOTOR_CHANNELS = ['C3', 'C1', 'Cz', 'C2', 'C4']  # Key motor cortex channels


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'simulation_running': False,
        'watch_mode': False,
        'env': None,
        'agent': None,
        'current_trial': 0,
        'results': {
            'rpe_history': [], 'reward_history': [], 'action_history': [],
            'policy_history': [], 'value_history': []
        },
        'last_intention': None,
        'last_rpe': 0,
        'last_state': None,
        'channel_importance': None,
        'simulation_speed': 0.05
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def generate_channel_importance(action: int, confidence: float) -> np.ndarray:
    """
    Generate simulated EEG channel importance based on motor imagery.
    
    In real BCI: This would use SHAP/Integrated Gradients on the actual model.
    Here we simulate realistic patterns based on neuroscience:
    - C3 activates for RIGHT hand imagery (contralateral)
    - C4 activates for LEFT hand imagery (contralateral)
    """
    importance = np.random.rand(22) * 0.3  # Base noise
    
    if action == 0:  # Left hand → Right motor cortex (C4, C2, C6)
        importance[11] += 0.7 * confidence  # C4
        importance[10] += 0.5 * confidence  # C2
        importance[12] += 0.4 * confidence  # C6
        importance[5] += 0.3 * confidence   # FC4
    else:  # Right hand → Left motor cortex (C3, C1, C5)
        importance[7] += 0.7 * confidence   # C3
        importance[8] += 0.5 * confidence   # C1
        importance[6] += 0.4 * confidence   # C5
        importance[1] += 0.3 * confidence   # FC3
    
    # Normalize
    importance = importance / np.max(importance)
    return importance


def create_brain_topography(importance: np.ndarray, action: int) -> go.Figure:
    """Create a realistic brain topography with anatomical regions."""
    # EEG channel positions (10-20 system, normalized coordinates)
    positions = {
        'Fz': (0.5, 0.85), 'FC3': (0.28, 0.72), 'FC1': (0.38, 0.72),
        'FCz': (0.5, 0.72), 'FC2': (0.62, 0.72), 'FC4': (0.72, 0.72),
        'C5': (0.12, 0.5), 'C3': (0.28, 0.5), 'C1': (0.38, 0.5),
        'Cz': (0.5, 0.5), 'C2': (0.62, 0.5), 'C4': (0.72, 0.5), 'C6': (0.88, 0.5),
        'CP3': (0.28, 0.35), 'CP1': (0.38, 0.35), 'CPz': (0.5, 0.35),
        'CP2': (0.62, 0.35), 'CP4': (0.72, 0.35),
        'P1': (0.38, 0.22), 'Pz': (0.5, 0.22), 'P2': (0.62, 0.22), 'POz': (0.5, 0.12)
    }
    
    # Brain region definitions for legend
    regions = {
        'Frontal (F)': {'color': '#3498db', 'channels': ['Fz']},
        'Frontal-Central (FC)': {'color': '#9b59b6', 'channels': ['FC3', 'FC1', 'FCz', 'FC2', 'FC4']},
        'Motor Cortex (C)': {'color': '#e74c3c', 'channels': ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6']},
        'Centro-Parietal (CP)': {'color': '#f39c12', 'channels': ['CP3', 'CP1', 'CPz', 'CP2', 'CP4']},
        'Parietal (P)': {'color': '#27ae60', 'channels': ['P1', 'Pz', 'P2', 'POz']}
    }
    
    fig = go.Figure()
    
    # Draw brain outline (more realistic shape)
    # Main head ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    head_x = 0.5 + 0.42 * np.cos(theta)
    head_y = 0.5 + 0.45 * np.sin(theta)
    fig.add_trace(go.Scatter(x=head_x, y=head_y, mode='lines',
                            line=dict(color='#2c3e50', width=3),
                            fill='toself', fillcolor='rgba(236, 240, 241, 0.3)',
                            name='Head', showlegend=False))
    
    # Nose indicator
    fig.add_trace(go.Scatter(x=[0.44, 0.5, 0.56], y=[0.94, 1.0, 0.94],
                            mode='lines', line=dict(color='#2c3e50', width=2),
                            showlegend=False))
    
    # Left ear
    ear_y = np.linspace(0.4, 0.6, 20)
    ear_x_left = 0.08 - 0.02 * np.sin((ear_y - 0.5) * np.pi / 0.1)
    fig.add_trace(go.Scatter(x=ear_x_left, y=ear_y, mode='lines',
                            line=dict(color='#2c3e50', width=2), showlegend=False))
    
    # Right ear
    ear_x_right = 0.92 + 0.02 * np.sin((ear_y - 0.5) * np.pi / 0.1)
    fig.add_trace(go.Scatter(x=ear_x_right, y=ear_y, mode='lines',
                            line=dict(color='#2c3e50', width=2), showlegend=False))
    
    # Draw brain regions as shaded areas
    # Left motor cortex region (for right hand control)
    left_motor_x = [0.12, 0.18, 0.28, 0.38, 0.38, 0.28, 0.18, 0.12]
    left_motor_y = [0.5, 0.62, 0.62, 0.55, 0.45, 0.38, 0.38, 0.5]
    left_color = 'rgba(231, 76, 60, 0.3)' if action == 1 else 'rgba(200, 200, 200, 0.15)'
    fig.add_trace(go.Scatter(x=left_motor_x, y=left_motor_y, mode='lines',
                            fill='toself', fillcolor=left_color,
                            line=dict(color='rgba(0,0,0,0)'),
                            name='Left Motor (Right Hand)', showlegend=True))
    
    # Right motor cortex region (for left hand control)
    right_motor_x = [0.62, 0.72, 0.82, 0.88, 0.88, 0.82, 0.72, 0.62]
    right_motor_y = [0.55, 0.62, 0.62, 0.5, 0.5, 0.38, 0.38, 0.45]
    right_color = 'rgba(231, 76, 60, 0.3)' if action == 0 else 'rgba(200, 200, 200, 0.15)'
    fig.add_trace(go.Scatter(x=right_motor_x, y=right_motor_y, mode='lines',
                            fill='toself', fillcolor=right_color,
                            line=dict(color='rgba(0,0,0,0)'),
                            name='Right Motor (Left Hand)', showlegend=True))
    
    # Plot EEG channels with importance-based colors
    x = [positions[ch][0] for ch in EEG_CHANNELS]
    y = [positions[ch][1] for ch in EEG_CHANNELS]
    sizes = 18 + importance * 28
    
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers+text',
        marker=dict(size=sizes, color=importance, colorscale='YlOrRd',
                   showscale=True, line=dict(width=1, color='#2c3e50'),
                   colorbar=dict(title='Channel<br>Importance', len=0.6, x=1.02)),
        text=EEG_CHANNELS,
        textposition='middle center',
        textfont=dict(size=7, color='#2c3e50', family='Arial Black'),
        hovertemplate='<b>%{text}</b><br>Importance: %{marker.color:.2%}<extra></extra>',
        name='EEG Channels', showlegend=False
    ))
    
    # Add anatomical labels
    fig.add_annotation(x=0.5, y=0.95, text="FRONT", showarrow=False,
                      font=dict(size=10, color='#7f8c8d'))
    fig.add_annotation(x=0.5, y=0.02, text="BACK", showarrow=False,
                      font=dict(size=10, color='#7f8c8d'))
    fig.add_annotation(x=0.02, y=0.5, text="L", showarrow=False,
                      font=dict(size=12, color='#7f8c8d', family='Arial Black'))
    fig.add_annotation(x=0.98, y=0.5, text="R", showarrow=False,
                      font=dict(size=12, color='#7f8c8d', family='Arial Black'))
    
    # Active hemisphere indicator
    if action == 0:  # Left hand → Right motor cortex
        active_text = "Active: Right Motor Cortex (controls Left hand)"
    else:  # Right hand → Left motor cortex
        active_text = "Active: Left Motor Cortex (controls Right hand)"
    
    fig.update_layout(
        title=dict(text=f"🧠 Brain Topography - Motor Imagery<br><sub>{active_text}</sub>",
                  x=0.5, font=dict(size=13)),
        xaxis=dict(visible=False, range=[-0.05, 1.15]),
        yaxis=dict(visible=False, range=[-0.02, 1.08], scaleanchor='x'),
        height=350,
        margin=dict(l=5, r=80, t=65, b=5),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5,
                   font=dict(size=9))
    )
    
    return fig


def create_rpe_gauge(rpe: float) -> go.Figure:
    """Create a gauge showing current RPE value."""
    color = '#27ae60' if rpe > 0 else '#e74c3c'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rpe,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Dopamine Signal (RPE)", 'font': {'size': 14}},
        delta={'reference': 0, 'increasing': {'color': '#27ae60'}, 'decreasing': {'color': '#e74c3c'}},
        gauge={
            'axis': {'range': [-2, 2], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': 'white',
            'borderwidth': 2,
            'steps': [
                {'range': [-2, 0], 'color': '#ffebee'},
                {'range': [0, 2], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 2},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))
    
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_policy_pie(policy: list) -> go.Figure:
    """Create a pie chart showing current policy."""
    fig = go.Figure(data=[go.Pie(
        labels=['← Left', 'Right →'],
        values=policy,
        hole=0.4,
        marker_colors=['#e74c3c', '#3498db'],
        textinfo='percent',
        textfont_size=14
    )])
    
    fig.update_layout(
        title=dict(text="Current Policy", x=0.5, font=dict(size=14)),
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5)
    )
    
    return fig


def create_live_rpe_chart(rpe_history: list, reversal_trial: int, max_display: int = 100) -> go.Figure:
    """Create live updating RPE chart."""
    if len(rpe_history) == 0:
        fig = go.Figure()
        fig.update_layout(height=250, title="RPE History (waiting for data...)")
        return fig
    
    # Show last N trials for better visualization
    display_rpe = rpe_history[-max_display:] if len(rpe_history) > max_display else rpe_history
    start_trial = max(0, len(rpe_history) - max_display)
    trials = list(range(start_trial, start_trial + len(display_rpe)))
    
    colors = ['#27ae60' if r > 0 else '#e74c3c' for r in display_rpe]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=trials, y=display_rpe, marker_color=colors, name='RPE'))
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    # Mark reversal if visible
    if reversal_trial >= start_trial and reversal_trial < start_trial + len(display_rpe):
        fig.add_vline(x=reversal_trial, line_dash="dash", line_color="purple", 
                     line_width=2, annotation_text="🔄 Reversal")
    
    fig.update_layout(
        title="⚡ Dopamine Signal (RPE) - Live",
        xaxis_title="Trial",
        yaxis_title="RPE",
        height=250,
        margin=dict(l=50, r=20, t=50, b=40)
    )
    
    return fig


def create_reward_trend(reward_history: list, window: int = 20) -> go.Figure:
    """Create reward rate trend chart."""
    if len(reward_history) < window:
        fig = go.Figure()
        fig.update_layout(height=200, title="Reward Trend (collecting data...)")
        return fig
    
    # Calculate rolling average
    rolling = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    x = list(range(window-1, len(reward_history)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=rolling, mode='lines', name='Success Rate',
                            line=dict(color='#f39c12', width=2),
                            fill='tozeroy', fillcolor='rgba(243, 156, 18, 0.2)'))
    
    fig.add_hline(y=0.6, line_dash="dash", line_color="#27ae60", 
                 annotation_text="Optimal (60%)")
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                 annotation_text="Chance (50%)")
    
    fig.update_layout(
        title="📈 Success Rate Trend",
        xaxis_title="Trial",
        yaxis_title="Rate",
        yaxis_range=[0.3, 0.8],
        height=200,
        margin=dict(l=50, r=20, t=50, b=40)
    )
    
    return fig


def run_single_trial():
    """Execute a single trial of the simulation."""
    if st.session_state.env is None or st.session_state.agent is None:
        return None
    
    # Get current state
    if st.session_state.current_trial == 0:
        state = st.session_state.env.reset()
    else:
        state = st.session_state.last_state
    
    # Get action from agent
    action = st.session_state.agent.get_action(state, deterministic=False)
    
    # Get confidence from policy
    policy = st.session_state.agent.policy_history[-1] if st.session_state.agent.policy_history else [0.5, 0.5]
    confidence = policy[action]
    
    # Execute action
    next_state, reward, done, info = st.session_state.env.step(action)
    
    # Update agent
    rpe = st.session_state.agent.update(state, action, reward, next_state, done)
    
    # Generate channel importance (simulated XAI)
    st.session_state.channel_importance = generate_channel_importance(action, confidence)
    
    # Store results
    st.session_state.results['rpe_history'].append(rpe)
    st.session_state.results['reward_history'].append(reward)
    st.session_state.results['action_history'].append(action)
    st.session_state.results['policy_history'].append(
        st.session_state.agent.policy_history[-1].copy()
    )
    st.session_state.results['value_history'].append(
        st.session_state.agent.value_history[-1] if st.session_state.agent.value_history else 0
    )
    
    # Update state
    st.session_state.last_state = next_state
    st.session_state.last_intention = 'Left' if action == 0 else 'Right'
    st.session_state.last_rpe = rpe
    st.session_state.current_trial += 1
    
    return {
        'trial': st.session_state.current_trial,
        'action': action,
        'intention': st.session_state.last_intention,
        'reward': reward,
        'rpe': rpe,
        'confidence': confidence,
        'reversal': info.get('reversal_occurred', False)
    }


def reset_simulation(n_trials, reversal_trial, p_left, p_right, lr, gamma, seed):
    """Reset the simulation with new parameters."""
    st.session_state.env = GamblingTaskEnv(
        p_left=p_left, p_right=p_right,
        reversal_trial=reversal_trial, seed=seed
    )
    
    st.session_state.agent = ActorCriticAgent(
        state_size=10, action_size=2,
        learning_rate=lr, gamma=gamma, seed=seed
    )
    
    st.session_state.current_trial = 0
    st.session_state.results = {
        'rpe_history': [], 'reward_history': [], 'action_history': [],
        'policy_history': [], 'value_history': []
    }
    st.session_state.last_intention = None
    st.session_state.last_rpe = 0
    st.session_state.last_state = st.session_state.env.reset()
    st.session_state.channel_importance = np.random.rand(22) * 0.3
    st.session_state.watch_mode = False


def get_interpretation(trial: int, rpe: float, reward_rate: float, reversal_trial: int, policy: list) -> str:
    """Generate human-readable interpretation of current state."""
    interpretations = []
    
    # Trial phase
    if trial < reversal_trial:
        phase = "🔵 Pre-Reversal Phase: Right option is better (60% reward)"
    elif trial == reversal_trial:
        phase = "🔄 REVERSAL JUST HAPPENED! Probabilities flipped - now Left is better!"
    elif trial < reversal_trial + 50:
        phase = "🟡 Adaptation Phase: Agent is learning the new reward structure"
    else:
        phase = "🟣 Post-Reversal Phase: Left option is now better (60% reward)"
    interpretations.append(phase)
    
    # RPE interpretation
    if abs(rpe) < 0.1:
        rpe_text = "RPE ≈ 0: Reward matched expectations (stable learning)"
    elif rpe > 0.5:
        rpe_text = f"RPE = {rpe:+.2f}: 🎉 Better than expected! Dopamine surge → reinforce this action"
    elif rpe < -0.5:
        rpe_text = f"RPE = {rpe:+.2f}: 😞 Worse than expected! Dopamine dip → avoid this action"
    elif rpe > 0:
        rpe_text = f"RPE = {rpe:+.2f}: Slightly better than expected"
    else:
        rpe_text = f"RPE = {rpe:+.2f}: Slightly worse than expected"
    interpretations.append(rpe_text)
    
    # Policy interpretation
    if policy[0] > 0.7:
        policy_text = "Policy: Strong preference for LEFT (>70%)"
    elif policy[1] > 0.7:
        policy_text = "Policy: Strong preference for RIGHT (>70%)"
    else:
        policy_text = "Policy: Still exploring (no strong preference yet)"
    interpretations.append(policy_text)
    
    # Performance
    if reward_rate > 0.55:
        perf = f"✅ Performance: {reward_rate:.1%} - Above chance, learning effectively!"
    elif reward_rate > 0.45:
        perf = f"⚖️ Performance: {reward_rate:.1%} - Around chance level"
    else:
        perf = f"⚠️ Performance: {reward_rate:.1%} - Below chance, still adapting"
    interpretations.append(perf)
    
    return "\n\n".join(interpretations)


def main():
    """Main Streamlit app."""
    init_session_state()
    
    # ===== HEADER =====
    st.markdown('<p class="main-header">🧠 Brain–Behavior Mapping Dashboard</p>', unsafe_allow_html=True)
    st.markdown("<center><i>Decode Neural Intent • Simulate Dopamine Learning • Watch the Brain Learn</i></center>", unsafe_allow_html=True)
    st.markdown("---")
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.header("⚙️ Simulation Settings")
        
        # Quick explanation
        with st.expander("ℹ️ What is this?", expanded=False):
            st.markdown("""
            This dashboard simulates how the brain learns through dopamine signals.
            
            Key concepts:
            - RPE (Reward Prediction Error): The "surprise" signal
              - Positive RPE = better than expected → 🎉
              - Negative RPE = worse than expected → 😞
            - Policy: The brain's decision strategy
            - Reversal: Reward rules flip mid-experiment
            
            Watch how the agent adapts when rewards change!
            """)
        
        st.subheader("📊 Experiment Setup")
        n_trials = st.slider("Total Trials", 200, 1000, 500, 50,
                            help="Number of decisions the agent will make")
        reversal_trial = st.slider("Reversal Point", 100, n_trials-50, n_trials//2, 25,
                                  help="When reward probabilities flip")
        
        col1, col2 = st.columns(2)
        with col1:
            p_right = st.slider("P(Right) wins", 0.5, 0.8, 0.6, 0.05,
                               help="Pre-reversal probability")
        with col2:
            # P(Left) is the complement: if Right wins 60%, Left wins 40%
            p_left = 1.0 - p_right
            st.metric("P(Left) wins", f"{p_left:.0%}")
        
        st.subheader("🧠 Agent Parameters")
        lr = st.select_slider("Learning Rate", 
                             options=[0.005, 0.01, 0.02, 0.05, 0.1],
                             value=0.02,
                             help="How fast the agent learns (higher = faster but less stable)")
        gamma = st.slider("Discount Factor (γ)", 0.9, 0.99, 0.95, 0.01,
                         help="How much future rewards matter")
        seed = st.number_input("Random Seed", 0, 999, 42)
        
        st.markdown("---")
        
        # ===== CONTROL BUTTONS =====
        st.subheader("🎮 Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", use_container_width=True):
                reset_simulation(n_trials, reversal_trial, p_left, p_right, lr, gamma, seed)
                st.rerun()
        with col2:
            if st.button("▶️ Step", use_container_width=True, disabled=st.session_state.current_trial >= n_trials):
                run_single_trial()
                st.rerun()
        
        # Watch Mode Controls
        st.markdown("##### 🎬 Watch Mode")
        watch_speed = st.select_slider(
            "Playback Speed",
            options=["Slow (2 min)", "Medium (1 min)", "Fast (30s)"],
            value="Medium (1 min)",
            help="How fast to run the simulation"
        )
        speed_map = {"Slow (2 min)": 0.12, "Medium (1 min)": 0.06, "Fast (30s)": 0.03}
        st.session_state.simulation_speed = speed_map[watch_speed]
        
        col1, col2 = st.columns(2)
        with col1:
            watch_btn = st.button("▶️ Watch", use_container_width=True, 
                                 disabled=st.session_state.current_trial >= n_trials)
        with col2:
            stop_btn = st.button("⏹️ Stop", use_container_width=True)
        
        if stop_btn:
            st.session_state.watch_mode = False
        
        # Run 50 trials button
        if st.button("⏩ Run 50 Trials", use_container_width=True, 
                    disabled=st.session_state.current_trial >= n_trials):
            for _ in range(min(50, n_trials - st.session_state.current_trial)):
                run_single_trial()
            st.rerun()
    
    # Initialize if needed
    if st.session_state.env is None:
        reset_simulation(n_trials, reversal_trial, p_left, p_right, lr, gamma, seed)
    
    # ===== MAIN CONTENT =====
    
    # Row 1: Current Status
    st.subheader("📍 Current State")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        progress = st.session_state.current_trial / n_trials
        st.metric("Progress", f"{st.session_state.current_trial}/{n_trials}")
        st.progress(progress)
    
    with col2:
        intention = st.session_state.last_intention or "—"
        intention_emoji = "👈" if intention == "Left" else ("👉" if intention == "Right" else "❓")
        st.metric("Decoded Intention", f"{intention_emoji} {intention}")
    
    with col3:
        rpe = st.session_state.last_rpe
        rpe_emoji = "🟢" if rpe > 0 else ("🔴" if rpe < 0 else "⚪")
        st.metric("Last RPE", f"{rpe_emoji} {rpe:+.3f}")
    
    with col4:
        total_reward = sum(st.session_state.results['reward_history'])
        trials_done = st.session_state.current_trial
        rate = total_reward / trials_done if trials_done > 0 else 0
        rate_emoji = "✅" if rate > 0.55 else ("⚠️" if rate > 0.45 else "❌")
        st.metric("Success Rate", f"{rate_emoji} {rate:.1%}")
    
    st.markdown("---")
    
    # Row 2: Brain Visualization + Policy
    col1, col2, col3 = st.columns([1.2, 0.8, 1])
    
    with col1:
        st.markdown("##### 🧠 Brain Activity (EEG Channels)")
        if st.session_state.channel_importance is not None:
            action = 0 if st.session_state.last_intention == "Left" else 1
            fig_brain = create_brain_topography(st.session_state.channel_importance, action)
            st.plotly_chart(fig_brain, use_container_width=True)
            
            # XAI Explanation
            with st.expander("💡 What am I seeing?"):
                st.markdown("""
                This shows which brain regions are most active during motor imagery:
                
                - Hot/red colors = High activation
                - C3 (left motor cortex) → Controls right hand
                - C4 (right motor cortex) → Controls left hand
                - Highlighted region shows the active motor area
                
                This contralateral (opposite-side) pattern is how real brains work!
                """)
    
    with col2:
        st.markdown("##### 🎯 Current Policy")
        if st.session_state.results['policy_history']:
            policy = st.session_state.results['policy_history'][-1]
            fig_policy = create_policy_pie(policy)
            st.plotly_chart(fig_policy, use_container_width=True)
        else:
            st.info("Run simulation to see policy")
    
    with col3:
        st.markdown("##### ⚡ Dopamine Gauge")
        fig_gauge = create_rpe_gauge(st.session_state.last_rpe)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Quick interpretation
        if st.session_state.last_rpe > 0.3:
            st.success("🎉 Positive surprise! Reinforcing this behavior.")
        elif st.session_state.last_rpe < -0.3:
            st.error("😞 Negative surprise! Learning to avoid this.")
        else:
            st.info("⚖️ As expected. Stable state.")
    
    st.markdown("---")
    
    # Row 3: Live Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rpe = create_live_rpe_chart(
            st.session_state.results['rpe_history'], 
            reversal_trial
        )
        st.plotly_chart(fig_rpe, use_container_width=True)
    
    with col2:
        fig_reward = create_reward_trend(st.session_state.results['reward_history'])
        st.plotly_chart(fig_reward, use_container_width=True)
    
    st.markdown("---")
    
    # Row 4: Interpretation Panel
    st.subheader("🔍 What's Happening?")
    
    if st.session_state.results['rpe_history']:
        policy = st.session_state.results['policy_history'][-1] if st.session_state.results['policy_history'] else [0.5, 0.5]
        reward_rate = np.mean(st.session_state.results['reward_history'][-50:]) if len(st.session_state.results['reward_history']) > 10 else 0.5
        
        interpretation = get_interpretation(
            st.session_state.current_trial,
            st.session_state.last_rpe,
            reward_rate,
            reversal_trial,
            policy
        )
        
        st.markdown(f"""
        <div class="explanation-box">
        {interpretation}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 Click Step or Watch to start the simulation and see interpretations!")
    
    # Row 5: Statistics Summary
    if st.session_state.current_trial > 0:
        st.markdown("---")
        st.subheader("📊 Session Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("📈 Overall")
            st.write(f"• Trials completed: {st.session_state.current_trial}")
            st.write(f"• Total rewards: {sum(st.session_state.results['reward_history'])}")
            st.write(f"• Mean RPE: {np.mean(st.session_state.results['rpe_history']):+.4f}")
        
        with col2:
            if st.session_state.current_trial > reversal_trial:
                st.markdown("🔵 Pre-Reversal")
                pre_rewards = st.session_state.results['reward_history'][:reversal_trial]
                pre_actions = st.session_state.results['action_history'][:reversal_trial]
                st.write(f"• Success rate: {np.mean(pre_rewards):.1%}")
                st.write(f"• Chose Right: {np.mean(pre_actions):.1%}")
                st.write(f"• Learned optimal? {'✅ Yes' if np.mean(pre_actions) > 0.55 else '❌ No'}")
        
        with col3:
            if st.session_state.current_trial > reversal_trial + 10:
                st.markdown("🟣 Post-Reversal")
                post_rewards = st.session_state.results['reward_history'][reversal_trial:]
                post_actions = st.session_state.results['action_history'][reversal_trial:]
                st.write(f"• Success rate: {np.mean(post_rewards):.1%}")
                st.write(f"• Chose Left: {1-np.mean(post_actions):.1%}")
                st.write(f"• Adapted? {'✅ Yes' if np.mean(post_actions) < 0.45 else '⏳ Learning...'}")
        
        # Row 6: Advanced XAI Tools (SHAP/MNE/MOABB)
        st.markdown("---")
        st.subheader("🔬 Advanced XAI Tools")
        
        xai_col1, xai_col2, xai_col3 = st.columns(3)
        
        with xai_col1:
            st.markdown("##### 📊 SHAP Analysis")
            if XAI_AVAILABLE and SHAP_AVAILABLE:
                st.success("✅ SHAP Available")
                st.caption("SHAP (SHapley Additive exPlanations) provides feature importance explanations for BCI model predictions.")
                with st.expander("Learn about SHAP"):
                    st.markdown("""
                    **SHAP** explains individual predictions by computing the contribution of each feature:
                    - Uses game-theoretic Shapley values
                    - Shows which EEG channels most influenced the prediction
                    - Red = increased prediction, Blue = decreased prediction
                    """)
            else:
                st.warning("⚠️ SHAP not available")
                st.caption("Install with: `pip install shap`")
        
        with xai_col2:
            st.markdown("##### 🧠 MNE Topomaps")
            if XAI_AVAILABLE and MNE_AVAILABLE:
                st.success("✅ MNE Available")
                st.caption("MNE provides neuroscience-standard EEG visualization with proper electrode positioning.")
                with st.expander("Learn about MNE"):
                    st.markdown("""
                    **MNE-Python** is the gold standard for EEG analysis:
                    - Standard 10-20 electrode positions
                    - Professional topographic maps
                    - Integrated with neuroscience pipelines
                    """)
            else:
                st.warning("⚠️ MNE not available")
                st.caption("Install with: `pip install mne`")
        
        with xai_col3:
            st.markdown("##### 📡 MOABB Datasets")
            if XAI_AVAILABLE and MOABB_AVAILABLE:
                st.success("✅ MOABB Available")
                st.caption("MOABB provides standardized access to BCI benchmark datasets like BNCI2014-001.")
                with st.expander("Learn about MOABB"):
                    st.markdown("""
                    **MOABB** (Mother of All BCI Benchmarks):
                    - Access to 50+ EEG datasets
                    - Standardized data format
                    - BNCI2014-001: 4-class motor imagery
                    - 9 subjects, 22 channels, 250Hz
                    """)
                # Show dataset info
                if st.button("📋 Show Dataset Info"):
                    dataset_info = get_dataset_info()
                    st.json(dataset_info)
            else:
                st.warning("⚠️ MOABB not available")
                st.caption("Install with: `pip install moabb`")
    
    # ===== WATCH MODE EXECUTION =====
    if watch_btn and not st.session_state.watch_mode:
        st.session_state.watch_mode = True
    
    # Watch mode placeholder for real-time updates
    watch_placeholder = st.empty()
    
    if st.session_state.watch_mode and st.session_state.current_trial < n_trials:
        with watch_placeholder.container():
            st.info(f"🎬 Watch Mode Active - Trial {st.session_state.current_trial}/{n_trials}")
            progress_bar = st.progress(st.session_state.current_trial / n_trials)
        
        # Run trials with delay
        while st.session_state.watch_mode and st.session_state.current_trial < n_trials:
            run_single_trial()
            time.sleep(st.session_state.simulation_speed)
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <center>
    <small>
    🧠 <b>Brain–Behavior Mapping</b> | Track 3 – Cognitive & Behavioral Analyses<br>
    Built with ❤️ by Person 3 | Integration, UX & Explainability
    </small>
    </center>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
