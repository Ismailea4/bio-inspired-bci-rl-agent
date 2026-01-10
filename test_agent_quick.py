# test_agent_quick.py
print("🧠 DÉMARRAGE DU TEST DE L'AGENT...")

try:
    from rpe_agent import ActorCriticAgent, softmax
    import numpy as np
    
    print("✅ Import réussi")
    
    # Créer l'agent
    agent = ActorCriticAgent(state_size=10, action_size=2, learning_rate=0.01)
    print("✅ Agent créé")
    
    # Test softmax
    logits = np.array([0.5, -0.5])
    probs = softmax(logits)
    print(f"✅ Softmax test: {probs} (sum={np.sum(probs):.2f})")
    
    # Test action selection
    state = np.random.randn(10)
    action = agent.get_action(state, deterministic=False)
    print(f"✅ Action sélectionnée: {action}")
    
    # Test RPE computation
    next_state = np.random.randn(10)
    delta = agent.compute_rpe(state, action, reward=1.0, state_next=next_state, done=False)
    print(f"✅ RPE calculé: δ = {delta:.4f}")
    
    # Test update
    delta_updated = agent.update(state, action, reward=1.0, state_next=next_state, done=False)
    print(f"✅ Agent mis à jour, nouveau δ: {delta_updated:.4f}")
    
    print(f"\n📊 Historique RPE: {len(agent.rpe_history)} valeurs")
    print(f"📊 Dernière V(s): {agent.value_history[-1]:.4f}")
    print("\n🎉 AGENT TESTÉ AVEC SUCCÈS!")
    
except Exception as e:
    print(f"\n❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
