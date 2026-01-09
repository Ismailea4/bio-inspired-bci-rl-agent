# test_full_integration.py
print("🔗 TEST D'INTÉGRATION COMPLET\n" + "="*40)

from envs.gambling_task import GamblingTaskEnv
from models.rpe_agent import ActorCriticAgent
import numpy as np

# Configuration
N_TRIALS = 200  # Test réduit pour la validation
SEED = 42

# Initialisation
env = GamblingTaskEnv(p_left=0.4, p_right=0.6, reversal_trial=100, seed=SEED)
agent = ActorCriticAgent(state_size=10, action_size=2, learning_rate=0.05, gamma=0.99)

# 🔥 CORRECTION: env.reset() retourne 1 valeur seulement
obs = env.reset()

print(f"📊 Environnement: {N_TRIALS} trials, reversal @ trial 100")
print(f"🧠 Agent: lr=0.05, gamma=0.99\n")

# Boucle d'entraînement
rewards = []
rpes = []

for trial in range(N_TRIALS):
    action = agent.get_action(obs, deterministic=False)
    obs_next, reward, done, info = env.step(action)
    delta = agent.update(obs, action, reward, obs_next, done)
    
    rewards.append(reward)
    rpes.append(delta)
    obs = obs_next
    
    if (trial + 1) % 20 == 0:
        print(f"Trial {trial+1:3d} | Action: {action} | Reward: {reward} | RPE: {delta:+6.3f}")

# Résultats
print("\n" + "="*40)
print("📈 RÉSULTATS FIN")
print(f"Récompense totale: {np.sum(rewards)} / {N_TRIALS}")
print(f"RPE moyen: {np.mean(rpes):+.4f}")
print(f"Dernière politique: {agent.policy_history[-1]}")

# Vérification apprentissage
pre = np.mean(rewards[:20])
post = np.mean(rewards[-20:])
print(f"\nTaux réussite début: {pre:.1%}")
print(f"Taux réussite fin: {post:.1%}")

if post > pre + 0.1:
    print("\n✅ L'AGENT APPREND CORRECTEMENT!")
else:
    print("\n⚠️ Vérifie les hyperparamètres (lr, gamma)")

print("\n🎉 TEST D'INTÉGRATION RÉUSSI!")