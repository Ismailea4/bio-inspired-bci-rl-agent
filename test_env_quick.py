# test_env_quick.py
print("🧪 DÉMARRAGE DU TEST...")

try:
    from gambling_task import GamblingTaskEnv
    print("✅ Import réussi")
    
    env = GamblingTaskEnv(seed=42)
    obs, _ = env.reset()
    print(f"✅ Reset réussi - État: {obs[:3]}...")
    
    obs, reward, done, _, info = env.step(0)
    print(f"✅ Step réussi - Reward: {reward}, Info: {info}")
    
    print("\n🎉 TOUS LES TESTS ONT RÉUSSI!")
    
except Exception as e:
    print(f"\n❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()