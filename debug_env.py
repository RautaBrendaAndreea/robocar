"""Debug script to inspect the simulator's behavior specs."""
from src.client.environment import create_environment, get_behavior, get_observations

env = create_environment()
behavior_name, spec = get_behavior(env)

print(f"Behavior name: {behavior_name}")
print(f"Observation shapes: {[obs.shape for obs in spec.observation_specs]}")
print(f"Action spec: {spec.action_spec}")
print(f"  Continuous size: {spec.action_spec.continuous_size}")
print(f"  Discrete branches: {spec.action_spec.discrete_branches}")

decision_steps, terminal_steps = get_observations(env, behavior_name)
print(f"\nDecision steps: {len(decision_steps)}")
if len(decision_steps) > 0:
    for i, obs in enumerate(decision_steps.obs):
        print(f"  Obs[{i}] shape: {obs.shape}, values: {obs[0][:5]}...")

env.close()
