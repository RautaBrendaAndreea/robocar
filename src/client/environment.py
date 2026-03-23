# connexion entre notre code et le simulateur unity

import json
import os
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# chemin vers l'app du simulateur sur mac
SIMULATOR_PATH = "/Users/Andreea/Downloads/BuildMac/RacingSimulator.app"


def load_agent_config(config_path="config/agents.json"):
    with open(config_path) as f:
        return json.load(f)


def create_environment(config_path="config/agents.json", port=5004, time_scale=1.0, simulator_path=None):
    # on cree un channel pour regler la vitesse du simu
    engine_channel = EngineConfigurationChannel()

    # on lance le simulateur et on se connecte dessus via grpc
    env = UnityEnvironment(
        file_name=simulator_path or SIMULATOR_PATH,
        side_channels=[engine_channel],
        additional_args=["--config-path", os.path.abspath(config_path)],
        worker_id=0,
        base_port=port,
        timeout_wait=300
    )

    # on regle la vitesse et le framerate du simu
    engine_channel.set_configuration_parameters(
        time_scale=time_scale,
        target_frame_rate=60
    )
    env.reset()
    return env


def get_behavior(env):
    # recupere le nom du behavior (genre "Agent0?team=0") et ses specs
    behavior_names = list(env.behavior_specs.keys())
    if not behavior_names:
        raise RuntimeError("No behaviors found in environment")
    return behavior_names[0], env.behavior_specs[behavior_names[0]]


def get_observations(env, behavior_name):
    # recupere les observations (raycasts) et l'etat de l'agent
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    return decision_steps, terminal_steps
