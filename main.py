# 3 modes: collect (tu conduis), train (l'ia apprend), drive (l'ia conduit)

import argparse
import numpy as np
from mlagents_envs.base_env import ActionTuple

from src.client.environment import create_environment, get_behavior, get_observations
from src.input.controller import InputController
from src.collector.data_collector import DataCollector
from src.ai.trainer import Trainer


def run_collect(config_path, time_scale):
    # mode collecte: tu pilotes la voiture et on enregistre tout
    env = create_environment(config_path, time_scale=time_scale)
    behavior_name, spec = get_behavior(env)
    controller = InputController()
    collector = DataCollector()

    print(f"Connected! Behavior: {behavior_name}")
    print(f"Observation shape: {spec.observation_specs[0].shape}")
    print(f"Action size: {spec.action_spec.continuous_size}")
    print("Collecting data... Press Q or close pygame window to stop.")
    try:
        while not controller.should_quit():
            decision_steps, _ = get_observations(env, behavior_name)
            if len(decision_steps) == 0:
                env.reset()
                continue
            # on recupere les raycasts de la voiture
            obs = decision_steps.obs[0][0]
            # on recupere ce que le joueur fait au clavier
            action = controller.get_action()
            # on sauvegarde le couple (observation, action)
            collector.record(obs, action)
            # on envoie l'action au simulateur
            env.set_actions(behavior_name, ActionTuple(continuous=action))
            env.step()
            if collector.size % 100 == 0 and collector.size > 0:
                print(f"  Collected {collector.size} samples...", end="\r")
    except (KeyboardInterrupt, Exception) as e:
        if not isinstance(e, KeyboardInterrupt):
            print(f"\nError: {e}")
    finally:
        collector.save()
        controller.close()
        try:
            env.close()
        except Exception:
            pass


def run_train(data_dir, epochs, hidden_size, lr, batch_size):
    # mode entrainement: on entraine le modele sur les donnees collectees
    trainer = Trainer(hidden_size=hidden_size, lr=lr, batch_size=batch_size)
    trainer.train(data_dir=data_dir, epochs=epochs)
    trainer.save_model()


def run_inference(config_path, model_path, time_scale):
    # mode conduite auto: le modele charge predit les actions tout seul
    model = Trainer.load_model(model_path)
    env = create_environment(config_path, time_scale=time_scale)
    behavior_name, spec = get_behavior(env)

    print("Running AI... Press Ctrl+C to stop.")
    try:
        while True:
            decision_steps, _ = get_observations(env, behavior_name)
            if len(decision_steps) == 0:
                env.reset()
                continue
            # l'ia voit les raycasts et decide quoi faire
            obs = decision_steps.obs[0][0]
            action = model.predict(obs)
            env.set_actions(behavior_name, ActionTuple(continuous=action.astype(np.float32)))
            env.step()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Robocar Racing AI")
    sub = parser.add_subparsers(dest="mode", required=True)

    # python main.py collect -> tu conduis et on enregistre
    collect = sub.add_parser("collect", help="Collect driving data")
    collect.add_argument("--config", default="config/agents.json")
    collect.add_argument("--time-scale", type=float, default=1.0)

    # python main.py train -> on entraine l'ia sur tes donnees
    train = sub.add_parser("train", help="Train the AI model")
    train.add_argument("--data-dir", default="data")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--batch-size", type=int, default=64)

    # python main.py drive -> l'ia conduit toute seule
    infer = sub.add_parser("drive", help="Run AI on the simulator")
    infer.add_argument("--config", default="config/agents.json")
    infer.add_argument("--model", default="models/driving_model.pth")
    infer.add_argument("--time-scale", type=float, default=1.0)

    args = parser.parse_args()

    if args.mode == "collect":
        run_collect(args.config, args.time_scale)
    elif args.mode == "train":
        run_train(args.data_dir, args.epochs, args.hidden_size, args.lr, args.batch_size)
    elif args.mode == "drive":
        run_inference(args.config, args.model, args.time_scale)


if __name__ == "__main__":
    main()
