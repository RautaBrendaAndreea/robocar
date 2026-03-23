# enregistre les donnees de conduite (observations + actions) dans des fichiers csv
# chaque session de collecte cree un nouveau fichier, et on peut tout charger d'un coup pour l'entrainement

import os
import numpy as np
import pandas as pd
from datetime import datetime


class DataCollector:
    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.observations = []
        self.actions = []

    def record(self, observation, action):
        # stocke une frame: ce que la voiture voit (rays) + ce que le joueur fait (throttle/steering)
        self.observations.append(observation.flatten())
        self.actions.append(action.flatten())

    def save(self, prefix="driving"):
        # save tout dans un csv avec un timestamp unique
        if not self.observations:
            print("No data to save.")
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        obs = np.array(self.observations)
        act = np.array(self.actions)
        # colonnes: ray_0, ray_1 ... ray_49, throttle, steering
        columns = [f"ray_{i}" for i in range(obs.shape[1])] + ["throttle", "steering"]
        df = pd.DataFrame(np.hstack([obs, act]), columns=columns)
        path = os.path.join(self.save_dir, f"{prefix}_{timestamp}.csv")
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} samples to {path}")
        self.clear()
        return path

    def clear(self):
        self.observations.clear()
        self.actions.clear()

    @property
    def size(self):
        return len(self.observations)

    @staticmethod
    def load_all(data_dir="data"):
        # charge tous les csv du dossier data et les merge en un seul dataframe
        files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files in {data_dir}")
        return pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in files], ignore_index=True)
