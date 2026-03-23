# Robocar - ia de conduite autonome

Une ia qui apprend a conduire en te regardant piloter sur un simulateur de course.

## Setup

```bash
pyenv local 3.10.12
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Comment ca marche

### 1. Collecter des donnees (toi tu conduis)
```bash
python main.py collect
```
- fleches du clavier pour piloter
- Q pour arreter et sauvegarder
- fais des tours propres, reste sur la piste
- plus t'as de donnees, mieux l'ia conduit

### 2. Entrainer l'ia
```bash
python main.py train --epochs 200
```
l'ia apprend a reproduire ta conduite a partir des donnees collectees.

### 3. Laisser l'ia conduire
```bash
python main.py drive
```
ctrl+c pour arreter.

### 4. Analyser les donnees (eda)
```bash
python notebooks/eda.py
```

## Options utiles

```bash
# changer le nombre d'epochs
python main.py train --epochs 500

# changer la taille du modele
python main.py train --hidden-size 128

# changer le learning rate
python main.py train --lr 0.0005

# utiliser un modele specifique
python main.py drive --model models/driving_model.pth
```

## structure du projet

```
src/client/       -> connexion au simulateur unity
src/input/        -> capture des touches clavier
src/collector/    -> enregistrement des donnees de conduite
src/ai/           -> modele + entrainement + metriques
config/           -> config des agents (fov, raycasts)
data/             -> csv des sessions de conduite
models/           -> modeles entraines sauvegardes
```

## tips

- une voiture lente qui reste sur la piste > une voiture rapide qui sort
- petit modele + gros dataset > gros modele + petit dataset
- vise un r2 > 0.6 pour une conduite correcte
- varie tes trajectoires pour que l'ia generalise mieux
