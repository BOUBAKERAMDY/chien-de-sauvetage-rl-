# Chien de sauvetage — RL robuste pour un robot quadrupède (Unitree Go1)

Projet d'apprentissage par renforcement : voir la proposition dans [`docs/proposition.md`](docs/proposition.md).

**Étape actuelle : simuler le chien dans son environnement** (sol plat normal, viewer 3D,
trot codé à la main). Les scénarios, l'entraînement PPO et l'évaluation viendront ensuite.

## Installation

```bash
make install        # crée .venv (Python 3.12) avec uv
make test           # tests unitaires
```

## Voir le chien

```bash
make view
# ou, avec une suite d'actions :
uv run mjpython scripts/view_dog.py --actions "avancer 1; tourner 90; avancer 1; reculer 0.5"
```

Clavier : ↑ avancer 0,5 m · ↓ reculer 0,3 m · ← → tourner 45° · Entrée stop.

## Organisation

```
configs/base.yaml              configuration (nom de l'environnement, pas de contrôle)
src/rescue_dog/
  config.py                    lecture des YAML
  envs/go1.py                  modèle MuJoCo du Go1 (le même que MuJoCo Playground)
  visualization/
    trot_gait.py               trot codé à la main (sans RL)
    actions.py                 actions précises : avancer, reculer, tourner, stop
  scenarios/                   à faire : « Mission sur la glace », « La patte blessée »
  training/                    à faire : PPO (Brax)
  evaluation/                  à faire : grilles de test
  utils/                       à faire : sauvegarde, logs, graines
scripts/view_dog.py            viewer 3D interactif
tests/                         tests (modèle, simulation, actions)
notebooks/                     à faire (Kaggle)
results/                       à faire (checkpoints, figures, vidéos)
docs/                          proposition, rapport, présentation, articles
```
