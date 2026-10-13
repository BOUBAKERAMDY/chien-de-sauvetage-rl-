"""Le chien Go1 dans son environnement (sol plat normal), avec le viewer MuJoCo.

    uv run mjpython scripts/view_dog.py
    uv run mjpython scripts/view_dog.py --actions "avancer 1; tourner 90; avancer 1; reculer 0.5"

Le chien marche avec un trot codé à la main (pas de RL pour l'instant).

Clavier (dans la fenêtre du viewer) — chaque appui ajoute une action, qui s'arrête seule :
    ↑       avancer de 0,5 m          ↓       reculer de 0,3 m
    ←       tourner de 45° à gauche   →       tourner de 45° à droite
    Entrée  stop (annule les actions en attente)
Souris : double-clic pour sélectionner un corps, Ctrl + clic droit glissé pour le pousser.
Le chien est remis debout automatiquement s'il reste tombé plus d'une seconde.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from rescue_dog.config import load_config
from rescue_dog.envs.go1 import TORSO_BODY_ID, load_go1_model
from rescue_dog.visualization.actions import Action, ActionController, parse_actions
from rescue_dog.visualization.trot_gait import TrotGait

KEY_ACTIONS = {  # codes GLFW -> action
    265: Action("avancer", 0.5),   # ↑
    264: Action("reculer", 0.3),   # ↓
    263: Action("tourner", 45),    # ←
    262: Action("tourner", -45),   # →
    257: Action("stop"),           # Entrée
}
FALL_RESET_S = 1.0  # durée à terre avant remise debout


def torso_pose(data) -> tuple[np.ndarray, float]:
    xmat = data.xmat[TORSO_BODY_ID]
    return data.qpos[:2].copy(), float(np.arctan2(xmat[3], xmat[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--actions", default="", help='ex. "avancer 1; tourner 90; reculer 0.5"')
    args = p.parse_args()

    cfg = load_config()["env"]
    model = load_go1_model(cfg["name"])
    data = mujoco.MjData(model)
    home = model.keyframe("home").id

    def reset():
        mujoco.mj_resetDataKeyframe(model, data, home)
        mujoco.mj_forward(model, data)

    reset()
    default_pose = model.keyframe("home").qpos[7:].copy()
    ctrl_dt = cfg["ctrl_dt"]
    sim_dt = model.opt.timestep
    n_substeps = int(round(ctrl_dt / sim_dt))
    gait = TrotGait(default_pose)
    actions = ActionController()
    actions.add_many(parse_actions(args.actions))
    fallen_s = 0.0

    def on_key(key: int) -> None:
        if key in KEY_ACTIONS:
            actions.add(KEY_ACTIONS[key])

    print("[viewer] Trot codé à la main (sans RL), sol plat normal.")
    print("[viewer] ↑ avancer 0,5 m | ↓ reculer 0,3 m | ← → tourner 45° | Entrée stop")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            t0 = time.time()
            v, w, s = actions.command(*torso_pose(data), ctrl_dt)
            for _ in range(n_substeps):  # le trot est calculé à chaque pas physique (250 Hz)
                data.ctrl[:] = gait.step(sim_dt, v, w, s)
                mujoco.mj_step(model, data)

            fallen_s = fallen_s + ctrl_dt if data.xmat[TORSO_BODY_ID][8] < 0.3 else 0.0
            if fallen_s > FALL_RESET_S:
                print("[viewer] Chute ! Remise debout.")
                actions.add(Action("stop"))
                gait = TrotGait(default_pose)
                reset()
                fallen_s = 0.0

            viewer.sync()
            time.sleep(max(0.0, ctrl_dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
