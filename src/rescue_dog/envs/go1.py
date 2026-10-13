"""Modèle MuJoCo du Go1, identique à celui de Playground (`Go1JoystickFlatTerrain`).

On charge le même XML et on réapplique les mêmes réglages que `Go1Env.__init__`
(mujoco_playground/_src/locomotion/go1/base.py) : pas de temps, gains PD.
"""

from __future__ import annotations

import mujoco
from mujoco_playground import registry
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants

FLOOR_GEOM_ID = 0  # le sol
TORSO_BODY_ID = 1  # le tronc ("trunk")


def load_go1_model(env_name: str = "Go1JoystickFlatTerrain") -> mujoco.MjModel:
    cfg = registry.get_default_config(env_name)
    xml = go1_constants.task_to_xml("flat_terrain")
    model = mujoco.MjModel.from_xml_string(xml.read_text(), assets=go1_base.get_assets())
    model.opt.timestep = cfg.sim_dt      # 0,004 s : physique à 250 Hz
    model.opt.ccd_iterations = 20
    model.dof_damping[6:] = cfg.Kd       # moteurs = contrôleurs PD vers un angle cible
    model.actuator_gainprm[:, 0] = cfg.Kp
    model.actuator_biasprm[:, 1] = -cfg.Kp
    return model
