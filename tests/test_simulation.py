import mujoco
import numpy as np

from rescue_dog.envs.go1 import TORSO_BODY_ID, load_go1_model
from rescue_dog.visualization.trot_gait import TrotGait


def simulate(seconds: float, v: float = 0.0) -> mujoco.MjData:
    model = load_go1_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    gait = TrotGait(model.keyframe("home").qpos[7:].copy())
    for _ in range(int(seconds / model.opt.timestep)):
        data.ctrl[:] = gait.step(model.opt.timestep, v, 0.0)
        mujoco.mj_step(model, data)
    return data


def test_model_loads():
    model = load_go1_model()
    assert model.nu == 12  # 4 pattes × 3 moteurs
    assert model.body(TORSO_BODY_ID).name == "trunk"
    assert np.isclose(model.opt.timestep, 0.004)


def test_dog_stands_still():
    data = simulate(2.0)
    assert data.xmat[TORSO_BODY_ID][8] > 0.9  # tronc à l'horizontale
    assert np.linalg.norm(data.qpos[:2]) < 0.05


def test_dog_walks_forward():
    data = simulate(4.0, v=1.0)
    assert data.xmat[TORSO_BODY_ID][8] > 0.9  # pas tombé
    assert data.qpos[0] > 0.5                 # a avancé (≈ 0,35 m/s)
