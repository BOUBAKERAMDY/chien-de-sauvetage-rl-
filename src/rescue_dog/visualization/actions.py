"""Actions précises pour le chien : avancer / reculer de d mètres, tourner de θ degrés, s'arrêter.

Chaque action est en boucle fermée sur la vraie pose du torse (position, cap) et s'arrête
d'elle-même une fois l'objectif atteint. Elle produit une commande (v, w, s) ∈ [-1, 1]³
(avance, rotation, pas de côté) pour `TrotGait`.

    ctrl = ActionController()
    ctrl.add_many(parse_actions("avancer 1; tourner 90; reculer 0.5"))
    v, w, s = ctrl.command(pos_xy, yaw, dt)   # à chaque pas de contrôle
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

KINDS = ("avancer", "reculer", "tourner", "stop")
HEADING_GAIN = 2.0     # correction de cap en marchant (w par radian d'erreur)
TURN_GAIN = 2.0        # w par radian d'erreur pendant une rotation
TURN_MIN_W = 0.6       # en dessous, la rotation n'avance plus
TURN_TOL = np.radians(3)
TURN_FORWARD_V = 0.15  # compense le recul du trot pendant une rotation sur place
SIDE_KP = 2.0          # pas de côté par mètre d'écart à la ligne
SIDE_KI = 1.0          # terme intégral : compense la dérive constante du trot (crabe)
SLOW_ZONE_M = 0.15     # ralentir à l'approche de la cible


@dataclass
class Action:
    kind: str
    value: float = 0.0  # mètres (avancer / reculer) ou degrés (tourner, > 0 = gauche)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"action inconnue '{self.kind}', attendu : {KINDS}")

    def __str__(self) -> str:
        unit = {"avancer": " m", "reculer": " m", "tourner": "°"}.get(self.kind)
        return self.kind if unit is None else f"{self.kind} {self.value:g}{unit}"


def parse_actions(text: str) -> list[Action]:
    """'avancer 1; tourner -90; stop' -> [Action(...), ...]"""
    actions = []
    for part in filter(None, (p.strip() for p in text.split(";"))):
        kind, *rest = part.split()
        actions.append(Action(kind, float(rest[0]) if rest else 0.0))
    return actions


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class ActionController:
    def __init__(self, verbose: bool = True):
        self.queue: deque[Action] = deque()
        self.current: Action | None = None
        self.verbose = verbose
        self._start_xy = np.zeros(2)
        self._target_yaw = 0.0
        self._side_integral = 0.0

    @property
    def idle(self) -> bool:
        return self.current is None and not self.queue

    def add(self, action: Action) -> None:
        if action.kind == "stop":
            self.queue.clear()
            self.current = None
            self._log("stop")
        else:
            self.queue.append(action)

    def add_many(self, actions: list[Action]) -> None:
        for a in actions:
            self.add(a)

    def command(self, xy: np.ndarray, yaw: float, dt: float) -> tuple[float, float, float]:
        """Commande (v, w, s) pour la pose actuelle ; passe à l'action suivante quand c'est fini."""
        if self.current is None:
            if not self.queue:
                return 0.0, 0.0, 0.0
            self._begin(self.queue.popleft(), xy, yaw)

        a = self.current
        if a.kind in ("avancer", "reculer"):
            sign = 1.0 if a.kind == "avancer" else -1.0
            heading = np.array([np.cos(self._target_yaw), np.sin(self._target_yaw)])
            normal = np.array([-heading[1], heading[0]])
            rel = xy - self._start_xy
            along, lateral = sign * float(rel @ heading), float(rel @ normal)
            remaining = abs(a.value) - along
            if remaining <= 0:
                return self._finish(f"{along:.2f} m parcourus, écart latéral {lateral:+.2f} m")
            speed = 1.0 if remaining > SLOW_ZONE_M else 0.5
            if a.kind == "reculer":
                return -speed, 0.0, 0.0  # le trot ne sait ni tourner ni se décaler en reculant
            # Cap : rotation. Écart à la ligne : pas de côté (PI), pour ne pas marcher en crabe.
            w = np.clip(HEADING_GAIN * wrap_angle(self._target_yaw - yaw), -1, 1)
            self._side_integral = np.clip(self._side_integral - lateral * dt, -0.25, 0.25)
            s = np.clip(-SIDE_KP * lateral + SIDE_KI * self._side_integral, -1, 1)
            return speed, float(w), float(s)

        # tourner
        err = wrap_angle(self._target_yaw - yaw)
        if abs(err) < TURN_TOL:
            return self._finish(f"cap atteint à {np.degrees(err):+.1f}° près")
        w = np.clip(TURN_GAIN * err, -1, 1)
        return TURN_FORWARD_V, float(np.sign(w) * max(abs(w), TURN_MIN_W)), 0.0

    def _begin(self, a: Action, xy: np.ndarray, yaw: float) -> None:
        self.current = a
        self._start_xy = xy.copy()
        self._target_yaw = wrap_angle(yaw + np.radians(a.value)) if a.kind == "tourner" else yaw
        self._side_integral = 0.0
        self._log(f"{a} ...")

    def _finish(self, detail: str) -> tuple[float, float, float]:
        self._log(f"{self.current} : fait ({detail})")
        self.current = None
        return 0.0, 0.0, 0.0

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[action] {msg}", flush=True)
