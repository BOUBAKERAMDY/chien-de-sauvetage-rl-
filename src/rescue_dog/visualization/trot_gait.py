"""Trot codé à la main (sans RL), pour piloter le chien dans le viewer avant d'avoir une politique.

Les pattes diagonales (FR+RL, FL+RR) bougent ensemble. Pour chaque patte, avec la phase φ :
    hanche : q_défaut − H·w·avant · cos(φ)         -> pas latéraux opposés avant/arrière = rotation
           + S·s · sin(φ)                          -> pas de côté (corrige la marche en crabe)
    cuisse : q_défaut + A·v · cos(φ)               -> balancement avant / arrière
    mollet : q_défaut − B · max(0, sin(φ))         -> lever le pied pendant le balancement

Paramètres réglés par recherche en simulation (sol nominal, μ = 0,6) : environ 0,35 m/s en avant,
0,15 m/s en arrière (tout droit), ~10 °/s en rotation. C'est un contrôleur en boucle ouverte :
il ne corrige pas son équilibre, tombe parfois dans les enchaînements brusques (≈ 4 séquences
aléatoires sur 20) et tombe sur la glace (μ = 0,1) — une bonne référence « naïve » face au RL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHASE_OFFSET = np.array([0.0, np.pi, np.pi, 0.0])  # FR, FL, RR, RL : trot diagonal
FRONT = np.array([1, 1, -1, -1])                   # +1 = patte avant


@dataclass
class TrotGait:
    default_pose: np.ndarray
    stride_fwd: float = 0.35      # A en avant, pour v = 1
    freq_fwd: float = 2.0         # Hz
    step_height_fwd: float = 0.6  # B (rad, mollet)
    stride_back: float = 0.10     # en arrière : petits pas rapides, sinon le chien bascule
    freq_back: float = 3.5
    step_height_back: float = 0.7
    hip_turn_fwd: float = 0.04    # H (rad, hanche) pour w = ±1
    hip_turn_back: float = 0.0    # pas de virage en arrière : le chien bascule
    hip_side: float = 0.05        # S (rad, hanche) pour s = ±1
    smoothing_s: float = 0.3      # constante de temps du lissage de la commande
    phase: float = 0.0
    v: float = 0.0                # commande lissée
    w: float = 0.0
    s: float = 0.0

    def step(self, dt: float, v: float, w: float, s: float = 0.0) -> np.ndarray:
        """Avance la phase de `dt` et retourne les 12 angles cibles.

        Appeler à chaque pas physique (250 Hz) : à 50 Hz la démarche devient instable.

        v ∈ [-1, 1] : avant (> 0) / arrière ; w ∈ [-1, 1] : rotation à gauche (> 0) / droite ;
        s ∈ [-1, 1] : pas de côté vers la gauche (> 0), seulement en marche avant.
        """
        # Les changements brusques de commande font tomber le chien : on les lisse.
        alpha = min(1.0, dt / self.smoothing_s)
        self.v += alpha * (v - self.v)
        self.w += alpha * (w - self.w)
        self.s += alpha * (s - self.s)
        v, w, s = self.v, self.w, self.s

        q = self.default_pose.copy()
        if abs(v) < 1e-2 and abs(w) < 1e-2:
            self.phase = 0.0
            return q
        forward = v >= 0
        self.phase += 2 * np.pi * (self.freq_fwd if forward else self.freq_back) * dt
        phi = self.phase + PHASE_OFFSET
        stride = (self.stride_fwd if forward else self.stride_back) * v
        step_height = self.step_height_fwd if forward else self.step_height_back
        hip_turn = self.hip_turn_fwd if forward else self.hip_turn_back
        q[0::3] -= hip_turn * w * FRONT * np.cos(phi)
        if forward:
            q[0::3] += self.hip_side * s * np.sin(phi)
        q[1::3] += stride * np.cos(phi)
        q[2::3] -= step_height * np.maximum(0.0, np.sin(phi))
        return q
