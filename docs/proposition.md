# Apprentissage par renforcement robuste pour un robot quadrupède face à un nouvel environnement ou à un corps endommagé

*Sous-titre de présentation : le chien de sauvetage, un robot quadrupède robuste à la charge, au sol glissant et aux blessures.*

---

## 1. Contexte et problématique

Un robot entraîné en simulation perd souvent en performance une fois déployé, parce que la simulation n'est jamais parfaite : friction, masse et force des moteurs diffèrent toujours un peu de la réalité (*sim-to-real gap*). Le robot peut aussi rencontrer des situations jamais vues, comme un sol gelé ou un moteur endommagé.

**Question de recherche.** Deux techniques populaires et éprouvées, la **randomisation de domaine** et l'**acteur-critique asymétrique**, permettent-elles à un robot chien de rester fonctionnel face à un nouvel environnement (sol glissant, charge lourde) ou à un corps endommagé (patte blessée), **sans réapprendre après le déploiement** ?

**Approche.** Tout l'apprentissage se fait en simulation. Le robot est ensuite testé avec une politique **figée**, dans des conditions qui vont au-delà de celles vues à l'entraînement (transfert *zero-shot*).

---

## 2. Les deux techniques

### 2.1 Randomisation de domaine (Domain Randomization)

**Idée :** les paramètres physiques $\xi$ (friction, masse, force des moteurs) sont des variables aléatoires. À chaque épisode, on tire un nouveau $\xi$, qui définit la dynamique du simulateur :

$$\xi \sim p(\xi), \qquad s_{t+1} \sim P_\xi(\cdot \mid s_t, a_t)$$

Au lieu de maximiser le retour dans un seul simulateur, on maximise le **retour moyen sur tous les simulateurs** :

$$\max_\theta \; \mathbb{E}_{\xi \sim p(\xi)} \left[ \mathbb{E}_{\pi_\theta, P_\xi} \left[ \sum_{t=0}^{T} \gamma^t \, r(s_t, a_t) \right] \right]$$

En pratique, chaque paramètre est tiré uniformément : $\xi_i \sim \mathcal{U}[\xi_i^{\min}, \xi_i^{\max}]$. Le monde réel, ou le scénario de test, est vu comme un tirage de plus. Le transfert fonctionne si le vrai $\xi$ se trouve dans le support de $p(\xi)$.

**Limites connues :**
- La randomisation ne couvre que ce qu'on a anticipé.
- Randomiser plus large n'est pas toujours mieux (Hamed et al., 2026).
- Une politique robuste partout tend à être prudente partout.

### 2.2 Acteur-critique asymétrique (Asymmetric Actor-Critic)

**Idée :** l'**acteur** n'utilise que ce que le vrai robot peut mesurer (observation $o_t$). Le **critique**, utilisé uniquement pendant l'entraînement, voit l'état complet du simulateur $s_t$, y compris les paramètres $\xi$ (vraie friction, vraie charge, moteur endommagé).

$$\text{Acteur : } \pi_\theta(a \mid o_t) \qquad \qquad \text{Critique : } V_\phi(s_t)$$

**Critique** (régression sur les retours), avec l'état privilégié :

$$L(\phi) = \mathbb{E}_t \left[ \left( V_\phi(s_t) - R_t \right)^2 \right]$$

**Avantage** (GAE), calculé avec le critique privilégié :

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t), \qquad \hat{A}_t = \sum_{l \geq 0} (\gamma \lambda)^l \, \delta_{t+l}$$

**Acteur** (objectif PPO avec clipping), qui n'utilise que $o_t$ :

$$\rho_t = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t)}, \qquad L(\theta) = \mathbb{E}_t \left[ \min \left( \rho_t \hat{A}_t, \; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \, \hat{A}_t \right) \right]$$

**Pourquoi ça aide :** avec la randomisation, le critique sait dans quel monde $\xi$ il se trouve. Ses estimations de valeur sont donc plus précises, les avantages aussi, et la politique apprend mieux. Au déploiement, on ne garde que l'acteur : aucune information privilégiée n'est nécessaire.

**Limite à mentionner :** un critique $V(s)$ peut être biaisé en observabilité partielle. Baisero & Amato (2022) proposent $V(h, s)$, où $h$ est l'historique des observations.

### 2.3 Comment les deux se combinent

> La randomisation tire $\xi$, le critique privilégié voit $\xi$, et l'acteur apprend à bien agir sans connaître $\xi$.

---

## 3. Environnement

| Élément | Choix |
|---|---|
| Simulateur | MuJoCo Playground (MJX, accéléré sur GPU) |
| Robot | Unitree Go1 (robot quadrupède) |
| Tâche | `Go1JoystickFlatTerrain` : marcher en suivant des commandes de joystick |
| Algorithme | PPO (Brax), le standard pour les robots quadrupèdes |
| Fréquence de contrôle | 50 Hz (une action toutes les 0,02 s) |
| Épisode | 1 000 pas (20 secondes), ou jusqu'à la chute |

### 3.1 États

**Acteur, 51 valeurs** (ce que le robot réel peut mesurer, avec du bruit) :

| Entrée | Taille |
|---|---|
| Vitesse linéaire du corps | 3 |
| Vitesse angulaire (gyroscope) | 3 |
| Direction de la gravité (inclinaison) | 3 |
| Angles des articulations (par rapport à la posture debout) | 12 |
| Vitesses des articulations | 12 |
| Action précédente | 12 |
| Commande joystick (avant, côté, rotation) | 3 |

**Critique, 132 valeurs (état privilégié) :** tout ce qui précède sans bruit, plus les forces de contact, les vitesses des pieds, les forces des moteurs, le temps en l'air des pieds et les poussées externes.

**Ajout de ce projet :** on ajoute à l'état privilégié les paramètres $\xi$ de chaque scénario (friction, charge, patte touchée, force du moteur).

### 3.2 Actions

**12 valeurs**, une par articulation (3 par patte × 4 pattes). Chaque valeur est un **décalage d'angle cible** par rapport à la posture debout :

$$q^{\text{cible}} = q^{\text{défaut}} + 0{,}5 \cdot a_t, \qquad a_t \in [-1, 1]^{12}$$

Les contrôleurs des moteurs amènent ensuite chaque articulation vers sa cible.

### 3.3 Récompense

La récompense de Playground combine 16 termes. Les principaux :

| Terme | Effet |
|---|---|
| + Suivi de la vitesse linéaire commandée | Aller où on lui dit |
| + Suivi de la vitesse de rotation commandée | Tourner où on lui dit |
| − Corps non horizontal | Rester droit |
| − Arrêt prématuré (chute) | Ne pas tomber |
| − Couple et énergie | Économiser l'effort |
| − Variation brutale des actions | Mouvements fluides |
| − Glissement des pieds | Ne pas glisser (important sur la glace) |
| + Temps en l'air des pieds | Bien lever les pattes |

Les commandes sont tirées au hasard : vitesse avant dans $[-1{,}5 ; 1{,}5]$ m/s, latérale dans $[-0{,}8 ; 0{,}8]$ m/s, rotation dans $[-1{,}2 ; 1{,}2]$ rad/s. Elles changent environ toutes les 5 secondes.

### 3.4 Randomisation déjà présente dans Playground (Go1)

| Paramètre | Plage |
|---|---|
| Friction du sol | 0,4 à 1,0 |
| Masses des segments | × 0,9 à 1,1 |
| Masse du torse | ± 1 kg |
| Position du centre de masse | ± 5 cm |
| Frottement des articulations | × 0,9 à 1,1 |

Le projet **étend** cette randomisation avec les paramètres des deux scénarios.

---

## 4. Scénarios

### 4.1 Scénario 1 : nouvel environnement, « Mission sur la glace » ⛸️🎒

**Histoire.** Un robot chien de sauvetage doit livrer un sac de secours en traversant un sol gelé. Le sol est de plus en plus glissant et le sac est plus lourd que prévu.

| Paramètre $\xi$ | Entraînement (randomisation) | Test (au-delà) |
|---|---|---|
| Friction du sol $\mu$ | 0,3 à 1,0 | 0,1 à 1,0 |
| Charge sur le dos $\Delta m$ | +0 à +3 kg | +0 à +5 kg |

- **Critique privilégié :** voit la vraie friction et la vraie charge. L'acteur ne voit que ses capteurs.
- **Vidéo attendue :** sans randomisation, le chien glisse et tombe. Avec les deux techniques, il raccourcit ses pas et reste debout.
- **Difficulté :** moyenne, avec deux paramètres combinés (tableau friction × charge).

### 4.2 Scénario 2 : corps endommagé, « La patte blessée » 🩼

**Histoire.** Pendant la mission, un débris endommage une patte : un moteur perd une partie de sa force. Le robot ne sait pas quelle patte est touchée et doit continuer à marcher.

| Paramètre $\xi$ | Entraînement (randomisation) | Test (au-delà) |
|---|---|---|
| Patte touchée $k$ | une des 4, tirée au hasard | les 4 testées séparément |
| Force du moteur endommagé $\eta$ | 50 à 100 % | 20 à 100 % |

La patte blessée est simulée en réduisant la force maximale (ou le gain) des moteurs de cette patte :

$$\tau_j^{\max} \leftarrow \eta \cdot \tau_j^{\max} \quad \text{pour les articulations } j \text{ de la patte } k$$

- **Critique privilégié :** voit quelle patte est blessée et à quel point. L'acteur doit le « deviner » à partir de ses mouvements.
- **Vidéo attendue :** sans randomisation, le chien s'effondre sur la patte faible. Avec les deux techniques, il boite mais avance.
- **Difficulté :** moyenne, car la patte touchée change à chaque épisode.

---

## 5. Protocole expérimental

### 5.1 Politiques comparées (pour chaque scénario)

| Politique | Randomisation de domaine | Critique privilégié |
|---|---|---|
| A | ❌ | ❌ |
| B | ✅ | ❌ |
| C | ❌ | ✅ |
| D | ✅ | ✅ |

Chaque politique est entraînée avec 3 graines aléatoires.

### 5.2 Données

**Pas de dataset à télécharger.** En apprentissage par renforcement, le robot génère ses propres données en agissant dans le simulateur. Chaque pas produit une transition $(s_t, a_t, r_t, s_{t+1})$. Des milliers de robots simulés tournent en parallèle et produisent des millions de transitions par entraînement.

**Jeu de test fixe (créé par nous) :**
- Scénario 1 : grille friction × charge, par exemple 5 × 5 = 25 conditions.
- Scénario 2 : 4 pattes × 5 niveaux de force = 20 conditions.
- La grille est fixée une fois et réutilisée pour toutes les politiques, pour une comparaison équitable.

### 5.3 Métriques

| Métrique | Ce qu'elle mesure |
|---|---|
| Taux de réussite | Le robot reste debout pendant 20 s |
| Erreur de suivi de vitesse | Il marche là où on lui dit |
| Nombre de chutes | La sécurité |
| Retour moyen | La performance globale |

### 5.4 Résultats attendus

- **Heatmaps de réussite** par politique (friction × charge ; patte × force).
- **Vidéos 3D** avant/après pour chaque scénario.
- **Hypothèse :** D > B > C > A, avec la randomisation qui compte le plus et le critique privilégié qui accélère l'apprentissage et améliore la robustesse.

---

## 6. Tech stack

| Outil | Rôle |
|---|---|
| **MuJoCo Playground** ([GitHub](https://github.com/google-deepmind/mujoco_playground)) | Environnement Go1 prêt à l'emploi, randomisation de domaine |
| **MuJoCo / MJX** | Moteur physique (MJX = version accélérée sur GPU) |
| **JAX** | Calcul sur GPU, simulation parallèle |
| **Brax** | Implémentation de PPO (avec critique asymétrique) |
| **Kaggle Notebooks** | GPU gratuit (environ 30 h/semaine, sessions jusqu'à 12 h) pour l'entraînement |
| **MacBook Air (CPU)** | Écriture du code, analyse, graphiques, visualisation 3D interactive |
| **mediapy** | Enregistrement des vidéos 3D dans le notebook |
| **MuJoCo viewer (`mjpython`)** | Visualisation 3D interactive sur Mac pour la démo |
| **Matplotlib / Seaborn** | Courbes d'apprentissage et heatmaps |
| **TensorBoard ou Weights & Biases** | Suivi des entraînements |
| **GitHub** | Code et configurations |

### Installation

```bash
pip install playground mediapy matplotlib
```

Sur Kaggle, pour le rendu vidéo sans écran :

```python
import os
os.environ["MUJOCO_GL"] = "egl"
```

Sur Mac, pour la visualisation interactive :

```bash
pip install mujoco
mjpython view_dog.py
```

### Temps d'entraînement

| Matériel | Un entraînement (≈ 200 M pas) |
|---|---|
| 2 × RTX 4090 (article Playground) | ≈ 5 min |
| GPU Kaggle gratuit (T4 / P100) | ≈ 30 à 90 min (estimation) |
| Mac CPU | Plusieurs jours (non réaliste) |

Pour l'expérience complète (2 scénarios × 4 politiques × 3 graines = 24 entraînements), compter environ **12 à 36 h de GPU** (estimation), soit une à deux semaines de quota Kaggle. Faire d'abord un test court (10 M pas) pour mesurer le vrai temps. Si 100 M pas suffisent, le coût est divisé par deux.

---

## 7. Planning (4 semaines)

| Semaine | Tâches | Terminé quand |
|---|---|---|
| 1 | Installer Playground sur Kaggle, faire marcher le Go1, lire les articles clés | Le chien marche, vidéo enregistrée |
| 2 | Ajouter les paramètres des scénarios à la randomisation et à l'état privilégié, entraîner A et B | A échoue sur la glace et la patte blessée |
| 3 | Entraîner C et D, lancer la grille de test | Heatmaps des 4 politiques |
| 4 | Vidéos, démo 3D, rapport, présentation | Projet rendu |

---

## 8. Références

**Randomisation de domaine**
1. J. Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017. arXiv:1703.06907
2. X. B. Peng, M. Andrychowicz, W. Zaremba, P. Abbeel, *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*, ICRA 2018. arXiv:1710.06537
3. I. Akkaya et al. (OpenAI), *Solving Rubik's Cube with a Robot Hand* (randomisation automatique), 2019. arXiv:1910.07113
4. X. Chen et al., *Understanding Domain Randomization for Sim-to-Real Transfer*, ICLR 2022. arXiv:2110.03239
5. *Reinforcement Learning with Adaptive Curriculum Dynamics Randomization for Fault-Tolerant Robot Control*, 2021. arXiv:2111.10005 (randomisation de pannes de moteurs sur un quadrupède)

**Acteur-critique asymétrique et PPO**

6. L. Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning*, RSS 2018. arXiv:1710.06542
7. J. Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain*, Science Robotics, 2020 (apprentissage privilégié sur robot quadrupède)
8. A. Baisero, C. Amato, *Unbiased Asymmetric Reinforcement Learning under Partial Observability*, AAMAS 2022
9. J. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017. arXiv:1707.06347
10. J. Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016. arXiv:1506.02438

**Robots quadrupèdes et outils**

11. N. Rudin, D. Hoeller, P. Reist, M. Hutter, *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*, CoRL 2022
12. A. Kumar, Z. Fu, D. Pathak, J. Malik, *RMA: Rapid Motor Adaptation for Legged Robots*, RSS 2021. arXiv:2107.04034 (perspective : aller au-delà de la randomisation)
13. K. Zakka et al., *MuJoCo Playground*, 2025. arXiv:2502.08844
14. H. Hamed, A. Naik, C. Bellinger, A. R. Mahmood, *Rethinking the Suitability of Reinforcement Learning Algorithms Under Practical Transfer Constraints*, Reinforcement Learning Journal, 2026. arXiv:2607.17326 (randomiser plus large n'est pas toujours mieux)
15. G. Turrisi, O. Pali, L. Oneto, C. Semini, *Mixture-of-Experts RL for Fault-Tolerant Legged Locomotion*, 2026. arXiv:2606.25965 (pannes de moteurs sur un Unitree Go2)
