# DAgger vs symmetric sampling (RLPD) dans SuperTuxKart

L'objectif est de comparer DAgger et le *symmetric sampling* de RLPD sur SuperTuxKart.

## Structure

```
src/
  config.py      hyper-paramètres communs (piste, limite de pas, graines, réseau, entraînement)
  env.py         création de l'environnement + conversions observation/action
  evaluate.py    évaluation d'une politique sur un tour
  policy.py      MLP commun à toutes les méthodes (entrées normalisées, sortie tanh)
  bc.py          behavior cloning (régression MSE, Adam)
  dagger.py      épisode de l'élève étiqueté par l'expert
  sac.py         Soft Actor-Critic (acteur tanh-gaussien, double critique avec LayerNorm)
  rlpd.py        buffers et symmetric sampling (50 % démonstrations, 50 % en ligne)
  expert/        politique experte (Agent1 du repo d'agents)
scripts/
  test_env.py      fait un tour avec l'expert ou un modèle (--model)
  collect_data.py  génère le dataset
  evaluate_expert.py  évalue l'expert
  train_bc.py      behavior cloning sur 10 graines
  train_dagger.py  DAgger sur 10 graines (beta_i = p^i, p dans config.py)
  train_rlpd.py    RLPD sur 10 graines (OFFLINE_RATIO = 0 : SAC seul)
analysis/
  results.py         statistiques, tests, figures et tableaux LaTeX -> runs/analysis/, report/figures/
  trajectory_gap.py  compare l'expert et un modèle pas à pas (--model, --label)
runs/            résultats (générés) : un dossier par méthode et par graine
report/          rapport ; figures générées dans report/figures/
notes/           notes théoriques
```

## Installation

```bash
uv sync
```

## Utilisation

```bash
uv run python scripts/test_env.py --render
uv run python scripts/collect_data.py
uv run python scripts/evaluate_expert.py
uv run python scripts/train_bc.py
uv run python scripts/train_dagger.py
uv run python scripts/train_rlpd.py
uv run python analysis/results.py
uv run python analysis/trajectory_gap.py --model runs/dagger_p0.5/seed_4/model.pt --label "DAgger p=0.5"
```

## Choix de modélisation

- **Environnement** : `supertuxkart/full-v0`, 1 kart, 1 tour, piste `cocoa_temple`, 1500 pas maximum (150 s).
- **Récompense** (celle de `pystk2_gymnasium`, avec 1 kart) : `r_t = (d_t - d_{t-1}) / 10 - 0.1 + 10 * 1[fin]`
  (progression moins une pénalité de temps, donc maximiser le retour revient à finir vite).
- **Actions** : `a = (steer, throttle) in [-1, 1]^2`. `throttle > 0` accélère, `< 0` freine.
- **Observation de l'élève** : vecteur de 28 nombres dans le repère du kart (vitesse, vecteur vers
  le centre de la piste, distance au centre, angle de braquage max, extrémités et largeurs des
  5 prochains segments).

## Protocole d'évaluation

- L'environnement est déterministe : la graine ne change pas la course, et l'expert fait le même tour à chaque fois.
  Attention : le mode `--render` modifie la simulation (contenu des boîtes cadeaux, trajectoire).
  Toutes les évaluations sont faites sans rendu ; le rendu sert uniquement à visualiser.
- Dataset : un tour de l'expert, sans perturbation (`collect_data.py`).
- Métrique principale : le retour du tour (équivalent au temps au tour s'il est fini, à la distance sinon).
  Métriques secondaires : tour fini ou non, temps au tour.
- Une politique déterministe est évaluée sur un seul épisode (graine 1000).
- 10 graines d'entraînement par méthode. L'unité statistique est la graine d'entraînement.
- Tests : Welch et Mann-Whitney U, IC 95 % par bootstrap, IQM, correction de Holm-Bonferroni.
