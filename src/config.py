TRACK = "cocoa_temple"
MAX_STEPS = 1500  # 150 s
DT = 0.1  # durée d'un pas pystk2
N_PATHS = 5

EXPERT_DATASET = "data/expert_dataset.npz"

# On est dans un environnement déterministe, un seul épisode suffit donc pour collecter ou évaluer une politique déterministe
COLLECT_SEED = 0
EVAL_SEED = 1000
TRAIN_SEEDS = list(range(10))

HIDDEN = 256
BATCH_SIZE = 256
LR = 1e-3
BC_STEPS = 2000  # nombre fixe de pas de gradient pour pouvoir comparer les coûts entre itérations de DAgger
LOG_EVERY = 100

DAGGER_ITERS = 20  # itérations après le BC initial ; 1 épisode de collecte par itération
DAGGER_P = 0.5  # beta_i = p^i (Ross et al. 2011) ; p = 0 : l'expert ne conduit plus après le BC

GAMMA = 0.99
TAU = 0.005
SAC_LR = 3e-4
INIT_ALPHA = 1.0  # valeur par défaut de SAC / RLPD
UTD = 10  # mises à jour du critique par pas d'environnement (RLPD : 20 avec 10 critiques)
RL_STEPS = 30_000
EVAL_EVERY = 1_000
OFFLINE_RATIO = 0.5  # symmetric sampling ; 0 : SAC sans données expert
