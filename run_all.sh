#!/usr/bin/env bash
# ==============================================================================
# run_all.sh — One-shot driver for the three PhysHGNet experiments:
#   (1) anchor visualisation (temperature field + moving anchors: GIF + PNGs)
#   (2) PhysHGNet vs DGNet anchor / node scaling benchmark
#   (3) PhysHGNet component ablation
#
# It also (optionally) generates the datasets and trains the models so the
# benchmark/visualisation use meaningful (trained) weights.
#
# Usage:
#   bash run_all.sh                 # full pipeline with defaults below
#   GPUS=0,1,2,3,4,5,6 EPOCHS=15 bash run_all.sh
#   SKIP_TRAIN=1 bash run_all.sh    # skip training (random/loaded weights only)
#   QUICK=1 bash run_all.sh         # tiny/fast settings for a smoke test
#
# Adjust the variables in the CONFIG block as needed.
# ==============================================================================
set -e

# ── CONFIG ────────────────────────────────────────────────────────────────
GPUS=${GPUS:-0,1,2,3,4,5,6}          # GPUs for DDP training (train_*.py); 7 cards
VIZ_GPU=${VIZ_GPU:-0}                # single GPU for viz / benchmark / ablation
DATA_DIR=${DATA_DIR:-data_laser_hardening}
MAIN_N=${MAIN_N:-2000}               # mesh used for viz + anchor-scaling + ablation
NODE_LIST=${NODE_LIST:-"500 1000 2000 4000"}   # meshes for node-scaling
EPOCHS=${EPOCHS:-15}
N_TRAJ=${N_TRAJ:-40}
N_TIMESTEPS=${N_TIMESTEPS:-121}

if [ "${QUICK:-0}" = "1" ]; then
  EPOCHS=3; N_TRAJ=6; N_TIMESTEPS=40; NODE_LIST="500 1000 2000"
  echo "[run_all] QUICK mode: EPOCHS=$EPOCHS N_TRAJ=$N_TRAJ N_TIMESTEPS=$N_TIMESTEPS"
fi

NPROC=$(echo "$GPUS" | awk -F, '{print NF}')
PH_CKPT="checkpoints/phys_hgnet/best_${MAIN_N}.pth"
DG_CKPT="checkpoints/dgnet/best_${MAIN_N}.pth"

echo "================= PhysHGNet experiment pipeline ================="
echo " GPUS=$GPUS (nproc=$NPROC)  VIZ_GPU=$VIZ_GPU  MAIN_N=$MAIN_N  EPOCHS=$EPOCHS"
echo " NODE_LIST=$NODE_LIST  DATA_DIR=$DATA_DIR"
echo "================================================================="

# ── 0. DATA GENERATION ──────────────────────────────────────────────────────
echo ">>> [0] Generating datasets ..."
for N in $MAIN_N $NODE_LIST; do
  f="${DATA_DIR}/pde_trajectories_${N}.h5"
  if [ -f "$f" ]; then
    echo "    exists: $f (skip)"
  else
    python generate_laser_data_aligned.py --n_nodes "$N" --n_traj "$N_TRAJ" \
        --n_timesteps "$N_TIMESTEPS" --out_dir "$DATA_DIR"
  fi
done

# ── 1. TRAINING (optional) ──────────────────────────────────────────────────
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  echo ">>> [1] SKIP_TRAIN=1 -> skipping training."
else
  echo ">>> [1a] Training PhysHGNet (full) ..."
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC train_physhgnet.py \
      --epochs $EPOCHS --n-nodes $MAIN_N --data-dir $DATA_DIR

  echo ">>> [1b] Training DGNet ..."
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC train_dgnet.py \
      --epochs $EPOCHS --n-nodes $MAIN_N --data-dir $DATA_DIR || \
      echo "    [warn] train_dgnet.py failed/!= expected args; DGNet ckpt may be absent."
fi

PH_ARG=""; [ -f "$PH_CKPT" ] && PH_ARG="--physhgnet-ckpt $PH_CKPT"
DG_ARG=""; [ -f "$DG_CKPT" ] && DG_ARG="--dgnet-ckpt $DG_CKPT"
VIZ_PH_ARG=""; [ -f "$PH_CKPT" ] && VIZ_PH_ARG="--ckpt $PH_CKPT"

# ── 2. ANCHOR VISUALISATION ─────────────────────────────────────────────────
echo ">>> [2] Anchor visualisation (GIF + screenshots) ..."
CUDA_VISIBLE_DEVICES=$VIZ_GPU python visualize_anchors.py \
    --n-nodes $MAIN_N --data-dir $DATA_DIR --traj-index 0 \
    --update-freq 1 --max-steps 80 --fps 8 \
    --out-dir viz_anchors $VIZ_PH_ARG

# ── 3. SCALING BENCHMARK ────────────────────────────────────────────────────
echo ">>> [3a] Anchor-count scaling (PhysHGNet vs DGNet reference) ..."
CUDA_VISIBLE_DEVICES=$VIZ_GPU python benchmark_scaling.py --mode anchor \
    --n-nodes $MAIN_N --data-dir $DATA_DIR \
    --anchors 16 32 64 128 256 \
    --out-dir bench_scaling $PH_ARG $DG_ARG

echo ">>> [3b] Node-size scaling (PhysHGNet vs DGNet) ..."
CUDA_VISIBLE_DEVICES=$VIZ_GPU python benchmark_scaling.py --mode nodes \
    --node-list $NODE_LIST --data-dir $DATA_DIR \
    --out-dir bench_scaling $PH_ARG $DG_ARG

# ── 4. ABLATION ─────────────────────────────────────────────────────────────
echo ">>> [4] Component ablation ..."
CUDA_VISIBLE_DEVICES=$VIZ_GPU python ablation_experiment.py \
    --n-nodes $MAIN_N --data-dir $DATA_DIR \
    --epochs $EPOCHS --batch-size 2 \
    --configs full no_c1 static_anchor no_c2 no_c3 no_vn baseline \
    --out-dir ablation_results

echo "================================================================="
echo " DONE. Artifacts:"
echo "   viz_anchors/anchor_evolution.gif + screenshot_*.png + montage"
echo "   bench_scaling/scaling_anchor.{csv,png}, scaling_nodes.{csv,png}"
echo "   ablation_results/ablation_results.csv, ablation_bar.png"
echo "================================================================="
