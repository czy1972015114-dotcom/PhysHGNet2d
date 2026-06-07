#!/usr/bin/env bash
# =============================================================================
# run_all.sh — PhysHGNet 全流程一键实验（可安心挂后台过夜）
#
# 步骤：
#   [0] 数据生成  N=2000 + 4000/6000/8000/10000
#   [1] 训练      PhysHGNet(full) + DGNet baseline (7 卡 DDP)
#   [2] 消融      full/no_c1/static_anchor/no_mg/no_c2/no_c3/no_vn/baseline
#   [3] 可视化    锚点随温度演化 GIF + 截图
#   [4] 推理速度  PhysHGNet vs DGNet 独立轨迹计时
#   [5] 大规模    N-scaling 训练+评测（DGNet OOM 捕获）
#
# 用法：
#   bash run_all.sh                        # 完整流程
#   QUICK=1 bash run_all.sh               # 冒烟测试 (~20 min)
#   SKIP_DATA=1 bash run_all.sh           # 数据已有，跳过生成
#   SKIP_TRAIN=1 bash run_all.sh          # 训练已完成，只跑评测
#   ONLY_STEP=2 bash run_all.sh           # 只跑第 2 步（消融）
#   GPUS=0,1,2,3 bash run_all.sh          # 只用 4 张卡
# =============================================================================

# ── 遇到未定义变量时报错，但单步失败不中止全流程（用 run_step 捕获）───────────
set -uo pipefail

# ── 配置（可通过环境变量覆盖）─────────────────────────────────────────────────
GPUS="${GPUS:-0,1,2,3,4,5,6}"
VIZ_GPU="${VIZ_GPU:-0}"
DATA_DIR="${DATA_DIR:-data_laser_hardening}"
MAIN_N="${MAIN_N:-2000}"
SCALE_LIST="${SCALE_LIST:-4000 6000 8000 10000}"
EPOCHS="${EPOCHS:-15}"
SCALE_EPOCHS="${SCALE_EPOCHS:-5}"
BATCH="${BATCH:-4}"
ABLATION_BATCH="${ABLATION_BATCH:-8}"
LR="${LR:-5e-4}"
M_ANCHORS="${M_ANCHORS:-64}"
TRAIN_STEPS="${TRAIN_STEPS:-7}"
N_TRAJ="${N_TRAJ:-40}"
N_TIMESTEPS="${N_TIMESTEPS:-121}"
LOG_DIR="${LOG_DIR:-logs}"

if [ "${QUICK:-0}" = "1" ]; then
    MAIN_N=500; EPOCHS=3; SCALE_EPOCHS=2
    N_TRAJ=6; N_TIMESTEPS=40; BATCH=2; ABLATION_BATCH=2
    SCALE_LIST="2000 4000"
    echo "[QUICK] 冒烟模式: N=$MAIN_N epochs=$EPOCHS"
fi

NPROC=$(echo "$GPUS" | awk -F, '{print NF}')
PH_CKPT="checkpoints/phys_hgnet/best_${MAIN_N}.pth"
DG_CKPT="checkpoints/dgnet/best_${MAIN_N}.pth"
ONLY_STEP="${ONLY_STEP:-all}"
mkdir -p "$LOG_DIR"

# ── 工具函数 ──────────────────────────────────────────────────────────────────
ts()  { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

should_run() {
    [ "$ONLY_STEP" = "all" ] || [ "$ONLY_STEP" = "$1" ]
}

# 运行一条命令，捕获错误，不中止全局
safe_run() {
    local step="$1"; shift
    local logfile="$LOG_DIR/step${step}.log"
    log ">>> 步骤 [$step] 开始 → 日志: $logfile"
    if "$@" 2>&1 | tee "$logfile"; then
        log "✓ 步骤 [$step] 成功"
        return 0
    else
        log "✗ 步骤 [$step] 失败，继续下一步（详见 $logfile）"
        return 1
    fi
}

# ── 打印摘要 ──────────────────────────────────────────────────────────────────
log "================================================================="
log " PhysHGNet 全流程实验  $(date '+%Y-%m-%d %H:%M')"
log " GPUS=$GPUS (nproc=$NPROC)  MAIN_N=$MAIN_N  EPOCHS=$EPOCHS"
log " SCALE_LIST=$SCALE_LIST  ONLY_STEP=$ONLY_STEP"
log "================================================================="

# ═════════════════════════════════════════════════════════════════════════════
# [0] 数据生成
# ═════════════════════════════════════════════════════════════════════════════
if should_run 0 && [ "${SKIP_DATA:-0}" != "1" ]; then
    log ">>> [0] 数据生成"
    for N in $MAIN_N $SCALE_LIST; do
        F="${DATA_DIR}/pde_trajectories_${N}.h5"
        if [ -f "$F" ]; then
            log "  已存在 $F（跳过）"
        else
            log "  生成 N=$N ..."
            safe_run 0_N${N} python3 generate_laser_data_aligned.py \
                --n_nodes "$N" --n_traj "$N_TRAJ" \
                --n_timesteps "$N_TIMESTEPS" --dt 0.5 \
                --out_dir "$DATA_DIR"
        fi
    done
fi

# ═════════════════════════════════════════════════════════════════════════════
# [1] 训练
# ═════════════════════════════════════════════════════════════════════════════
if should_run 1 && [ "${SKIP_TRAIN:-0}" != "1" ]; then

    log ">>> [1a] 训练 PhysHGNet (full, $NPROC 卡 DDP)"
    safe_run 1a \
        env CUDA_VISIBLE_DEVICES="$GPUS" \
        torchrun --nproc_per_node="$NPROC" train_physhgnet.py \
            --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
            --n-nodes "$MAIN_N" --data-dir "$DATA_DIR" \
            --m-anchors "$M_ANCHORS" --train-time-steps "$TRAIN_STEPS"

    log ">>> [1b] 训练 DGNet baseline ($NPROC 卡 DDP)"
    safe_run 1b \
        env CUDA_VISIBLE_DEVICES="$GPUS" \
        torchrun --nproc_per_node="$NPROC" train_dgnet.py \
            --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
            --n-nodes "$MAIN_N" --data-dir "$DATA_DIR" \
            --train-time-steps "$TRAIN_STEPS"
fi

# ckpt 参数（有就传，没有也不报错）
PH_ARG=""; [ -f "$PH_CKPT" ] && PH_ARG="--physhgnet-ckpt $PH_CKPT"
DG_ARG=""; [ -f "$DG_CKPT" ] && DG_ARG="--dgnet-ckpt $DG_CKPT"
VZ_ARG=""; [ -f "$PH_CKPT" ] && VZ_ARG="--ckpt $PH_CKPT"

# ═════════════════════════════════════════════════════════════════════════════
# [2] 消融实验
# ═════════════════════════════════════════════════════════════════════════════
if should_run 2; then
    log ">>> [2] 消融实验 ($NPROC 卡 DDP)"
    safe_run 2 \
        env CUDA_VISIBLE_DEVICES="$GPUS" \
        torchrun --nproc_per_node="$NPROC" ablation_experiment.py \
            --n-nodes "$MAIN_N" --data-dir "$DATA_DIR" \
            --epochs "$EPOCHS" --batch-size "$ABLATION_BATCH" --lr "$LR" \
            --m-anchors "$M_ANCHORS" --train-time-steps "$TRAIN_STEPS" \
            --configs full no_c1 static_anchor no_mg no_c2 no_c3 no_vn baseline \
            --out-dir ablation_results
fi

# ═════════════════════════════════════════════════════════════════════════════
# [3] 锚点可视化
# ═════════════════════════════════════════════════════════════════════════════
if should_run 3; then
    log ">>> [3] 锚点可视化 (单卡 GPU $VIZ_GPU)"
    safe_run 3 \
        env CUDA_VISIBLE_DEVICES="$VIZ_GPU" \
        python3 visualize_anchors.py \
            --n-nodes "$MAIN_N" --data-dir "$DATA_DIR" \
            --traj-index 0 --update-freq 1 --max-steps 80 \
            --fps 8 --out-dir viz_anchors $VZ_ARG
fi

# ═════════════════════════════════════════════════════════════════════════════
# [4] 推理速度（必须单卡单进程，排除通信干扰）
# ═════════════════════════════════════════════════════════════════════════════
if should_run 4; then
    log ">>> [4] 推理速度 (单卡 GPU $VIZ_GPU)"
    safe_run 4 \
        env CUDA_VISIBLE_DEVICES="$VIZ_GPU" \
        python3 inference_speed.py \
            --n-nodes "$MAIN_N" --data-dir "$DATA_DIR" \
            --traj-index -1 --repeats 10 --warmup 3 \
            --m-anchors "$M_ANCHORS" --out-dir speed_results \
            $PH_ARG $DG_ARG
fi

# ═════════════════════════════════════════════════════════════════════════════
# [5] 大规模 N-scaling（对每个 N 用 DDP 完整训练 + 评测）
# ═════════════════════════════════════════════════════════════════════════════
if should_run 5; then
    log ">>> [5] N-scaling: $SCALE_LIST ($NPROC 卡 DDP，每个 N 训 $SCALE_EPOCHS epoch)"
    safe_run 5 \
        env CUDA_VISIBLE_DEVICES="$GPUS" \
        torchrun --nproc_per_node="$NPROC" benchmark_scaling.py \
            --node-list $SCALE_LIST \
            --data-dir "$DATA_DIR" \
            --epochs "$SCALE_EPOCHS" \
            --batch-size "$BATCH" --lr "$LR" \
            --m-anchors "$M_ANCHORS" --train-steps "$TRAIN_STEPS" \
            --out-dir bench_scaling
fi

# ═════════════════════════════════════════════════════════════════════════════
# 完成
# ═════════════════════════════════════════════════════════════════════════════
log "================================================================="
log " 全部步骤已触发  $(date '+%Y-%m-%d %H:%M')"
log " 产出："
log "   logs/            各步骤完整日志"
log "   checkpoints/     phys_hgnet/best_*.pth  dgnet/best_*.pth"
log "   ablation_results/ ablation_results.csv + ablation_bar.png"
log "   viz_anchors/      anchor_evolution.gif + screenshots"
log "   speed_results/    inference_speed.csv + .png"
log "   bench_scaling/    scaling_nodes.csv + .png"
log "================================================================="
