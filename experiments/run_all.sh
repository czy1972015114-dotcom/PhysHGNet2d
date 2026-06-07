#!/usr/bin/env bash
# =============================================================================
# run_all.sh —— PhysHGNet vs DGNet 全实验自动化运行脚本（7 卡 DDP）
#
# 用法：
#   bash run_all.sh <step>
# step 取值：
#   all   —— 依次跑 data -> 2.1 -> 2.2 -> 2.3 -> 2.4 -> 2.5 -> 2.6
#   data  —— 2.0 数据生成（含 2.5 的未知测试集）
#   2.1   —— 基础测试（N=2000，两模型各 40 epoch）
#   2.2   —— PhysHGNet 消融（N=2000，各 20 epoch）
#   2.3   —— scaling（各 N，两模型各 40 epoch，OOM 如实记录）
#   2.4   —— 锚点 scaling（N=4000，m=64/128/256/512，仅 PhysHGNet，各 20 epoch）
#   2.5   —— 推理基准（N=4000，未知轨迹：误差 / 速度 / 显存）
#   2.6   —— 锚点可视化（N=4000 最佳 PhysHGNet）
#
# 注意：本脚本需在仓库根目录下运行，且 experiments/ 内的脚本随仓库一同放置。
#       7 卡 DDP 通过 torchrun --nproc_per_node=7 + CUDA_VISIBLE_DEVICES 实现。
# =============================================================================

set -u  # 不使用 -e：训练 OOM 时我们要捕获并继续，而非中断整个脚本

# ---------------- 全局配置（可用环境变量覆盖）----------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
NGPU="${NGPU:-7}"
MASTER_PORT="${MASTER_PORT:-29511}"

EPOCHS_BASE="${EPOCHS_BASE:-40}"     # 2.1 / 2.3
EPOCHS_ABL="${EPOCHS_ABL:-20}"       # 2.2 / 2.4
BATCH="${BATCH:-4}"                  # 默认 batch（每卡）
SCALE_BATCH="${SCALE_BATCH:-2}"      # 大 N 时降到的 batch
LR="${LR:-5e-4}"

DATA_DIR="${DATA_DIR:-data_laser_hardening}"
TEST_DIR="${TEST_DIR:-data_laser_hardening_test}"
RESULTS="${RESULTS:-results}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints}"
LOG_DIR="${LOG_DIR:-logs}"

N_TRAJ="${N_TRAJ:-40}"               # 每个 N 生成的轨迹数（训练集）
N_TEST_TRAJ="${N_TEST_TRAJ:-10}"     # 2.5 未知测试集轨迹数
N_TIMESTEPS="${N_TIMESTEPS:-121}"
DT="${DT:-0.5}"

PY="${PY:-python}"
TORCHRUN="torchrun --nproc_per_node=${NGPU} --master_port=${MASTER_PORT}"

mkdir -p "$RESULTS" "$LOG_DIR"

ALL_N=(2000 4000 6000 8000 10000 12000)

# ---------------- 工具函数 ----------------
log()  { echo -e "\033[1;34m[run_all]\033[0m $*"; }
warn() { echo -e "\033[1;33m[run_all]\033[0m $*"; }

# 运行一条训练命令并捕获 OOM；参数：<日志文件> <model> <n_nodes> <命令...>
run_with_oom_guard() {
  local logf="$1"; shift
  local model="$1"; shift
  local n="$1"; shift
  log "执行: $* (log -> $logf)"
  "$@" 2>&1 | tee "$logf"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    if grep -qiE "out of memory|CUDA out of memory|OutOfMemory" "$logf"; then
      warn "检测到 OOM: model=$model N=$n —— 如实记录到 $RESULTS/oom_${model}_${n}.json"
      cat > "$RESULTS/oom_${model}_${n}.json" <<EOF
{"experiment": "${model}_${n}", "model": "${model}", "n_nodes": ${n}, "split": "val", "status": "OOM"}
EOF
    else
      warn "命令非 OOM 失败（rc=$rc）: model=$model N=$n，请查看 $logf"
    fi
    return "$rc"
  fi
  return 0
}

batch_for_n() {  # 大 N 自动降 batch
  local n="$1"
  if [ "$n" -ge 8000 ]; then echo "$SCALE_BATCH"; else echo "$BATCH"; fi
}

# ---------------- 2.0 数据生成 ----------------
gen_data() {
  log "===== 2.0 数据生成（训练集，$DATA_DIR）====="
  for n in "${ALL_N[@]}"; do
    if [ -f "$DATA_DIR/pde_trajectories_${n}.h5" ]; then
      log "已存在 $DATA_DIR/pde_trajectories_${n}.h5，跳过"
    else
      $PY generate_laser_data_aligned.py \
        --n_nodes "$n" --n_traj "$N_TRAJ" --n_timesteps "$N_TIMESTEPS" \
        --dt "$DT" --out_dir "$DATA_DIR" --seed 42 --no_store_L \
        2>&1 | tee "$LOG_DIR/gen_${n}.log"
    fi
  done

  log "===== 2.0 生成未知测试集（N=4000, seed=999, $TEST_DIR）用于 2.5 ====="
  if [ -f "$TEST_DIR/pde_trajectories_4000.h5" ]; then
    log "已存在 $TEST_DIR/pde_trajectories_4000.h5，跳过"
  else
    $PY generate_laser_data_aligned.py \
      --n_nodes 4000 --n_traj "$N_TEST_TRAJ" --n_timesteps "$N_TIMESTEPS" \
      --dt "$DT" --out_dir "$TEST_DIR" --seed 999 --no_store_L \
      2>&1 | tee "$LOG_DIR/gen_test_4000.log"
  fi
}

# ---------------- 训练封装 ----------------
# train_phys <exp_name> <n_nodes> <epochs> <m_anchors> [额外消融flag...]
train_phys() {
  local exp="$1"; local n="$2"; local ep="$3"; local m="$4"; shift 4
  local bs; bs=$(batch_for_n "$n")
  run_with_oom_guard "$LOG_DIR/train_${exp}_${n}.log" "physhgnet" "$n" \
    $TORCHRUN train_physhgnet.py \
      --exp-name "$exp" --epochs "$ep" --batch-size "$bs" --lr "$LR" \
      --m-anchors "$m" --n-nodes "$n" --data-dir "$DATA_DIR" "$@"
}

# train_dgnet <exp_name> <n_nodes> <epochs>
train_dgnet() {
  local exp="$1"; local n="$2"; local ep="$3"
  local bs; bs=$(batch_for_n "$n")
  run_with_oom_guard "$LOG_DIR/train_${exp}_${n}.log" "dgnet" "$n" \
    $TORCHRUN train_dgnet.py \
      --exp-name "$exp" --epochs "$ep" --batch-size "$bs" --lr "$LR" \
      --n-nodes "$n" --data-dir "$DATA_DIR"
}

# eval_model <physhgnet|dgnet> <ckpt> <n_nodes> <out_json> [--data-path X --split Y]
eval_model() {
  local model="$1"; local ckpt="$2"; local n="$3"; local out="$4"; shift 4
  if [ ! -f "$ckpt" ]; then
    warn "ckpt 不存在（可能因 OOM 未训出）: $ckpt —— 跳过评测"
    return 0
  fi
  $PY experiments/eval_metrics.py --model "$model" --ckpt "$ckpt" \
    --n-nodes "$n" --data-dir "$DATA_DIR" --out "$out" "$@" \
    2>&1 | tee "$LOG_DIR/eval_$(basename "$out" .json).log"
}

# ---------------- 2.1 基础测试 ----------------
run_21() {
  log "===== 2.1 基础测试 N=2000，两模型各 ${EPOCHS_BASE} epoch ====="
  train_phys  "phys_hgnet" 2000 "$EPOCHS_BASE" 64
  train_dgnet "dgnet"      2000 "$EPOCHS_BASE"
  eval_model physhgnet "$CKPT_ROOT/phys_hgnet/best_2000.pth" 2000 "$RESULTS/eval_2.1_phys_2000.json" --split val
  eval_model dgnet     "$CKPT_ROOT/dgnet/best_2000.pth"      2000 "$RESULTS/eval_2.1_dgnet_2000.json" --split val
}

# ---------------- 2.2 消融实验 ----------------
run_22() {
  log "===== 2.2 PhysHGNet 消融 N=2000，各 ${EPOCHS_ABL} epoch ====="
  # full
  train_phys "phys_hgnet_full" 2000 "$EPOCHS_ABL" 64
  eval_model physhgnet "$CKPT_ROOT/phys_hgnet_full/best_2000.pth" 2000 "$RESULTS/eval_2.2_full_2000.json" --split val
  # -C1 物理感知锚点
  train_phys "phys_hgnet_noC1" 2000 "$EPOCHS_ABL" 64 --no-c1
  eval_model physhgnet "$CKPT_ROOT/phys_hgnet_noC1/best_2000.pth" 2000 "$RESULTS/eval_2.2_noC1_2000.json" --split val
  # -C2 可学习粗算子
  train_phys "phys_hgnet_noC2" 2000 "$EPOCHS_ABL" 64 --no-c2
  eval_model physhgnet "$CKPT_ROOT/phys_hgnet_noC2/best_2000.pth" 2000 "$RESULTS/eval_2.2_noC2_2000.json" --split val
  # -C3 双尺度 GNN 修正
  train_phys "phys_hgnet_noC3" 2000 "$EPOCHS_ABL" 64 --no-c3
  eval_model physhgnet "$CKPT_ROOT/phys_hgnet_noC3/best_2000.pth" 2000 "$RESULTS/eval_2.2_noC3_2000.json" --split val
  # -VN 虚拟节点
  train_phys "phys_hgnet_noVN" 2000 "$EPOCHS_ABL" 64 --no-vn
  eval_model physhgnet "$CKPT_ROOT/phys_hgnet_noVN/best_2000.pth" 2000 "$RESULTS/eval_2.2_noVN_2000.json" --split val
  # baseline：去掉 C1+C2+C3
  train_phys "phys_hgnet_baseline" 2000 "$EPOCHS_ABL" 64 --no-c1 --no-c2 --no-c3
  eval_model physhgnet "$CKPT_ROOT/phys_hgnet_baseline/best_2000.pth" 2000 "$RESULTS/eval_2.2_baseline_2000.json" --split val
}

# ---------------- 2.3 scaling 测试 ----------------
run_23() {
  log "===== 2.3 scaling，各 N，两模型各 ${EPOCHS_BASE} epoch（OOM 如实记录）====="
  for n in "${ALL_N[@]}"; do
    log "----- N=$n -----"
    # PhysHGNet
    if [ -f "$CKPT_ROOT/phys_hgnet/best_${n}.pth" ]; then
      log "PhysHGNet best_${n}.pth 已存在，跳过训练"
    else
      train_phys "phys_hgnet" "$n" "$EPOCHS_BASE" 64
    fi
    eval_model physhgnet "$CKPT_ROOT/phys_hgnet/best_${n}.pth" "$n" "$RESULTS/eval_2.3_phys_${n}.json" --split val
    # DGNet（大 N 可能 OOM）
    if [ -f "$CKPT_ROOT/dgnet/best_${n}.pth" ]; then
      log "DGNet best_${n}.pth 已存在，跳过训练"
    else
      train_dgnet "dgnet" "$n" "$EPOCHS_BASE"
    fi
    eval_model dgnet "$CKPT_ROOT/dgnet/best_${n}.pth" "$n" "$RESULTS/eval_2.3_dgnet_${n}.json" --split val
  done
}

# ---------------- 2.4 锚点 scaling ----------------
run_24() {
  log "===== 2.4 锚点 scaling N=4000，m=64/128/256/512，仅 PhysHGNet，各 ${EPOCHS_ABL} epoch ====="
  for m in 64 128 256 512; do
    log "----- m_anchors=$m -----"
    train_phys "phys_hgnet_m${m}" 4000 "$EPOCHS_ABL" "$m"
    eval_model physhgnet "$CKPT_ROOT/phys_hgnet_m${m}/best_4000.pth" 4000 \
      "$RESULTS/eval_2.4_m${m}_4000.json" --split val
  done
}

# ---------------- 2.5 推理基准 ----------------
run_25() {
  log "===== 2.5 推理基准 N=4000（未知轨迹）====="
  local phys_ckpt="$CKPT_ROOT/phys_hgnet/best_4000.pth"
  local dgnet_ckpt="$CKPT_ROOT/dgnet/best_4000.pth"
  if [ ! -f "$phys_ckpt" ] || [ ! -f "$dgnet_ckpt" ]; then
    warn "缺少 N=4000 的 best ckpt（需先跑 2.3 的 N=4000）。phys=$phys_ckpt dgnet=$dgnet_ckpt"
  fi
  $PY experiments/benchmark_inference.py \
    --phys-ckpt "$phys_ckpt" --dgnet-ckpt "$dgnet_ckpt" \
    --data-path "$TEST_DIR/pde_trajectories_4000.h5" \
    --batch-size 1 --warmup 2 --out "$RESULTS/benchmark_4000.json" \
    2>&1 | tee "$LOG_DIR/benchmark_4000.log"
}

# ---------------- 2.6 锚点可视化 ----------------
run_26() {
  log "===== 2.6 锚点可视化 N=4000（最佳 PhysHGNet）====="
  local phys_ckpt="$CKPT_ROOT/phys_hgnet/best_4000.pth"
  if [ ! -f "$phys_ckpt" ]; then
    warn "缺少 $phys_ckpt（需先跑 2.3 的 N=4000）"
  fi
  $PY experiments/visualize_anchors.py \
    --ckpt "$phys_ckpt" --n-nodes 4000 \
    --data-path "$DATA_DIR/pde_trajectories_4000.h5" \
    --traj-index 0 --time-window 60 --stride 2 \
    --coarse-update-freq 1 --out-dir "$RESULTS" \
    2>&1 | tee "$LOG_DIR/visualize_anchors_4000.log"
}

collect() {
  $PY experiments/collect_results.py --results-dir "$RESULTS" || true
}

# ---------------- 调度 ----------------
STEP="${1:-all}"
case "$STEP" in
  data) gen_data ;;
  2.1)  run_21; collect ;;
  2.2)  run_22; collect ;;
  2.3)  run_23; collect ;;
  2.4)  run_24; collect ;;
  2.5)  run_25; collect ;;
  2.6)  run_26 ;;
  all)
    gen_data
    run_21
    run_22
    run_23
    run_24
    run_25
    run_26
    collect
    ;;
  *)
    echo "未知步骤: $STEP"
    echo "可选: all | data | 2.1 | 2.2 | 2.3 | 2.4 | 2.5 | 2.6"
    exit 1
    ;;
esac

log "步骤 '$STEP' 完成。结果见 $RESULTS/ ，日志见 $LOG_DIR/ 。"
