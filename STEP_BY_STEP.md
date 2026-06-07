# 所有实验的多卡分布式运行指令

## 约定

```bash
# 进入仓库目录
cd /home/caiziyue/.local/PhysHGNetv2

# 公共变量（按需修改）
GPUS="0,1,2,3,4,5,6"     # 7 张卡
N=7                        # nproc_per_node = 卡数
MAIN_N=2000
DATA=data_laser_hardening
EPOCHS=15
BATCH=4
LR=5e-4
M=64                       # m_anchors

PH_CKPT=checkpoints/phys_hgnet/best_${MAIN_N}.pth
DG_CKPT=checkpoints/dgnet/best_${MAIN_N}.pth
```

---

## [0] 数据生成（CPU，不需要多卡）

```bash
# 主实验 N=2000
python3 generate_laser_data_aligned.py \
    --n_nodes 2000 --n_traj 40 --n_timesteps 121 --dt 0.5 \
    --out_dir data_laser_hardening

# 大规模 scaling 数据（分别跑，每个约 5-20 分钟）
python3 generate_laser_data_aligned.py --n_nodes 4000  --n_traj 40 --n_timesteps 121 --dt 0.5 --out_dir data_laser_hardening
python3 generate_laser_data_aligned.py --n_nodes 6000  --n_traj 40 --n_timesteps 121 --dt 0.5 --out_dir data_laser_hardening
python3 generate_laser_data_aligned.py --n_nodes 8000  --n_traj 40 --n_timesteps 121 --dt 0.5 --out_dir data_laser_hardening
python3 generate_laser_data_aligned.py --n_nodes 10000 --n_traj 40 --n_timesteps 121 --dt 0.5 --out_dir data_laser_hardening
```

---

## [1a] 训练 PhysHGNet — 7 卡 DDP

```bash
# full config（首次运行）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    train_physhgnet.py \
    --epochs 15 --batch-size 4 --lr 5e-4 \
    --n-nodes 2000 --data-dir data_laser_hardening \
    --m-anchors 64 --train-time-steps 7

# 关掉多重网格预条件（调试消融）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    train_physhgnet.py --no-mg \
    --epochs 15 --batch-size 4 --n-nodes 2000 --data-dir data_laser_hardening

# 关掉物理锚点
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    train_physhgnet.py --no-c1 \
    --epochs 15 --batch-size 4 --n-nodes 2000 --data-dir data_laser_hardening

# 静态锚点
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    train_physhgnet.py --static-anchors \
    --epochs 15 --batch-size 4 --n-nodes 2000 --data-dir data_laser_hardening

# 关掉 C2（粗算子）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    train_physhgnet.py --no-c2 \
    --epochs 15 --batch-size 4 --n-nodes 2000 --data-dir data_laser_hardening

# 关掉 C3（双尺度 GNN）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    train_physhgnet.py --no-c3 \
    --epochs 15 --batch-size 4 --n-nodes 2000 --data-dir data_laser_hardening

# 关掉虚拟节点
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    train_physhgnet.py --no-vn \
    --epochs 15 --batch-size 4 --n-nodes 2000 --data-dir data_laser_hardening

# baseline（全关）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    train_physhgnet.py --no-c1 --no-mg --no-c2 --no-c3 --no-vn --static-anchors \
    --epochs 15 --batch-size 4 --n-nodes 2000 --data-dir data_laser_hardening
```

---

## [1b] 训练 DGNet — 7 卡 DDP

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    train_dgnet.py \
    --epochs 15 --batch-size 4 --lr 5e-4 \
    --n-nodes 2000 --data-dir data_laser_hardening
```

---

## [2] 消融实验 — 7 卡 DDP（各 config 内部全卡并行）

```bash
# 完整 8 个 config（推荐）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    ablation_experiment.py \
    --n-nodes 2000 --data-dir data_laser_hardening \
    --epochs 15 --batch-size 2 --lr 5e-4 \
    --m-anchors 64 --train-time-steps 7 \
    --configs full no_c1 static_anchor no_mg no_c2 no_c3 no_vn baseline \
    --out-dir ablation_results

# 只跑 3 个关键 config（快速验证）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    ablation_experiment.py \
    --n-nodes 2000 --data-dir data_laser_hardening \
    --epochs 5 --batch-size 2 \
    --configs full no_mg baseline \
    --out-dir ablation_results_quick
```

---

## [3] 锚点可视化（单卡，推理不需要多卡）

```bash
CUDA_VISIBLE_DEVICES=0 python3 visualize_anchors.py \
    --n-nodes 2000 --data-dir data_laser_hardening \
    --traj-index 0 --update-freq 1 --max-steps 80 \
    --fps 8 --out-dir viz_anchors \
    --ckpt checkpoints/phys_hgnet/best_2000.pth
```

---

## [4] 推理速度（单卡，计时要求单进程）

```bash
CUDA_VISIBLE_DEVICES=0 python3 inference_speed.py \
    --n-nodes 2000 --data-dir data_laser_hardening \
    --traj-index -1 --repeats 10 --warmup 3 \
    --m-anchors 64 --out-dir speed_results \
    --physhgnet-ckpt checkpoints/phys_hgnet/best_2000.pth \
    --dgnet-ckpt     checkpoints/dgnet/best_2000.pth
```

> 推理速度测试**必须单卡单进程**，多卡会引入通信开销，测出的延迟不反映真实推理性能。

---

## [5a] 大规模 N-scaling（单卡，DGNet OOM 捕获）

```bash
CUDA_VISIBLE_DEVICES=0 python3 benchmark_scaling.py \
    --mode nodes \
    --node-list 4000 6000 8000 10000 \
    --data-dir data_laser_hardening \
    --t-bench 11 --batch-size 1 --repeats 3 \
    --out-dir bench_scaling \
    --physhgnet-ckpt checkpoints/phys_hgnet/best_2000.pth \
    --dgnet-ckpt     checkpoints/dgnet/best_2000.pth
```

---

## [5b] 锚点数量 scaling（单卡）

```bash
CUDA_VISIBLE_DEVICES=0 python3 benchmark_scaling.py \
    --mode anchor \
    --n-nodes 8000 \
    --anchors 32 64 128 256 512 \
    --data-dir data_laser_hardening \
    --t-bench 11 --repeats 3 \
    --out-dir bench_scaling \
    --physhgnet-ckpt checkpoints/phys_hgnet/best_2000.pth
```

---

## 快速调试命令

```bash
# 冒烟测试：3 epoch，只用 2 条轨迹数据，验证整条链路通畅
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    train_physhgnet.py --epochs 3 --batch-size 2 \
    --n-nodes 2000 --data-dir data_laser_hardening

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 \
    train_dgnet.py --epochs 3 --batch-size 2 \
    --n-nodes 2000 --data-dir data_laser_hardening

# 验证模型可实例化
python3 -c "
from dgnet import DGNet; from phys_hgnet import PhysHGNet
dg = DGNet({'spatial_dim':2,'feature_dim':1,'output_dim':1})
ph = PhysHGNet({'use_physics_anchor':True,'use_mg_precond':True})
print('DGNet  参数量:', f'{dg.num_parameters():,}')
print('PhysHG 参数量:', f'{ph.num_parameters():,}')
"

# 检查 checkpoint
python3 -c "
import torch, sys
ck = torch.load(sys.argv[1], map_location='cpu')
print('val_mse:', ck.get('val_mse'))
print('val_rne:', ck.get('val_rne'))
print('config:', ck.get('config'))
" checkpoints/phys_hgnet/best_2000.pth

# 检查数据文件
python3 -c "
import h5py, sys
with h5py.File(sys.argv[1]) as f:
    ks = list(f.keys()); g = f[ks[0]]
    print(f'轨迹数={len(ks)}')
    for k in g: print(f'  {k}: {g[k].shape}')
" data_laser_hardening/pde_trajectories_2000.h5
```

---

## 一键运行（run_all.sh）

```bash
# 完整流程
GPUS=0,1,2,3,4,5,6 bash run_all.sh

# 冒烟测试（约 15 分钟验证全流程）
QUICK=1 GPUS=0,1,2,3,4,5,6 bash run_all.sh

# 数据已有，只训练 + 评测
SKIP_DATA=1 GPUS=0,1,2,3,4,5,6 bash run_all.sh

# 训练已有，只跑 4 个评测实验
SKIP_DATA=1 SKIP_TRAIN=1 GPUS=0,1,2,3,4,5,6 bash run_all.sh

# 只跑单步（1=训练 2=消融 3=可视化 4=速度 5=scaling）
ONLY=2 GPUS=0,1,2,3,4,5,6 bash run_all.sh
ONLY=4 GPUS=0,1,2,3,4,5,6 bash run_all.sh
```

---

## 哪些步骤用多卡，哪些只用单卡？

| 步骤 | 多卡 or 单卡 | 原因 |
|---|---|---|
| [0] 数据生成 | 单卡（CPU 任务） | 无 GPU 需求 |
| [1] 训练 | **7 卡 DDP** | 数据量大，加速训练 |
| [2] 消融 | **7 卡 DDP** | 每个 config 内部 DDP，训 8 个 config |
| [3] 可视化 | 单卡 | 推理 + 画图，单卡够用 |
| [4] 推理速度 | **必须单卡** | 计时要排除通信开销 |
| [5] Scaling | 单卡 | 基准测试需要单进程计时 |
