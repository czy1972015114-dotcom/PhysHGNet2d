"""
heat_equation_benchmark.py — Self-contained comparison on 2D heat equation.

Compares Original DGNet (dense O(N²)) vs Structured DGNet (sparse O(E+Nm)).

Usage:
    python heat_equation_benchmark.py --model structured --n_nodes 2000 --epochs 50
    python heat_equation_benchmark.py --model original   --n_nodes 2000 --epochs 50
    python heat_equation_benchmark.py --scaling_test
    python heat_equation_benchmark.py --scaling_test --include_large
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import json
import os


# ══════════════════════════════════════════════════════════════
# Data Generation
# ══════════════════════════════════════════════════════════════

def generate_mesh_2d(n_nodes, seed=42):
    torch.manual_seed(seed)
    coords = torch.rand(n_nodes, 2)
    k = min(8, n_nodes - 1)
    dist = torch.cdist(coords, coords)
    dist.fill_diagonal_(float('inf'))
    _, knn = dist.topk(k, dim=1, largest=False)

    edges_set = set()
    src = torch.arange(n_nodes).unsqueeze(1).expand(n_nodes, k).reshape(-1)
    dst = knn.reshape(-1)
    for s, d in zip(src.tolist(), dst.tolist()):
        edges_set.add((min(s, d), max(s, d)))
    edges = torch.tensor(list(edges_set), dtype=torch.long)

    # Faces (simplified)
    faces_list = []
    for i in range(min(n_nodes, 200)):
        nb = knn[i].tolist()
        for j in range(len(nb)):
            for l in range(j + 1, len(nb)):
                if nb[j] in knn[nb[l]].tolist():
                    faces_list.append(sorted([i, nb[j], nb[l]]))
    if not faces_list:
        faces_list = [[0, 1, 2]]
    faces = torch.tensor(faces_list[:n_nodes * 2], dtype=torch.long)

    vols = torch.ones(n_nodes) / n_nodes
    ntype = torch.zeros(n_nodes, dtype=torch.long)
    bnd = (coords[:, 0] < 0.02) | (coords[:, 0] > 0.98) | \
          (coords[:, 1] < 0.02) | (coords[:, 1] > 0.98)
    ntype[bnd] = 1
    return coords, edges, faces, vols, ntype


def build_graph_laplacian(coords, edges, kappa=1.0):
    N = coords.shape[0]
    rev = edges.flip(1)
    bi = torch.cat([edges, rev], 0)
    d = (coords[bi[:, 1]] - coords[bi[:, 0]]).norm(dim=-1).clamp(min=1e-8)
    w = kappa / d
    L = torch.zeros(N, N)
    L[bi[:, 0], bi[:, 1]] = w
    L[range(N), range(N)] = -L.sum(1)
    return L


def generate_heat_data(n_nodes, n_traj, n_steps=30, dt=0.01, kappa=0.05, seed=42):
    coords, edges, faces, vols, ntype = generate_mesh_2d(n_nodes, seed)
    N = coords.shape[0]
    L = build_graph_laplacian(coords, edges, kappa)
    I = torch.eye(N)
    A = I - (dt / 2) * L
    B_op = I + (dt / 2) * L
    lu, piv = torch.linalg.lu_factor(A)

    batches = []
    for ti in range(n_traj):
        torch.manual_seed(seed + ti + 1000)
        cx, cy = 0.3 + 0.4 * torch.rand(2)
        sig = 0.08 + 0.04 * torch.rand(1).item()
        r2 = (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2
        u0 = torch.exp(-r2 / (2 * sig ** 2))
        src = 0.01 * torch.randn(n_steps, N)
        traj = torch.zeros(n_steps, N)
        traj[0] = u0
        u = u0
        for t in range(n_steps - 1):
            rhs = B_op @ u + (dt / 2) * (src[t] + src[t + 1])
            u = torch.linalg.lu_solve(lu, piv, rhs.unsqueeze(-1)).squeeze(-1)
            traj[t + 1] = u

        bnd_idx = (ntype == 1).nonzero(as_tuple=True)[0]
        binfo = {}
        if len(bnd_idx) > 0:
            binfo = {'dirichlet': {'indices': bnd_idx,
                                   'values': torch.zeros(len(bnd_idx))}}
        batches.append({
            'nodes': coords, 'edges': edges, 'faces': faces,
            'node_volumes': vols, 'node_type': ntype, 'L_physics': L,
            'initial_conditions': traj[0:1].unsqueeze(0).unsqueeze(-1),
            'source_terms': src.unsqueeze(0).unsqueeze(-1),
            'time_points': torch.arange(n_steps, dtype=torch.float32) * dt,
            'node_features': traj.unsqueeze(0).unsqueeze(-1),
            'boundary_info': binfo,
        })
    return batches


# ══════════════════════════════════════════════════════════════
# Original DGNet (simplified) — dense N×N operator
#
# This faithfully mirrors the O(N²) memory bottleneck of the
# real DGNet's OperatorCorrector:
#   GNN → per-edge δ values → dense N×N δL → dense LU
# ══════════════════════════════════════════════════════════════

from torch_geometric.nn import MessagePassing


class _OrigMPNN(MessagePassing):
    def __init__(self, dim, aggr='mean'):
        super().__init__(aggr=aggr)
        self.mlp = nn.Sequential(nn.Linear(3 * dim, dim), nn.ReLU(),
                                  nn.Linear(dim, dim))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        if edge_attr.shape[-1] != x_i.shape[-1]:
            edge_attr = nn.functional.pad(
                edge_attr, (0, x_i.shape[-1] - edge_attr.shape[-1]))
        return self.mlp(torch.cat([x_i, x_j, edge_attr], -1))


class OriginalDGNet(nn.Module):
    """
    Simplified Original DGNet — uses DENSE N×N correction matrix.
    At large N this causes O(N²) memory and O(N³) solve time.
    """
    def __init__(self, config):
        super().__init__()
        sd = config.get('spatial_dim', 2)
        hd = config.get('operator_hidden_dim', 32)
        self.output_dim = config.get('output_dim', 1)

        # Operator corrector GNN
        self.node_enc = nn.Sequential(
            nn.Linear(sd + 1, hd), nn.ReLU(), nn.Linear(hd, hd))
        self.edge_enc = nn.Sequential(
            nn.Linear(sd + 1, hd), nn.ReLU(), nn.Linear(hd, hd))
        self.mp_layers = nn.ModuleList([_OrigMPNN(hd) for _ in range(2)])
        self.mp_norms = nn.ModuleList([nn.LayerNorm(hd) for _ in range(2)])
        self.edge_corr = nn.Sequential(
            nn.Linear(3 * hd, hd), nn.ReLU(), nn.Linear(hd, 1))
        self.scale = nn.Parameter(torch.tensor(1e-3))

        # Nonlinear dynamics GNN
        self.nl_enc = nn.Sequential(
            nn.Linear(sd + self.output_dim, hd), nn.ReLU(), nn.Linear(hd, hd))
        self.nl_mp = nn.ModuleList([_OrigMPNN(hd) for _ in range(2)])
        self.nl_norms = nn.ModuleList([nn.LayerNorm(hd) for _ in range(2)])
        self.nl_dec = nn.Sequential(
            nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, self.output_dim))

    def forward(self, batch):
        nodes = batch['nodes']
        edges = batch['edges']
        st = batch['source_terms']
        ic = batch['initial_conditions']
        tp = batch['time_points']
        Lp = batch['L_physics']
        device = nodes.device
        B, T, N, C = st.shape
        dt = float(tp[1] - tp[0]) if T > 1 else 0.01

        # ─── Operator correction → DENSE N×N (the bottleneck) ───
        bi_e = torch.cat([edges, edges.flip(1)], 0).T.to(device)
        ea_raw = torch.cat([
            nodes[bi_e[1]] - nodes[bi_e[0]],
            (nodes[bi_e[1]] - nodes[bi_e[0]]).norm(dim=-1, keepdim=True)], -1)

        h = self.node_enc(torch.cat([nodes, batch['node_volumes'].unsqueeze(-1)], -1))
        ea_enc = self.edge_enc(ea_raw)
        for layer, norm in zip(self.mp_layers, self.mp_norms):
            h = h + layer(h, bi_e, ea_enc)
            h = norm(h)

        src, dst = bi_e
        corr_in = torch.cat([h[src], h[dst], ea_enc], -1)
        dv = self.scale * torch.tanh(self.edge_corr(corr_in).squeeze(-1))

        # ★ DENSE N×N matrix — THIS is the O(N²) memory cost ★
        dL = torch.zeros(N, N, device=device)
        dL[src, dst] = dv
        dL = dL - torch.diag(dL.sum(1))

        L_h = Lp + dL  # Dense N×N

        I_mat = torch.eye(N, device=device)
        A_mat = I_mat - (dt / 2) * L_h    # Dense N×N
        B_mat = I_mat + (dt / 2) * L_h    # Dense N×N

        lu, piv = torch.linalg.lu_factor(A_mat)

        u_cur = ic[:, 0] if ic.dim() == 4 else ic
        steps = [u_cur]

        for t in range(T - 1):
            # Nonlinear dynamics
            nl_h = self.nl_enc(torch.cat([
                nodes.unsqueeze(0).expand(B, -1, -1), u_cur], -1))
            for layer, norm in zip(self.nl_mp, self.nl_norms):
                nl_h = nl_h + layer(nl_h, bi_e,
                    ea_enc.unsqueeze(0).expand(B, -1, -1).reshape(B * ea_enc.shape[0], -1)
                    if False else ea_enc)
                nl_h = norm(nl_h)
            r_uk = self.nl_dec(nl_h)

            # Build RHS and solve
            ch_list = []
            for c in range(C):
                b_list = []
                for b in range(B):
                    rhs_bc = (B_mat @ u_cur[b, :, c]) + \
                             (dt / 2) * (st[b, t, :, c] + st[b, t+1, :, c]) + \
                             dt * r_uk[b, :, c]
                    sol = torch.linalg.lu_solve(lu, piv, rhs_bc.unsqueeze(-1)).squeeze(-1)
                    b_list.append(sol)
                ch_list.append(torch.stack(b_list, 0))
            u_next = torch.stack(ch_list, -1)
            steps.append(u_next)
            u_cur = u_next.detach()

        return {'u_final': torch.stack(steps, 1)}


class SimpleLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        u = pred['u_final']
        l1 = self.mse(u[:, 1], target[:, 1])
        lT = self.mse(u[:, -1], target[:, -1])
        return {'total_loss': l1 + lT,
                'first_step_loss': l1.item(),
                'final_step_loss': lT.item()}


# ══════════════════════════════════════════════════════════════
# Benchmark Runner
# ══════════════════════════════════════════════════════════════

def run_single(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 64)
    print(f"  N={args.n_nodes}  Model={args.model}  Device={device}")
    print("=" * 64)

    # Data
    print("\n[1] Generating data...")
    t0 = time.time()
    batches = generate_heat_data(args.n_nodes, args.n_train + args.n_val,
                                  args.n_timesteps, seed=42)
    print(f"    {len(batches)} traj, N={args.n_nodes}, T={args.n_timesteps}, "
          f"{time.time()-t0:.1f}s")
    train_b = batches[:args.n_train]
    val_b   = batches[args.n_train:]

    # Model
    print("\n[2] Building model...")
    cfg = {
        'spatial_dim': 2, 'feature_dim': 1, 'output_dim': 1,
        'operator_hidden_dim': 32, 'operator_num_layers': 2,
        'residual_hidden_dim': 64, 'residual_num_layers': 3,
        'm_anchors': args.m_anchors, 'q_local': 4, 'k_coarse': 6,
        'cg_max_iter': 100, 'cg_tol': 1e-8, 'tau': 0.1,
        'coarse_num_layers': 2, 'L_physics_scale': 1.0,
    }

    if args.model == "structured":
        from structured_dgnet import StructuredDGNet, StructuredDGNetLoss
        model = StructuredDGNet(cfg).to(device)
        loss_fn = StructuredDGNetLoss(cfg)
        mname = f"Structured (m={args.m_anchors})"
    else:
        model = OriginalDGNet(cfg).to(device)
        loss_fn = SimpleLoss(cfg)
        mname = "Original (dense N×N)"

    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    {mname}   params={npar:,}")

    # Optimizer — separate lr for alpha gates if structured
    if args.model == "structured" and hasattr(model, 'raw_alpha_loc'):
        alpha_p = [model.raw_alpha_loc, model.raw_alpha_coarse]
        alpha_ids = {id(p) for p in alpha_p}
        other_p = [p for p in model.parameters()
                   if id(p) not in alpha_ids and p.requires_grad]
        opt = torch.optim.Adam([
            {'params': other_p, 'lr': args.lr},
            {'params': alpha_p, 'lr': args.lr * 10}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    print(f"\n[3] Training {args.epochs} epochs...")
    print(f"    {'Ep':>4} | {'Loss':>10} | {'RelErr':>10} | {'Time':>6} | {'GPU MB':>8}")
    print("    " + "-" * 55)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    best, losses, etimes = float('inf'), [], []

    for ep in range(args.epochs):
        model.train()
        eloss, t0 = 0.0, time.time()

        for batch in train_b:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
            opt.zero_grad()
            try:
                pred = model(bd)
                ld = loss_fn(pred, bd['node_features'])
                ld['total_loss'].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                eloss += ld['total_loss'].item()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n    *** OOM at epoch {ep+1}! ***")
                    torch.cuda.empty_cache()
                    return {"model": mname, "n_nodes": args.n_nodes,
                            "status": "OOM", "epoch": ep, "n_params": npar}
                raise

        eloss /= max(len(train_b), 1)
        dt_ep = time.time() - t0
        etimes.append(dt_ep)
        losses.append(eloss)
        best = min(best, eloss)
        peak = torch.cuda.max_memory_allocated(device) / 1024**2 \
               if device.type == 'cuda' else 0

        # Validate
        model.eval()
        verr = 0.0
        with torch.no_grad():
            for batch in val_b:
                bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
                pred = model(bd)
                tgt = bd['node_features']
                verr += (torch.norm(pred['u_final'][:, -1] - tgt[:, -1]) /
                         (torch.norm(tgt[:, -1]) + 1e-8)).item()
        verr /= max(len(val_b), 1)

        if (ep + 1) % max(1, args.epochs // 10) == 0 or ep == 0:
            print(f"    {ep+1:>4} | {eloss:>10.6f} | {verr:>10.6f} | "
                  f"{dt_ep:>5.1f}s | {peak:>7.0f}")

    avg_t = np.mean(etimes) if etimes else 0
    print(f"\n{'='*64}")
    print(f"  RESULTS — {mname} (N={args.n_nodes})")
    print(f"  Best loss:    {best:.8f}")
    print(f"  Final error:  {verr:.6f}")
    print(f"  Peak GPU:     {peak:.0f} MB")
    print(f"  Avg time/ep:  {avg_t:.2f}s")
    print(f"  Params:       {npar:,}")
    print(f"{'='*64}")

    return {"model": mname, "n_nodes": args.n_nodes, "status": "OK",
            "best_loss": best, "final_error": verr,
            "peak_gpu_mb": peak, "avg_time": avg_t, "n_params": npar}


# ══════════════════════════════════════════════════════════════
# Scaling Test
# ══════════════════════════════════════════════════════════════

def run_scaling(args):
    print("\n" + "=" * 70)
    print("  SCALING TEST: Original (Dense) vs Structured (Sparse+LowRank)")
    print("=" * 70)

    Ns = [500, 1000, 2000, 5000]
    if args.include_large:
        Ns += [10000, 20000]

    results = []
    for N in Ns:
        for mt in ["original", "structured"]:
            print(f"\n{'─'*50}")
            print(f"  N={N}, Model={mt}")
            print(f"{'─'*50}")
            args.n_nodes = N
            args.model = mt
            args.epochs = 10
            args.m_anchors = min(64, max(8, N // 16))
            try:
                r = run_single(args)
            except Exception as e:
                print(f"  FAILED: {e}")
                r = {"model": mt, "n_nodes": N,
                     "status": f"FAIL:{str(e)[:40]}"}
            results.append(r)

    # Summary
    print("\n\n" + "=" * 85)
    print("  SCALING SUMMARY")
    print("=" * 85)
    print(f"  {'N':>6} | {'Model':>25} | {'Status':>6} | "
          f"{'Loss':>10} | {'GPU MB':>8} | {'Time':>8} | {'Params':>10}")
    print("  " + "-" * 82)
    for r in results:
        s = r.get('status', '?')
        ls = f"{r.get('best_loss',0):.8f}" if s == "OK" else s[:10]
        gp = f"{r.get('peak_gpu_mb',0):.0f}" if s == "OK" else "-"
        tm = f"{r.get('avg_time',0):.2f}s" if s == "OK" else "-"
        pr = f"{r.get('n_params',0):,}" if 'n_params' in r else "-"
        print(f"  {r['n_nodes']:>6} | {r['model']:>25} | {s:>6} | "
              f"{ls:>10} | {gp:>8} | {tm:>8} | {pr:>10}")
    print("=" * 85)

    with open("scaling_results.json", 'w') as f:
        safe = [{k: v for k, v in r.items() if k != 'train_losses'}
                for r in results]
        json.dump(safe, f, indent=2, default=str)
    print(f"\n  Saved to scaling_results.json")


# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["original", "structured"], default="structured")
    p.add_argument("--n_nodes", type=int, default=2000)
    p.add_argument("--n_train", type=int, default=8)
    p.add_argument("--n_val", type=int, default=2)
    p.add_argument("--n_timesteps", type=int, default=30)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--m_anchors", type=int, default=32)
    p.add_argument("--scaling_test", action="store_true")
    p.add_argument("--include_large", action="store_true")
    args = p.parse_args()

    if args.scaling_test:
        run_scaling(args)
    else:
        run_single(args)
