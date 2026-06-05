"""Dataset structures, transforms, and loaders.

Shared by both DGNet and PhysHGNet. Reads pde_trajectories.h5 and serves
fixed-length trajectory chunks.
"""

import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional


class DGGraph:
    """Container for one PDE trajectory on a graph mesh."""

    def __init__(self, nodes, edges, faces, node_features, source_terms,
                 initial_condition, time_points, edge_attr=None,
                 boundary_info=None, node_type=None, **kwargs):
        self.nodes = nodes
        self.edges = edges
        self.faces = faces
        self.node_features = node_features
        self.source_terms = source_terms
        self.initial_condition = initial_condition
        self.time_points = time_points
        self.edge_attr = edge_attr
        self.boundary_info = boundary_info or {}
        self.node_type = (node_type if node_type is not None
                          else torch.zeros(nodes.shape[0], dtype=torch.long,
                                           device=nodes.device))

        if boundary_info:
            self._setup_boundary_conditions(boundary_info)
        self._compute_geometric_properties()

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _setup_boundary_conditions(self, boundary_info):
        for bc_type, bc_data in boundary_info.items():
            if bc_type == 'dirichlet' and 'indices' in bc_data:
                self.node_type[bc_data['indices']] = 1
            elif bc_type == 'neumann' and 'target_indices' in bc_data:
                self.node_type[bc_data['target_indices']] = 2

    def _compute_geometric_properties(self):
        edge_i, edge_j = self.edges[:, 0], self.edges[:, 1]
        edge_vectors = self.nodes[edge_j] - self.nodes[edge_i]
        self.edge_distances = torch.norm(edge_vectors, dim=1)
        self.node_volumes = self._estimate_node_volumes()

    def _estimate_node_volumes(self):
        num_nodes = self.nodes.shape[0]
        device = self.nodes.device
        face_vertices = self.nodes[self.faces]
        v0, v1, v2 = face_vertices[:, 0, :], face_vertices[:, 1, :], face_vertices[:, 2, :]
        edge1, edge2 = v1 - v0, v2 - v0
        spatial_dim = self.nodes.shape[1]
        if spatial_dim == 2:
            triangle_areas = 0.5 * torch.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
        elif spatial_dim == 3:
            triangle_areas = 0.5 * torch.norm(torch.cross(edge1, edge2, dim=1), dim=1)
        else:
            raise ValueError(f"Volume calculation for spatial_dim={spatial_dim} not implemented.")
        area_per_vertex = triangle_areas / 3.0
        node_volumes = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        node_volumes.scatter_add_(0, self.faces.flatten(), area_per_vertex.repeat_interleave(3))
        return node_volumes


class DGPdeDataset(Dataset):
    """Load and chunk PDE trajectories from HDF5."""

    def __init__(self, data_path: str, train_time_steps: int,
                 max_samples: Optional[int] = None, rank: int = 0,
                 trajectory_keys: Optional[List[str]] = None):
        self.data_path = data_path
        self.train_time_steps = train_time_steps
        self.samples = []
        self.rank = rank
        self.trajectory_keys = trajectory_keys
        self._chunk_and_load_data()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _chunk_and_load_data(self):
        if not self.train_time_steps or self.train_time_steps <= 0:
            raise ValueError("train_time_steps must be a positive integer")

        with h5py.File(self.data_path, 'r') as f:
            traj_keys = sorted(list(f.keys()))
            if self.trajectory_keys is not None:
                missing = [k for k in self.trajectory_keys if k not in f]
                if missing:
                    raise KeyError(f"Trajectories not found in HDF5: {missing}")
                traj_keys = [k for k in traj_keys if k in self.trajectory_keys]

            for traj_key in traj_keys:
                traj_group = f[traj_key]
                nodes = torch.from_numpy(traj_group['nodes'][:]).float()
                edges = torch.from_numpy(traj_group['edges'][:]).long()
                faces = torch.from_numpy(traj_group['faces'][:]).long()
                full_node_features = torch.from_numpy(traj_group['node_features'][:]).float()
                full_source_terms = torch.from_numpy(traj_group['source_terms'][:]).float()
                full_time_points = torch.from_numpy(traj_group['time_points'][:]).float()

                boundary_info = {}
                if 'boundary_info' in traj_group:
                    bc_group = traj_group['boundary_info']
                    if 'dirichlet' in bc_group:
                        dg = bc_group['dirichlet']
                        boundary_info['dirichlet'] = {
                            'indices': torch.from_numpy(dg['indices'][:]).long(),
                            'values': torch.from_numpy(dg['values'][:]).float()
                        }
                    if 'neumann' in bc_group:
                        ng = bc_group['neumann']
                        boundary_info['neumann'] = {
                            'source_indices': torch.from_numpy(ng['source_indices'][:]).long(),
                            'target_indices': torch.from_numpy(ng['target_indices'][:]).long(),
                        }

                T = full_node_features.shape[0]
                M = self.train_time_steps
                for i in range(T // M):
                    start_idx = i * M
                    end_idx = start_idx + M
                    chunk = DGGraph(
                        nodes=nodes,
                        edges=edges,
                        faces=faces,
                        node_features=full_node_features[start_idx:end_idx],
                        source_terms=full_source_terms[start_idx:end_idx],
                        initial_condition=full_node_features[start_idx],
                        time_points=full_time_points[start_idx:end_idx],
                        boundary_info=boundary_info,
                        trajectory_id=traj_key,
                    )
                    self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def dg_collate_fn(batch_list: List[DGGraph]) -> Dict[str, Any]:
    """Pack a list of DGGraph samples into one batch dict."""
    sample = batch_list[0]
    return {
        'batch_size': len(batch_list),
        'num_nodes': sample.nodes.shape[0],
        'num_timesteps': sample.node_features.shape[0],
        'nodes': sample.nodes,
        'edges': sample.edges,
        'faces': sample.faces,
        'edge_attr': sample.edge_attr,
        'node_volumes': sample.node_volumes,
        'node_features': torch.stack([g.node_features for g in batch_list]),
        'source_terms': torch.stack([g.source_terms for g in batch_list]),
        'initial_conditions': torch.stack([g.initial_condition for g in batch_list]),
        'time_points': sample.time_points,
        'boundary_info': sample.boundary_info,
        'node_type': sample.node_type,
        'trajectory_ids': [getattr(g, 'trajectory_id', i) for i, g in enumerate(batch_list)]
    }


def create_dg_loader(dataset: DGPdeDataset, batch_size: int = 4,
                     shuffle: bool = True, num_workers: int = 2,
                     pin_memory: bool = True,
                     sampler=None, **kwargs) -> DataLoader:
    """Create a DataLoader for DG trajectory chunks."""
    if sampler is not None:
        shuffle = False
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dg_collate_fn,
        sampler=sampler,
        **kwargs
    )
