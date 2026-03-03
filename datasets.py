"""PyTorch Dataset classes for AIS trajectory data."""

import numpy as np
import torch
from torch.utils.data import Dataset


class AISDataset(Dataset):
    """Dataset for normalized AIS trajectories.

    Each element is a dict with:
        "mmsi": vessel identifier (int)
        "traj": (N, 5) array of [LAT, LON, SOG, COG, TIMESTAMP],
                where LAT/LON/SOG/COG are normalized to [0, 1).
    """

    def __init__(self, l_data, max_seqlen=96, dtype=torch.float32,
                 device=torch.device("cpu")):
        self.max_seqlen = max_seqlen
        self.device = device
        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        V = self.l_data[idx]
        m_v = V["traj"][:, :4]  # lat, lon, sog, cog
        m_v = np.clip(m_v, 0.0, 0.9999)

        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen, 4), dtype=np.float32)
        seq[:seqlen, :] = m_v[:seqlen, :]
        seq = torch.tensor(seq, dtype=torch.float32)

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.0

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(int(V["mmsi"]), dtype=torch.int)
        time_start = torch.tensor(int(V["traj"][0, 4]), dtype=torch.long)

        return seq, mask, seqlen, mmsi, time_start


class AISDataset_grad(Dataset):
    """Dataset that returns positions AND position gradients (deltas)."""

    def __init__(self, l_data, dlat_max=0.04, dlon_max=0.04,
                 max_seqlen=96, dtype=torch.float32,
                 device=torch.device("cpu")):
        self.dlat_max = dlat_max
        self.dlon_max = dlon_max
        self.dpos_max = np.array([dlat_max, dlon_max])
        self.max_seqlen = max_seqlen
        self.device = device
        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        V = self.l_data[idx]
        m_v = V["traj"][:, :4]
        m_v = np.clip(m_v, 0.0, 0.9999)

        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen, 4), dtype=np.float32)

        # lat and lon
        seq[:seqlen, :2] = m_v[:seqlen, :2]

        # dlat and dlon (position gradients)
        dpos = (m_v[1:, :2] - m_v[:-1, :2] + self.dpos_max) / (2 * self.dpos_max)
        dpos = np.concatenate((dpos[:1, :], dpos), axis=0)
        dpos = np.clip(dpos, 0.0, 0.9999)
        seq[:seqlen, 2:] = dpos[:seqlen, :2]

        seq = torch.tensor(seq, dtype=torch.float32)

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.0

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(int(V["mmsi"]), dtype=torch.int)
        time_start = torch.tensor(int(V["traj"][0, 4]), dtype=torch.long)

        return seq, mask, seqlen, mmsi, time_start
