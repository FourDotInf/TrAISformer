"""TrAISformer — Generative transformer for AIS trajectory prediction.

Adapted for Piraeus/Saronic Gulf AIS data (University of Piraeus dataset).
Training on 2017+2018, validation/testing on 2019.

Reference: https://arxiv.org/abs/2109.03958
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
import logging

import torch
from torch.utils.data import DataLoader

import models
import trainers
import datasets
import utils
from config_trAISformer import Config

cf = Config()
TB_LOG = cf.tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter()

utils.set_seed(42)


def load_raw_pickle(path):
    """Load a pickle file produced by piraeus_preprocess.py.

    Returns a list of numpy arrays, each of shape (seq_len, 6)
    with columns [LAT, LON, SOG, COG, TIMESTAMP, MMSI].
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_trajectories(raw_trajs, cf):
    """Convert raw numpy trajectories to the dict format the model expects.

    Each raw trajectory is (N, 6): [LAT, LON, SOG, COG, TIMESTAMP, MMSI].
    Output is a list of dicts: {"mmsi": int, "traj": (N, 5)} where traj
    columns are [LAT, LON, SOG, COG, TIMESTAMP] normalized to [0, 1).
    """
    lat_range = cf.lat_max - cf.lat_min
    lon_range = cf.lon_max - cf.lon_min

    converted = []
    for raw in raw_trajs:
        mmsi = int(raw[0, 5])

        lat_norm = (raw[:, 0] - cf.lat_min) / lat_range
        lon_norm = (raw[:, 1] - cf.lon_min) / lon_range
        sog_norm = np.clip(raw[:, 2], 0, cf.sog_max) / cf.sog_max
        cog_norm = raw[:, 3] / 360.0
        timestamp = raw[:, 4]

        traj = np.stack([lat_norm, lon_norm, sog_norm, cog_norm, timestamp], axis=1).astype(np.float32)

        # Clip to [0, 1) — discard trajectories with points outside the ROI
        if (traj[:, :4] < 0).any() or (traj[:, :4] >= 1.0).any():
            # Clip rather than discard — clamp stray points
            traj[:, :4] = np.clip(traj[:, :4], 0.0, 0.9999)

        converted.append({"mmsi": mmsi, "traj": traj})

    return converted


def evaluate(model, dataloader, cf, device):
    """Run ensemble evaluation on the test set and return per-step errors in km."""
    init_seqlen = cf.init_seqlen
    # Prediction horizon: up to 3 hours at 10-min intervals = 18 steps
    max_seqlen = init_seqlen + 6 * 4  # 18 prediction steps beyond init

    lat_range = cf.lat_max - cf.lat_min
    lon_range = cf.lon_max - cf.lon_min

    # For converting normalized coords back to radians for haversine
    v_ranges = torch.tensor([lat_range, lon_range, 0, 0], device=device)
    v_roi_min = torch.tensor([cf.lat_min, cf.lon_min, 0, 0], device=device)

    model.eval()
    l_min_errors, l_mean_errors, l_masks = [], [], []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating")
    with torch.no_grad():
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            seqs_init = seqs[:, :init_seqlen, :].to(device)
            masks = masks[:, :max_seqlen].to(device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros(
                (batchsize, max_seqlen - init_seqlen, cf.n_samples), device=device
            )

            for i_sample in range(cf.n_samples):
                preds = trainers.sample(
                    model,
                    seqs_init,
                    max_seqlen - init_seqlen,
                    temperature=1.0,
                    sample=True,
                    sample_mode=cf.sample_mode,
                    r_vicinity=cf.r_vicinity,
                    top_k=cf.top_k,
                )
                inputs = seqs[:, :max_seqlen, :].to(device)

                # Convert normalized [0,1) back to degrees, then to radians
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180

                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, i_sample] = d[:, init_seqlen:]

            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, init_seqlen:])

    l_min = [x.values for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    return pred_errors.detach().cpu().numpy()


def plot_prediction_errors(pred_errors, savedir):
    """Plot prediction error vs time horizon."""
    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors)) / 6  # 6 steps per hour

    plt.plot(v_times, pred_errors, linewidth=2)

    # Mark 1h, 2h, 3h if available
    for hours, step in [(1, 6), (2, 12), (3, 18)]:
        if step < len(pred_errors):
            err = pred_errors[step]
            plt.plot(hours, err, "o", markersize=8)
            plt.plot([hours, hours], [0, err], "r--", alpha=0.5)
            plt.plot([0, hours], [err, err], "r--", alpha=0.5)
            plt.text(hours + 0.1, err - 0.3, f"{err:.2f} km", fontsize=10)

    plt.xlabel("Prediction Horizon (hours)")
    plt.ylabel("Prediction Error (km)")
    plt.title("TrAISformer — Piraeus Trajectory Prediction Error")
    plt.xlim([0, max(4, v_times[-1] + 0.5)])
    plt.ylim([0, max(pred_errors) * 1.2 + 1])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "prediction_error.png"), dpi=150)
    plt.close()
    logging.info(f"Saved prediction error plot to {savedir}prediction_error.png")


if __name__ == "__main__":

    device = cf.device
    init_seqlen = cf.init_seqlen

    # Logging
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print(f"Created output directory: {cf.savedir}")
    else:
        print(f"Output directory: {cf.savedir}")
    utils.new_log(cf.savedir, "log")

    logging.info(f"Device: {device}")
    logging.info(f"Config: {cf.dataset_name}, mode={cf.mode}, "
                 f"lat_size={cf.lat_size}, lon_size={cf.lon_size}, "
                 f"sog_size={cf.sog_size}, cog_size={cf.cog_size}")

    # Load and normalize data
    moving_threshold = 0.05  # in normalized SOG space
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}

    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        logging.info(f"Loading {datapath}...")

        raw_trajs = load_raw_pickle(datapath)
        logging.info(f"  Raw trajectories: {len(raw_trajs)}")

        l_data = normalize_trajectories(raw_trajs, cf)
        del raw_trajs

        # Filter out stationary starts and NaN trajectories
        for V in l_data:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except IndexError:
                moving_idx = len(V["traj"]) - 1
            V["traj"] = V["traj"][moving_idx:, :]

        Data[phase] = [
            x for x in l_data
            if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen
        ]
        logging.info(f"  After filtering: {len(Data[phase])} trajectories "
                     f"(removed {len(l_data) - len(Data[phase])})")
        del l_data

        # Create PyTorch datasets
        # max_seqlen + 1 because we use inputs = x[:-1], targets = x[1:]
        if cf.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = datasets.AISDataset_grad(
                Data[phase], max_seqlen=cf.max_seqlen + 1, device=cf.device
            )
        else:
            aisdatasets[phase] = datasets.AISDataset(
                Data[phase], max_seqlen=cf.max_seqlen + 1, device=cf.device
            )

        shuffle = phase != "test"
        aisdls[phase] = DataLoader(
            aisdatasets[phase],
            batch_size=cf.batch_size,
            shuffle=shuffle,
            num_workers=cf.num_workers,
        )

    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen
    logging.info(f"Train: {len(aisdatasets['train'])}, "
                 f"Valid: {len(aisdatasets['valid'])}, "
                 f"Test: {len(aisdatasets['test'])}")

    # Model
    model = models.TrAISformer(cf, partition_model=None)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {n_params:,}")

    # Training
    if cf.retrain:
        trainer = trainers.Trainer(
            model,
            aisdatasets["train"],
            aisdatasets["valid"],
            cf,
            savedir=cf.savedir,
            device=cf.device,
            aisdls=aisdls,
            INIT_SEQLEN=init_seqlen,
        )
        trainer.train()

    # Evaluation
    logging.info("Loading best model for evaluation...")
    model.load_state_dict(torch.load(cf.ckpt_path, map_location=device))
    model.to(device)

    pred_errors = evaluate(model, aisdls["test"], cf, device)

    # Print results
    for hours, step in [(1, 6), (2, 12), (3, 18)]:
        if step < len(pred_errors):
            logging.info(f"Prediction error at {hours}h: {pred_errors[step]:.4f} km")

    plot_prediction_errors(pred_errors, cf.savedir)
    logging.info("Done!")
