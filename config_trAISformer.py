"""Configuration for TrAISformer trained on Piraeus/Saronic Gulf AIS data."""

import os
import torch


class Config:
    retrain = True
    tb_log = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training
    max_epochs = 50
    batch_size = 32
    n_samples = 16

    early_stopping_patience = 10

    # Sequence lengths
    init_seqlen = 18
    max_seqlen = 120
    min_seqlen = 36

    dataset_name = "piraeus"

    # Region of Interest: Saronic Gulf / Piraeus
    # Tight bounds around the actual data with small margin
    lat_min = 37.4
    lat_max = 38.1
    lon_min = 22.9
    lon_max = 24.1
    sog_max = 30.0   # knots, clip outliers above this

    # Quantization sizes (number of discrete bins per attribute)
    # ~1 km/bin resolution, matching the original paper's ratio
    lat_size = 80    # 0.7° / 80 ≈ 970m resolution
    lon_size = 135   # 1.2° / 135 ≈ 980m resolution
    sog_size = 30    # 1 knot per bin
    cog_size = 72    # 5° per bin

    # Embedding sizes per attribute
    n_lat_embd = 256
    n_lon_embd = 256
    n_sog_embd = 128
    n_cog_embd = 128

    # Model and sampling
    mode = "pos"
    sample_mode = "pos_vicinity"
    top_k = 10
    r_vicinity = 20

    # Blur
    blur = True
    blur_learnable = False
    blur_loss_w = 1.0
    blur_n = 2
    if not blur:
        blur_n = 0
        blur_loss_w = 0

    # Data paths (relative to src/)
    datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
    trainset_name = "piraeus_train.pkl"
    validset_name = "piraeus_val.pkl"
    testset_name = "piraeus_test.pkl"

    # Derived model parameters
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
    n_head = 8
    n_layer = 8
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    # Optimization
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = True
    warmup_tokens = 512 * 20
    final_tokens = 260e9
    num_workers = 4

    # Output paths
    filename = (
        f"{dataset_name}"
        f"-{mode}-{sample_mode}-{top_k}-{r_vicinity}"
        f"-blur-{blur}-{blur_learnable}-{blur_n}-{blur_loss_w}"
        f"-data_size-{lat_size}-{lon_size}-{sog_size}-{cog_size}"
        f"-embd_size-{n_lat_embd}-{n_lon_embd}-{n_sog_embd}-{n_cog_embd}"
        f"-head-{n_head}-{n_layer}"
        f"-bs-{batch_size}"
        f"-lr-{learning_rate}"
        f"-seqlen-{init_seqlen}-{max_seqlen}"
    )
    savedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", filename, "")
    ckpt_path = os.path.join(savedir, "model.pt")
