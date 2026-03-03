"""Training loop and sampling functions for TrAISformer.

Reference: https://github.com/karpathy/minGPT
"""

import os
import math
import logging

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils

logger = logging.getLogger(__name__)


@torch.no_grad()
def sample(model, seqs, steps, temperature=1.0, sample=False,
           sample_mode="pos_vicinity", r_vicinity=20, top_k=None):
    """Autoregressively sample future trajectory steps from the model."""
    max_seqlen = model.get_max_seqlen()
    model.eval()
    for k in range(steps):
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]

        logits, _ = model(seqs_cond)
        d2inf_pred = torch.zeros((logits.shape[0], 4)).to(seqs.device) + 0.5

        logits = logits[:, -1, :] / temperature

        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)

        if sample_mode in ("pos_vicinity",):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])
            lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = utils.top_k_logits(lat_logits, top_k)
            lon_logits = utils.top_k_logits(lon_logits, top_k)
            sog_logits = utils.top_k_logits(sog_logits, top_k)
            cog_logits = utils.top_k_logits(cog_logits, top_k)

        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)

        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes

        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)

    return seqs


class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    ckpt_path = None
    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config,
                 savedir=None, device=torch.device("cpu"), aisdls=None,
                 INIT_SEQLEN=0):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir
        self.device = device
        self.model = model.to(device)
        self.aisdls = aisdls or {}
        self.INIT_SEQLEN = INIT_SEQLEN

    def save_checkpoint(self, best_epoch):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        aisdls, INIT_SEQLEN = self.aisdls, self.INIT_SEQLEN
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            n_batches = len(loader)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_n = 0, 0

            for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
                seqs = seqs.to(self.device)
                masks = masks[:, :-1].to(self.device)

                with torch.set_grad_enabled(is_train):
                    logits, loss = model(seqs, masks=masks, with_targets=True)
                    loss = loss.mean()
                    losses.append(loss.item())

                d_loss += loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens += (seqs >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    pbar.set_description(
                        f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}")

            if is_train:
                logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, lr {lr:e}.")
            else:
                logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}.")

            if not is_train:
                return float(np.mean(losses))

        best_loss = float('inf')
        self.tokens = 0
        best_epoch = 0
        patience = getattr(config, "early_stopping_patience", 10)
        epochs_without_improvement = 0

        for epoch in range(config.max_epochs):
            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)

            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                self.save_checkpoint(best_epoch + 1)
            else:
                epochs_without_improvement += 1

            # Sample and plot predictions each epoch
            self._plot_samples(model, epoch)

            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping at epoch {epoch + 1} "
                             f"(no improvement for {patience} epochs). "
                             f"Best epoch: {best_epoch + 1}, best val loss: {best_loss:.5f}")
                break

        # Save final model
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logging.info(f"Last epoch: {epoch + 1:03d}, saving model to {self.config.ckpt_path}")
        save_path = self.config.ckpt_path.replace("model.pt", f"model_{epoch + 1:03d}.pt")
        torch.save(raw_model.state_dict(), save_path)

    def _plot_samples(self, model, epoch):
        """Generate and plot sample predictions during training."""
        if "test" not in self.aisdls:
            return

        raw_model = model.module if hasattr(self.model, "module") else model
        try:
            seqs, masks, seqlens, mmsis, time_starts = next(iter(self.aisdls["test"]))
        except StopIteration:
            return

        n_plots = 7
        init_seqlen = self.INIT_SEQLEN
        seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)
        preds = sample(raw_model,
                       seqs_init,
                       96 - init_seqlen,
                       temperature=1.0,
                       sample=True,
                       sample_mode=self.config.sample_mode,
                       r_vicinity=self.config.r_vicinity,
                       top_k=self.config.top_k)

        img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')
        plt.figure(figsize=(9, 6), dpi=150)
        cmap = plt.cm.get_cmap("jet")
        preds_np = preds.detach().cpu().numpy()
        inputs_np = seqs.detach().cpu().numpy()

        for idx in range(n_plots):
            c = cmap(float(idx) / n_plots)
            try:
                seqlen = seqlens[idx].item()
            except (IndexError, RuntimeError):
                continue
            plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], color=c)
            plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0],
                     "o", markersize=3, color=c)
            plt.plot(inputs_np[idx][:seqlen, 1], inputs_np[idx][:seqlen, 0],
                     linestyle="-.", color=c)
            plt.plot(preds_np[idx][init_seqlen:, 1], preds_np[idx][init_seqlen:, 0],
                     "x", markersize=4, color=c)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("Longitude (normalized)")
        plt.ylabel("Latitude (normalized)")
        plt.title(f"Epoch {epoch + 1}")
        plt.savefig(img_path, dpi=150)
        plt.close()
