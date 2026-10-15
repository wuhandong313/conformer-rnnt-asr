###
# Author: Kai Li
# Date: 2022-05-26 18:09:54
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-05-27 11:34:45
###
"""LightningModule：把分离网 / 损失 / loader 交给 Trainer 调度。

Trainer 调 ``training_step`` / ``validation_step``；本文件只定义「每步算什么」。
说明见 ``代码解析/pl.Trainer是干什么用的.md``。
"""

from collections.abc import MutableMapping

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def flatten_dict(d, parent_key="", sep="_"):
    """嵌套 dict 压平，供 TensorBoard hparams 使用。"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class AudioLightningModule(pl.LightningModule):
    """Trainer 会调下面这些钩子（本类已覆写的打勾）。

    训练循环：``training_step`` →（自动 backward / step）
    验证循环：``validation_step`` → ``on_validation_epoch_end``
    装配：``configure_optimizers``、``train_dataloader``、``val_dataloader``
    存盘：``on_save_checkpoint``；前向：``forward``

    未覆写、需要时再加：``test_step`` / ``on_train_epoch_end`` /
    ``on_before_optimizer_step`` / ``on_load_checkpoint`` 等。
    Trainer 侧的 ModelCheckpoint / EarlyStopping 不是本类方法，见 audio_train.py。
    """
    def __init__(
        self,
        audio_model=None,
        video_model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        self.speedperturb = self._build_speed_perturb()
        self.default_monitor = "val_loss/dataloader_idx_0"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _build_speed_perturb(self):
        """仅 SpeedAug=true 时 import speechbrain，避免一启动就拉整库。"""
        if not self.config.get("training", {}).get("SpeedAug", False):
            return None
        from speechbrain.augment.time_domain import SpeedPerturb

        return SpeedPerturb(
            self.config["datamodule"]["data_config"]["sample_rate"],
            speeds=[95, 100, 105],
        )

    def forward(self, wav, mouth=None):
        """Lightning 钩子：``self(wav)`` 或 ``validation_step`` 里调。只跑分离网，不算损失。"""
        return self.audio_model(wav)

    # ------------------------------------------------------------------
    # 配置 / batch
    # ------------------------------------------------------------------

    def _phoneme_cfg(self) -> dict:
        return self.config.get("phoneme_loss") or {}

    def _alpha_value(self) -> float:
        model = getattr(self, "audio_model", None)
        alpha = getattr(model, "alpha", None) if model is not None else None
        if alpha is None:
            return 0.0
        if torch.is_tensor(alpha):
            return float(alpha.detach().cpu().item())
        return float(alpha)

    def _use_mixed_phoneme_loss(self) -> bool:
        """前 signal_frac 的 epoch 只用 PIT；之后用 combine_signal_phoneme_loss。"""
        cfg = self._phoneme_cfg()
        schedule = (cfg.get("schedule") or "none").lower()
        if schedule in ("none", "", "pit_only"):
            return False
        if schedule not in ("half_then_mix", "signal_then_mix"):
            return False
        audionet_name = (self.config.get("audionet") or {}).get("audionet_name", "")
        if audionet_name != "TDANetPhoneme" and not cfg.get("force", False):
            return False
        # α=0 时对齐实验 3：波形走 mask_net，CE/ArtV 也不回传到主干
        if cfg.get("require_alpha", False) and self._alpha_value() == 0.0:
            return False
        max_epochs = int(self.trainer.max_epochs) if self.trainer else 1
        frac = float(cfg.get("signal_frac", 0.5))
        switch_at = int(max_epochs * frac)
        # epoch 从 0 计：前半 [0, switch_at)，后半 [switch_at, max)
        return int(self.current_epoch) >= max(switch_at, 0)

    def _unpack_batch(self, batch):
        """``(mix, src, meta)``。meta 为 dict（含 clip_start/stop、utt_key），或旧的文件名 str。"""
        if len(batch) < 3:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
        mixtures, targets, meta = batch[0], batch[1], batch[2]
        if not isinstance(meta, dict):
            meta = {"filename": meta, "clip_start": None, "clip_stop": None, "utt_key": ""}
        return mixtures, targets, meta

    def _log_epoch(self, name, value, prog_bar=False):
        self.log(
            name,
            value,
            on_epoch=True,
            prog_bar=prog_bar,
            sync_dist=True,
            logger=True,
        )

    # ------------------------------------------------------------------
    # 训练步
    # ------------------------------------------------------------------

    def _maybe_speed_augment(self, mixtures, targets):
        """对各源做语速扰动后截齐，再用加和当 mix。见笔记/语速增强SpeedAug是干什么的.md。"""
        if self.config["training"]["SpeedAug"] != True:
            return mixtures, targets
        with torch.no_grad():
            new_targets = []
            min_len = -1
            for i in range(targets.shape[1]):
                new_target = self.speedperturb(targets[:, i, :])
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                elif new_target.shape[-1] < min_len:
                    min_len = new_target.shape[-1]
            out = torch.zeros(
                targets.shape[0],
                targets.shape[1],
                min_len,
                device=targets.device,
                dtype=torch.float,
            )
            for i, new_target in enumerate(new_targets):
                out[:, i, :] = new_targets[i][:, 0:min_len]
            mixtures = out.sum(1)
        return mixtures, out

    def _phoneme_bank(self):
        from ..models.tdanet_phoneme.losses import (
            PhonemeAnnotationBank,
            resolve_phoneme_root,
        )

        if getattr(self, "_phoneme_bank_obj", None) is None:
            self._phoneme_bank_obj = PhonemeAnnotationBank(
                resolve_phoneme_root(self.config)
            )
        return self._phoneme_bank_obj

    def _loss_signal_only(self, mixtures, targets):
        est_sources = self(mixtures)
        loss = self.loss_func["train"](est_sources, targets)
        self._log_epoch("train_signal_loss", loss)
        self._log_epoch("train_phoneme_loss", loss.new_zeros(()))
        self._log_epoch("train_ce_loss", loss.new_zeros(()))
        self._log_epoch("train_artv_loss", loss.new_zeros(()))
        self._log_epoch("train_phoneme_mix", 0.0)
        return loss

    def _unpack_phoneme_forward(self, mixtures):
        """兼容 ``(wave, logits)`` 与实验 5 的 ``(wave, logits, artv)``。"""
        out = self.audio_model(mixtures, return_phoneme=True)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            return out[0], out[1], out[2]
        est, logits = out
        return est, logits, None

    def _phoneme_csv_path(self):
        exp_dir = (self.config.get("main_args") or {}).get("exp_dir")
        if not exp_dir:
            return None
        from pathlib import Path

        return Path(exp_dir).parent / "test_artvector_loss.csv"

    def _maybe_dump_artv(
        self,
        *,
        split: str,
        utt_keys,
        frame_ids,
        target_artv,
        gauss_artv,
        pred_artv,
        dump_fraction: float,
        epoch: int | None = None,
    ) -> None:
        if dump_fraction <= 0 or pred_artv is None:
            return
        from ..models.tdanet_phoneme.losses import append_artv_csv

        path = self._phoneme_csv_path()
        if path is None:
            return
        if epoch is None and getattr(self, "trainer", None) is not None:
            epoch = int(self.trainer.current_epoch)
        if isinstance(utt_keys, str):
            utt_keys = [utt_keys]
        ids_np = frame_ids.detach().cpu().numpy()
        tgt_np = target_artv.detach().cpu().numpy()
        g_np = gauss_artv.detach().cpu().numpy()
        p_np = pred_artv.detach().cpu().numpy()
        se_np = ((p_np - g_np) ** 2).mean(axis=-1)
        b, n_src = ids_np.shape[:2]
        for bi in range(b):
            key = utt_keys[bi] if bi < len(utt_keys) else ""
            parts = [p for p in str(key).split(",") if p] or [str(key)]
            for si in range(n_src):
                append_artv_csv(
                    path,
                    split=split,
                    utt_keys=parts,
                    src=si,
                    frame_ids=ids_np[bi, si],
                    target=tgt_np[bi, si],
                    gauss=g_np[bi, si],
                    pred=p_np[bi, si],
                    frame_loss=se_np[bi, si],
                    dump_fraction=dump_fraction,
                    epoch=epoch,
                )

    def _loss_signal_and_phoneme(
        self,
        mixtures,
        targets,
        meta,
        dump_split=None,
        log_metrics=True,
        pit_key: str = "train",
        metric_prefix: str = "train",
    ):
        """波形 PIT + 音素 CE + 高斯 ArtV MSE。

        Returns:
            (total, signal_loss, phoneme_term) 。
            ``phoneme_term = ce_weight·CE + artv_weight·ArtV``。
        """
        from ..models.tdanet_phoneme.losses import (
            PhonemeCELoss,
            align_artv_to_len,
            align_ids_to_len,
            apply_perm,
            artv_mse_loss,
            build_frame_targets_from_clips,
            combine_signal_ce_artv_loss,
            gaussianize_artv_batch,
        )

        ph_cfg = self._phoneme_cfg()
        ignore_index = int(ph_cfg.get("ignore_index", -100))
        lam = float(ph_cfg.get("lambda", 0.1))
        ce_weight = float(ph_cfg.get("ce_weight", 1.0))
        artv_weight = float(ph_cfg.get("artv_weight", 1.0))
        est_sources, phoneme_logits, artv_pred = self._unpack_phoneme_forward(
            mixtures
        )
        pit_fn = self.loss_func[pit_key]
        signal_loss, perm = pit_fn(est_sources, targets, return_perm=True)

        zero = phoneme_logits.new_zeros(())
        ce_loss = zero
        artv_loss = zero
        if self.config.get("training", {}).get("SpeedAug") is True:
            pass
        else:
            n_src = phoneme_logits.shape[1]
            sample_rate = int(
                self.config["datamodule"]["data_config"]["sample_rate"]
            )
            utt_keys = meta.get("utt_key") or []
            if isinstance(utt_keys, str):
                utt_keys = [utt_keys]
            clip_start = meta.get("clip_start")
            clip_stop = meta.get("clip_stop")
            if clip_start is not None and clip_stop is not None:
                # 全分辨率目标：与波形时长成比例的帧数用 clip 对齐到
                # 一个足够密的网格，再插值到探针 T_bn。
                t_grid = max(int(phoneme_logits.shape[2]) * int(
                    getattr(self.audio_model, "downsample_ratio", 8) or 8
                ), phoneme_logits.shape[2])
                frame_ids, frame_artv = build_frame_targets_from_clips(
                    self._phoneme_bank(),
                    utt_keys,
                    clip_start,
                    clip_stop,
                    sample_rate,
                    t_grid,
                    n_src,
                    ignore_index=ignore_index,
                    device=phoneme_logits.device,
                    dtype=phoneme_logits.dtype
                    if artv_pred is None
                    else artv_pred.dtype,
                )
                gauss = gaussianize_artv_batch(
                    frame_artv,
                    frame_ids,
                    ignore_index=ignore_index,
                    sigma_scale=float(ph_cfg.get("gauss_sigma_scale", 0.25)),
                )
                t_bn = phoneme_logits.shape[2]
                frame_ids_bn = align_ids_to_len(frame_ids, t_bn)
                gauss_bn = align_artv_to_len(gauss, t_bn)
                target_bn = align_artv_to_len(frame_artv, t_bn)
                logits = apply_perm(phoneme_logits, perm)
                frame_ids_bn = apply_perm(frame_ids_bn, perm)
                gauss_bn = apply_perm(gauss_bn, perm)
                target_bn = apply_perm(target_bn, perm)
                ce_loss = PhonemeCELoss(ignore_index=ignore_index)(
                    logits, frame_ids_bn
                )
                if artv_pred is not None:
                    pred = apply_perm(artv_pred, perm)
                    artv_loss = artv_mse_loss(
                        pred, gauss_bn, frame_ids_bn, ignore_index=ignore_index
                    )
                    frac = float(ph_cfg.get("dump_fraction", 0) or 0)
                    if dump_split and frac > 0:
                        self._maybe_dump_artv(
                            split=dump_split,
                            utt_keys=utt_keys,
                            frame_ids=frame_ids_bn,
                            target_artv=target_bn,
                            gauss_artv=gauss_bn,
                            pred_artv=pred,
                            dump_fraction=frac,
                        )

        phoneme_term = ce_weight * ce_loss + artv_weight * artv_loss
        loss = combine_signal_ce_artv_loss(
            signal_loss,
            ce_loss,
            artv_loss,
            lam,
            artv_weight,
            ce_weight,
        )
        if log_metrics:
            self._log_epoch(f"{metric_prefix}_signal_loss", signal_loss)
            self._log_epoch(f"{metric_prefix}_phoneme_loss", phoneme_term)
            self._log_epoch(f"{metric_prefix}_ce_loss", ce_loss)
            self._log_epoch(f"{metric_prefix}_artv_loss", artv_loss)
            if metric_prefix == "train":
                self._log_epoch("train_phoneme_mix", 1.0)
        return loss, signal_loss, phoneme_term

    def training_step(self, batch, batch_nb):
        """Lightning 钩子：每个 train batch 调一次。须返回带 ``loss`` 的 dict，Trainer 据此反传。"""
        mixtures, targets, meta = self._unpack_batch(batch)
        mixtures, targets = self._maybe_speed_augment(mixtures, targets)
        if self._use_mixed_phoneme_loss():
            loss, _, _ = self._loss_signal_and_phoneme(mixtures, targets, meta)
        else:
            loss = self._loss_signal_only(mixtures, targets)
        self._log_epoch("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    # ------------------------------------------------------------------
    # 验证 / 测试（val_dataloader 返回 [val, test] 两个 loader）
    # ------------------------------------------------------------------

    def _pit_val_loss(self, batch):
        mixtures, targets, _ = self._unpack_batch(batch)
        est_sources = self(mixtures)
        return self.loss_func["val"](est_sources, targets)

    def _should_eval_test_loader(self) -> bool:
        return int(self.trainer.current_epoch) % 10 == 0

    def validation_step(self, batch, batch_nb, dataloader_idx):
        """Lightning 钩子：每个 val batch 调一次。``dataloader_idx`` 对应 ``val_dataloader`` 返回的列表下标。"""
        if dataloader_idx == 0:
            mixtures, targets, meta = self._unpack_batch(batch)
            if self._use_mixed_phoneme_loss():
                frac = float((self._phoneme_cfg() or {}).get("dump_fraction", 0) or 0)
                dump = frac > 0 and self._should_eval_test_loader()
                mix_loss, signal_loss, phoneme_term = self._loss_signal_and_phoneme(
                    mixtures,
                    targets,
                    meta,
                    dump_split="val" if dump else None,
                    log_metrics=True,
                    pit_key="val",
                    metric_prefix="val",
                )
                self._log_epoch("val_loss", signal_loss, prog_bar=True)
                self._log_epoch("val_mix_loss", mix_loss)
                self.validation_step_outputs.append(signal_loss)
                return {"val_loss": signal_loss}
            loss = self._pit_val_loss(batch)
            self._log_epoch("val_loss", loss, prog_bar=True)
            self._log_epoch("val_signal_loss", loss)
            self.validation_step_outputs.append(loss)
            return {"val_loss": loss}
        if self._should_eval_test_loader() and dataloader_idx == 1:
            tloss = self._pit_val_loss(batch)
            self._log_epoch("test_loss", tloss, prog_bar=True)
            self.test_step_outputs.append(tloss)
            return {"test_loss": tloss}

    def _gathered_mean(self, tensors):
        avg_loss = torch.stack(tensors).mean()
        return torch.mean(self.all_gather(avg_loss))

    def on_validation_epoch_end(self):
        """Lightning 钩子：本 epoch 全部 ``validation_step`` 结束后。这里汇总 val/test 并写 lr、SI-SNR。"""
        val_loss = self._gathered_mean(self.validation_step_outputs)
        lr = self.optimizer.param_groups[0]["lr"]
        self.log("lr", lr, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("learning_rate", lr, on_epoch=True, sync_dist=True)
        self.log("val_pit_sisnr", -val_loss, on_epoch=True, sync_dist=True)

        if self._should_eval_test_loader() and self.test_step_outputs:
            test_loss = self._gathered_mean(self.test_step_outputs)
            self.log("test_pit_sisnr", -test_loss, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()
        self.test_step_outputs.clear()

    # ------------------------------------------------------------------
    # 优化器 / DataLoader / ckpt
    # ------------------------------------------------------------------

    def _normalize_scheduler(self, sched):
        if not isinstance(sched, dict):
            if isinstance(sched, ReduceLROnPlateau):
                sched = {"scheduler": sched, "monitor": self.default_monitor}
            return sched
        sched.setdefault("monitor", self.default_monitor)
        sched.setdefault("frequency", 1)
        if sched["interval"] == "batch":
            sched["interval"] = "step"
        assert sched["interval"] in ("epoch", "step"), (
            "Scheduler interval should be either step or epoch"
        )
        return sched

    def configure_optimizers(self):
        """Lightning 钩子：fit 开始时问一次。返回优化器，或 ``[opt], [scheduler]``。"""
        if self.scheduler is None:
            return self.optimizer
        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]
        epoch_schedulers = [self._normalize_scheduler(s) for s in self.scheduler]
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        """Lightning 钩子：每个 epoch 取训练集。本仓库在 audio_train 里已建好 loader，这里直接交出去。"""
        return self.train_loader

    def val_dataloader(self):
        """Lightning 钩子：验证数据。列表下标 0=验证集，1=测试集（每 10 个 epoch 评一次）。"""
        return [self.val_loader, self.test_loader]

    def on_save_checkpoint(self, checkpoint):
        """Lightning 钩子：写 ckpt 前。把 yaml 配置塞进 checkpoint，便于续训/导出。"""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """压平配置，并把 None / list 转成 TensorBoard 能记的类型。"""
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
