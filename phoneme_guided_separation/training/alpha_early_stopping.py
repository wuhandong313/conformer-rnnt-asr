"""第一次 val 早停后把模型 alpha 设为配置值并续训，第二次早停才真正结束。"""

from __future__ import annotations

from typing import Any

import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.rank_zero import rank_zero_info


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(metrics: dict, *keys: str) -> float | None:
    for key in keys:
        if key not in metrics:
            continue
        val = _as_float(metrics[key])
        if val is not None:
            return val
    return None


def snapshot_early_stop(trainer, callback) -> dict[str, Any]:
    """记录一次早停：是否触发、第几轮（Lightning ``epoch`` 从 0 计）、当时损失。"""
    metrics = getattr(trainer, "callback_metrics", None) or {}
    monitor = getattr(callback, "monitor", None) or "val_loss"
    current = _metric(metrics, monitor, "val_loss/dataloader_idx_0", "val_loss")
    best = _as_float(getattr(callback, "best_score", None))
    epoch = int(getattr(trainer, "current_epoch", 0) or 0)
    val_loss = current
    info: dict[str, Any] = {
        "triggered": True,
        "epoch": epoch,
        "best_val_loss": best,
        "val_loss": val_loss,
        "val_signal_loss": _metric(
            metrics, "val_signal_loss/dataloader_idx_0", "val_signal_loss"
        ),
        "val_phoneme_loss": _metric(
            metrics, "val_phoneme_loss/dataloader_idx_0", "val_phoneme_loss"
        ),
        "val_ce_loss": _metric(metrics, "val_ce_loss/dataloader_idx_0", "val_ce_loss"),
        "val_artv_loss": _metric(
            metrics, "val_artv_loss/dataloader_idx_0", "val_artv_loss"
        ),
        "val_mix_loss": _metric(
            metrics, "val_mix_loss/dataloader_idx_0", "val_mix_loss"
        ),
        "val_pit_sisnr": _metric(metrics, "val_pit_sisnr"),
        "monitor": monitor,
    }
    if info["val_signal_loss"] is None:
        info["val_signal_loss"] = val_loss
    if info["val_pit_sisnr"] is None and val_loss is not None:
        info["val_pit_sisnr"] = -val_loss
    return info


def _not_triggered() -> dict[str, Any]:
    return {"triggered": False, "epoch": None, "best_val_loss": None, "val_loss": None}


class TwoPhaseAlphaEarlyStopping(EarlyStopping):
    """phase=0 触发时：加载最优 ckpt、``set_alpha(alpha_after)``、重置 patience。

    ``start_phase=1``：跳过第一阶段（消融：已有第一次早停权重）。

    Lightning 2.6 默认 ``check_on_train_epoch_end=True``（每个 epoch 验一次时），
    真正 ``should_stop`` 写在 ``on_train_epoch_end``。只钩 validation 会拦不住，
    第一次 patience 到了训练就结束，α 一直为 0、音素损失也不会开。
    """

    def __init__(
        self,
        *args,
        alpha_after: float = 0.5,
        start_phase: int = 0,
        **kwargs,
    ):
        # False：在 on_validation_end 判定（本类已拦截）。仍同时钩 on_train_epoch_end，
        # 防止 Lightning setup() 或旧 ckpt 把检查点改回 train epoch end。
        kwargs.setdefault("check_on_train_epoch_end", False)
        super().__init__(*args, **kwargs)
        self.alpha_after = float(alpha_after)
        self.phase = int(start_phase)
        self.first_stop_info: dict[str, Any] | None = None
        self.second_stop_info: dict[str, Any] | None = None
        if self.phase >= 1 and self.first_stop_info is None:
            self.first_stop_info = {
                "triggered": True,
                "epoch": None,
                "best_val_loss": None,
                "val_loss": None,
                "skipped": True,
                "alpha_set_to": self.alpha_after,
            }

    def state_dict(self):
        state = super().state_dict()
        if isinstance(state, dict):
            state["alpha_es_phase"] = self.phase
            state["alpha_es_first_stop"] = self.first_stop_info
            state["alpha_es_second_stop"] = self.second_stop_info
        return state

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict):
            if "alpha_es_phase" in state_dict:
                self.phase = int(state_dict.pop("alpha_es_phase"))
            if "alpha_es_first_stop" in state_dict:
                self.first_stop_info = state_dict.pop("alpha_es_first_stop")
            if "alpha_es_second_stop" in state_dict:
                self.second_stop_info = state_dict.pop("alpha_es_second_stop")
        return super().load_state_dict(state_dict)

    def to_report(self, trainer=None) -> dict[str, Any]:
        first = self.first_stop_info or _not_triggered()
        second = self.second_stop_info
        if second is None and trainer is not None:
            stopped = bool(getattr(trainer, "should_stop", False))
            stopped_epoch = int(getattr(self, "stopped_epoch", 0) or 0)
            if self.phase == 1 and (stopped or stopped_epoch > 0):
                second = snapshot_early_stop(trainer, self)
                self.second_stop_info = second
        if second is None:
            second = _not_triggered()
        return {
            "two_phase": True,
            "alpha_after": self.alpha_after,
            "start_phase": self.phase if first.get("skipped") else 0,
            "first": first,
            "second": second,
        }

    def _reset_patience(self):
        self.wait_count = 0
        self.stopped_epoch = 0
        if hasattr(self, "stopping_reason_message"):
            self.stopping_reason_message = None
        try:
            from pytorch_lightning.callbacks.early_stopping import EarlyStoppingReason

            if hasattr(self, "stopping_reason"):
                self.stopping_reason = EarlyStoppingReason.NOT_STOPPED
        except Exception:
            pass
        if str(self.mode) == "min":
            self.best_score = torch.tensor(float("inf"))
        else:
            self.best_score = torch.tensor(float("-inf"))

    @staticmethod
    def _reset_checkpoint_scores(trainer):
        fill = float("inf")
        for cb in getattr(trainer, "checkpoint_callbacks", []) or []:
            if hasattr(cb, "best_model_score"):
                cb.best_model_score = None
            if hasattr(cb, "kth_value"):
                try:
                    cb.kth_value = torch.tensor(fill)
                except Exception:
                    pass

    def _record_if_stopped(self, trainer):
        if getattr(trainer, "sanity_checking", False):
            return
        if not getattr(trainer, "should_stop", False):
            return
        if self.phase == 0 and self.first_stop_info is None:
            info = snapshot_early_stop(trainer, self)
            info["alpha_set_to"] = self.alpha_after
            self.first_stop_info = info
            rank_zero_info(
                "Alpha early-stop first: epoch={} best_val_loss={} val_loss={}".format(
                    info.get("epoch"),
                    info.get("best_val_loss"),
                    info.get("val_loss"),
                )
            )
        elif self.phase == 1 and self.second_stop_info is None:
            self.second_stop_info = snapshot_early_stop(trainer, self)
            rank_zero_info(
                "Alpha early-stop second: epoch={} best_val_loss={} val_loss={}".format(
                    self.second_stop_info.get("epoch"),
                    self.second_stop_info.get("best_val_loss"),
                    self.second_stop_info.get("val_loss"),
                )
            )

    def _maybe_enter_phase2(self, trainer, pl_module):
        self._record_if_stopped(trainer)
        if not getattr(trainer, "should_stop", False) or self.phase != 0:
            return
        trainer.should_stop = False
        self.phase = 1
        self._reset_patience()

        ckpt_path = None
        for cb in trainer.checkpoint_callbacks:
            path = getattr(cb, "best_model_path", None)
            if path:
                ckpt_path = path
                break
        if ckpt_path:
            state = torch.load(ckpt_path, map_location=pl_module.device)
            sd = state.get("state_dict", state)
            pl_module.load_state_dict(sd)
            rank_zero_info(f"Alpha early-stop: restored {ckpt_path}")

        model = getattr(pl_module, "audio_model", None)
        if model is not None and hasattr(model, "set_alpha"):
            model.set_alpha(self.alpha_after)
            rank_zero_info(
                f"Alpha early-stop phase-1: alpha={self.alpha_after}, continue training"
            )

        self._reset_checkpoint_scores(trainer)

    def on_fit_start(self, trainer, pl_module):
        """续训：ckpt 里 patience 已耗尽但 phase 仍是 0 → 立刻进第二阶段，不要再停一次。"""
        if hasattr(EarlyStopping, "on_fit_start"):
            super().on_fit_start(trainer, pl_module)
        if self.phase != 0 or self.first_stop_info is not None:
            return
        stopped = int(getattr(self, "stopped_epoch", 0) or 0) > 0
        exhausted = self.wait_count >= int(self.patience)
        if not (stopped or exhausted):
            return
        rank_zero_info(
            "Alpha early-stop: resume with patience already exhausted "
            f"(wait_count={self.wait_count}, stopped_epoch={self.stopped_epoch}); "
            "treat as first stop and enter phase-1"
        )
        trainer.should_stop = True
        self._maybe_enter_phase2(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        self._maybe_enter_phase2(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        self._maybe_enter_phase2(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(EarlyStopping, "on_validation_epoch_end"):
            super().on_validation_epoch_end(trainer, pl_module)
        self._maybe_enter_phase2(trainer, pl_module)


def collect_early_stop_report(trainer) -> dict[str, Any]:
    """训练结束时写入 ``train_report.yaml`` 的早停摘要。"""
    cbs = getattr(trainer, "callbacks", None) or []
    two = next((c for c in cbs if isinstance(c, TwoPhaseAlphaEarlyStopping)), None)
    if two is not None:
        return two.to_report(trainer)
    single = next((c for c in cbs if isinstance(c, EarlyStopping)), None)
    if single is None:
        return {"two_phase": False, "first": _not_triggered(), "second": _not_triggered()}
    stopped_epoch = int(getattr(single, "stopped_epoch", 0) or 0)
    triggered = bool(getattr(trainer, "should_stop", False)) or stopped_epoch > 0
    if triggered:
        first = snapshot_early_stop(trainer, single)
        if stopped_epoch:
            first["epoch"] = stopped_epoch
    else:
        first = _not_triggered()
    return {"two_phase": False, "first": first, "second": _not_triggered()}
