"""Streaming recognition evaluation CLI.

Usage:
  python streaming_eval.py configs/streaming.yaml \
      --checkpoint .../val-naive-h2-frz-sl1-post-epoch=00-step=1000.ckpt \
      --max-samples 100 --check-offline
"""
import argparse
import logging
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")


def build_model(config, checkpoint, device):
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    model = LibriSpeechRNNTModule.from_config(config)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model = model.to(device).eval()
    return model


def per(pred, ref):
    from difflib import SequenceMatcher
    s = SequenceMatcher(None, pred, ref)
    m = sum(blk.size for blk in s.get_matching_blocks())
    return (max(len(pred), len(ref)) - m) / max(1, len(ref))


def iterate(dataset, max_samples):
    for idx, sample in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break
        yield idx, sample


def main():
    p = argparse.ArgumentParser(description="Streaming recognition eval")
    p.add_argument("config")
    p.add_argument("--checkpoint", "-c", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--check-offline", action="store_true")
    p.add_argument("--preview", action="store_true", default=True)
    p.add_argument("--no-preview", dest="preview", action="store_false")
    p.add_argument("--chunk-frames", type=int, default=None)
    p.add_argument("--left-context-frames", type=int, default=None)
    p.add_argument("--right-context-frames", type=int, default=None)
    p.add_argument("--preview-heads", type=int, default=None)
    args = p.parse_args()

    from common import load_config
    config = load_config(args.config)
    st = config.get("streaming", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, args.checkpoint, device)

    sc = ChunkPlan_vals(args, st)
    dcfg = config["data"]
    from conformer_rnnt.lightning_module import LIBRISPEECH
    dataset = LIBRISPEECH(config["paths"]["dataset_path"],
                          url=dcfg.get("test_subset", "dev-clean"))

    run_eval(model, dataset, dcfg, config["paths"], device, args, sc)


def ChunkPlan_vals(args, st):
    def val(x, key, dflt):
        return x if x is not None else st.get(key, dflt)
    return {
        "chunk_frames": val(args.chunk_frames, "chunk_frames", 16),
        "left_context_frames": val(args.left_context_frames, "left_context_frames", 64),
        "right_context_frames": val(args.right_context_frames, "right_context_frames", 4),
        "preview": args.preview,
        "preview_heads": args.preview_heads or st.get("preview_heads", 2),
    }


def run_eval(model, dataset, dcfg, paths, device, args, sc):
    from conformer_rnnt.streaming.features import StreamingFeatureExtractor
    from conformer_rnnt.streaming.recognizer import StreamingRecognizer
    from conformer_rnnt.spec_decode import greedy_decode

    total_per, tok_hits, tok_tot, n = 0.0, 0, 0, 0
    t0 = time.time()
    total_audio = 0.0
    for idx, sample in iterate(dataset, args.max_samples):
        wav, sr, transcript = sample[0], sample[1], sample[2]
        total_audio += wav.size(-1) / float(sr)
        wav = wav.squeeze(0) if wav.dim() == 2 else wav
        ex = StreamingFeatureExtractor(
            sample_rate=dcfg["sample_rate"], n_fft=dcfg["n_fft"],
            n_mels=dcfg["n_mels"], hop_length=dcfg["hop_length"],
            global_stats_path=paths["global_stats_path"])
        mel = ex.compute_all(wav).to(device)
        blank = model.blank_idx
        rec = StreamingRecognizer(model=model.model, mel=mel,
                                  chunk_frames=sc["chunk_frames"],
                                  left_context_frames=sc["left_context_frames"],
                                  right_context_frames=sc["right_context_frames"],
                                  stride=4, blank_idx=blank,
                                  preview=sc["preview"],
                                  preview_heads=sc["preview_heads"])
        pred = rec.run_offline()
        ref_ids = model.sp_model.encode(transcript.lower().replace(" ", ""))
        total_per += per(pred, ref_ids)
        if args.check_offline:
            feat = torch.cat([mel.unsqueeze(0), torch.zeros(1, 4, mel.size(1),
                                                            device=device)], dim=1)
            off = greedy_decode(model.model, feat,
                                torch.tensor([mel.size(0)], dtype=torch.int32,
                                             device=device),
                                blank)
            hits = sum(a == b for a, b in zip(pred, off))
            tok_hits += hits
            tok_tot += max(len(pred), len(off))
        n += 1
        pace = (time.time() - t0) / n
        logging.info("[%3d] PER %.3f | cum PER %.4f | %.2fs/sample",
                     idx, total_per / n, total_per / n, pace)
    if n:
        dec_sec = time.time() - t0
        logging.info("=== streaming eval ===\n"
                     "PER %.4f | RTF %.3f (decode wallclock/audio) | samples %d",
                     total_per / n, dec_sec / max(total_audio, 1e-6), n)
        if args.check_offline:
            logging.info("token match vs offline greedy: %.3f%%",
                         100.0 * tok_hits / max(1, tok_tot))


if __name__ == "__main__":
    main()