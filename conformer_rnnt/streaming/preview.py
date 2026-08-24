"""Display-only Medusa lookahead. Never feeds back into the decoder state.

``predict_heads`` is a pure function of (enc window, g_last); draft heads are
read and discarded, so decoder state is untouched.

Acceptance convention (matches the offline post-emit semantics): at preview time
the predictor is at emission count ``u = len(decoder.out)`` and head ``k`` (1-based)
predicts the token at final-output position ``u + k``. That is, ``drafts[0]``
(head 1) predicts the token greedy will emit one step after the current one.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class MedusaPreview:
    def __init__(self, *, model, max_heads: Optional[int] = None) -> None:
        self.model = model
        self.max_heads = max_heads

    def lookahead(self, decoder, enc: torch.Tensor) -> List[int]:
        heads = self.model.medusa_heads
        if heads is None:
            return []
        k = self.max_heads if self.max_heads is not None else heads.num_heads
        lookahead = int(getattr(heads, "lookahead_frames", 0))
        t = decoder.t
        want = max(1, 1 + lookahead)
        window = enc[:, t: t + want, :]
        if window.size(1) == 0:
            return []
        if window.size(1) < want:
            window = F.pad(window, (0, 0, 0, want - window.size(1)))
        _, future = self.model.predict_heads(window, decoder.g_last)
        return [int(f.argmax(dim=-1).item()) for f in future[:k]]


def preview_acceptance(preview_log: List[Tuple[int, List[int]]],
                       final_out: List[int]) -> List[Dict[str, float]]:
    """Per-head acceptance of preview drafts vs the final greedy output.

    Semantics match offline ``draft_after_emit``: a draft is sampled right after
    an emission, when the predictor has advanced to position ``u = len(out)``.
    Head ``k`` (1-based; drafts[0] = head 1) predicts the next emission(s):
    head 1 -> ``final_out[u]``, head k -> ``final_out[u + k - 1]``. So a preview
    entry's draft at index j is checked against ``final_out[u + j]``.
    """
    total: Dict[int, int] = {}
    hit: Dict[int, int] = {}
    for u, drafts in preview_log:
        for j, d in enumerate(drafts):
            idx = u + j
            total[j] = total.get(j, 0) + 1
            if idx < len(final_out) and d == final_out[idx]:
                hit[j] = hit.get(j, 0) + 1
    out = []
    for j in sorted(total):
        tj = total[j]
        hj = hit.get(j, 0)
        out.append({"head": j + 1, "accepted": hj, "drafts": tj,
                    "rate": hj / float(tj)})
    return out