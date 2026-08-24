"""Display-only Medusa lookahead. Never feeds back into the decoder state.

``predict_heads`` is a pure function of (enc window, g_last); draft heads are
read and discarded, so decoder state is untouched.
"""
from typing import List, Optional

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