"""Check conservative stage-aware batch selection and completion validation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.server.probe_phone69_training_memory import select_batches
from scripts.server import phone69_placement_pipeline as pipeline


class PlacementLaunchTests(unittest.TestCase):
    def test_stage_memory_bound_uses_all_layers_without_a_retention_target(self):
        base = SimpleNamespace(_url='train-clean-100', _walker=['1-1-1', '1-1-2'])
        dataset = SimpleNamespace(base_dataset=base, batches=[[0, 1]],
                                  _transcript=lambda fid, cache: 'AA B C' if fid.endswith('1') else 'D E')
        model = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda value: value.split()))
        config = dict(model=dict(time_reduction_stride=1, sequence_mode='offline', num_layers=10,
                                 num_heads=8, num_symbols=69,
                                 compression=dict(stages=[{'after_layer': 2}, {'after_layer': 5}])),
                      data=dict(hop_length=160))
        samples = {'train-clean-100/1-1-1': 1600, 'train-clean-100/1-1-2': 3200}
        row = select_batches([dataset], samples, model, config)[0]
        self.assertEqual(row['attention_elements'], 2 * 8 * 10 * 21 ** 2)
        self.assertEqual(row['joint_elements'], 2 * 21 * (3 + 1) * 70)
        legacy = deepcopy(config)
        legacy['model'].pop('num_layers')
        legacy['model'].update(num_layers_block1=2, num_layers_block2=8, compression={})
        self.assertEqual(select_batches([dataset], samples, model, legacy), [row])

    def test_completion_requires_full_state_and_final_boundary(self):
        definition = {'version': 2, 'stride': 1}
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, 'REPO', Path(directory)):
            root = Path(directory) / 'experiment'
            root.mkdir()
            checkpoint = dict(state_dict={'weight': torch.ones(1)}, optimizer_states=[{'state': {}}],
                              lr_schedulers=[{'last_epoch': 4}], offline_native_definition=definition,
                              loops={'fit_loop': {'epoch_progress': {'current': {'ready': 50, 'completed': 50}}}},
                              phone69_epoch_boundary={'resume_completed_epochs': 50})
            best, boundary = root / 'best.ckpt', root / 'boundary.ckpt'
            torch.save(checkpoint, best)
            torch.save(checkpoint, boundary)
            (root / 'run.json').write_text(json.dumps(dict(status='trained', epoch=50, global_step=4,
                offline_native_definition=definition, best_checkpoint=str(best), epoch_boundary_checkpoint=str(boundary))))
            item = {'experiment': 'experiment', 'scheme': 'mel'}
            self.assertEqual(pipeline.check_complete(item, definition)['epoch'], 50)
            checkpoint['phone69_epoch_boundary']['resume_completed_epochs'] = 49
            torch.save(checkpoint, boundary)
            with self.assertRaises(AssertionError):
                pipeline.check_complete(item, definition)


if __name__ == '__main__':
    unittest.main()
