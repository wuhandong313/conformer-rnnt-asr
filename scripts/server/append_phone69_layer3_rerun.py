#!/usr/bin/env python3
"""Run one fresh L3 repeat after the 256-wide stride-4 queue."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.server.append_phone69_placement_queue import matches, predecessor_done, read
from scripts.server.phone69_placement_supplement import PlacementQueue, load_plan, save, sha
from scripts.server.phone69_stride4_dim256 import AUDIT as STRIDE4_AUDIT


PLAN = ROOT / 'configs/phone69_layer3_rerun_20261119.json'
PRIOR = ROOT / STRIDE4_AUDIT
PRIOR_SCRIPT = 'phone69_stride4_dim256.py'
NAME = 'layer3_rerun'


def repeat_recipe():
    path, plan = load_plan(PLAN)
    item, = plan['experiments']
    actual = yaml.safe_load((ROOT / item['config']).read_text())
    expected = yaml.safe_load((ROOT / item['baseline_config']).read_text())
    old_path = expected['paths']['exp_dir']
    expected['paths']['exp_dir'] = old_path.removesuffix('_supp20261112') + '_rerun20261119'
    expected['paths']['from_checkpoint'] = None
    budget = expected['data']['max_token_limit']
    search = plan['batch_search']
    if (item['name'] != NAME or item['action'] != 'train' or item['positions'] != [3]
            or actual != expected or not budget == search['minimum'] == search['initial'] == search['maximum']):
        raise ValueError('Expected one fresh L3 repeat with the archived recipe and fixed batch budget')
    return path, plan, actual


def verify_stride4_completion(state, completion):
    if (state.get('experiment') != 'stride4_dim256' or state.get('status') != 'complete'
            or not completion or completion.get('epoch') != 50
            or any(state.get(key) != value for key, value in completion.items())):
        raise RuntimeError('The stride-4 completion record does not match the queue')
    boundary = completion['epoch_boundary_metadata']
    best = Path(completion['best_checkpoint'])
    if (boundary['resume_completed_epochs'] != 50 or not boundary['resume_at_epoch_boundary']
            or boundary['resume_global_step'] != completion['global_step']
            or not Path(completion['epoch_boundary_checkpoint']).is_file()
            or not best.is_file() or sha(best) != completion['best_checkpoint_sha256']):
        raise RuntimeError('The stride-4 completion record lacks verified final and best checkpoints')
    return completion


def wait_for_stride4(queue, budget):
    state_path = PRIOR / 'queue_state.json'
    initial = read(state_path)
    if not initial or initial.get('experiment') != 'stride4_dim256':
        raise RuntimeError('The expected stride-4 controller has not registered')
    pid = initial['controller_pid']
    while True:
        alive = matches(pid, PRIOR_SCRIPT)
        state = read(state_path)
        if not state or state['controller_pid'] != pid or state['code_commit'] != initial['code_commit']:
            raise RuntimeError('The preceding stride-4 controller identity changed')
        if predecessor_done(state, alive):
            record = verify_stride4_completion(state, read(PRIOR / 'completion.json'))
            save(queue.audit / 'predecessor_verified.json', record)
            return
        queue.state('waiting_for_predecessor', experiment=NAME, positions=[3], budget=budget,
                    predecessor_experiment='stride4_dim256', predecessor_pid=pid,
                    predecessor_status=state['status'], predecessor_training_pid=state.get('training_pid'))
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('validate', 'run'), required=True)
    args = parser.parse_args()
    path, _, config = repeat_recipe()
    budget = config['data']['max_token_limit']
    if args.mode == 'validate':
        print(json.dumps(dict(experiment=NAME, after='stride4_dim256', budget=budget, seed=config['seed'])))
        return

    experiment = Path(config['paths']['exp_dir'])
    if experiment.parent != ROOT or experiment.exists():
        raise ValueError('Expected a fresh experiment directory in the training checkout')
    os.chdir(ROOT)
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0',
                      OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8')
    queue = PlacementQueue(path)
    try:
        wait_for_stride4(queue, budget)
        queue.unchanged()
        queue.run()
    except BaseException as error:
        queue.state('failed', experiment=NAME, error=repr(error))
        raise


if __name__ == '__main__':
    main()
