#!/usr/bin/env python3
"""Check an IID-distance null for native peak+1/valley centre selection.

This CPU-only combinatorial analysis does not load a model, audio or config.
Continuous IID distances are a diagnostic assumption, not an ASR result or a
target retention rate. Strict comparisons intentionally exclude plateau ties.
"""
import argparse
from fractions import Fraction
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import random


def centres(distances):
    if len(distances) < 3:
        return [0]
    selected = set()
    for i in range(1, len(distances) - 1):
        left, value, right = distances[i - 1:i + 2]
        if value > left and value > right:
            selected.add(i + 1)
        if value < left and value < right:
            selected.add(i)
    return sorted(selected) or [0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    counts = dict(orderings=0, peak=0, valley=0, intersection=0, union=0)
    for a, b, c, d in permutations(range(4)):
        peak = a < b > c
        valley = b > c < d
        for key, value in dict(orderings=1, peak=peak, valley=valley,
                               intersection=peak and valley, union=peak or valley).items():
            counts[key] += int(value)
    assert counts == dict(orderings=24, peak=8, valley=8, intersection=5, union=11)
    probability = Fraction(counts['union'], counts['orderings'])
    rng = random.Random(42)
    simulations = []
    for n in (1000, 10000, 100000):
        curve = [rng.random() for _ in range(n - 1)]
        simulations.append(dict(hidden_frames=n, kept_centres=len(centres(curve)),
                                observed_K_over_n=len(centres(curve)) / n))
    curve = [rng.random() for _ in range(1000)]
    base = centres(curve)
    transforms = {'affine': lambda x: 3 * x + 1,
                  'square_on_positive_domain': lambda x: x * x,
                  'exponential': math.exp}
    invariant = {name: centres([transform(x) for x in curve]) == base
                 for name, transform in transforms.items()}
    assert all(invariant.values())
    families = {
        'monotonic_ramp': [i / 998 for i in range(999)],
        'alternating': [i % 2 for i in range(999)],
        'smooth_eight_cycles': [0.5 + 0.4 * math.sin(2 * math.pi * 8 * i / 998)
                               for i in range(999)],
    }
    report = dict(
        analysis_date='2026-10-18', seed=42,
        purpose='diagnostic_null_not_a_config_parameter_or_neural_ASR_result',
        assumptions=['continuous IID distances for the probability calculation',
                     'interior frame; finite-sequence endpoints are excluded from the limit',
                     'strict extrema; no plateau ties', 'mapping: peak -> p+1, valley -> v'],
        exact_order_counts=counts, interior_keep_probability=str(probability),
        interior_keep_probability_decimal=float(probability),
        simulations=simulations, monotonic_transform_checks=invariant,
        curve_family_checks={name: dict(hidden_frames=len(values) + 1,
                                       kept_centres=len(centres(values)))
                             for name, values in families.items()},
        learned_predictor_parameters=640 * 32 + 32 + 32 * 5 + 5,
        proposed_capacity_control_parameters=128 * 80 + 80 + 80 * 128 + 128,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_files_modified=False,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(valid=True, out=str(args.out), probability=str(probability)), indent=2))


if __name__ == '__main__':
    main()
