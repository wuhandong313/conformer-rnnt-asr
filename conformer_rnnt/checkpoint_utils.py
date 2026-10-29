"""Check full checkpoint eligibility when a phone-token batch budget changes."""

from pathlib import Path


def phone69_resume_metadata(checkpoint, batch_budget, expected_definition=None):
    if not checkpoint.get('optimizer_states') or not checkpoint.get('lr_schedulers'):
        raise ValueError('Resume requires optimizer and scheduler state')
    if expected_definition is not None and checkpoint.get('offline_native_definition') != expected_definition:
        raise ValueError('Checkpoint does not match the configured offline compression stages')
    old_budget = checkpoint.get('hyper_parameters', {}).get('max_token_limit')
    if old_budget is None:
        raise ValueError('Checkpoint does not record its batch budget')
    progress = checkpoint['loops']['fit_loop']['epoch_progress']['current']
    at_boundary = progress['ready'] == progress['completed']
    if old_budget != batch_budget and not at_boundary:
        raise ValueError('Changing batch budget requires a completed epoch boundary')
    return {
        'previous_max_token_limit': old_budget,
        'resume_global_step': checkpoint['global_step'],
        'resume_completed_epochs': progress['completed'],
        'resume_at_epoch_boundary': at_boundary,
    }


def select_phone69_resume_checkpoint(paths, batch_budget, expected_definition=None):
    import torch

    candidates = sorted({Path(path) for path in paths}, key=lambda path: path.stat().st_mtime, reverse=True)
    errors = []
    for path in candidates:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        try:
            metadata = phone69_resume_metadata(checkpoint, batch_budget, expected_definition)
        except ValueError as error:
            errors.append('{}: {}'.format(path, error))
            print('Skipping incompatible resume checkpoint: {}'.format(errors[-1]), flush=True)
            continue
        return str(path), metadata
    if candidates:
        raise ValueError('No safe resume checkpoint: ' + '; '.join(errors))
    return None, {}
