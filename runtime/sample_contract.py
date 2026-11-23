import math
def validate_sample(sample):
    tokens = list(sample.tokens)
    length = int(sample.response_length)
    mask = list(sample.loss_mask or [])
    probs = sample.rollout_log_probs
    if length < 0 or length > len(tokens) or len(mask) != length:
        raise ValueError('response/mask length mismatch')
    if any(v not in (0, 1) for v in mask):
        raise ValueError('loss mask must be binary')
    if sample.remove_sample:
        if any(mask):
            raise ValueError('excluded sample still has trainable tokens')
        return {'excluded': True, 'trainable_tokens': 0}
    if probs is None or len(probs) != length:
        raise ValueError('missing/misaligned generation logprobs; never backfill for TIS')
    if any(not math.isfinite(float(p)) for p in probs):
        raise ValueError('nonfinite generation logprob')
    contexts = (sample.metadata or {}).get('clawgym_nanobot', {}).get('opsd_turns', [])
    covered = set()
    response_start = len(tokens) - length
    for turn in contexts:
        start, end = int(turn['output_start']), int(turn['output_end'])
        if not 0 <= start < end <= length:
            raise ValueError('invalid output span')
        positions = set(range(start, end))
        if positions & covered or any(mask[i] != 1 for i in positions):
            raise ValueError('duplicate output or tool/context token marked as output')
        if tokens[:response_start + start] != list(turn['prompt_ids']):
            raise ValueError('training context differs from actual generation input')
        if tokens[response_start + start:response_start + end] != list(turn['output_ids']):
            raise ValueError('training output differs from sampled token ids')
        expected = turn.get('output_log_probs')
        if expected is None or len(expected) != end-start or any(not math.isfinite(float(v)) for v in expected):
            raise ValueError('missing per-turn original output logprobs')
        if any(abs(float(a)-float(b)) > 1e-6 for a, b in zip(probs[start:end], expected)):
            raise ValueError('training logprobs differ from sampled token logprobs')
        covered |= positions
    if covered != {i for i, value in enumerate(mask) if value}:
        raise ValueError('trainable token lacks exact generation provenance')
    return {'excluded': False, 'trainable_tokens': len(covered), 'turns': len(contexts)}

