"""Atomic small cursor commits; no model or optimizer serialization changes."""
import io
import os
from pathlib import Path
import tempfile


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != payload:
            raise OSError('Cursor readback does not match serialized state')
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def save_cursor(path, state):
    import torch

    for name in ('sample_offset', 'epoch_id', 'sample_group_index', 'sample_index'):
        value = state[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('Invalid rollout cursor: ' + name)
    stream = io.BytesIO()
    torch.save(state, stream)
    payload = stream.getvalue()
    restored = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    for name in ('sample_offset', 'epoch_id', 'sample_group_index', 'sample_index'):
        if restored[name] != state[name]:
            raise ValueError('Rollout cursor serialization mismatch: ' + name)
    atomic_bytes(path, payload)


def bundle_cursor(checkpoint, state_root, rollout_id):
    """Attach the already committed cursor before HF retention/replication."""
    import torch

    name = 'global_dataset_state_dict_' + str(int(rollout_id)) + '.pt'
    source = Path(state_root) / 'rollout' / name
    destination = Path(checkpoint) / 'rollout' / name
    if source.is_symlink() or not source.is_file():
        raise ValueError('Missing committed rollout cursor: ' + str(source))
    payload = source.read_bytes()
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    for key in ('sample_offset', 'epoch_id', 'sample_group_index', 'sample_index'):
        if isinstance(state[key], bool) or not isinstance(state[key], int) or state[key] < 0:
            raise ValueError('Invalid committed cursor: ' + key)
    if destination.exists() and destination.read_bytes() != payload:
        raise ValueError('Existing bundled cursor conflicts with committed cursor')
    atomic_bytes(destination, payload)
