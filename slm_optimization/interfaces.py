"""Correction-data and device contracts; no vendor drivers or legacy SLM classes are imported."""
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Protocol
import numpy as np


class SLMDevice(Protocol):
    """Caller-provided device accepting only encoded grayscale frames of hardware shape."""
    shape: tuple[int, int]

    def display(self, frame: np.ndarray) -> None:
        """Display a uint8 grayscale frame; the caller manages device connection and shutdown."""
        ...


@dataclass
class CorrectionData:
    """Optional corrections with explicit planes and units; None disables a correction."""
    slm_phase: np.ndarray | None = None
    target_phase: np.ndarray | None = None
    spot_amplitude: np.ndarray | None = None
    spot_phase: np.ndarray | None = None
    beam_width: np.ndarray | None = None
    lut: np.ndarray | None = None


def validate_corrections(data: CorrectionData, config) -> None:
    """Validate shapes and values without inferring units or modifying caller arrays.

    Args:
        data: CorrectionData with finite real arrays in non-None fields. Phases
            use radians, amplitude/width use multipliers, and LUT values use grayscale.
            Fields are replaced by float64 copies; copy the record first if needed.
        config: RunConfig defining hardware/target shapes, modes, and sample count.

    Returns:
        None. Invalid shapes, value ranges, or unsupported TEM01 widths raise errors.
        load_corrections validates unit strings; direct arrays must follow these units.
    """
    shapes = {'slm_phase': config.hardware_shape, 'target_phase': config.shape,
              'beam_width': (config.spot_count, 2)}
    count = config.spot_count * (2 if config.mode == 'tem01' else 1)
    for name in ('spot_amplitude', 'spot_phase'):
        shapes[name] = (count,)
    for item in fields(data):
        value = getattr(data, item.name)
        if value is None:
            continue
        if np.iscomplexobj(value):
            raise ValueError(f'{item.name} must be real-valued')
        value = np.array(value, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(value)):
            raise ValueError(f'{item.name} contains non-finite values')
        if item.name in shapes and value.shape != shapes[item.name]:
            raise ValueError(f'{item.name} shape must be {shapes[item.name]}')
        if item.name == 'beam_width' and np.any(value <= 0):
            raise ValueError('beam_width factors must be positive')
        if item.name == 'beam_width' and config.mode == 'tem01' and not np.all(value == 1):
            raise ValueError('TEM01 width correction requires a calibrated lobe-position model; use TEM00')
        if item.name == 'spot_amplitude' and (np.any(value < 0) or not np.any(value > 0)):
            raise ValueError('spot_amplitude must be nonnegative and nonzero')
        if item.name == 'lut' and (value.ndim != 1 or len(value) < 2 or np.any((value < 0) | (value > 255))):
            raise ValueError('lut must be a one-dimensional gray-level table')
        setattr(data, item.name, value)


def load_corrections(config) -> CorrectionData:
    """Load enabled corrections; phases require radians or an explicit grayscale-to-phase scale."""
    data = CorrectionData()
    valid = {f.name for f in fields(data)}
    for name, entry in config.corrections.items():
        if name not in valid:
            raise ValueError(f'unknown correction {name}')
        if entry is None or entry.get('enabled', True) is False:
            continue
        path = Path(entry['path'])
        if not path.is_absolute():
            path = Path(config.preset_dir) / path
        unit = entry.get('unit')
        expected = 'rad' if 'phase' in name else ('gray' if name == 'lut' else 'factor')
        if unit != expected and not ('phase' in name and unit == 'gray' and entry.get('levels')):
            raise ValueError(f'{name} unit must be {expected}, or explicit gray/levels for phase')
        if path.suffix.lower() == '.npy':
            value = np.load(path, allow_pickle=False)
        elif path.suffix.lower() == '.npz':
            with np.load(path, allow_pickle=False) as z:
                value = z[entry.get('key', 'data')].copy()
        elif path.suffix.lower() in ('.txt', '.csv'):
            value = np.loadtxt(path, delimiter=',' if path.suffix.lower() == '.csv' else None)
        elif path.suffix.lower() == '.bmp':
            from PIL import Image
            with Image.open(path) as image:
                value = np.asarray(image).copy()
        else:
            raise ValueError(f'unsupported correction format: {path.suffix}')
        if unit == 'gray' and 'phase' in name:
            value = value / float(entry['levels']) * (2 * np.pi)
        if 'patch_size' in entry:
            value = np.repeat(np.repeat(value, entry['patch_size'], 0), entry['patch_size'], 1)
            canvas = np.zeros(config.hardware_shape)
            area = int(entry['canvas_size'])
            if area > min(canvas.shape) or max(value.shape) > area:
                raise ValueError('patch correction does not fit canvas')
            y, x = [(n - area) // 2 for n in canvas.shape]
            canvas[y:y+value.shape[0], x:x+value.shape[1]] = value
            value = canvas
        setattr(data, name, value)
    validate_corrections(data, config)
    return data
