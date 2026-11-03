"""File presets and run configuration; shapes and positions use (row, col), and phases use radians."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json
import math
import sys

PRESET_DIR = Path(__file__).resolve().parent.parent / 'presets'


@dataclass
class RunConfig:
    """All numerical parameters for a reproducible run; experimental corrections are disabled by default."""
    name: str = 'custom'
    description: str = ''
    hardware_shape: tuple[int, int] = (1024, 1272)
    pad_factor: int = 2
    pixel_pitch_m: tuple[float, float] = (12.5e-6, 12.5e-6)
    wavelength_m: float = 411e-9
    focal_length_m: float | None = None
    output_spacing_m: tuple[float, float] | None = None  # Physical display-scale metadata; not used to construct masks.
    array_shape: tuple[int, int] = (11, 11)
    mode: str = 'tem00'
    pitch: float = 0.0055 * math.sqrt(2)
    positions: list | None = None  # Normalized (row, col) center offsets relative to the image center.
    angles_deg: list | None = None
    input_width: float = 0.45
    waist_scale: float = 0.55 / 0.62
    waist_reference_rows: int = 1024  # Reference size in the original empirical width formula, not the padded row count.
    waist_reference_beam: float = 0.55
    curvature: float = 0.575
    curvature_factors: tuple[float, float] = (1.49 * 0.978, 1.15 * 1.02)
    mask_radii: tuple[float, float, float, float] = (10., 45., 50., 2.)  # Pixels; divide by min(shape) to obtain a normalized radius.
    mask_edge: float = 4.  # Full cosine-transition width, in the same units as mask_radii.
    box_half_fraction: float = 0.2
    patch_threshold: float = 4e-2  # Local selection: I > this ratio * target intensity at the sampling point.
    seed: int = 42
    seed_strategy: str = 'mixed'
    seed_curve: float = 0.005
    first_round: bool = True
    learning_rate: float = 0.002
    max_steps: int = 6000
    target_loss: float = 0.01
    patience: int = 6000
    record_every: int = 100
    device: str = 'auto'
    dtype: str = 'float32'
    repeats: int = 1
    crop_size: int = 720
    steering_deg: tuple[float, float] = (0.20, -0.27)
    phase_offsets_deg: tuple[float, ...] = (0., 120., 240.)
    lut_levels: int = 102
    display_offset_deg: float = 0.
    corrections: dict = field(default_factory=dict)
    output_root: str = 'runs'
    preset_dir: str = ''

    @property
    def shape(self) -> tuple[int, int]:
        """Return the computational-plane shape; padding does not change the hardware shape."""
        return tuple(n * self.pad_factor for n in self.hardware_shape)

    @property
    def spot_count(self) -> int:
        """Return the number of mode centers; TEM01 has twice as many evaluation points."""
        return len(self.positions) if self.positions is not None else math.prod(self.array_shape)

    def snapshot(self) -> dict:
        """Return the complete configuration for saving, including values omitted by the preset."""
        return asdict(self)


def load_preset(name_or_path, *, overrides=None) -> RunConfig:
    """Load a JSON preset by name or path, then apply explicit overrides.

    Args:
        name_or_path: Name in presets (optional .json suffix), or a JSON file path.
        overrides: Optional dictionary keyed by RunConfig fields. Replace each
            field as a whole; nested dictionaries such as corrections are not merged.

    Returns:
        Validated RunConfig; preset_dir resolves relative correction paths from the preset directory.
    """
    path = Path(name_or_path)
    if not path.is_file():
        path = PRESET_DIR / (str(name_or_path) + ('' if path.suffix else '.json'))
        if not path.is_file():
            path = Path(sys.prefix)/'share/slm-optimization/presets'/path.name
    values = json.loads(path.read_text(encoding='utf-8'))
    values.update(overrides or {})
    unknown = values.keys() - {f.name for f in fields(RunConfig)}
    if unknown:
        raise ValueError(f'unknown configuration fields: {sorted(unknown)}')
    for key in ('hardware_shape', 'pixel_pitch_m', 'array_shape', 'curvature_factors',
                'mask_radii', 'steering_deg', 'phase_offsets_deg', 'output_spacing_m'):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    values['preset_dir'] = str(path.resolve().parent)
    config = RunConfig(**values)
    validate_config(config)
    return config


def validate_config(config: RunConfig) -> None:
    """Validate dimensions, physical quantities, and iteration parameters before allocating large arrays."""
    for name in ('hardware_shape', 'array_shape'):
        value = getattr(config, name)
        if len(value) != 2 or any(not isinstance(n, int) or n <= 0 for n in value):
            raise ValueError(f'{name} must contain two positive integers')
    for name in ('pad_factor', 'max_steps', 'record_every', 'patience', 'repeats', 'waist_reference_rows'):
        n = getattr(config, name)
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if config.mode not in ('tem00', 'tem01') or config.dtype not in ('float32', 'float64'):
        raise ValueError('unsupported mode or dtype')
    for name in ('pixel_pitch_m', 'output_spacing_m'):
        value = getattr(config, name)
        if value is not None and (len(value) != 2 or any(not math.isfinite(x) or x <= 0 for x in value)):
            raise ValueError(f'{name} must contain positive finite values')
    if len(config.mask_radii) != 4 or any(x <= 0 or not math.isfinite(x) for x in config.mask_radii):
        raise ValueError('mask_radii must contain four positive radii')
    if not 0 < config.mask_edge < 2 * config.mask_radii[2]:
        raise ValueError('invalid smooth mask edge')
    if not 0 < config.box_half_fraction <= .5:
        raise ValueError('invalid ROI dimensions')
    if not math.isfinite(config.patch_threshold) or not 0 < config.patch_threshold < 1:
        raise ValueError('patch_threshold must be finite and strictly between 0 and 1')
    for name in ('steering_deg', 'curvature_factors'):
        values = getattr(config, name)
        if len(values) != 2 or not all(math.isfinite(v) for v in values):
            raise ValueError(f'{name} must contain two finite values')
    if any(abs(v) >= 90 for v in config.steering_deg):
        raise ValueError('steering_deg must lie strictly between -90 and 90')
    if not config.phase_offsets_deg or not all(math.isfinite(v) for v in config.phase_offsets_deg):
        raise ValueError('phase_offsets_deg must contain finite values')
    if not all(math.isfinite(v) for v in (config.curvature, config.seed_curve, config.target_loss)):
        raise ValueError('curvature, seed_curve and target_loss must be finite')
    for name in ('input_width', 'waist_scale', 'waist_reference_beam', 'learning_rate', 'wavelength_m'):
        if not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if config.focal_length_m is not None and (not math.isfinite(config.focal_length_m) or config.focal_length_m <= 0):
        raise ValueError('focal_length_m must be positive')
    if not 0 < config.crop_size <= min(config.hardware_shape) or config.crop_size % 2:
        raise ValueError('crop_size must be positive, even, and fit hardware')
    if not 1 <= config.lut_levels <= 255:
        raise ValueError('lut_levels must fit uint8')
    if config.display_offset_deg not in config.phase_offsets_deg:
        raise ValueError('display_offset_deg must be an exported variant')
