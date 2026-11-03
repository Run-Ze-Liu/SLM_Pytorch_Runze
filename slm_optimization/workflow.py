"""Single and batch workflows with saved outputs; experimental devices connect only through explicit interfaces."""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid
import numpy as np
from PIL import Image
from .config import validate_config
from .interfaces import load_corrections, validate_corrections, CorrectionData, SLMDevice
from .optics import build_problem
from .optimization import optimize_phase, OptimizationResult


@dataclass
class RunResult:
    """One optimization result, intermediate phases, encoded frames, and the saved run directory."""
    optimization: OptimizationResult
    products: dict
    frames: dict
    run_dir: Path
    problem: object


@dataclass
class RunSummary:
    """Keep only paths and scalars in batch summaries to avoid accumulating full fields in host memory."""
    run_dir: Path
    steps: int
    seed: int
    metrics: dict


def postprocess_phase(phase, config, *, corrections=None):
    """Crop, add steering/correction, and retain central-circle phase in the original order.

    Args:
        phase: 2D real NumPy phase on the computational grid, in radians, usually optimized.
        config: RunConfig providing hardware shape, steering angles, wavelength, and region size.
        corrections: Optional CorrectionData; only hardware_shape slm_phase in radians is used here.
            Apply other corrections at their respective stages.

    Returns:
        NumPy dictionary: cropped_phase is the hardware-centered crop of optimized phase;
        hardware_phase adds steering; export_phase adds correction and region processing.
        All have hardware_shape, before global offsets or grayscale encoding; no amplitude aperture is applied.
    """
    rows, cols = config.hardware_shape
    y, x = np.array(phase.shape)//2-np.array(config.hardware_shape)//2
    cropped = np.array(phase[y:y+rows, x:x+cols], copy=True)
    yy, xx = np.indices(config.hardware_shape)
    gradients = []
    for axis, (angle, pitch) in enumerate(zip(config.steering_deg, config.pixel_pitch_m)):
        delta = 2*np.pi*pitch/config.wavelength_m*np.tan(np.deg2rad(angle))
        ramp = (yy if axis == 0 else xx)*delta
        gradients.append(np.mod(ramp+np.pi*((yy+xx) % 2), 2*np.pi))
    # Original ArrayModulator.add wrapped after each phase addition; preserve the checkerboard cancellation order.
    hardware = np.mod(np.mod(gradients[0]+gradients[1], 2*np.pi)+cropped, 2*np.pi)
    corrected = hardware.copy()
    if corrections is not None and corrections.slm_phase is not None:
        corrected += corrections.slm_phase
    size = config.crop_size
    cy, cx = rows//2, cols//2
    y1, x1 = cy-size//2, cx-size//2
    export = np.zeros(config.hardware_shape)
    patch = corrected[y1:y1+size, x1:x1+size].copy()
    py, px = np.indices((size, size))
    patch[(py-size//2)**2+(px-size//2)**2 > (size//2)**2] = 0
    export[y1:y1+size, x1:x1+size] = patch
    return {'cropped_phase': cropped, 'hardware_phase': hardware, 'export_phase': export}


def _encode_phase(phase, config, lut):
    """Encode grayscale once so saved files and SLM display use the same frame.

    Args:
        phase: Finite real NumPy phase in radians, including the chosen global offset.
        config: RunConfig; use linear lut_levels encoding when lut is None.
        lut: Optional 1D grayscale table sampled uniformly over [0, 2*pi]; interpolate here,
            without reapplying the LUT during saving or device display.

    Returns:
        uint8 grayscale array with the same shape as phase.
    """
    if not np.isrealobj(phase) or not np.all(np.isfinite(phase)):
        raise ValueError('phase frame must contain finite real radians')
    fraction = np.mod(phase+2*np.pi, 2*np.pi)/(2*np.pi)
    if lut is None:
        return (fraction*config.lut_levels).astype(np.uint8)
    return np.interp(fraction, np.linspace(0, 1, len(lut)), lut).astype(np.uint8)


def save_run(result, run_dir, config):
    """Save complete parameters, initial phase, outputs, and explicitly timed metrics.

    Args:
        result: RunResult with NumPy optimization results, stage phases, frames, and optical problem.
        run_dir: Already-created run directory; fixed filenames are overwritten on repeated calls.
        config: Actual RunConfig, including preset defaults and user overrides.

    Returns:
        None; write NPY, BMP, JSON, and JSONL without rerunning optimization or metrics.
    """
    run_dir = Path(run_dir)
    opt = result.optimization
    arrays = {'initial_phase': opt.initial_phase, 'target_field': result.problem.target,
              'phase_fft': opt.phase, 'reported_field': opt.reported_field,
              'reconstructed_field': opt.reconstructed_field,
              'phase_hardware': result.products['hardware_phase'],
              'phase_export': result.products['export_phase'], 'spots': result.problem.spots}
    for name, array in arrays.items():
        np.save(run_dir/f'{name}.npy', array, allow_pickle=False)
    for angle, frame in result.frames.items():
        Image.fromarray(frame).save(run_dir/f'phase_mask0_{angle:g}.bmp')
    metadata = {'steps': opt.steps, 'phase_iteration': opt.steps,
                'reported_field_iteration': opt.steps-1, 'seed': opt.seed_metadata,
                'mask_geometry': 'physical_circle_original_major_axis',
                'reported': opt.metrics['reported'], 'reconstructed': opt.metrics['reconstructed'],
                'exported_phase_metrics': None}
    (run_dir/'metrics.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    (run_dir/'config.json').write_text(json.dumps(config.snapshot(), indent=2), encoding='utf-8')
    (run_dir/'history.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in opt.history), encoding='utf-8')


def run_once(config, *, corrections=None, observer=None, slm: SLMDevice | None = None):
    """Execute and save a complete run with no experimental files or hardware by default.

    Args:
        config: RunConfig defining one numerical workflow and its output_root.
        corrections: Optional CorrectionData arrays, taking priority over config.corrections.
            If None, load enabled correction paths from the config;
            an empty path configuration means no corrections.
        observer: Optional observer(completed_steps, metrics), with the optimize_phase contract;
            receives only detached scalars.
        slm: Optional SLMDevice with hardware_shape; display one configured uint8 frame.
            None avoids device access; the caller manages connection and shutdown.

    Returns:
        RunResult with optimization results, stage phases, encoded frames, and a unique run_dir.
        Save automatically; failures after directory creation write failure.json and reraise.
    """
    validate_config(config)
    if slm is not None and tuple(slm.shape) != config.hardware_shape:
        raise ValueError('SLM device shape does not match hardware_shape')
    root = Path(config.output_root).resolve()
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:10]
    run_dir = root/name
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        data = load_corrections(config) if corrections is None else CorrectionData(**vars(corrections))
        validate_corrections(data, config)
        problem = build_problem(config, corrections=data)
        opt = optimize_phase(problem, config, observer=observer)
        products = postprocess_phase(opt.phase, config, corrections=data)
        frames = {angle: _encode_phase(products['export_phase']+np.deg2rad(angle), config, data.lut)
                  for angle in config.phase_offsets_deg}
        result = RunResult(opt, products, frames, run_dir, problem)
        save_run(result, run_dir, config)
        if slm is not None:
            slm.display(frames[config.display_offset_deg].copy())
        return result
    except Exception as error:
        (run_dir/'failure.json').write_text(json.dumps({'type': type(error).__name__,
            'message': str(error), 'config': config.snapshot()}, indent=2), encoding='utf-8')
        raise


def run_batch(configs, *, output_root, repeats, observer=None):
    """Call the single-run entry point serially with independent NumPy random substreams.

    Args:
        configs: Iterable of RunConfig objects; run each configuration repeats times.
        output_root: Batch output root overriding each configuration output_root.
        repeats: Positive repetition count; individual configuration repeats values are ignored.
        observer: Optional callback forwarded to run_once; do not retain graphs across runs.

    Returns:
        RunSummary list in configuration/repetition order, containing paths, steps, child seeds,
        and metrics without full fields; failures propagate as in run_once.
    """
    if repeats <= 0:
        raise ValueError('repeats must be positive')
    results = []
    for config in configs:
        children = np.random.SeedSequence(config.seed).spawn(repeats)
        for child in children:
            cfg = replace(config, output_root=str(output_root), seed=int(child.generate_state(1)[0]))
            result = run_once(cfg, observer=observer)
            results.append(RunSummary(result.run_dir, result.optimization.steps, cfg.seed,
                                      result.optimization.metrics))
            del result
    # Return lightweight batch summaries to avoid retaining 20 full-size targets and fields.
    return results
