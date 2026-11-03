"""NumPy initialization and Torch optimization; phase uses scheme A: circular error before interpolation."""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from .optics import propagate_ortho


@dataclass
class OptimizationResult:
    """Final phase and fields from two update stages, each stored as an independent NumPy array."""
    phase: np.ndarray
    initial_phase: np.ndarray
    reported_field: np.ndarray
    reconstructed_field: np.ndarray
    steps: int
    history: list[dict]
    seed_metadata: dict
    metrics: dict


@dataclass
class _TensorProblem:
    """Torch constants and sampling indices uploaded once, distinct from the CPU optical data record."""
    amplitude: torch.Tensor
    target_amplitude: torch.Tensor
    target_phase: torch.Tensor
    compensation: torch.Tensor
    masks: dict
    grid: torch.Tensor
    patch_y: torch.Tensor  # (L,), row indices of all local selected pixels.
    patch_x: torch.Tensor  # (L,), column indices paired elementwise with the row indices.
    patch_lengths: torch.Tensor  # (K,), contiguous group lengths summing to L.
    patch_weight: torch.Tensor  # (L,), fixed ion-region weights.
    patch_count: torch.Tensor  # (K,), group weight sums clamped to at least 1 for division.
    sampled_amplitude: torch.Tensor


def make_initial_phase(problem, config, *, rng):
    """Implement four legacy seed strategies with an independent NumPy random stream.

    Args:
        problem: CPU OpticalProblem supplying the target field with design curvature.
        config: RunConfig specifying seed strategy and curvature parameters; does not reseed rng.
        rng: Caller-created numpy.random.Generator controlling the draws.

    Returns:
        (phase, metadata): real NumPy phase in radians on the computational grid,
        not necessarily wrapped to [-pi, pi]; metadata records strategy and sampled factors.
    """
    back = propagate_ortho(problem.target, inverse=True)
    angle = np.angle(back)
    magnitude = abs(back)/abs(back).max()
    strategies = {'estimate_noise': 0, 'quadratic': 2, 'tilt_mix': 3, 's1': 4}
    if config.seed_strategy == 'mixed':
        strategy = int(rng.choice([0, 2, 3, 4]))
    elif config.seed_strategy in strategies:
        strategy = strategies[config.seed_strategy]
    else:
        raise ValueError(f'unknown seed strategy {config.seed_strategy}')
    curve = float(rng.uniform(.1, 2))*config.seed_curve
    noise = float(rng.uniform(.0001, .05))
    rows, cols = problem.amplitude.shape
    y = (np.arange(rows)-rows/2+.5)[:, None] 
    x = (np.arange(cols)-cols/2+.5)[None, :]        #generate pixel coordinates with the center (0,0)
    meta = {'source': 'numpy', 'seed': config.seed, 'strategy': strategy,
            'curve': curve, 'noise_scale': noise}
    if strategy == 0:       #add random phase noise
        phase = angle + 2*np.pi*rng.normal(0, noise, (rows, cols))
    elif strategy == 2:     #add random curvature noise
        phase = curve*(y*y+x*x)     #elipse on physical space
    elif strategy == 3:     # Randomly select backpropagated or linear-tilt phase; retain more field phase in high-amplitude regions.
        factor = rng.random((rows, cols))*float(rng.uniform(.05, .2))
        phase = np.where(factor > magnitude, curve*(y+x), angle)
    else:
        tilt = curve*(y+x)  # Mix magnitude-weighted field phase with a random linear-tilt phase.
        phase = magnitude*np.mod(angle+tilt, 2*np.pi)-tilt
    return phase, meta


def _sample(field: torch.Tensor, grid: torch.Tensor, *,
            phase_reference: torch.Tensor | None = None) -> torch.Tensor:
    """Bilinearly sample scalar maps, computing pixelwise circular errors in phase mode.

    Args:
        field: Real Torch tensor (rows, cols). Ordinary mode accepts amplitude,
            intensity, or other scalar maps; phase_reference selects output phase in radians.
        grid: Torch tensor (1, 1, N, 2), with normalized (x, y) coordinates in [-1, 1];
            dtype and device must match field.
        phase_reference: Optional target phase of matching shape, dtype, and device, in radians.
            Samples 1-cos(field-phase_reference), rather than interpolating phase angles.

    Returns:
        N samples or circular errors, preserving zeros, align_corners=True, and gradients.
    """
    if phase_reference is not None:
        if phase_reference.shape != field.shape:
            raise ValueError('phase_reference must match field shape')
        field = 1-torch.cos(field-phase_reference)
    return F.grid_sample(field[None, None], grid, mode='bilinear',
                         padding_mode='zeros', align_corners=True).reshape(-1)


def _patch_indices(problem, sampled_intensity, threshold):
    """Build unpadded grouped pixel indices using nearest-sample assignment and intensity.

    Args:
        problem: CPU OpticalProblem; spots are pixel (row, col), and ion holds region weights.
        sampled_intensity: K target intensities sampled directly by bilinear interpolation.
        threshold: Relative intensity threshold from config.patch_threshold.

    Returns:
        (y, x, lengths): flat grouped pixel indices and K group counts.
        Use normalized-coordinate distances; ties go to the earlier sample. Count each pixel once.
        Zero-intensity samples receive no pixels; retain denominator protection and averaging for empty groups.
    """
    power = abs(problem.target)**2
    sampled_intensity = np.asarray(sampled_intensity)
    active = sampled_intensity > 0
    count = len(problem.spots)
    if not active.any():
        raise ValueError('all target sampling intensities are zero')
    y, x = np.nonzero((power > threshold*sampled_intensity[active].min()) &
                      (abs(problem.masks['ion']) > 0))
    centers = np.asarray(problem.spots)/np.array(power.shape)
    owners = np.empty(len(y), dtype=np.int64)
    # Chunk distances to avoid a full-image-pixels * K tensor; execute only once in _prepare.
    for begin in range(0, len(y), 2048):
        end = begin+2048
        distance = (y[begin:end, None]/power.shape[0]-centers[None, :, 0])**2
        distance += (x[begin:end, None]/power.shape[1]-centers[None, :, 1])**2
        distance[:, ~active] = np.inf
        owners[begin:end] = distance.argmin(axis=1)
    keep = power[y, x] > threshold*sampled_intensity[owners]
    y, x, owners = y[keep], x[keep], owners[keep]
    order = np.argsort(owners, kind='stable')
    return y[order], x[order], np.bincount(owners, minlength=count)


def _prepare(problem, config):
    """Copy and upload fixed arrays once, avoiding full-image transfers inside the loop.

    Args:
        problem: CPU OpticalProblem with target, input amplitude, masks, and pixel spots.
        config: Matching RunConfig defining device, dtype, and relative intensity threshold.

    Returns:
        _TensorProblem with fixed Torch tensors, normalized (x, y) sampling grid,
        unpadded intensity-selected pixel indices, group counts, weights, and sampled target amplitudes.
    """
    device = config.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, config.dtype)
    a = torch.tensor(problem.amplitude, dtype=dtype, device=device)
    ta = torch.tensor(abs(problem.target), dtype=dtype, device=device)
    tp = torch.tensor(np.angle(problem.target), dtype=dtype, device=device)
    compensation = torch.tensor(problem.compensation, dtype=dtype, device=device)
    masks = {k: torch.tensor(v, dtype=dtype, device=device) for k, v in problem.masks.items()}
    spots = torch.tensor(problem.spots, dtype=torch.float64, device=device)
    grid = spots.to(dtype).clone()
    grid[:, 0] = grid[:, 0]/(a.shape[0]-1)*2-1
    grid[:, 1] = grid[:, 1]/(a.shape[1]-1)*2-1      #move the original point in the center to align _sample()
    grid = grid[None, None].flip(-1)        #->(1,1,M,2) and the coordinate convert from (y,x)(see optics.py (rows,cols))) to (x,y)
    sampled_intensity = _sample(ta.square(), grid).cpu().numpy()
    py, px, lengths = _patch_indices(problem, sampled_intensity, config.patch_threshold)
    py = torch.tensor(py, dtype=torch.long, device=device)
    px = torch.tensor(px, dtype=torch.long, device=device)
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    weight = abs(masks['ion'][py, px])
    count = torch.segment_reduce(weight, 'sum', lengths=lengths).clamp_min(1)
    return _TensorProblem(a, ta, tp, compensation, masks, grid, py, px, lengths, weight, count,
                          _sample(ta/ta.max(), grid))


def _forward(amplitude, phase):
    """Preserve original propagation: unnormalized FFT, division by H*W, then power normalization.

    Args:
        amplitude: 2D Torch input amplitude of shape (rows, cols).
        phase: Real Torch phase matching amplitude shape/device/dtype, in radians, with optional gradients.

    Returns:
        Image-plane complex Torch amplitude with unit total discrete power.
    """
    out = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(amplitude*torch.exp(1j*phase))))
    out = out/(amplitude.shape[0]*amplitude.shape[1])
    return out/torch.sqrt(torch.sum(torch.abs(out)**2))


def compute_loss(field, problem, *, step, config):
    """Compute four loss terms and their staged sum; phase uses scheme A.

    Args:
        field: Complex image-plane amplitude from _forward, not an intensity map.
        problem: _TensorProblem from _prepare, not CPU OpticalProblem;
            target, masks, and indices must match the field shape and device.
        step: Zero-based iteration index selecting the early or late loss stage.
        config: RunConfig; first_round determines the stage transition.

    Returns:
        Dictionary of total/overlap/amplitude/phase/local differentiable Torch scalars.
        The four individual terms exclude their stage weights.
    """
    amplitude = abs(field)
    amplitude = amplitude/(amplitude.max()+1e-18)
    phase = torch.angle(field)
    # fidelity to target image by mask
    out = amplitude*torch.exp(1j*phase)*problem.masks['smooth']
    target = problem.target_amplitude*torch.exp(1j*problem.target_phase)*problem.masks['smooth']
    num = torch.sum(torch.conj(target)*out)
    den = torch.sqrt(torch.sum(abs(out)**2))*torch.sqrt(torch.sum(abs(target)**2))
    overlap = 100*(1-torch.real(num/(den+1e-12))) #1e-12 is to stablize the loss where target intensity is very low
    # beam sampled amplitude penalty
    sampled = _sample(amplitude, problem.grid)
    expected = problem.sampled_amplitude
    amp_loss = torch.mean((sampled/(sampled.mean()+1e-12)-expected/(expected.mean()+1e-12))**2) #1e-12 also for stablizing
    # Target phase already includes design curvature; pixelwise subtraction cancels the common curvature.
    # Apply cosine before interpolation so 2*pi branch jumps cannot create spurious fractional phase offsets.
    phase_loss = _sample(phase, problem.grid, phase_reference=problem.target_phase).mean()
    # local phase nonuniformity loss
    patch = torch.remainder(phase[problem.patch_y, problem.patch_x]-
                                problem.compensation[problem.patch_y, problem.patch_x], 2*torch.pi)
    sine = torch.segment_reduce(torch.sin(patch)*problem.patch_weight, 'sum', lengths=problem.patch_lengths)
    cosine = torch.segment_reduce(torch.cos(patch)*problem.patch_weight, 'sum', lengths=problem.patch_lengths)
    resultant = (torch.hypot(sine, cosine)/problem.patch_count).clamp(1e-7, 1.-1e-7)
    local_loss = torch.sqrt(-2*torch.log(resultant)+1e-7).mean()
    # sum up
    total = overlap
    if step >= (2000 if config.first_round else 200):
        total = total+500*local_loss+200*amp_loss+10000*phase_loss
    return dict(total=total, overlap=overlap, amplitude=amp_loss, phase=phase_loss, local=local_loss)


def evaluate(field, problem):
    """Evaluate regional complex-field fidelity and efficiency separately from the loss.

    Args:
        field: 2D NumPy complex image-plane amplitude, usually reported_field or
            reconstructed_field, with the same shape as problem.target.
        problem: CPU OpticalProblem providing the target and region weights.

    Returns:
        Dictionary of Python floats; fidelity is None if a region has zero energy.
        Metrics describe the optimization grid, not reconstruction after hardware postprocessing.
    """
    u = np.asarray(field)
    t = problem.target
    result = {}
    power = np.sum(abs(u)**2)
    for name in ('box', 'ion', 'smooth'):
        weight = problem.masks[name]
        if name == 'smooth':
            weight = weight**2  # The original smooth fidelity multiplies both fields by the mask.
        den = np.sum(abs(u)**2*weight)*np.sum(abs(t)**2*weight)
        result[f'fidelity_{name}'] = float(abs(np.sum(np.conj(u)*t*weight))**2/den) if den > 0 else None
    for name in ('efficiency', 'ion'):
        result[f'power_{name}'] = float(np.sum(abs(u)**2*problem.masks[name])/power)
    return result


def optimize_phase(problem, config, *, initial_phase=None, observer=None):
    """Run Adam with explicit initial-phase priority and original stop/report timing.

    Args:
        problem: CPU OpticalProblem built by build_problem, not a legacy Wu object.
        config: RunConfig defining device, precision, Adam, stopping, and recording.
        initial_phase: Optional real NumPy phase matching the computational grid, in radians;
            takes priority over random initialization and is copied independently.
        observer: Optional observer(completed_steps, metrics), with one-based steps
            and detached scalars, called at recording intervals, stopping, or the final step.

    Returns:
        OptimizationResult. phase is after the final update; reported_field is before
        that update; reconstructed_field is propagated again from final phase.
        Arrays are CPU NumPy; history and both sets of metrics are recorded separately.
    """
    if initial_phase is None:
        initial_phase, meta = make_initial_phase(problem, config, rng=np.random.default_rng(config.seed))
    else:
        meta = {'source': 'provided'}
    initial_phase = np.array(initial_phase, copy=True)
    if initial_phase.shape != problem.amplitude.shape or not np.all(np.isfinite(initial_phase)):
        raise ValueError('initial_phase must be finite and match FFT shape')
    meta['phase_sha256'] = hashlib.sha256(initial_phase.tobytes()).hexdigest()
    data = _prepare(problem, config)
    phase = torch.tensor(initial_phase, device=data.amplitude.device,
                         dtype=data.amplitude.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([phase], lr=config.learning_rate)
    history = []
    # The original double best_loss comparison increments patience every step and never restores best_phi; preserve its step-limit semantics.
    for step in range(min(config.max_steps, config.patience)):
        optimizer.zero_grad()
        field = _forward(data.amplitude, phase)
        losses = compute_loss(field, data, step=step, config=config)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('non-finite optimization loss')
        losses['total'].backward()
        optimizer.step()
        stop = losses['overlap'].item() < config.target_loss
        if (step+1) % config.record_every == 0 or stop or step+1 == min(config.max_steps, config.patience):
            row = {'step': step+1, **{k: v.detach().item() for k, v in losses.items()}}
            history.append(row)
            if observer is not None:
                observer(step+1, row.copy())
        if stop:
            break
    with torch.no_grad():
        final = _forward(data.amplitude, phase).cpu().numpy().copy()
    reported = field.detach().cpu().numpy().copy()
    return OptimizationResult(phase.detach().cpu().numpy().copy(), initial_phase, reported,
                              final, step+1, history, meta,
                              {'reported': evaluate(reported, problem), 'reconstructed': evaluate(final, problem)})
