"""Phase optimization with one shared field gather and validated segment lengths."""
from dataclasses import dataclass

import torch

from .optimization import OptimizationResult, _TensorProblem, _forward, _optimize_phase
from .optimization import _prepare as _prepare_base


@dataclass
class _SharedProblem(_TensorProblem):
    """Fixed union indices and maps from each loss support into the shared field."""
    support_indices: torch.Tensor  # (U,), unique flattened image-plane indices.
    overlap_map: torch.Tensor  # (O,), shared indices in overlap-weight order.
    sample_map: torch.Tensor  # (K, 16), shared indices in bicubic-neighbor order.
    patch_map: torch.Tensor  # (L,), shared indices in grouped local-pixel order.


def _prepare(problem, config):
    """Reuse current optical preparation, validate lengths, and cache one index union.

    Args:
        problem: CPU OpticalProblem containing the target, masks, and sampling points.
        config: RunConfig selecting the device, precision, and local threshold.

    Returns:
        _SharedProblem. Treat its indices and lengths as fixed throughout optimization.
    """
    data = _prepare_base(problem, config)
    lengths = data.patch_lengths.detach().cpu()
    if (lengths.ndim != 1 or torch.any(lengths < 0)
            or lengths.sum().item() != data.patch_weight.numel()):
        raise ValueError('patch lengths must be nonnegative and cover all local weights')

    cols = data.amplitude.shape[1]
    oy, ox, _ = data.overlap
    sy, sx, _ = data.sample_pixel
    indices = torch.cat((oy*cols+ox, (sy*cols+sx).flatten(),
                         data.patch_y*cols+data.patch_x))
    support, inverse = torch.unique(indices, sorted=True, return_inverse=True)
    # Include every branch, including zero-weight boundary neighbors, in this union.
    overlap_map, sample_map, patch_map = inverse.split(
        [oy.numel(), sy.numel(), data.patch_y.numel()])
    return _SharedProblem(**vars(data), support_indices=support, overlap_map=overlap_map,
                          sample_map=sample_map.reshape(sy.shape), patch_map=patch_map)


def compute_loss(field, problem, *, step, config):
    """Preserve the current loss formulas while sharing the full-field gradient writeback.

    Args:
        field: Complex image-plane Torch amplitude matching the prepared grid.
        problem: _SharedProblem returned by this module's _prepare.
        step: Zero-based Adam iteration controlling the auxiliary-loss stage.
        config: Matching RunConfig supplying first_round.

    Returns:
        Differentiable total, overlap, amplitude, phase, and local scalar losses.
    """
    selected = field.reshape(-1)[problem.support_indices]
    _, _, ow = problem.overlap
    ofield = selected[problem.overlap_map]
    oamplitude = abs(ofield)
    scale = oamplitude.max()+1e-18
    out = oamplitude*torch.exp(1j*torch.angle(ofield))*ow/scale
    num = torch.sum(problem.cache['overlap_target_conj']*out)
    den = torch.sqrt(torch.sum(abs(out)**2))*problem.cache['overlap_target_norm']
    overlap = 100*(1-torch.real(num/(den+1e-12)))

    _, _, sw = problem.sample_pixel
    sfield = selected[problem.sample_map]
    sampled = (abs(sfield)*sw).sum(dim=1)/scale
    expected = problem.cache['sampled_target_amplitude']
    amp_loss = torch.mean((sampled/(sampled.mean()+1e-12)
                           - expected/(expected.mean()+1e-12))**2)
    phase_error = 1-torch.cos(torch.angle(sfield)-problem.cache['sample_target_phase'])
    phase_loss = (phase_error*sw).sum(dim=1).mean()

    patch = torch.remainder(torch.angle(selected[problem.patch_map])
                            - problem.cache['patch_compensation'], 2*torch.pi)
    # Lengths were checked once in _prepare; skip repeated GPU-to-CPU validation.
    sine = torch.segment_reduce(torch.sin(patch)*problem.patch_weight,
                                'sum', lengths=problem.patch_lengths, unsafe=True)
    cosine = torch.segment_reduce(torch.cos(patch)*problem.patch_weight,
                                  'sum', lengths=problem.patch_lengths, unsafe=True)
    resultant = (torch.hypot(sine, cosine)/problem.patch_count).clamp(1e-7, 1.-1e-7)
    local_loss = torch.sqrt(-2*torch.log(resultant)+1e-7).mean()

    total = overlap
    if step >= (2000 if config.first_round else 200):
        total = total+500*local_loss+200*amp_loss+10000*phase_loss
    return dict(total=total, overlap=overlap, amplitude=amp_loss, phase=phase_loss, local=local_loss)


def optimize_phase(problem, config, *, initial_phase=None, observer=None) -> OptimizationResult:
    """Run the current Adam loop using shared sampling and fixed segment lengths.

    Args:
        problem: CPU OpticalProblem built for config.
        config: RunConfig defining Adam, stopping, precision, and recording.
        initial_phase: Optional NumPy phase on the FFT grid, copied by the shared loop.
        observer: Optional callback receiving completed steps and recorded loss scalars.

    Returns:
        The standard OptimizationResult with final phase, fields, history, and metrics.
    """
    return _optimize_phase(problem, config, initial_phase=initial_phase, observer=observer,
                           prepare=_prepare, loss_fn=compute_loss)
