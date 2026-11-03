"""NumPy optical preparation: input, target, empirical curvature, and normalized circular regions."""
from dataclasses import dataclass
import numpy as np
from .config import RunConfig, validate_config
from .interfaces import CorrectionData, validate_corrections


@dataclass
class OpticalProblem:
    """Fixed optical data on the CPU; spots and centers use (row, col) order."""
    amplitude: np.ndarray
    target: np.ndarray
    centers: np.ndarray
    spots: np.ndarray
    compensation: np.ndarray
    masks: dict[str, np.ndarray]
    config: RunConfig


def propagate_ortho(field: np.ndarray, *, inverse=False) -> np.ndarray:
    """Preserve original orthonormal propagation and shift order for odd dimensions.

    Args:
        field: 2D NumPy complex amplitude of shape (rows, cols), not intensity.
        inverse: False propagates from SLM to image plane; True applies the inverse.

    Returns:
        Same-shaped complex amplitude with discrete power conserved by orthonormal FFT normalization.
    """
    if inverse:
        return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(field), norm='ortho'))
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field), norm='ortho'))


def _input_gaussian(config: RunConfig) -> np.ndarray:
    """Generate the original Gaussian amplitude on the full grid without strict zero-padding outside a hardware aperture."""
    rows, cols = config.shape #(2048, 2544)
    width = config.input_width/config.pad_factor
    a = np.exp(-np.linspace(-1, 1, rows)**2/width**2)[:, None]
    a = a * np.exp(-np.linspace(-1, 1, cols)**2/(width*rows/cols)**2)[None, :]
    return a / np.sqrt(np.sum(a*a))


def _spot_layout(config: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    """Construct regular or arbitrary subpixel centers; angles use the legacy image-coordinate convention in degrees."""
    if config.positions is None:
        ny, nx = config.array_shape
        y, x = np.meshgrid(np.linspace(-(ny-1)/2, (ny-1)/2, ny)*config.pitch,
                           np.linspace(-(nx-1)/2, (nx-1)/2, nx)*config.pitch, indexing='ij')
        offsets = np.column_stack((y.ravel(), x.ravel()))
    else:
        offsets = np.asarray(config.positions, dtype=float)
    if offsets.shape != (config.spot_count, 2) or not np.all(np.isfinite(offsets)):
        raise ValueError('positions must be finite (count,2) normalized offsets')
    if np.any((offsets < -.5) | (offsets >= .5)):
        raise ValueError('positions outside image grid')
    angles = np.full(len(offsets), 90.) if config.angles_deg is None else np.asarray(config.angles_deg)
    if angles.shape != (len(offsets),) or not np.all(np.isfinite(angles)):
        raise ValueError('angles_deg must contain one finite angle per mode')
    return (offsets + .5)*np.array(config.shape), angles


def _target_window(config, center, widths):
    """Return a beam-centered window using NumPy half-open indexing.

    Args:
        config: RunConfig providing shape and mode.
        center: Beam center (row, col), allowing subpixel coordinates.
        widths: Normalized local-axis widths (sx, sy), not pixels or meters.

    Returns:
        (y1, y2, x1, x2), including all global samples inside the symmetric continuous
        bounds and clipped to the grid. Half-width is 5*max(sx, sy), times sqrt(2)
        for TEM01; use ceil below and floor above, including the upper sample without shifting global FFT samples.
    """
    half = 5*max(widths)*(np.sqrt(2) if config.mode == 'tem01' else 1.)
    shape = np.asarray(config.shape)
    center = np.asarray(center)
    # Allow floating-point roundoff so mathematically integral window boundaries are not omitted.
    roundoff = 4*np.finfo(float).eps*shape
    lower = center-half*shape-roundoff
    upper = center+half*shape+roundoff
    start = np.maximum(0, np.ceil(lower)).astype(int)
    stop = np.minimum(shape, np.floor(upper).astype(int)+1)
    return start[0], stop[0], start[1], stop[1]


def _target_array(config, centers, angles, corrections):
    """Accumulate local beams with centered windows and original field/peak normalization.

    Args:
        config: RunConfig defining the grid shape, TEM mode, and empirical widths.
        centers: NumPy array (M, 2) for M mode centers in image-plane (row, col)
            pixel coordinates; fractional values are allowed, not normalized offsets.
        angles: M angles in degrees; at 90 degrees the local u axis points right.
            Increasing angles rotate clockwise when image rows increase downward.
        corrections: Validated CorrectionData. beam_width (M, 2) contains local u/v
            width multipliers; spot_amplitude and spot_phase contain amplitude factors
            and phases in radians, with M entries for TEM00 or 2*M for TEM01.
            None uses unit factors or zero phase; each TEM01 pair shares its even phase.

    Returns:
        (target, spots): target is a complex128 field of config.shape with unit
        total discrete power, before design curvature or target_phase corrections.
        spots are (row, col) evaluation points, shape (M, 2) or (2*M, 2).
        Each TEM01 pair gives analytic peaks at u=+/-sqrt(2)*sx and v=0;
        these need not equal discrete maxima or maxima after inter-beam interference.
    """
    rows, cols = config.shape
    scale = 2/(np.pi*config.waist_reference_beam*config.waist_reference_rows)*config.waist_scale
    target = np.zeros(config.shape, np.complex128)
    widths = np.ones((len(centers), 2)) if corrections.beam_width is None else corrections.beam_width
    count = len(centers)*(2 if config.mode == 'tem01' else 1)
    amps = np.ones(count) if corrections.spot_amplitude is None else corrections.spot_amplitude
    phases = np.zeros(count) if corrections.spot_phase is None else corrections.spot_phase
    spots = []
    for k, ((row, col), angle) in enumerate(zip(centers, angles)):
        sx, sy = scale*widths[k]  # Local u/v axes, matching the original w[0]/w[1] convention.
        y0, x0 = row/rows, col/cols
        y1, y2, x1, x2 = _target_window(config, (row, col), (sx, sy))
        if x1 >= x2 or y1 >= y2:
            raise ValueError('beam has empty support on this grid')
        # stop already includes the upper sample; arange preserves 1/cols and 1/rows spacing and matches lobe splitting.
        x = (np.arange(x1, x2)/cols)[None, :]
        y = (np.arange(y1, y2)/rows)[:, None]
        theta = np.deg2rad(angle-90)
        u = np.cos(theta)*(x-x0)+np.sin(theta)*(y-y0)
        v = -np.sin(theta)*(x-x0)+np.cos(theta)*(y-y0)
        if config.mode == 'tem00':
            beam = np.exp(-u*u/(2*sx*sx)-v*v/(2*sy*sy))
            weight = amps[k]*np.exp(1j*phases[k])
            spots.append([row, col])
        else:
            beam = 2*u/sx*np.exp(-u*u/(4*sx*sx)-v*v/(4*sy*sy))
            # Use i/N for both field and lobe splitting; original profile_v1.py:1073-1090 used inclusive endpoints.
            # Both lobes still share phases[2*k], preserving the original phase-weight convention.
            a = np.deg2rad(angle)
            split = -np.sin(a)*(np.arange(x1, x2)[None, :]/cols-x0)
            split = split + np.cos(a)*(np.arange(y1, y2)[:, None]/rows-y0) #split=-u
            weight = np.where(split > 0, amps[2*k], amps[2*k+1])*np.exp(1j*phases[2*k])
            # Analytic peaks are u=+/-sqrt(2)*sx, v=0; convert normalized y/x to row/column pixels separately.
            # Original profile_v1.py:1160,1174-1175 used the column scale on both axes.
            peak_offset = sx*np.sqrt(2)
            delta = np.array([rows*np.cos(a), -cols*np.sin(a)])*peak_offset
            spots.extend([np.array([row, col])+delta, np.array([row, col])-delta])
        peak = beam.max()
        if peak <= 0:
            raise ValueError('beam is undersampled or has no positive peak')
        target[y1:y2, x1:x2] += beam/peak*weight
    power = np.sum(abs(target)**2)
    if power <= 0:
        raise ValueError('target has zero power')
    return target/np.sqrt(power), np.asarray(spots)


def _radial_mask(shape, spots, radius, *, edge=None):
    """Build circular regions in normalized image coordinates using local bounding boxes.

    Args:
        shape: Image-plane sample counts (rows, cols).
        spots: Pixel centers (row, col), shape (N, 2), allowing subpixel positions.
        radius: Pixel-valued radius divided by min(rows, cols) for normalization;
            its pixel value applies to the shorter axis, independent of physical pitch.
        edge: Positive full cosine-transition width in radius units; None gives hard edges.

    Returns:
        Mask of the requested shape: Boolean union for hard edges, or float32
        weights in [0, 1] for smooth edges, taking the maximum in overlaps.
    """
    rows, cols = shape
    size = min(rows, cols)
    r_norm = radius/size
    edge_norm = edge/size if edge is not None else None
    outer_norm = r_norm + (edge_norm/2 if edge_norm is not None else 0)
    mask = np.zeros(shape, np.float32 if edge is not None else bool)
    for row, col in np.asarray(spots, dtype=np.float32):
        y1, y2 = max(0, int(np.ceil(row-outer_norm*rows))), min(rows, int(np.floor(row+outer_norm*rows))+1)
        x1, x2 = max(0, int(np.ceil(col-outer_norm*cols))), min(cols, int(np.floor(col+outer_norm*cols))+1)
        y = (np.arange(y1, y2)[:, None]-row)/rows
        x = (np.arange(x1, x2)[None, :]-col)/cols
        distance = np.hypot(y, x)
        weight = distance <= r_norm if edge is None else .5*(1-np.cos(np.pi*np.clip((outer_norm-distance)/edge_norm, 0, 1)))
        np.maximum(mask[y1:y2, x1:x2], weight, out=mask[y1:y2, x1:x2])
    return mask


def _curvature_phase(config):
    """Preserve experimental a/c coefficients and center at shape/2; return the unwrapped design phase."""
    rows, cols = config.shape
    y = np.arange(rows)-rows/2
    x = np.arange(cols)-cols/2
    cy, cx = config.curvature_factors
    return config.curvature*np.pi*(cy/rows*y[:, None]**2 + cx/cols*x[None, :]**2)


def build_problem(config: RunConfig, *, corrections=None) -> OpticalProblem:
    """Build the complete numerical problem without file or hardware dependencies.

    Args:
        config: RunConfig defining the grid, input, array, curvature, and regions.
        corrections: CorrectionData arrays or None; no paths are read. None means
            no experimental correction. Arrays are validated and copied, not modified.

    Returns:
        CPU NumPy OpticalProblem with normalized input amplitude, complex target
        including design curvature and optional target_phase, pixel positions, and masks.
        compensation is wrapped design curvature in radians, excluding extra target_phase.
    """
    validate_config(config)
    corrections = corrections or CorrectionData()
    # Copy the record so validation cannot replace or modify caller-provided arrays.
    corrections = CorrectionData(**vars(corrections))
    validate_corrections(corrections, config)
    amplitude = _input_gaussian(config)
    centers, angles = _spot_layout(config)
    target, spots = _target_array(config, centers, angles, corrections)
    curve = _curvature_phase(config)
    factor = np.exp(1j*curve)
    target *= factor
    if corrections.target_phase is not None:
        target *= np.exp(1j*corrections.target_phase)
    rows, cols = config.shape
    if np.any(spots < 0) or np.any(spots > np.array(config.shape)-1):
        raise ValueError('spot sampling point would leave image grid')
    r_eff, r_ion, r_smooth, r_small = config.mask_radii
    small_spots = spots.reshape(-1, 2, 2).mean(axis=1) if config.mode == 'tem01' else spots
    masks = {'efficiency': _radial_mask(config.shape, spots, r_eff),
             'ion': _radial_mask(config.shape, spots, r_ion),
             'smooth': _radial_mask(config.shape, spots, r_smooth, edge=config.mask_edge),
             'small': _radial_mask(config.shape, small_spots, r_small)}
    box = np.zeros(config.shape, bool)
    h = config.box_half_fraction
    box[int(rows/2-rows*h):int(rows/2+rows*h), int(cols/2-cols*h):int(cols/2+cols*h)] = True
    masks['box'] = box
    if not np.any(abs(target[box]) > 0):
        raise ValueError('box ROI has no target power')
    return OpticalProblem(amplitude, target, centers, spots, np.angle(factor), masks, config)
