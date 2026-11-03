"""Explicit entry point for the clean SLM workflow; importing this module does not start calculations."""
import argparse
from slm_optimization.config import load_preset
from slm_optimization.workflow import run_batch


def main(argv=None):
    """Load presets and a few overrides, then run the batch and print result paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', required=True, help='presets name or JSON file path')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--output', default='runs')
    parser.add_argument('--threads', type=int, help='Explicitly set the number of CPU processes')
    args = parser.parse_args(argv)
    overrides = {key: val for key, val in {'max_steps': args.steps, 'repeats': args.repeats,
                 'seed': args.seed, 'device': args.device}.items() if val is not None}
    config = load_preset(args.preset, overrides=overrides)
    if args.threads is not None:
        import torch
        torch.set_num_threads(args.threads)
    results = run_batch([config], output_root=args.output, repeats=config.repeats)
    for result in results:
        print(result.run_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
