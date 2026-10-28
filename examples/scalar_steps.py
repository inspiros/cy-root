"""Choose scalar solvers and replay their actual histories.

python examples/scalar_steps.py --methods brent secant householder
python examples/scalar_steps.py --save assets/scalar_steps.gif
Use --save FILE.png/FILE.svg --step N for a still figure.
"""
import argparse
from pathlib import Path

from cyroot import as_root_snapshot
from cyroot.native import Bisection, BracketSolver, Chebyshev, Halley, Householder, Newton, Secant

if __package__:
    from .plotting import scalar_figure, save_gif
else:
    from plotting import scalar_figure, save_gif


def function(x):
    return x**3-x-2.


def derivative(x):
    return 3*x*x-1.


def second_derivative(x):
    return 6*x


# Problem/solver selection stays here; plotting.py only consumes copied records.
METHODS = {
    'bisect': ('Bisection', lambda **opts: Bisection(function, 1., 2., **opts)),
    'brent': ('Brent', lambda **opts: BracketSolver(function, 1., 2., method='brent', **opts)),
    'secant': ('Secant', lambda **opts: Secant(function, 1., 2., **opts)),
    'newton': ('Newton', lambda **opts: Newton(function, derivative, 1.9, **opts)),
    'halley': ('Halley', lambda **opts: Halley(function, derivative, second_derivative, 1.9, **opts)),
    'chebyshev': ('Chebyshev', lambda **opts: Chebyshev(function, derivative, second_derivative, 1.9, **opts)),
    'householder': ('Householder (d=3)', lambda **opts: Householder(
        function, (derivative, second_derivative, lambda x: 6.), 1.9, **opts)),
}
DEFAULT_METHODS = ('bisect', 'newton', 'halley')


def record_histories(methods=DEFAULT_METHODS):
    histories = {}
    for method in methods:
        label, construct = METHODS[method]
        solver = construct(history=128, max_iter=100, etol=1e-12, ertol=0.)
        solver.run()
        if solver.history_truncated:
            raise RuntimeError(f'{label}: increase history capacity to retain every step.')
        histories[label] = tuple(map(as_root_snapshot, solver.history))
    return histories


def create_figure(histories, step=0, *, interactive=True):
    return scalar_figure(histories, function, xlim=(.9, 2.12), ylim=(-3., 5.6),
        title=r'Scalar roots, one recorded step at a time',
        subtitle=r'$f(x)=x^3-x-2$  ·  brackets / starting pairs [1, 2] or an initial point of 1.9',
        models={'Newton': 'tangent', 'Halley': 'halley'}, step=step, interactive=interactive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=DEFAULT_METHODS,
                        help='Choose one to three methods (default: bisect newton halley).')
    parser.add_argument('--save', type=Path, help='Save GIF animation or PNG/SVG still instead of opening a GUI.')
    parser.add_argument('--step', type=int, default=0, help='Iteration for a still figure (default: 0).')
    args = parser.parse_args()
    if not 1 <= len(args.methods) <= 3 or len(set(args.methods)) != len(args.methods):
        parser.error('Choose one to three distinct methods per figure.')
    try:
        import matplotlib
        if args.save:
            matplotlib.use('Agg')
    except ImportError:
        parser.error("Install plotting dependencies with pip install 'cy-root[examples]'.")
    histories = record_histories(args.methods)
    try:
        figure, update, slider = create_figure(histories, args.step, interactive=not args.save)
    except ValueError as error:
        parser.error(str(error))
    for name, history in histories.items():
        last = history[-1]
        print(f'{name}: iterations={last.iters}, calls={last.f_calls}, '
              f'residual={last.error:.3g}, stop={last.termination_reason}')
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        if args.save.suffix.lower() == '.gif':
            save_gif(figure, update, max(h[-1].iters for h in histories.values()), args.save)
        else:
            figure.savefig(args.save, dpi=160)
        print(f'Saved {args.save.resolve()}.')
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
