"""Choose 2D solvers and replay their actual histories.

python examples/vector_steps.py --methods broyden lbfgs
python examples/vector_steps.py --save assets/vector_steps.gif
python examples/vector_steps.py --view surface
Use --save FILE.png/FILE.svg --step N for a still figure.
"""
import argparse
from pathlib import Path

from cyroot import OptimizationSolver, as_root_snapshot
from cyroot.native import BroydenGood, VectorChebyshev, VectorNewton

if __package__:
    from .plotting import vector_figure, surface_figure, save_gif
else:
    from plotting import vector_figure, surface_figure, save_gif


def residual(point):
    x, y = point
    return (10.*(y-x*x), 1.-x)


def jacobian(point):
    x, _ = point
    return ((-20.*x, 10.), (-1., 0.))


def hessian(point):
    return (((-20., 0.), (0., 0.)), ((0., 0.), (0., 0.)))


METHODS = ('newton', 'chebyshev', 'broyden', 'levenberg_marquardt', 'bfgs', 'lbfgs', 'powell_dogleg')
LABELS = dict(zip(METHODS, ('Newton', 'Chebyshev', 'Broyden (good)', 'Levenberg–Marquardt',
                          'BFGS', 'L-BFGS', 'Powell dogleg')))
DEFAULT_METHODS = ('newton', 'powell_dogleg')


def record_histories(methods=DEFAULT_METHODS):
    histories = {}
    for method in methods:
        options = dict(history=512, max_iter=400, etol=1e-10, ertol=0.)
        if method == 'newton':
            solver = VectorNewton(residual, jacobian, (-1.2, 1.), **options)
        elif method == 'chebyshev':
            solver = VectorChebyshev(residual, jacobian, hessian, (-1.2, 1.), **options)
        elif method == 'broyden':
            solver = BroydenGood(residual, jacobian((-1.2, 1.)), (-1.2, 1.), **options)
        elif method in METHODS:
            solver = OptimizationSolver(method, residual, (-1.2, 1.), jacobian=jacobian, **options)
        else:
            raise ValueError(f'Unknown vector method {method!r}.')
        solver.run()
        if solver.history_truncated:
            raise RuntimeError(f'{method}: increase history capacity to retain every iteration.')
        histories[LABELS[method]] = tuple(map(as_root_snapshot, solver.history))
    return histories


def create_figure(histories, step=0, *, interactive=True, view='contour'):
    if view == 'surface':
        return surface_figure(histories, residual, bounds=(-1.5, 1.4, -4.2, 1.8),
            title='Roots lie on the zero residual plane',
            subtitle=r'$z=\|F(x,y)\|_\infty$  ·  all residual components vanish at z = 0',
            root=(1., 1.), step=step, interactive=interactive)
    if view != 'contour':
        raise ValueError(f'Unknown view {view!r}.')
    return vector_figure(histories, residual, bounds=(-1.5, 1.4, -4.2, 1.8), detail=(-1.5, 1.4, -.4, 1.8),
        title='Different paths to the same root',
        subtitle=r'$F(x,y)=(10(y-x^2),1-x)$  ·  start (-1.2, 1)  ·  root (1, 1)',
        root=(1., 1.), step=step, interactive=interactive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=DEFAULT_METHODS,
                        help='Choose one to three methods (default: newton powell_dogleg).')
    parser.add_argument('--view', choices=('contour', 'surface'), default='contour',
                        help='2D contours or a 3D residual-norm surface (default: contour).')
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
        figure, update, slider = create_figure(histories, args.step, interactive=not args.save, view=args.view)
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
