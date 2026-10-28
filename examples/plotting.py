"""Small example renderers for copied histories; not a cyroot plotting API.

Pass ``{label: tuple(map(as_root_snapshot, solver.history))}`` and a plot-only
callable. These helpers never advance a solver or evaluate its providers.
"""
from pathlib import Path

import numpy as np

PALETTE = ('#3377b4', '#cd7237', '#408270', '#9263a5', '#b45264', '#677b86')
INK = '#263b46'


def _setup(histories, step):
    if len(histories) > 3:
        raise ValueError('Choose one to three histories per figure.')
    if not histories or any(not history for history in histories.values()):
        raise ValueError('Supply at least one nonempty recorded history.')
    maximum = max(history[-1].iters for history in histories.values())
    if not 0 <= step <= maximum:
        raise ValueError(f'step must be between 0 and {maximum}.')
    return maximum, dict(zip(histories, (PALETTE[i % len(PALETTE)] for i in range(len(histories)))))


def _prefix(history, iteration):
    return history[:max(i for i, state in enumerate(history) if state.iters <= iteration)+1]


def _style(axis):
    axis.tick_params(labelsize=10, colors=INK)
    axis.xaxis.label.set(size=11, color=INK)
    axis.yaxis.label.set(size=11, color=INK)
    axis.title.set(size=12, color=INK)
    axis.spines[['top', 'right']].set_visible(False)
    axis.spines[['left', 'bottom']].set_color('#c4cdd3')
    axis.grid(alpha=.16, color='#7f939e')


def _status(state):
    if state.termination_reason is None:
        return 'running'
    if state.residual_converged:
        return 'root reached'
    return state.termination_reason.value.replace('_', ' ')


def _slider(fig, maximum, step, update, interactive):
    if not interactive:
        return None
    from matplotlib.widgets import Slider
    ax = fig.add_axes((.20, .055, .62, .022))
    slider = Slider(ax, 'Iteration', 0, max(1, maximum), valinit=step, valstep=1, valfmt='%0.0f')
    slider.label.set_fontsize(11)
    slider.valtext.set_fontsize(11)
    slider.on_changed(update)
    return slider


def scalar_figure(histories, function, *, xlim, ylim, title, subtitle='',
                  models=None, step=0, interactive=True):
    """Scalar geometry and residuals; model kinds are explicit, never inferred.

    ``models`` maps labels to 'tangent' or 'halley'. Only use these for methods
    whose steps have that geometry. Next-point markers always use stored data.
    ``function`` receives a NumPy plot grid during setup only.
    """
    import matplotlib.pyplot as plt

    maximum, colors = _setup(histories, step)
    models = {} if models is None else models
    grid = np.linspace(*xlim, 600)
    curve = function(grid)
    fig = plt.figure(figsize=(max(8., 4.1*len(histories)), 7.6), facecolor='white')
    layout = fig.add_gridspec(2, len(histories), height_ratios=(1.5, 1.))
    geometry = [fig.add_subplot(layout[0, i]) for i in range(len(histories))]
    convergence = fig.add_subplot(layout[1, :])
    fig.subplots_adjust(left=.075, right=.98, bottom=.16 if interactive else .11,
                        top=.82, wspace=.27, hspace=.56)
    fig.suptitle(title, fontsize=19, color=INK, y=.975)
    fig.text(.5, .915, subtitle, ha='center', fontsize=11, color='#5c6f7a')
    progress = fig.text(.5, .865, '', ha='center', fontsize=10, color='#5c6f7a')

    def update(iteration):
        iteration = int(iteration)
        progress.set_text(f'RECORDED ITERATION {iteration:02d} / {maximum:02d}')
        convergence.clear()
        for axis, (name, history) in zip(geometry, histories.items()):
            axis.clear()
            prefix = _prefix(history, iteration)
            state = prefix[-1]
            color = colors[name]
            axis.plot(grid, curve, color=INK, lw=1.8, label=r'$f(x)$')
            axis.axhline(0., color=INK, lw=1.5, zorder=1)
            bracket = getattr(state.diagnostics, 'bracket', None)
            if bracket is not None:
                axis.axvspan(*bracket, color=color, alpha=.13, label='Bracket')
                axis.plot(bracket, state.diagnostics.f_bracket, 's', color=color, ms=4)
            model = models.get(name)
            if len(prefix) < len(history) and model:
                slope = state.diagnostics.df_root
                offset = grid-state.root
                if model == 'tangent':
                    values = state.f_root+slope*offset
                    label = 'Tangent'
                elif model == 'halley':
                    denominator = 1.-state.diagnostics.d2f_root*offset/(2.*slope)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        values = state.f_root+slope*offset/denominator
                    values[np.abs(denominator) < 1e-8] = np.nan
                    label = 'Osculating hyperbola'
                else:
                    raise ValueError(f'Unknown scalar geometry {model!r}.')
                axis.plot(grid, values, '--', color=color, lw=1.5, label=label)
            if len(prefix) < len(history):
                following = history[len(prefix)]
                axis.plot((following.root, following.root), (0., following.f_root), ':', color=color, lw=1.)
                axis.plot(following.root, 0., 'v', mfc='white', mec=color, ms=7, label='Next iterate')
            axis.plot(state.root, state.f_root, 'o', color=color, ms=6, zorder=5)
            axis.set(xlim=xlim, ylim=ylim, xlabel='x', ylabel='f(x)', title=name)
            axis.text(.03, .97, f'k = {state.iters}  ·  |f| = {state.error:.1e}\n{_status(state)}',
                      va='top', transform=axis.transAxes, fontsize=10, color=color,
                      bbox=dict(facecolor='white', edgecolor='none', alpha=.92))
            axis.legend(loc='lower right', fontsize=9, frameon=False)
            _style(axis)
            convergence.semilogy([s.iters for s in history], [max(s.error, 1e-16) for s in history],
                                 color=color, lw=1., alpha=.18)
            convergence.semilogy([s.iters for s in prefix], [max(s.error, 1e-16) for s in prefix],
                                 '-o', color=color, lw=1.8, ms=3, label=name)
        convergence.set(xlim=(-.5, maximum+.5), ylim=(1e-17, 10.), xlabel='Recorded iteration',
                        ylabel=r'$|f(x_k)|$', title='Residual convergence')
        convergence.legend(loc='upper right', ncol=min(len(histories), 4), fontsize=10, frameon=False)
        _style(convergence)
        fig.canvas.draw_idle()

    fig.text(.5, .012, 'Actual solver histories · faint curves show the complete run · exact zeros shown at 10⁻¹⁶',
             ha='center', fontsize=10, color='#5c6f7a')
    slider = _slider(fig, maximum, step, update, interactive)
    update(step)
    return fig, update, slider


def vector_figure(histories, residual, *, bounds, detail, title, subtitle='',
                  root=None, step=0, interactive=True):
    """2D paths over a residual norm background and a full-width convergence plot.

    ``residual`` receives NumPy mesh grids during setup only. No optimizer-
    specific fields are needed. Each record is a stored outer iteration.
    """
    import matplotlib.pyplot as plt

    maximum, colors = _setup(histories, step)
    fig = plt.figure(figsize=(11.8, 8.1), facecolor='white')
    layout = fig.add_gridspec(2, 2, height_ratios=(1.55, 1.))
    full, zoom = fig.add_subplot(layout[0, 0]), fig.add_subplot(layout[0, 1])
    convergence = fig.add_subplot(layout[1, :])
    fig.subplots_adjust(left=.08, right=.92, bottom=.16 if interactive else .11,
                        top=.80, wspace=.26, hspace=.48)
    fig.suptitle(title, fontsize=19, color=INK, y=.975)
    fig.text(.5, .915, subtitle, ha='center', fontsize=11, color='#5c6f7a')
    # Small per-method status labels replace the large diagnostic quadrant.
    labels = {}
    for i, (name, color) in enumerate(colors.items()):
        x = .5 if len(histories) == 1 else .08+.86*i/(len(histories)-1)
        align = 'center' if len(histories) == 1 else 'left' if i == 0 else 'right' if i == len(histories)-1 else 'center'
        labels[name] = fig.text(x, .868, '', ha=align, va='center', fontsize=10.5, color=color)
    paths = {}
    for axis, limits in ((full, bounds), (zoom, detail)):
        xx, yy = np.meshgrid(np.linspace(*limits[:2], 260), np.linspace(*limits[2:], 260))
        values = residual((xx, yy))
        norm = np.maximum.reduce([np.abs(value) for value in values])
        contours = axis.contourf(xx, yy, np.log10(np.maximum(norm, 1e-4)),
                                levels=np.linspace(-4., 2., 31), cmap='Greys_r', alpha=.55)
        axis.set(xlim=limits[:2], ylim=limits[2:], xlabel='x', ylabel='y')
        for name, color in colors.items():
            line, = axis.plot([], [], '-o', color=color, lw=1.9, ms=3.5, label=name)
            current, = axis.plot([], [], 'o', mfc='white', mec=color, mew=1.7, ms=8, zorder=5)
            paths[axis, name] = line, current
            start = histories[name][0].root
            axis.plot(*start, 's', color=INK, ms=4, zorder=6)
        if root is not None:
            axis.plot(*root, '*', color=INK, mec='white', mew=.6, ms=14, zorder=6)
        _style(axis)
    full.set_title('Full trajectory', fontsize=12, color=INK)
    zoom.set_title('Valley detail', fontsize=12, color=INK)
    full.legend(loc='lower left', fontsize=10, frameon=False)
    bar = fig.colorbar(contours, ax=(full, zoom), fraction=.025, pad=.02, ticks=(-4., -2., 0., 2.))
    bar.ax.set_yticklabels((r'$\leq10^{-4}$', r'$10^{-2}$', '1', '100'))
    bar.set_label(r'$\|F\|_\infty$', fontsize=11, color=INK)
    outside = zoom.text(.025, .025, '', transform=zoom.transAxes, fontsize=10, color=INK,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.9))

    def update(iteration):
        iteration = int(iteration)
        convergence.clear()
        offscreen = []
        for name, history in histories.items():
            prefix = _prefix(history, iteration)
            state = prefix[-1]
            points = np.asarray([item.root for item in prefix])
            for axis in (full, zoom):
                line, current = paths[axis, name]
                line.set_data(points[:, 0], points[:, 1])
                current.set_data((state.root[0],), (state.root[1],))
            x, y = state.root
            if not (detail[0] <= x <= detail[1] and detail[2] <= y <= detail[3]):
                offscreen.append(name)
            labels[name].set_text(f'{name}  /  k={state.iters}  /  {_status(state)}\n'
                                  f'‖F‖∞ = {state.error:.2g}')
            convergence.semilogy([s.iters for s in history], [max(s.error, 1e-16) for s in history],
                                 color=colors[name], lw=1., alpha=.18)
            convergence.semilogy([s.iters for s in prefix], [max(s.error, 1e-16) for s in prefix],
                                 '-o', color=colors[name], lw=1.8, ms=3, label=name)
        outside.set_text(', '.join(offscreen)+' outside detail' if offscreen else '')
        convergence.set(xlim=(-.5, maximum+.5), ylim=(1e-17, 100.), xlabel='Recorded outer iteration',
                        ylabel=r'$\|F(x_k)\|_\infty$', title='Residual convergence')
        convergence.legend(loc='upper right', ncol=min(len(histories), 4), fontsize=10, frameon=False)
        _style(convergence)
        fig.canvas.draw_idle()

    fig.text(.5, .012, 'Actual solver histories · contours sampled separately · exact zeros shown at 10⁻¹⁶',
             ha='center', fontsize=10, color='#5c6f7a')
    slider = _slider(fig, maximum, step, update, interactive)
    update(step)
    return fig, update, slider


def surface_figure(histories, residual, *, bounds, title, subtitle='',
                   root=None, step=0, interactive=True):
    """3D residual-norm surface; the zero-height plane denotes residual roots.

    A vector residual has several component values. Its infinity norm supplies
    a single nonnegative height and is zero exactly when all components vanish.
    Plot-only grid sampling occurs once; path heights use the recorded errors.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.patches import Patch

    maximum, palette = _setup(histories, step)
    x = np.linspace(*bounds[:2], 110)
    y = np.linspace(*bounds[2:], 110)
    if root is not None:
        x, y = np.unique(np.append(x, root[0])), np.unique(np.append(y, root[1]))
    xx, yy = np.meshgrid(x, y)
    values = residual((xx, yy))
    norm = np.maximum.reduce([np.abs(value) for value in values])
    fig = plt.figure(figsize=(11.8, 9.2), facecolor='white')
    layout = fig.add_gridspec(2, 1, height_ratios=(2.7, 1.))
    axis = fig.add_subplot(layout[0], projection='3d', computed_zorder=False)
    convergence = fig.add_subplot(layout[1])
    fig.subplots_adjust(left=.085, right=.90, bottom=.14 if interactive else .10,
                        top=.81, hspace=.12)
    axis.set_position((.16, .35, .65, .47))
    fig.suptitle(title, fontsize=19, color=INK, y=.975)
    fig.text(.5, .925, subtitle, ha='center', fontsize=11, color='#5c6f7a')
    labels = {}
    for i, (name, color) in enumerate(palette.items()):
        position = .5 if len(histories) == 1 else .085+.815*i/(len(histories)-1)
        align = 'center' if len(histories) == 1 else 'left' if i == 0 else 'right' if i == len(histories)-1 else 'center'
        labels[name] = fig.text(position, .862, '', ha=align, va='center', fontsize=10.5, color=color)
    height_limit = max(float(np.max(norm)), max(s.error for history in histories.values() for s in history))
    mapping = cm.ScalarMappable(norm=colors.LogNorm(vmin=1e-3, vmax=max(1., height_limit)), cmap='viridis')
    axis.plot_surface(xx, yy, norm, facecolors=mapping.to_rgba(np.maximum(norm, 1e-3)),
                      rcount=80, ccount=80, linewidth=0., antialiased=True,
                      shade=False, alpha=.55, zorder=1)
    floor_x, floor_y = np.meshgrid(bounds[:2], bounds[2:])
    axis.plot_surface(floor_x, floor_y, np.zeros_like(floor_x), color='#9cc9dc',
                      alpha=.14, shade=False, zorder=0)
    axis.plot((bounds[0], bounds[1], bounds[1], bounds[0], bounds[0]),
              (bounds[2], bounds[2], bounds[3], bounds[3], bounds[2]), (0.,)*5,
              color='#729eaf', lw=.9, zorder=2)
    paths = {}
    for name, color in palette.items():
        # Overlay the recorded paths so the sampled surface cannot hide them.
        line, = axis.plot([], [], [], '-o', color=color, lw=2.2, ms=4, label=name, zorder=3)
        current, = axis.plot([], [], [], 'o', mfc='white', mec=color, mew=1.7, ms=8, zorder=4)
        paths[name] = line, current
    if root is not None:
        axis.plot((root[0],), (root[1],), (0.,), '*', color=INK, mec='white', mew=.5, ms=13, zorder=5)
    axis.set(xlim=bounds[:2], ylim=bounds[2:], zlim=(0., height_limit*1.03),
             xlabel='x', ylabel='y', zlabel=r'$z=\|F(x,y)\|_\infty$')
    axis.view_init(elev=30, azim=-125)
    axis.set_box_aspect((1.3, 1.25, .85))
    axis.tick_params(labelsize=9, colors=INK)
    for label in (axis.xaxis.label, axis.yaxis.label, axis.zaxis.label):
        label.set(size=11, color=INK)
    handles, names = axis.get_legend_handles_labels()
    handles.append(Patch(facecolor='#9cc9dc', edgecolor='#729eaf', alpha=.5))
    names.append('Zero residual plane: z = 0')
    axis.legend(handles, names, loc='upper left', bbox_to_anchor=(-.22, .99), fontsize=10, frameon=False)
    bar = fig.colorbar(mapping, cax=fig.add_axes((.85, .42, .015, .30)))
    bar.set_label('Residual norm (color)', fontsize=10, color=INK)

    def update(iteration):
        convergence.clear()
        for name, history in histories.items():
            prefix = _prefix(history, int(iteration))
            state = prefix[-1]
            points = np.asarray([item.root for item in prefix])
            errors = np.asarray([item.error for item in prefix])
            line, current = paths[name]
            line.set_data_3d(points[:, 0], points[:, 1], errors)
            current.set_data_3d((state.root[0],), (state.root[1],), (state.error,))
            labels[name].set_text(f'{name}  /  k={state.iters}  /  {_status(state)}\n'
                                  f'‖F‖∞ = {state.error:.2g}')
            convergence.semilogy([s.iters for s in history], [max(s.error, 1e-16) for s in history],
                                 color=palette[name], lw=1., alpha=.18)
            convergence.semilogy([s.iters for s in prefix], [max(s.error, 1e-16) for s in prefix],
                                 '-o', color=palette[name], lw=1.8, ms=3, label=name)
        convergence.set(xlim=(-.5, maximum+.5), ylim=(1e-17, 100.), xlabel='Recorded outer iteration',
                        ylabel=r'$\|F(x_k)\|_\infty$', title='Residual convergence')
        convergence.legend(loc='upper right', ncol=min(len(histories), 4), fontsize=10, frameon=False)
        _style(convergence)
        fig.canvas.draw_idle()

    fig.text(.5, .013, 'Roots have zero height · a nonzero local minimum is not a root · exact zeros below shown at 10⁻¹⁶',
             ha='center', fontsize=10, color='#5c6f7a')
    slider = _slider(fig, maximum, step, update, interactive)
    update(step)
    return fig, update, slider


def save_gif(figure, update, last_iteration, path, *, fps=7):
    """Export stored-state replay with a short pause at either end, using Pillow."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [0]*3+list(range(last_iteration+1))+[last_iteration]*7
    animation = FuncAnimation(figure, update, frames=frames, interval=1000/fps, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=90)
