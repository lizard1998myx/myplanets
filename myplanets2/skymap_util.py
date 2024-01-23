import matplotlib.pyplot as plt
import numpy as np


def plot_constellation_legacy(ax, cat_df, con_id, color, polar=False, south=False, **kwargs):
    for con in con_id:
        for line in con:
            ra_deg = cat_df['RA'].values[line] * 180 / np.pi
            dec_deg = cat_df['DE'].values[line] * 180 / np.pi
            if not polar:
                plot(ax=ax, ra_deg=ra_deg, dec_deg=dec_deg,
                     color=color, **kwargs)
            else:
                plot_polar(ax=ax, ra_deg=ra_deg, dec_deg=dec_deg,
                           color=color,
                           south=south, **kwargs)


def plot_constellation_new(ax, lines_ra_deg, lines_dec_deg, color,
                           polar=False, south=False, **kwargs):
    for ra_deg, dec_deg in zip(lines_ra_deg, lines_dec_deg):
        if not polar:
            plot(ax=ax, ra_deg=ra_deg, dec_deg=dec_deg,
                 color=color, **kwargs)
        else:
            plot_polar(ax=ax, ra_deg=ra_deg, dec_deg=dec_deg,
                       color=color,
                       south=south, **kwargs)


def plot_stars_legacy(ax, cat_df, size_func=None, zorder=3, polar=False, south=False, **kwargs):
    if size_func is None:
        size_func = lambda x: 100 * x ** 2

    kwargs_0 = {'ax': ax,
                'ra_deg': cat_df['RA'].values * 180 / np.pi,
                'dec_deg': cat_df['DE'].values * 180 / np.pi,
                's': size_func(cat_df['pix_size']),
                'color': cat_df['RGB'], 'zorder': zorder}

    kwargs_0.update(kwargs)

    if not polar:
        scatter(**kwargs_0)
    else:
        scatter_polar(south=south, **kwargs_0)


def plot_stars_new(ax, ra_deg, dec_deg, size, color, zorder=3, polar=False, south=False, **kwargs):
    if not polar:
        scatter(ax=ax, ra_deg=ra_deg, dec_deg=dec_deg, s=size, color=color, zorder=zorder, **kwargs)
    else:
        scatter_polar(ax=ax, ra_deg=ra_deg, dec_deg=dec_deg, s=size,
                      color=color, zorder=zorder, south=south, **kwargs)


# via 20230614 skymap obj.ipynb
def get_fig_ax(ymax=70, color=None, grid=True):
    fig, ax = plt.subplots(figsize=(36, ymax * 2 / 10))
    set_ax(ax=ax, ymax=ymax, color=color, grid=grid)
    return fig, ax


def set_ax(ax, ymax, outside=False, color=None, grid=True):
    ax.set_aspect('equal')
    if color is not None:
        ax.patch.set_color(color)

    ax.set_ylim(-ymax, ymax)
    if outside:
        ax.set_xlim(0, 360)
    else:
        ax.set_xlim(360, 0)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_yticks(np.append(-1 * np.arange(0, ymax + 1, 30)[1:], np.arange(0, ymax + 1, 30)))
    ax.grid(grid)


def get_ax_polar(rmax=60, south=False, outside=True, color=None, grid=True):
    # North outside, +1; South outside, -1
    # FT+, TT-, TF+, FF-
    ax = plt.subplot(111, projection='polar')
    # projection = 'polar' 指定为极坐标
    set_ax_polar(ax=ax, rmax=rmax, south=south, outside=outside, color=color, grid=grid)
    return ax


def set_ax_polar(ax, rmax, south=False, outside=True, color=None, grid=True):
    if color is not None:
        ax.patch.set_color(color)
    ax.grid(grid)  # 是否有网格
    ax.set_rlim(0, rmax)
    if south ^ outside:
        ax.set_theta_direction(+1)
    else:
        ax.set_theta_direction(-1)

    ax.set_thetagrids(range(0, 360, 30))

    r_girds = np.arange(30, rmax, 30)
    if not south:  # north
        ax.set_rgrids(r_girds, 90 - r_girds)
    else:
        ax.set_rgrids(r_girds, r_girds - 90)


# via 20230614 skymap obj.ipynb
# x as ra_deg, y as dec_deg
def _break_line(x, y, bound=360, max_dx=300):
    assert len(x) == len(y)
    # x = np.array(x)
    # y = np.array(y)

    if len(x) <= 1:
        return [(x, y)]

    break_points = np.where(abs(x[:-1] - x[1:]) > max_dx)[0]
    break_points = [0] + list(break_points) + [len(x) - 2]

    broken_x_list = []  # element is xlist of a broken line
    broken_y_list = []

    for j in range(len(break_points) - 1):
        broken_x_list.append(np.array(x[break_points[j]:break_points[j + 1] + 2]))
        broken_y_list.append(np.array(y[break_points[j]:break_points[j + 1] + 2]))

    # problematic for short lines
    for i, broken_x in enumerate(broken_x_list):
        if i != 0:
            if broken_x[0] - broken_x[1] > max_dx:
                broken_x[0] -= bound
            elif broken_x[0] - broken_x[1] < -max_dx:
                broken_x[0] += bound
        if broken_x[-1] - broken_x[-2] > max_dx:
            broken_x[-1] -= bound
        elif broken_x[-1] - broken_x[-2] < -max_dx:
            broken_x[-1] += bound

    results = []
    for broken_x, broken_y in zip(broken_x_list, broken_y_list):
        results.append((np.array(broken_x), np.array(broken_y)))
    # element is a broken line (x_array, y_array)
    return results


def plot(ax, ra_deg, dec_deg, **kwargs):
    broken_lines = _break_line(ra_deg, dec_deg)
    for x, y in broken_lines:
        ax.plot(x, y, **kwargs)


def animate(ax, ra_deg, dec_deg, color, linestyle=None, marker='o', **kwargs):
    broken_lines = _break_line(ra_deg, dec_deg)
    sc, = ax.plot([], [], color=color, marker=marker)
    ln_list = []
    for _ in range(len(broken_lines)):
        ln_list.append(ax.plot([], [], color=color, linestyle=linestyle, **kwargs)[0])

    i_points = []
    i_start = 0
    for line in broken_lines:
        n_points = len(line[0])
        i_points.append(np.arange(i_start, i_start + n_points))
        i_start = i_start + n_points - 2  # 2 overlap

    def ani(frame):
        sc.set_data(ra_deg[frame], dec_deg[frame])

        for i, ln in enumerate(ln_list):
            idx = i_points[i] <= frame
            ln.set_data(broken_lines[i][0][idx], broken_lines[i][1][idx])

        return tuple(ln_list + [sc])
        # return tuple(ln_list)

    return ani


def scatter(ax, ra_deg, dec_deg, **kwargs):
    ax.scatter(ra_deg, dec_deg, **kwargs)


def plot_polar(ax, ra_deg, dec_deg, south=False, **kwargs):
    if not south:  # north
        ax.plot(ra_deg * np.pi / 180, 90 - dec_deg, **kwargs)
    else:
        ax.plot(ra_deg * np.pi / 180, 90 + dec_deg, **kwargs)


def animate_polar(ax, ra_deg, dec_deg, color, linestyle=None, marker='o', south=False, **kwargs):
    ra_polar = ra_deg * np.pi / 180
    if not south:  # north
        dec_polar = 90 - dec_deg
    else:
        dec_polar = 90 + dec_deg

    ln, = ax.plot([], [], color=color, linestyle=linestyle, **kwargs)
    sc, = ax.plot([], [], color=color, marker=marker)

    def ani(frame):
        ln.set_data(ra_polar[:frame + 1], dec_polar[:frame + 1])
        sc.set_data(ra_polar[frame], dec_polar[frame])
        return ln, sc

    return ani


def scatter_polar(ax, ra_deg, dec_deg, south=False, **kwargs):
    if not south:  # north
        ax.scatter(ra_deg * np.pi / 180, 90 - dec_deg, **kwargs)
    else:
        ax.scatter(ra_deg * np.pi / 180, 90 + dec_deg, **kwargs)


# ecliptic, solar trajectory
# via 20230520 sky map.ipynb
def get_ecliptic_deg(alpha_deg=23 + 26 / 60, approximate=False):
    if approximate:
        ra = np.linspace(0, 2 * np.pi, 1000)
        return ra * 180 / np.pi, np.sin(ra) * (alpha_deg)  # good fits actually
    else:  # precise value
        alpha = alpha_deg * np.pi / 180

        t0 = np.linspace(0, 2 * np.pi, 1000)
        x0 = np.cos(t0)
        y0 = np.sin(t0)
        z0 = 0 * t0

        x1 = x0
        y1 = y0 * np.cos(alpha) - z0 * np.sin(alpha)
        z1 = y0 * np.sin(alpha) + z0 * np.cos(alpha)

        ra = np.arctan(y1 / x1)
        ra[x1 < 0] += np.pi
        ra[(x1 >= 0) & (y1 < 0)] += 2 * np.pi
        if ra[-1] == 0:
            ra[-1] += 2 * np.pi
        de = np.arcsin(z1)

        return ra * 180 / np.pi, de * 180 / np.pi


def merge_animate(*animate_functions):
    def ani(frame):
        result = []
        for ani_sub in animate_functions:
            result += list(ani_sub(frame=frame))
        return tuple(result)

    return ani
