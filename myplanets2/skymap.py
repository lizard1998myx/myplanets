import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from .skymap_util import (plot_constellation_legacy, plot_stars_legacy, plot, plot_polar,
                          plot_stars_new, plot_constellation_new,
                          animate, animate_polar, merge_animate, scatter, scatter_polar,
                          set_ax, set_ax_polar)
from .catalog import StarCatalog
from .models import BaseModel, cal_radec


# to-do list
# 1. easy version
# 2. normal version
# 3. pinyin version
# 4. animation
# 5. documentation

# to-do: save figure?

class SkyMapBase:
    def __init__(self):
        self.max_xyz = 0
        self.ymax = 0
        self.rmax = 0

        self.fig = None
        self.ax_xy = None
        self.ax_xz = None
        self.ax_yz = None
        self.ax_north = None
        self.ax_south = None
        self.ax_map = None

        self.ani_list = []  # list of animation functions

    def plot_constellation(self, lines_ra_deg, lines_dec_deg, color, **kwargs):
        if self.ax_map is not None:
            plot_constellation_new(ax=self.ax_map, lines_ra_deg=lines_ra_deg,
                                   lines_dec_deg=lines_dec_deg, color=color, **kwargs)
        if self.ax_north is not None:
            plot_constellation_new(ax=self.ax_north, lines_ra_deg=lines_ra_deg,
                                   lines_dec_deg=lines_dec_deg, color=color, polar=True, **kwargs)
        if self.ax_south is not None:
            plot_constellation_new(ax=self.ax_south, lines_ra_deg=lines_ra_deg,
                                   lines_dec_deg=lines_dec_deg, color=color, polar=True, south=True, **kwargs)

    # easy utility
    def plot_constellation_easy(self, star_cat: StarCatalog, color='black', **kwargs):
        self.plot_constellation(**star_cat.get_cons(), color=color, **kwargs)

    def plot_constellation_legacy(self, cat_df, con_id, color, **kwargs):
        # map, north and south polar
        if self.ax_map is not None:
            plot_constellation_legacy(ax=self.ax_map, cat_df=cat_df, con_id=con_id, color=color, **kwargs)
        if self.ax_north is not None:
            plot_constellation_legacy(ax=self.ax_north, cat_df=cat_df, con_id=con_id, color=color, polar=True,
                                      **kwargs)
        if self.ax_south is not None:
            plot_constellation_legacy(ax=self.ax_south, cat_df=cat_df, con_id=con_id, color=color, polar=True,
                                      south=True, **kwargs)

    def plot_stars(self, ra_deg, dec_deg, size, color, size_func=None, **kwargs):
        if size_func is None:
            size_func = lambda x: 100 * x ** 2

        if self.ax_map is not None:
            plot_stars_new(ax=self.ax_map, ra_deg=ra_deg, dec_deg=dec_deg,
                           size=size_func(size), color=color, **kwargs)
        if self.ax_north is not None:
            plot_stars_new(ax=self.ax_north, ra_deg=ra_deg, dec_deg=dec_deg,
                           size=size_func(size), color=color, polar=True, **kwargs)
        if self.ax_south is not None:
            plot_stars_new(ax=self.ax_south, ra_deg=ra_deg, dec_deg=dec_deg,
                           size=size_func(size), color=color, polar=True, south=True, **kwargs)

    # easy utility
    def plot_stars_easy(self, star_cat: StarCatalog, **kwargs):
        kwargs_0 = star_cat.get_stars()
        kwargs_0.update(kwargs)
        self.plot_stars(**kwargs_0)

    def plot_stars_legacy(self, cat_df, size_func=lambda x: 10 * x ** 2, **kwargs):
        # map, north and south polar
        if self.ax_map is not None:
            plot_stars_legacy(ax=self.ax_map, cat_df=cat_df, size_func=size_func, **kwargs)
        if self.ax_north is not None:
            plot_stars_legacy(ax=self.ax_north, cat_df=cat_df, size_func=size_func, polar=True,
                              **kwargs)
        if self.ax_south is not None:
            plot_stars_legacy(ax=self.ax_south, cat_df=cat_df, size_func=size_func, polar=True,
                              south=True, **kwargs)

    def plot_radec(self, ra_deg, dec_deg, **kwargs):
        if self.ax_map is not None:
            plot(ax=self.ax_map, ra_deg=ra_deg, dec_deg=dec_deg, **kwargs)
        if self.ax_north is not None:
            plot_polar(ax=self.ax_north, ra_deg=ra_deg, dec_deg=dec_deg, **kwargs)
        if self.ax_south is not None:
            plot_polar(ax=self.ax_south, ra_deg=ra_deg, dec_deg=dec_deg, south=True, **kwargs)

    def scatter_radec(self, ra_deg, dec_deg, **kwargs):
        if self.ax_map is not None:
            scatter(ax=self.ax_map, ra_deg=ra_deg, dec_deg=dec_deg, **kwargs)
        if self.ax_north is not None:
            scatter_polar(ax=self.ax_north, ra_deg=ra_deg, dec_deg=dec_deg, **kwargs)
        if self.ax_south is not None:
            scatter_polar(ax=self.ax_south, ra_deg=ra_deg, dec_deg=dec_deg, south=True, **kwargs)

    def plot_xyz(self, x, y, z, **kwargs):
        if self.ax_xy is not None:
            self.ax_xy.plot(x, y, **kwargs)
            self.ax_xz.plot(x, z, **kwargs)
            self.ax_yz.plot(y, z, **kwargs)

    def scatter_xyz(self, x, y, z, **kwargs):
        if self.ax_xy is not None:
            self.ax_xy.scatter(x, y, **kwargs)
            self.ax_xz.scatter(x, z, **kwargs)
            self.ax_yz.scatter(y, z, **kwargs)

    # easy utility
    def plot_model_easy(self, t, target_model: BaseModel, color, obs_model=None, marker='.', **kwargs):
        xyz = target_model.xyz(t)  # n, 3
        self.plot_xyz(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], color=color, **kwargs)
        self.scatter_xyz(x=xyz[-1, 0], y=xyz[-1, 1], z=xyz[-1, 2], color=color, marker=marker)
        if obs_model is not None:
            radec = cal_radec(xyz_list=xyz, xyz_list_obs=obs_model.xyz(t))  # n, 2
            if len(radec) > 1:
                self.plot_radec(radec[:, 0], radec[:, 1], color=color, **kwargs)
            self.scatter_radec(radec[-1, 0], radec[-1, 1], color=color, marker=marker)

    def animate_radec(self, ra_deg, dec_deg, color, marker='o', **kwargs):
        if self.ax_map is not None:
            self.ani_list.append(animate(ax=self.ax_map, ra_deg=ra_deg, dec_deg=dec_deg,
                                         color=color, marker=marker, **kwargs))
        if self.ax_north is not None:
            self.ani_list.append(animate_polar(ax=self.ax_north,
                                               ra_deg=ra_deg, dec_deg=dec_deg,
                                               color=color, marker=marker,
                                               **kwargs))
        if self.ax_south is not None:
            self.ani_list.append(animate_polar(ax=self.ax_south,
                                               ra_deg=ra_deg, dec_deg=dec_deg,
                                               color=color, marker=marker,
                                               south=True, **kwargs))

    def animate_xyz(self, x, y, z, color, marker='o', **kwargs):
        if self.ax_xy is not None:
            ln_xy, = self.ax_xy.plot([], [], color=color, **kwargs)
            ln_xz, = self.ax_xz.plot([], [], color=color, **kwargs)
            ln_yz, = self.ax_yz.plot([], [], color=color, **kwargs)
            sc_xy, = self.ax_xy.plot([], [], color=color, marker=marker)
            sc_xz, = self.ax_xz.plot([], [], color=color, marker=marker)
            sc_yz, = self.ax_yz.plot([], [], color=color, marker=marker)

            def ani(frame):
                ln_xy.set_data(x[:frame + 1], y[:frame + 1])
                ln_xz.set_data(x[:frame + 1], z[:frame + 1])
                ln_yz.set_data(y[:frame + 1], z[:frame + 1])
                # return ln0, ln1, ln2
                sc_xy.set_data(x[frame], y[frame])
                sc_xz.set_data(x[frame], z[frame])
                sc_yz.set_data(y[frame], z[frame])
                return ln_xy, ln_xz, ln_yz, sc_xy, sc_xz, sc_yz

            self.ani_list.append(ani)

    def animate_model_easy(self, t, target_model: BaseModel, color, obs_model=None,
                           marker='o', **kwargs):
        assert len(t) > 1, 'animation requires a series of time'
        xyz = target_model.xyz(t)  # n, 3
        self.animate_xyz(*xyz.T, color=color, marker=marker, **kwargs)
        if obs_model is not None:
            radec = cal_radec(xyz_list=xyz, xyz_list_obs=obs_model.xyz(t))  # shape (n, 2)
            self.animate_radec(*radec.T, color=color, marker=marker, **kwargs)

    def animate_text(self, tlist, **kwargs):
        # works only if ax_xy or ax_map is not None
        if self.ax_xy is not None:
            kwargs_use = {'x': -0.8 * self.max_xyz, 'y': 0.8 * self.max_xyz, 's': ''}
            kwargs_use.update(kwargs)
            txt = self.ax_xy.text(**kwargs_use)

            def ani(frame):
                txt.set_text(tlist[frame].strftime(r'%Y-%m-%d'))
                return txt,

            self.ani_list.append(ani)

        elif self.ax_map is not None:
            kwargs_use = {'x': 350, 'y': 0.8 * self.ymax, 's': ''}
            kwargs_use.update(kwargs)
            txt = self.ax_map.text(**kwargs_use)

            def ani(frame):
                txt.set_text(tlist[frame].strftime(r'%Y-%m-%d'))
                return txt,

            self.ani_list.append(ani)

    def legend(self, tags, colors, linestyles=None, ax=None, loc='upper right'):
        if ax is not None:
            pass
        elif self.ax_yz is not None:
            ax = self.ax_yz
        elif self.ax_map is not None:
            ax = self.ax_map
        elif self.ax_south is not None:
            ax = self.ax_south
        else:
            print('legend: no ax to plot')
            return

        line_elements = []

        n = len(tags)
        assert len(colors) == n, 'len(colors) != len(tags)'
        if linestyles is None:
            linestyles = ['solid'] * n
        else:
            assert len(linestyles) == n, 'len(linestyles) != len(tags)'

        for i in range(n):
            line_elements.append(mpl.lines.Line2D([0], [0], color=colors[i],
                                                  linestyle=linestyles[i],
                                                  label=tags[i]))

        ax.legend(handles=line_elements, loc=loc)

    def show(self):
        if self.fig is not None:
            self.fig.tight_layout()
            plt.show()
        else:
            print('[warning] no figure to show.')

    def animation(self, frames, interval, blit=True, **kwargs):
        if self.fig is not None:
            self.fig.tight_layout()
            return FuncAnimation(fig=self.fig, func=merge_animate(*self.ani_list),
                                 frames=frames, interval=interval, blit=blit,
                                 **kwargs)
        else:
            print('[warning] no figure to animate.')

    def animation_easy(self, n: int, filename, length_seconds=None,
                       fps=30, blit=True, inf_length=False, **kwargs):
        # kwargs here is different from animation kwargs
        if length_seconds is None:
            length_seconds = n/fps
            print(f'[animation] expected length {length_seconds:.2f} sec')
        n_frame_total = length_seconds * fps
        n = int(n)
        if n <= 0:
            raise ValueError('no frames to animate')
        elif n < n_frame_total:
            length_seconds = n/fps
            skip = 1
            print(f'[warning] no enough frames, reset length to {length_seconds:.2f} sec')
        else:
            skip = int(n // n_frame_total)

        # manual check
        # ~10 times generation time
        if not inf_length:
            assert length_seconds < 2*60, 'video length too long! 20min+ generation time'

        ani = self.animation(frames=range(0, n, skip), interval=1000/fps,
                             blit=blit)

        if ani is not None:
            ani.save(filename, **kwargs)


class SkyMap(SkyMapBase):
    def __init__(self, ax_xyz=True, ax_map=True, ax_polar=True,
                 color_xyz=None, color_map=None, color_polar=None,
                 max_xyz=10, ymax=75, rmax=60, outside=False):
        super().__init__()

        assert ymax <= 90 and rmax <= 90, 'max declination must be smaller than 90'

        if ax_xyz and ax_map and ax_polar:  # Full - 3D Map Polar
            hratios = [5, 15 * (2 * ymax / 360) + 1, 7.5]
            fig = plt.figure(figsize=(15, np.sum(hratios)))
            gs = fig.add_gridspec(nrows=3, ncols=6, height_ratios=hratios)

            ax_xy = fig.add_subplot(gs[0, 0:2])
            ax_xz = fig.add_subplot(gs[0, 2:4])
            ax_yz = fig.add_subplot(gs[0, 4:6])
            ax_map = fig.add_subplot(gs[1, :])
            ax_north = fig.add_subplot(gs[2, 0:3], projection='polar')
            ax_south = fig.add_subplot(gs[2, 3:6], projection='polar')

        elif ax_xyz and ax_map:  # 3D Map
            hratios = [5, 15 * (2 * ymax / 360) + 1]
            fig = plt.figure(figsize=(15, np.sum(hratios)))
            gs = fig.add_gridspec(nrows=2, ncols=3,
                                  height_ratios=hratios)

            ax_xy = fig.add_subplot(gs[0, 0])
            ax_xz = fig.add_subplot(gs[0, 1])
            ax_yz = fig.add_subplot(gs[0, 2])
            ax_map = fig.add_subplot(gs[1, :])

        elif ax_xyz and ax_polar:  # 3D Polar
            print('[warning] 3D+Polar is experimental, might fail.')
            hratios = [5, 7.5]
            fig = plt.figure(figsize=(15, np.sum(hratios)))
            gs = fig.add_gridspec(nrows=2, ncols=6, height_ratios=hratios)

            ax_xy = fig.add_subplot(gs[0, 0:2])
            ax_xz = fig.add_subplot(gs[0, 2:4])
            ax_yz = fig.add_subplot(gs[0, 4:6])
            ax_north = fig.add_subplot(gs[1, 0:3], projection='polar')
            ax_south = fig.add_subplot(gs[1, 3:6], projection='polar')

        elif ax_xyz:  # 3D
            fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
            ax_xy = axs[0]
            ax_xz = axs[1]
            ax_yz = axs[2]

        elif ax_map and ax_polar:  # MapPolar
            hratios = [15 * (2 * ymax / 360) + 1, 7.5]
            fig = plt.figure(figsize=(15, np.sum(hratios)))
            gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=hratios)

            ax_map = fig.add_subplot(gs[0, :])
            ax_north = fig.add_subplot(gs[1, 0], projection='polar')
            ax_south = fig.add_subplot(gs[1, 1], projection='polar')

        elif ax_map:  # Map
            fig, ax_map = plt.subplots(figsize=(15, 15 * (2 * ymax / 360) + 1))

        elif ax_polar:  # Polar
            print('[warning] polar only is experimental, might fail.')
            fig = plt.figure(figsize=(15, 7.5))
            gs = fig.add_gridspec(nrows=1, ncols=2)

            ax_north = fig.add_subplot(gs[0], projection='polar')
            ax_south = fig.add_subplot(gs[1], projection='polar')

        else:
            print('[warning] no figure to create.')
            fig = None

        # initializations
        self.max_xyz = max_xyz
        self.ymax = ymax
        self.rmax = rmax
        self.fig = fig

        if ax_xyz:
            ax_xy.set(xlabel='X', ylabel='Y')
            ax_xz.set(xlabel='X', ylabel='Z')
            ax_yz.set(xlabel='Y', ylabel='Z')

            for ax in [ax_xy, ax_xz, ax_yz]:
                ax.set_xlim(-max_xyz, max_xyz)
                ax.set_ylim(-max_xyz, max_xyz)
                ax.grid(True)
                if color_xyz is not None:
                    ax.patch.set_color(color_xyz)

            self.ax_xy = ax_xy
            self.ax_xz = ax_xz
            self.ax_yz = ax_yz

        if ax_map:
            set_ax(ax=ax_map, ymax=ymax, color=color_map, outside=outside)
            self.ax_map = ax_map

        if ax_polar:
            set_ax_polar(ax=ax_north, rmax=rmax, color=color_polar, outside=outside)
            set_ax_polar(ax=ax_south, rmax=rmax, color=color_polar, south=True, outside=outside)
            self.ax_north = ax_north
            self.ax_south = ax_south
