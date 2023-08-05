# via 20230708 modular map.ipynb

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class SkyMapper():
    def __init__(self):
        pass

    @staticmethod
    def plot_constellation(ax, cat_df, con_id, color, polar=False, south=False, **kwargs):
        for con in con_id:
            for line in con:
                ra_deg = cat_df['RA'].values[line]*180/np.pi
                dec_deg = cat_df['DE'].values[line]*180/np.pi
                if not polar:
                    SkyMapper.plot(ax=ax, ra_deg=ra_deg, dec_deg=dec_deg, 
                                   color=color, **kwargs)
                else:
                    SkyMapper.plot_polar(ax=ax, ra_deg=ra_deg, dec_deg=dec_deg,
                                         color=color, 
                                         south=south, **kwargs)

    @staticmethod
    def plot_stars(ax, cat_df, size_func=None, zorder=3, polar=False, south=False, **kwargs):
        if size_func is None:
            size_func = lambda x: 100*x**2

        kwargs_0 = {'ax': ax,
                    'ra_deg': cat_df['RA'].values*180/np.pi,
                    'dec_deg': cat_df['DE'].values*180/np.pi,
                    's': size_func(cat_df['pix_size']),
                    'color': cat_df['RGB'], 'zorder': zorder}
        
        kwargs_0.update(kwargs)
        
        if not polar:
            SkyMapper.scatter(**kwargs_0)
        else:
            SkyMapper.scatter_polar(south=south, **kwargs_0)

    # via 20230614 skymap obj.ipynb
    @staticmethod
    def get_fig_ax(ymax=70, dark=False, grid=True):
        fig, ax = plt.subplots(figsize=(36, ymax*2/10))
        SkyMapper.set_ax(ax=ax, ymax=ymax, dark=dark, grid=grid)
        return fig, ax
    
    @staticmethod
    def set_ax(ax, ymax, outside=False, dark=False, grid=True):
        ax.set_aspect('equal')
        if dark:
            ax.patch.set_color('black')

        ax.set_ylim(-ymax, ymax)
        if outside:
            ax.set_xlim(0, 360)
        else:
            ax.set_xlim(360, 0)
        ax.set_xticks(np.arange(0, 361, 30))
        ax.set_yticks(np.append(-1*np.arange(0, ymax+1, 30)[1:], np.arange(0, ymax+1, 30)))
        ax.grid(grid)
    
    @staticmethod
    def get_ax_polar(rmax=60, south=False, outside=True, dark=False, grid=True):
        # North outside, +1; South outside, -1
        # FT+, TT-, TF+, FF-
        ax = plt.subplot(111, projection='polar')
        # projection = 'polar' 指定为极坐标
        SkyMapper.set_ax_polar(ax=ax, rmax=rmax, south=south, outside=outside, dark=dark, grid=grid)
        return ax
    
    @staticmethod
    def set_ax_polar(ax, rmax, south=False, outside=True, dark=False, grid=True):
        if dark:
            ax.patch.set_color('black')
        ax.grid(grid)  # 是否有网格
        ax.set_rlim(0, rmax)
        if south ^ outside:
            ax.set_theta_direction(+1)
        else:
            ax.set_theta_direction(-1)

        ax.set_thetagrids(range(0, 360, 30))

        r_girds = np.arange(30, rmax, 30)
        if not south:  # north
            ax.set_rgrids(r_girds, 90-r_girds)
        else:
            ax.set_rgrids(r_girds, r_girds-90)
    
    # via 20230614 skymap obj.ipynb
    # x as ra_deg, y as dec_deg
    @staticmethod
    def _break_line(x, y, bound=360, max_dx=300):
        assert len(x) == len(y)
        # x = np.array(x)
        # y = np.array(y)
        
        break_points = np.where(abs(x[:-1] - x[1:]) > max_dx)[0]
        break_points = [0] + list(break_points) + [len(x) - 2]

        broken_x_list = []  # element is xlist of a broken line
        broken_y_list = []

        for j in range(len(break_points) - 1):
            broken_x_list.append(np.array(x[break_points[j]:break_points[j+1] + 2]))
            broken_y_list.append(np.array(y[break_points[j]:break_points[j+1] + 2]))

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

    @staticmethod
    def plot(ax, ra_deg, dec_deg, **kwargs):
        broken_lines = SkyMapper._break_line(ra_deg, dec_deg)
        for x, y in broken_lines:
            ax.plot(x, y, **kwargs)

    @staticmethod
    def animate(ax, ra_deg, dec_deg, color, linestyle=None, marker='o'):
        broken_lines = SkyMapper._break_line(ra_deg, dec_deg)
        sc, = ax.plot([], [], color=color, marker=marker)
        ln_list = []
        for _ in range(len(broken_lines)):
            ln_list.append(ax.plot([], [], color=color, linestyle=linestyle)[0])

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

    @staticmethod
    def scatter(ax, ra_deg, dec_deg, **kwargs):
        ax.scatter(ra_deg, dec_deg, **kwargs)

    @staticmethod
    def plot_polar(ax, ra_deg, dec_deg, south=False, **kwargs):
        if not south:  # north
            ax.plot(ra_deg*np.pi/180, 90-dec_deg, **kwargs)
        else:
            ax.plot(ra_deg*np.pi/180, 90+dec_deg, **kwargs)
    
    @staticmethod
    def animate_polar(ax, ra_deg, dec_deg, color, linestyle=None, marker='o', south=False):
        ra_polar = ra_deg*np.pi/180
        if not south:  # north
            dec_polar = 90 - dec_deg
        else:
            dec_polar = 90 + dec_deg

        ln, = ax.plot([], [], color=color, linestyle=linestyle)
        sc, = ax.plot([], [], color=color, marker=marker)

        def ani(frame):
            ln.set_data(ra_polar[:frame+1], dec_polar[:frame+1])
            sc.set_data(ra_polar[frame], dec_polar[frame])
            return ln, sc
        
        return ani

    @staticmethod
    def scatter_polar(ax, ra_deg, dec_deg, south=False, **kwargs):
        if not south:  # north
            ax.scatter(ra_deg*np.pi/180, 90-dec_deg, **kwargs)
        else:
            ax.scatter(ra_deg*np.pi/180, 90+dec_deg, **kwargs)

    # ecliptic, solar trajectory
    # via 20230520 sky map.ipynb
    @staticmethod
    def get_ecliptic_deg(alpha_deg=23+26/60, approximate=False):
        if approximate:
            ra = np.linspace(0, 2*np.pi, 1000)
            return ra*180/np.pi, np.sin(ra)*(alpha_deg)  # good fits actually
        else:  # precise value
            alpha = alpha_deg*np.pi/180

            t0 = np.linspace(0, 2*np.pi, 1000)
            x0 = np.cos(t0)
            y0 = np.sin(t0)
            z0 = 0*t0

            x1 = x0
            y1 = y0*np.cos(alpha) - z0*np.sin(alpha)
            z1 = y0*np.sin(alpha) + z0*np.cos(alpha)

            ra = np.arctan(y1/x1)
            ra[x1 < 0] += np.pi
            ra[(x1 >= 0) & (y1 < 0)] += 2*np.pi
            if ra[-1] == 0:
                ra[-1] += 2*np.pi
            de = np.arcsin(z1)

            return ra*180/np.pi, de*180/np.pi
        
    @staticmethod
    def merge_animate(*animate_functions):
        def ani(frame):
            result = []
            for ani_sub in animate_functions:
                result += list(ani_sub(frame=frame))
            return tuple(result)
        return ani
    

class FigureBase():
    def __init__(self):
        self.max_xyz = 0

        self.fig = None
        self.ax_xy = None
        self.ax_xz = None
        self.ax_yz = None
        self.ax_north = None
        self.ax_south = None
        self.ax_map = None

    def plot_constellation(self, cat_df, con_id, color, **kwargs):
        # map, north and south polar
        if self.ax_map is not None:
            SkyMapper.plot_constellation(ax=self.ax_map, cat_df=cat_df, con_id=con_id, color=color, **kwargs)
        if self.ax_north is not None:
            SkyMapper.plot_constellation(ax=self.ax_north, cat_df=cat_df, con_id=con_id, color=color, polar=True, **kwargs)
        if self.ax_south is not None:
            SkyMapper.plot_constellation(ax=self.ax_south, cat_df=cat_df, con_id=con_id, color=color, polar=True, south=True, **kwargs)

    def plot_stars(self, cat_df, size_func=lambda x: 10*x**2, **kwargs):
        # map, north and south polar
        if self.ax_map is not None:
            SkyMapper.plot_stars(ax=self.ax_map, cat_df=cat_df, size_func=size_func, **kwargs)
        if self.ax_north is not None:
            SkyMapper.plot_stars(ax=self.ax_north, cat_df=cat_df, size_func=size_func, polar=True, **kwargs)
        if self.ax_south is not None:
            SkyMapper.plot_stars(ax=self.ax_south, cat_df=cat_df, size_func=size_func, polar=True, south=True, **kwargs)

    def plot_radec(self, ra_deg, dec_deg, **kwargs):
        if self.ax_map is not None:
            SkyMapper.plot(ax=self.ax_map, ra_deg=ra_deg, dec_deg=dec_deg, **kwargs)
        if self.ax_north is not None:
            SkyMapper.plot_polar(ax=self.ax_north, ra_deg=ra_deg, dec_deg=dec_deg, **kwargs)
        if self.ax_south is not None:
            SkyMapper.plot_polar(ax=self.ax_south, ra_deg=ra_deg, dec_deg=dec_deg, south=True, **kwargs)

    def animate_radec(self, ra_deg, dec_deg, color, linestyle=None, marker='o'):
        ani_list = []
        if self.ax_map is not None:
            ani_list.append(SkyMapper.animate(ax=self.ax_map, ra_deg=ra_deg, dec_deg=dec_deg, 
                                              color=color, linestyle=linestyle, marker=marker))
        if self.ax_north is not None:
            ani_list.append(SkyMapper.animate_polar(ax=self.ax_north, 
                                                    ra_deg=ra_deg, dec_deg=dec_deg, 
                                                    color=color, linestyle=linestyle, 
                                                    marker=marker))
        if self.ax_south is not None:
            ani_list.append(SkyMapper.animate_polar(ax=self.ax_south, 
                                                    ra_deg=ra_deg, dec_deg=dec_deg, 
                                                    color=color, linestyle=linestyle, 
                                                    marker=marker, south=True))
        return SkyMapper.merge_animate(*ani_list)

    def scatter_radec(self, ra_deg, dec_deg, **kwargs):
        if self.ax_map is not None:
            SkyMapper.scatter(ax=self.ax_map, ra_deg=ra_deg, dec_deg=dec_deg, **kwargs)
        if self.ax_north is not None:
            SkyMapper.scatter_polar(ax=self.ax_north, ra_deg=ra_deg, dec_deg=dec_deg, **kwargs)
        if self.ax_south is not None:
            SkyMapper.scatter_polar(ax=self.ax_south, ra_deg=ra_deg, dec_deg=dec_deg, south=True, **kwargs)

    def plot_xyz(self, x, y, z, **kwargs):
        if self.ax_xy is not None:
            self.ax_xy.plot(x, y, **kwargs)
            self.ax_xz.plot(x, z, **kwargs)
            self.ax_yz.plot(y, z, **kwargs)

    def animate_xyz(self, x, y, z, color, linestyle=None, marker='o'):
        if self.ax_xy is not None:
            ln_xy, = self.ax_xy.plot([], [], color=color, linestyle=linestyle)
            ln_xz, = self.ax_xz.plot([], [], color=color, linestyle=linestyle)
            ln_yz, = self.ax_yz.plot([], [], color=color, linestyle=linestyle)
            sc_xy, = self.ax_xy.plot([], [], color=color, marker=marker)
            sc_xz, = self.ax_xz.plot([], [], color=color, marker=marker)
            sc_yz, = self.ax_yz.plot([], [], color=color, marker=marker)
            
            def ani(frame):
                ln_xy.set_data(x[:frame+1], y[:frame+1])
                ln_xz.set_data(x[:frame+1], z[:frame+1])
                ln_yz.set_data(y[:frame+1], z[:frame+1])
                # return ln0, ln1, ln2
                sc_xy.set_data(x[frame], y[frame])
                sc_xz.set_data(x[frame], z[frame])
                sc_yz.set_data(y[frame], z[frame])
                return ln_xy, ln_xz, ln_yz, sc_xy, sc_xz, sc_yz
            
            return ani
    
    def animate_text(self, tlist, **kwargs):
        if self.ax_xy is not None:
            kwargs_use = {'x': -0.8*self.max_xyz, 'y': 0.8*self.max_xyz, 's': ''}
            kwargs_use.update(kwargs)
            txt = self.ax_xy.text(**kwargs_use)
            def ani(frame):
                txt.set_text(tlist[frame].strftime(r'%Y-%m-%d'))
                return txt,
            return ani

    def scatter_xyz(self, x, y, z, **kwargs):
        if self.ax_xy is not None:
            self.ax_xy.scatter(x, y, **kwargs)
            self.ax_xz.scatter(x, z, **kwargs)
            self.ax_yz.scatter(y, z, **kwargs)

    def legend(self, tags, colors, linestyles=None, ax=None, loc='upper right'):
        if ax is None:
            assert self.ax_yz is not None, 'YZ axis do not exist'
            ax = self.ax_yz
        
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


class FigureFull(FigureBase):
    def __init__(self, max_xyz=10, ymax=75, rmax=60, dark=True, outside=False):
        FigureBase.__init__(self)
        
        hratios = [5, 15*(2*ymax/360)+1, 7.5]
        fig = plt.figure(figsize=(15, np.sum(hratios)))
        gs = fig.add_gridspec(nrows=3, ncols=6, height_ratios=hratios)
        ax_xy = fig.add_subplot(gs[0, 0:2])
        ax_xz = fig.add_subplot(gs[0, 2:4])
        ax_yz = fig.add_subplot(gs[0, 4:6])

        ax_north = fig.add_subplot(gs[2, 0:3], projection='polar')
        ax_south = fig.add_subplot(gs[2, 3:6], projection='polar')
        ax_map = fig.add_subplot(gs[1, :])

        SkyMapper.set_ax_polar(ax=ax_north, rmax=rmax, dark=dark, outside=outside)
        SkyMapper.set_ax_polar(ax=ax_south, rmax=rmax, dark=dark, south=True, outside=outside)
        SkyMapper.set_ax(ax=ax_map, ymax=ymax, dark=dark, outside=outside)

        ax_xy.set(xlabel='X', ylabel='Y')
        ax_xz.set(xlabel='X', ylabel='Z')
        ax_yz.set(xlabel='Y', ylabel='Z')

        for ax in [ax_xy, ax_xz, ax_yz]:
            ax.set_xlim(-max_xyz, max_xyz)
            ax.set_ylim(-max_xyz, max_xyz)
            ax.grid(True)

        self.max_xyz = max_xyz

        self.fig = fig
        self.ax_xy = ax_xy
        self.ax_xz = ax_xz
        self.ax_yz = ax_yz
        self.ax_north = ax_north
        self.ax_south = ax_south
        self.ax_map = ax_map

class Figure3D(FigureBase):
    def __init__(self, max_xyz=10):
        FigureBase.__init__(self)
        
        fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
        ax_xy = axs[0]
        ax_xz = axs[1]
        ax_yz = axs[2]

        ax_xy.set(xlabel='X', ylabel='Y')
        ax_xz.set(xlabel='X', ylabel='Z')
        ax_yz.set(xlabel='Y', ylabel='Z')

        for ax in [ax_xy, ax_xz, ax_yz]:
            ax.set_xlim(-max_xyz, max_xyz)
            ax.set_ylim(-max_xyz, max_xyz)
            ax.grid(True)

        self.max_xyz = max_xyz

        self.fig = fig
        self.ax_xy = ax_xy
        self.ax_xz = ax_xz
        self.ax_yz = ax_yz

class FigureMap(FigureBase):
    def __init__(self, ymax=75, dark=True, outside=False):
        FigureBase.__init__(self)
        
        fig, ax_map = plt.subplots(figsize=(15, 15*(2*ymax/360)+1))
        SkyMapper.set_ax(ax=ax_map, ymax=ymax, dark=dark, outside=outside)

        self.fig = fig
        self.ax_map = ax_map

class FigureMapPolar(FigureBase):
    def __init__(self, ymax=75, rmax=60, dark=True, outside=False):
        FigureBase.__init__(self)
        
        hratios = [15*(2*ymax/360)+1, 7.5]
        fig = plt.figure(figsize=(15, np.sum(hratios)))
        gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=hratios)

        ax_map = fig.add_subplot(gs[0, :])
        ax_north = fig.add_subplot(gs[1, 0], projection='polar')
        ax_south = fig.add_subplot(gs[1, 1], projection='polar')

        SkyMapper.set_ax_polar(ax=ax_north, rmax=rmax, dark=dark, outside=outside)
        SkyMapper.set_ax_polar(ax=ax_south, rmax=rmax, dark=dark, south=True, outside=outside)
        SkyMapper.set_ax(ax=ax_map, ymax=ymax, dark=dark, outside=outside)

        self.fig = fig
        self.ax_north = ax_north
        self.ax_south = ax_south
        self.ax_map = ax_map

class Figure3DMap(FigureBase):
    def __init__(self, max_xyz=10, ymax=75, dark=True, outside=False):
        FigureBase.__init__(self)
        
        hratios = [5, 15*(2*ymax/360)+1]
        fig = plt.figure(figsize=(15, np.sum(hratios)))
        gs = fig.add_gridspec(nrows=2, ncols=3, 
                              height_ratios=hratios)

        ax_xy = fig.add_subplot(gs[0, 0])
        ax_xz = fig.add_subplot(gs[0, 1])
        ax_yz = fig.add_subplot(gs[0, 2])
        ax_map = fig.add_subplot(gs[1, :])

        SkyMapper.set_ax(ax=ax_map, ymax=ymax, dark=dark, outside=outside)

        ax_xy.set(xlabel='X', ylabel='Y')
        ax_xz.set(xlabel='X', ylabel='Z')
        ax_yz.set(xlabel='Y', ylabel='Z')

        for ax in [ax_xy, ax_xz, ax_yz]:
            ax.set_xlim(-max_xyz, max_xyz)
            ax.set_ylim(-max_xyz, max_xyz)
            ax.grid(True)

        self.max_xyz = max_xyz

        self.fig = fig
        self.ax_map = ax_map
        self.ax_xy = ax_xy
        self.ax_xz = ax_xz
        self.ax_yz = ax_yz