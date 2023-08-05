import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import datetime
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body, get_moon
from scipy.optimize import leastsq, fmin, minimize, differential_evolution, brute
import random, time
from mpl_toolkits import mplot3d


class OrbitalModel:
    def __init__(self):
        pass

    def coord_0(self, time):  # 黄道坐标，轨道平面
        return [0, 0, 0]

    def coord_1(self, time):  # 赤道坐标，赤道平面
        alpha = 0.4095  # 黄赤交角 23.461 deg -> rad
        x0, y0, z0 = self.coord_0(time)
        x1 = x0
        y1 = np.cos(alpha) * y0 - np.sin(alpha) * z0
        z1 = np.sin(alpha) * y0 + np.cos(alpha) * z0
        return [x1, y1, z1]

    def coord_deg(self, earth_model, time):
        x1, y1, z1 = np.array(self.coord_1(time)) - np.array(earth_model.coord_1(time))
        if x1 > 0:
            ra = np.arctan(y1/x1) * 180 / np.pi
        elif x1 < 0:
            ra = np.arctan(y1/x1) * 180 / np.pi + 180
        elif y1 > 0:  # x1 == 0
            ra = 90
        elif y1 < 0:
            ra = 270
        else:
            ra = 0
        ra = ra % 360
        dec = np.arctan(z1/(x1**2 + y1**2)**0.5) * 180 / np.pi
        return [ra, dec]
    
    def get_xyz(self, time_series):  # 赤道坐标
        results = []
        for t in time_series:
            results.append(self.coord_1(time=t))
        return np.array(results).T

    def get_radec(self, earth_model, time_series):  # deg
        results = []
        for t in time_series:
            results.append(self.coord_deg(earth_model=earth_model, time=t))
        return np.array(results).T

    def error(self, time_series, target, astropy_model='de430'):
        error = 0
        for time in time_series:
            model_coord = self.coord_1(time)
            true_coord = get_body_barycentric(target,
                                              Time(time.strftime('%Y-%m-%d %H:%M')),
                                              ephemeris=astropy_model)
            error += (model_coord[0] - true_coord.x.value) ** 2
            error += (model_coord[1] - true_coord.y.value) ** 2
            error += (model_coord[2] - true_coord.z.value) ** 2
        return error ** 0.5


class PreciseModel(OrbitalModel):
    def __init__(self, target, ephemeris='de430'):
        OrbitalModel.__init__(self)
        self.target = target
        self.ephemeris = ephemeris

    def coord_1(self, time):
        coord = get_body_barycentric(self.target,
                                     Time(time.strftime('%Y-%m-%d %H:%M')),
                                     ephemeris=self.ephemeris)
        return coord.xyz.to('AU').value

    def coord_0(self, time):
        alpha = 0.4095  # 黄赤交角 23.461 deg -> rad
        x1, y1, z1 = self.coord_1(time)
        x0 = x1
        y0 = np.cos(alpha) * y1 + np.sin(alpha) * z1
        z0 = - np.sin(alpha) * y1 + np.cos(alpha) * z1
        return [x0, y0, z0]


class CircleModel(OrbitalModel):
    def __init__(self, r_AU=1, period_days=365.25, theta_0_deg=180):
        OrbitalModel.__init__(self)
        self.r = r_AU  # AU, no * 1.496e8
        self.period_days = period_days
        self.theta_0 = theta_0_deg * np.pi / 180  # rad
        self.day_initial = datetime.datetime.fromisoformat('2022-03-21')

    def coord_0(self, time):  # 黄道坐标
        delta_t = (time - self.day_initial).days
        theta = (delta_t / self.period_days) * 2 * np.pi + self.theta_0
        return [self.r * np.cos(theta), self.r * np.sin(theta), 0]  # x, y, z


class EclipseModel(OrbitalModel):
    def __init__(self, a_AU=1, e=0.0167, period_days=365.25, theta_0_deg=180 - (114.2 + 180),
                 assending_deg=0, perihelion_deg=114.2, inclination_deg=0):
        OrbitalModel.__init__(self)

        # geometry
        self.a = a_AU  # AU, no * 1.496e8
        self.e = e  # 偏心率
        self.b = (self.a ** 2 * (1 - self.e ** 2)) ** 0.5
        self.area = np.pi * self.a * self.b

        # oribital
        self.period_days = period_days
        self.assending = assending_deg * np.pi / 180  # rad, 升交点黄经
        self.perihelion = perihelion_deg * np.pi / 180  # rad, 近日点幅角
        self.inclination = inclination_deg * np.pi / 180  # rad

        # initial conditions
        self.theta_0 = theta_0_deg * np.pi / 180  # rad
        self.day_initial = datetime.datetime.fromisoformat('2022-03-21')

        # kinematics
        self.area_speed = self.area / self.period_days
        self.theta_t = self._get_theta_func_of_time()

    # 参数方程 r(theta)
    def rho(self, theta):
        return (self.a * (1 - self.e ** 2)) / (1 - self.e * np.cos(theta))

    # get theta(time)
    def _get_theta_func_of_time(self):
        theta_list = np.linspace(self.theta_0, self.theta_0 + 2 * np.pi + 0.1, 1000)  # add 0.1 to avoid bug
        area_list = []

        def delta_area(theta):
            return 0.5 * self.rho(theta) ** 2

        for theta in theta_list:
            area_list.append(integrate.quad(delta_area, self.theta_0, theta)[0])

        time_list = np.array(area_list) / self.area_speed

        return interp1d(time_list, theta_list, kind='linear')  # interpolated function

    def coord_plane(self, time):  # coordinate in orbit plane
        delta_t = (time - self.day_initial).days % self.period_days
        theta = self.theta_t(delta_t)
        rho = self.rho(theta)
        # 半长轴为X轴
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        beta = self.perihelion + np.pi
        return [x * np.cos(beta) - y * np.sin(beta),
                x * np.sin(beta) + y * np.cos(beta)]  # 旋转到升交点为x轴

    def coord_0(self, time):
        x_plane, y_plane = self.coord_plane(time)
        # 旋转黄道平面坐标（保持升交点为X轴）
        x_rot = x_plane
        y_rot = np.cos(self.inclination) * y_plane
        z_rot = np.sin(self.inclination) * y_plane
        # 黄道平面坐标（修正升交点黄经，以春分日地球->太阳连线为X轴）
        x_0 = np.cos(self.assending) * x_rot - np.sin(self.assending) * y_rot
        y_0 = np.sin(self.assending) * x_rot + np.cos(self.assending) * y_rot
        z_0 = z_rot
        return [x_0, y_0, z_0]


def plot_models(*models: OrbitalModel, curve_t=list(range(200)), scatter_t=[0, 10, 30]):
    def get_coords(m, t_list):
        coord_list = []
        for t in t_list:
            coord_list.append(m.coord_1(datetime.datetime.now() + datetime.timedelta(days=t)))
        return np.array(coord_list)

    ax = plt.subplot()

    for m in models:
        ax.plot(*get_coords(m, curve_t).T[:2])
        ax.scatter(*get_coords(m, scatter_t).T[:2])

    ax.scatter([0], [0], marker='+')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


def plot_models_3d(*models: OrbitalModel, curve_t=list(range(200)), scatter_t=[0, 10, 30]):
    def get_coords(m, t_list):
        coord_list = []
        for t in t_list:
            coord_list.append(m.coord_1(datetime.datetime.now() + datetime.timedelta(days=t)))
        return np.array(coord_list)

    ax = plt.axes(projection='3d')

    for m in models:
        ax.plot(*get_coords(m, curve_t).T)
        ax.scatter(*get_coords(m, scatter_t).T)

    ax.scatter([0], [0], [0], marker='+')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


def plot_models_3d_label(model_dict, curve_t=list(range(80)), scatter_t=[0, 10, 30]):
    def get_coords(m, t_list):
        coord_list = []
        for t in t_list:
            coord_list.append(m.coord_1(datetime.datetime.now() + datetime.timedelta(days=t)))
        return np.array(coord_list)

    ax = plt.axes(projection='3d')

    for m_name, m in model_dict.items():
        ax.plot(*get_coords(m, curve_t).T, label=m_name)
        ax.scatter(*get_coords(m, scatter_t).T)

    ax.scatter([0], [0], [0], marker='+')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.show()


# optimizing
# target: a, e, T, inc, asd, per, theta0_circular, theta0_elliptical
# orbits = {'mercury': (0.387, 0.206, 87.97, 7.00, 48.33, 29.2, 336.7, 61.2),
#           'venus': (0.723, 0.00648, 224.7, 3.39, 76.68, 56.3, 223.8, 271.0),
#           'earth': (1.0, 0.0172, 365.25, 0, 0, 114.2, 179.2, 245.8),
#           'mars': (1.524, 0.0932, 686.97, 1.85, 49.56, 287, 289.1, 122.7),
#           'jupiter': (5.204, 0.0483, 4331, 1.303, 100.46, 273, 349.7, 152.6),
#           'saturn': (9.583, 0.0534, 10824, 2.485, 113.665, 338, 324.5, 48.4),
#           'uranus': (19.218, 0.046, 30882, 0.771, 74.1, 93.8, 51.7, 59.2),
#           'neptune': (30.3, 0.0143, 60896, 1.768, 131.784, 251, 1.5, 155.9)}

orbits = {'mercury': (0.38700886,  0.2097759, 87.99030232,
                      7.00982693, 48.14651942, 25.67752864,
                      334.04430974, 65.40242943),
          'venus': (0.722564471, 0.00587386838, 224.674639,
                    3.39619006, 76.6941430, 26.3000000,
                    223.84244825, 301.000000),
          'earth': (1.00031741,  0.0183136279,  365.227808,
                    0.0, -0.459099970,  89.5774922,
                    178.35688127, 271.805193),
          'mars': (1.52403457, 0.0940906030, 686.957555,
                   1.83348709, 49.8432306, 287.599209,
                   291.52798276, 1.21813507e+02),
          'jupiter': (5.19884550, 0.0482563006, 4333.26879,
                      1.30820318, 101.697915, 273.425456,
                      350.44769549, 150.973894),
          'saturn': (9.53665940, 0.0544091603, 10755.0185,
                     2.49742564, 114.038572, 338.400763,
                     323.9132159, 4.46787659e+01),
          'uranus': (19.1878471, 0.0475513687, 30680.6986,
                     0.764464345, 75.7254062, 95.2757505,
                     47.86594222, 52.9864176),
          'neptune': (30.0735728, 0.00894016313, 60200.4221,
                      1.78650794, 132.322962, 271.650205,
                      353.24657499, 128.666096)}


class OrbitOptimizer:
    def __init__(self, target_name, a, e, T, inc, asd, per, theta_0_c, theta_0_e):
        self.target_name = target_name
        self.a = a
        self.e = e
        self.T = T
        self.inc = inc
        self.asd = asd
        self.per = per
        self.theta_0_c = theta_0_c
        self.theta_0_e = theta_0_e

    def get_time_series(self, n_points=5):
        t_list = list(range(int(3 * self.T)))
        if len(t_list) > n_points:
            t_list = random.sample(t_list, n_points)
        time_series = []
        for i in t_list:
            time_series.append(datetime.datetime.now() + datetime.timedelta(days=i))
        return time_series

    def circular_model0(self, theta_0=None):
        if theta_0 is None:
            theta_0 = self.theta_0_c
        return CircleModel(r_AU=self.a, period_days=self.T, theta_0_deg=theta_0)

    def elliptical_model0(self, theta_0=None):
        if theta_0 is None:
            theta_0 = self.theta_0_e
        return EclipseModel(a_AU=self.a, e=self.e, period_days=self.T,
                            inclination_deg=self.inc,
                            assending_deg=self.asd, perihelion_deg=self.per,
                            theta_0_deg=theta_0)

    def get_circular_theta0(self):
        time_start = time.time()
        time_series = self.get_time_series()
        result = differential_evolution(lambda p: self.circular_model0(p[0]).error(time_series=time_series,
                                                                                   target=self.target_name,
                                                                                   astropy_model='de430'),
                                        bounds=[(0, 360)])
        print(f'== time taken {(time.time() - time_start)/60:.1f}min ==')
        print(result)

    def minimize_circular(self, n_points=30):
        time_start = time.time()
        time_series = self.get_time_series(n_points=n_points)
        result = minimize(lambda p: CircleModel(r_AU=p[0],
                                                period_days=p[1],
                                                theta_0_deg=p[2]).error(time_series=time_series,
                                                                        target=self.target_name,
                                                                        astropy_model='de430'),
                          (self.a, self.T, self.theta_0_c),
                          bounds=[(0.9*self.a, 1.1*self.a),
                                  (0.9*self.T, 1.1*self.T),
                                  (self.theta_0_c - 30, self.theta_0_c + 30)])
        error_0 = CircleModel(r_AU=self.a, period_days=self.T,
                              theta_0_deg=self.theta_0_c).error(time_series=time_series,
                                                                target=self.target_name,
                                                                astropy_model='de430')
        error_1 = CircleModel(r_AU=result.x[0], period_days=result.x[1],
                              theta_0_deg=result.x[2]).error(time_series=time_series,
                                                             target=self.target_name,
                                                             astropy_model='de430')
        print(f'Error {error_0} -> {error_1} ({100*error_1/error_0:.2f}%)')
        print(f'== time taken {(time.time() - time_start)/60:.1f}min ==')
        print(result)

    def get_elliptical_theta0(self):
        time_start = time.time()
        time_series = self.get_time_series()
        result = differential_evolution(lambda p: self.elliptical_model0(p[0]).error(time_series=time_series,
                                                                                     target=self.target_name,
                                                                                     astropy_model='de430'),
                                        bounds=[(0, 360)])
        print(f'== time taken {(time.time() - time_start)/60:.1f}min ==')
        print(result)

    def minimize_elliptical(self, n_points=30):
        time_start = time.time()
        time_series = self.get_time_series(n_points=n_points)
        result = minimize(lambda p: EclipseModel(a_AU=p[0],
                                                 e=p[1],
                                                 period_days=p[2],
                                                 inclination_deg=p[3],
                                                 assending_deg=p[4],
                                                 perihelion_deg=p[5],
                                                 theta_0_deg=p[6]).error(time_series=time_series,
                                                                         target=self.target_name,
                                                                         astropy_model='de430'),
                          (self.a, self.e, self.T, self.inc,
                           self.asd, self.per, self.theta_0_e),
                          bounds=[(0.9*self.a, 1.1*self.a),
                                  (0.5 * self.e, 1.5 * self.e),
                                  (0.9*self.T, 1.1*self.T),
                                  (0, self.inc + 2),
                                  (self.asd - 30, self.asd + 30),
                                  (self.per - 30, self.per + 30),
                                  (self.theta_0_e - 30, self.theta_0_e + 30)])
        error_0 = EclipseModel(a_AU=self.a, e=self.e, period_days=self.T,
                               inclination_deg=self.inc, assending_deg=self.asd,
                               perihelion_deg=self.per,
                               theta_0_deg=self.theta_0_e).error(time_series=time_series,
                                                                 target=self.target_name,
                                                                 astropy_model='de430')
        error_1 = EclipseModel(a_AU=result.x[0], e=result.x[1], period_days=result.x[2],
                               inclination_deg=result.x[3], assending_deg=result.x[4],
                               perihelion_deg=result.x[5],
                               theta_0_deg=result.x[6]).error(time_series=time_series,
                                                              target=self.target_name,
                                                              astropy_model='de430')
        print(f'Error {error_0} -> {error_1} ({100*error_1/error_0:.2f}%)')
        print(f'== time taken {(time.time() - time_start)/60:.1f}min ==')
        print(result)

    def minimize_elliptical_advance(self, n_points=30):
        time_start = time.time()
        time_series = self.get_time_series(n_points=n_points)
        result = differential_evolution(lambda p: EclipseModel(a_AU=p[0],
                                                               e=p[1],
                                                               period_days=p[2],
                                                               inclination_deg=p[3],
                                                               assending_deg=p[4],
                                                               perihelion_deg=p[5],
                                                               theta_0_deg=p[6]
                                                               ).error(time_series=time_series,
                                                                       target=self.target_name,
                                                                       astropy_model='de430'),
                                        bounds=[(0.9*self.a, 1.1*self.a),
                                                (0.5 * self.e, 1.5 * self.e),
                                                (0.9*self.T, 1.1*self.T),
                                                (0, self.inc + 2),
                                                (self.asd - 30, self.asd + 30),
                                                (self.per - 30, self.per + 30),
                                                (self.theta_0_e - 30, self.theta_0_e + 30)])
        error_0 = EclipseModel(a_AU=self.a, e=self.e, period_days=self.T,
                               inclination_deg=self.inc, assending_deg=self.asd,
                               perihelion_deg=self.per,
                               theta_0_deg=self.theta_0_e).error(time_series=time_series,
                                                                 target=self.target_name,
                                                                 astropy_model='de430')
        error_1 = EclipseModel(a_AU=result.x[0], e=result.x[1], period_days=result.x[2],
                               inclination_deg=result.x[3], assending_deg=result.x[4],
                               perihelion_deg=result.x[5],
                               theta_0_deg=result.x[6]).error(time_series=time_series,
                                                              target=self.target_name,
                                                              astropy_model='de430')
        print(f'Error {error_0} -> {error_1} ({100*error_1/error_0:.2f}%)')
        print(f'== time taken {(time.time() - time_start)/60:.1f}min ==')
        print(result)


    def get_elliptical_theta0_b(self):
        time_start = time.time()
        time_series = self.get_time_series()
        result = minimize(lambda p: self.elliptical_model0(p[0]).error(time_series=time_series,
                                                                       target=self.target_name,
                                                                       astropy_model='de430'),
                          (180,), bounds=[(0, 360)])
        print(f'== time taken {(time.time() - time_start)/60:.1f}min ==')
        print(result)
        return result


def auto_optimize():
    for target, target_params in orbits.items():
        print(f'start {target} - [{datetime.datetime.now().strftime("%H:%M:%S")}]\n')
        oot = OrbitOptimizer(target, *target_params)
        oot.get_circular_theta0()
        print()
        # oot.get_elliptical_theta0()
        # print()


def auto_minimize(n_points=100):
    for target, target_params in orbits.items():
        print(f'start {target} - [{datetime.datetime.now().strftime("%H:%M:%S")}]\n')
        oot = OrbitOptimizer(target, *target_params)
        oot.minimize_circular(n_points=n_points)
        print()
        oot.minimize_elliptical(n_points=n_points)
        print()


def auto_minimize_advanced(n_points=100):
    for target, target_params in orbits.items():
        print(f'start {target} - [{datetime.datetime.now().strftime("%H:%M:%S")}]\n')
        oot = OrbitOptimizer(target, *target_params)
        oot.minimize_elliptical_advance(n_points=n_points)
        print()


def plot_error(target, t0, t1, error=True):
    oot = OrbitOptimizer(target, *orbits[target])
    oot_earth = OrbitOptimizer('earth', *orbits['earth'])
    time_series = []
    for t in range(t0, t1):
        time_series.append(datetime.datetime.now() + datetime.timedelta(days=t))

    target_model = {'p': PreciseModel(target=target),
                    'c': oot.circular_model0(),
                    'e': oot.elliptical_model0()}
    earth_model = {'p': PreciseModel(target='earth'),
                   'c': oot_earth.circular_model0(),
                   'e': oot_earth.elliptical_model0()}

    ra = {'p': [], 'c': [], 'e': []}
    dec = {'p': [], 'c': [], 'e': []}

    for time in time_series:
        for m in ['p', 'c', 'e']:
            ra_0, dec_0 = target_model[m].coord_deg(earth_model=earth_model[m],
                                                    time=time)
            ra[m].append(ra_0)
            dec[m].append(dec_0)

    if error:
        ra_error_c = np.array(ra['c']) - np.array(ra['p'])
        dec_error_c = np.array(dec['c']) - np.array(dec['p'])
        ra_error_e = np.array(ra['e']) - np.array(ra['p'])
        dec_error_e = np.array(dec['e']) - np.array(dec['p'])

        ra_error_c = (ra_error_c + 180) % 360 - 180
        ra_error_e = (ra_error_e + 180) % 360 - 180

        fig, axs = plt.subplots(nrows=2, sharex=True)
        axs[0].plot(ra_error_c, label='circular')
        axs[0].plot(ra_error_e, label='elliptical')
        axs[0].legend()
        axs[0].grid()
        axs[0].set_title(target)
        axs[1].plot(dec_error_c)
        axs[1].plot(dec_error_e)
        axs[1].grid()
        fig.tight_layout()
        plt.show()
    else:
        fig, axs = plt.subplots(nrows=2, sharex=True)
        for m in ['p', 'c', 'e']:
            axs[0].plot(ra[m], label=m)
            axs[1].plot(dec[m])
        axs[0].legend()
        axs[0].set_title(target)
        axs[0].grid()
        axs[1].grid()
        fig.tight_layout()
        plt.show()