import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
from collections.abc import Iterable
import datetime, pytz


# to update radec, deal with (0, 0) input

def _convert_time(ti):
    """
    convert time into datetime.datetime object
    :param ti: date or number, str
    :return: datetime.datetime object
    """
    if isinstance(ti, int):
        return datetime.datetime(year=ti, month=1, day=1)
    elif isinstance(ti, float):
        t = datetime.datetime(year=int(ti), month=1, day=1)
        t += datetime.timedelta((ti - int(ti))*365.2425)
        return t
    elif isinstance(ti, str):
        return datetime.datetime.fromisoformat(ti)
    elif isinstance(ti, datetime.date):
        return datetime.datetime(year=ti.year,
                                 month=ti.month,
                                 day=ti.day)
    elif isinstance(ti, datetime.datetime):
        return ti
    elif isinstance(ti, Iterable):
        raise ValueError
    else:
        raise ValueError


def _convert_xyz_to_equatorial(x0, y0, z0):
    alpha = 0.4095  # 黄赤交角 23.461 deg -> rad
    x1 = x0
    y1 = np.cos(alpha) * y0 - np.sin(alpha) * z0
    z1 = np.sin(alpha) * y0 + np.cos(alpha) * z0
    return x1, y1, z1


def cal_radec(xyz_list, xyz_list_obs):
    """
    calculate (ra, dec) coordinates
    ! problematic with delta_xyz == 0
    :param xyz_list: shape (n, 3)
    :param xyz_list_obs: shape (n, 3)
    :return: coordinates in degrees, shape (n, 2)
    """
    if len(xyz_list) != len(xyz_list_obs):  # n
        raise ValueError('xyz_list & xyz_list_obs not match')

    rel_xyz = np.array(xyz_list) - np.array(xyz_list_obs)  # shape (n, 3)
    x, y, z = rel_xyz.T
    radec = np.zeros(shape=(len(xyz_list), 2), dtype=float)  # shape (n, 2)

    # x > 0
    idx = x > 0
    radec[:, 0][idx] = np.arctan(y[idx] / x[idx]) * 180 / np.pi

    # x < 0
    idx = x < 0
    radec[:, 0][idx] = np.arctan(y[idx] / x[idx]) * 180 / np.pi + 180

    # x == 0, y > 0
    idx = (x == 0) & (y > 0)
    radec[:, 0][idx] = 90

    # x == 0, y < 0
    idx = (x == 0) & (y < 0)
    radec[:, 0][idx] = 270

    # x == 0, y == 0 -> ra = 0
    # !
    # this might be problematic -> try to improve
    # buggy for dec as well

    radec[:, 0] = radec[:, 0] % 360
    radec[:, 1] = np.arctan(z / (x ** 2 + y ** 2) ** 0.5) * 180 / np.pi

    return radec  # degree (n, 2)


def cal_period(m, a):
    """
    Kepler's 3rd law
    :param m: center object in solar mass
    :param a: radius / semi-major axis in AU
    :return: period in days
    """
    return 365.25*(a**3/m)**0.5


def time_sequence(t0, t1, n=100):
    """
    Populate a time sequence (forwards/backwards)
    :param t0: initial time
    :param t1: terminal time
    :param n: number of time points in sequence, require positive
    :return:
    """
    t0 = _convert_time(t0)
    t1 = _convert_time(t1)
    tlist = []
    dt = t1 - t0
    for k in np.linspace(0, 1, n):
        tlist.append(t0 + k * dt)
    return tlist


class BaseModel:
    def _xyz(self, ti):
        """
        calculate position at given ti
        :param ti: datetime.datetime object
        :return: 3d tuple, xyz in AU
        """
        return 0., 0., 0.

    def xyz(self, t):
        if isinstance(t, Iterable) and not isinstance(t, str):
            # note: str is Iterable
            xyz_list = []
            for ti in t:
                xyz_list.append(self._xyz(ti=_convert_time(ti)))
            return np.array(xyz_list)  # shape (n, 3)
        else:
            return np.array([self._xyz(ti=_convert_time(t))])  # shape (1, 3)


class StaticModel(BaseModel):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.static_xyz = (x, y, z)

    def _xyz(self, ti):
        return self.static_xyz


class CircularModel(BaseModel):
    def __init__(self, a, period, t0, time_initial=None):
        """
        Circular orbit model
        :param a: orbital radius, in AU
        :param period: orbital period, in days
        :param t0: initial phase angle theta, in degree
        :param time_initial: initial time, default 2022.3.21
        """
        self.a = a  # AU
        self.period = period  # day
        self.t0 = t0 * np.pi / 180  # degree -> rad

        if time_initial is None:
            self.time_initial = datetime.datetime(year=2022, month=3, day=21)
        else:
            self.time_initial = _convert_time(time_initial)

    def _xyz(self, ti):
        dt = (ti - self.time_initial).days
        theta = (dt / self.period) * 2 * np.pi + self.t0
        return _convert_xyz_to_equatorial(x0=self.a * np.cos(theta),
                                          y0=self.a * np.sin(theta),
                                          z0=0)  # x, y, z


class EllipticalModel(CircularModel):
    def __init__(self, a, period, t0, e, ascending, perihelion, inclination, time_initial=None):
        """
        Keplerian elliptical orbit model
        :param a: orbital semi-major axis, in AU
        :param period: orbital period, in days
        :param t0: initial phase angle theta, in degree
        :param e: eccentricity ratio
        :param ascending: longitude of ascending node, in degree
        :param perihelion: argument of perihelion, in degree
        :param inclination: inclination angle, in degree
        :param time_initial: initial time, default 2022.3.21
        """
        super().__init__(a=a, period=period, t0=t0, time_initial=time_initial)

        # geometry
        self.e = e
        self.b = (self.a ** 2 * (1 - self.e ** 2)) ** 0.5
        self.area = np.pi * self.a * self.b

        # orbital, deg -> rad
        self.ascending = ascending * np.pi / 180
        self.perihelion = perihelion * np.pi / 180
        self.inclination = inclination * np.pi / 180

        # kinematics
        self.area_speed = self.area / self.period
        self.func_theta = self._get_func_theta()

    def _rho(self, theta):
        """
        parametric equation rho(theta)
        :param theta: phase angle in rad
        :return: rho in AU
        """
        return (self.a * (1 - self.e ** 2)) / (1 - self.e * np.cos(theta))

    def _get_func_theta(self, n_bins=1000):
        """
        get a function of theta(t) according to Kepler's 2nd law
        :param n_bins: sampling of theta, higher if highly eccentric
        :return: function theta(t), t is numeric, in days
        """
        theta_list = np.linspace(self.t0, self.t0 + 2 * np.pi + 0.1, n_bins)  # add 0.1 to avoid bug
        area_list = []

        def delta_area(theta):
            return 0.5 * self._rho(theta) ** 2

        for theta_i in theta_list:
            area_list.append(integrate.quad(delta_area, self.t0, theta_i)[0])

        time_list = np.array(area_list) / self.area_speed

        return interp1d(time_list, theta_list, kind='linear')  # interpolated function

    def _xyz_plane(self, ti):  # coordinate in orbit plane
        delta_t = (ti - self.time_initial).total_seconds() / (24*3600)
        delta_t %= self.period  # days
        theta = self.func_theta(delta_t)
        rho = self._rho(theta)
        # 半长轴为X轴
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        beta = self.perihelion + np.pi
        return (x * np.cos(beta) - y * np.sin(beta),
                x * np.sin(beta) + y * np.cos(beta))  # 旋转到升交点为x轴

    def _xyz(self, ti):
        x_plane, y_plane = self._xyz_plane(ti=ti)
        # 旋转黄道平面坐标（保持升交点为X轴）
        x_rot = x_plane
        y_rot = np.cos(self.inclination) * y_plane
        z_rot = np.sin(self.inclination) * y_plane
        # 黄道平面坐标（修正升交点黄经，以春分日地球->太阳连线为X轴）
        x = np.cos(self.ascending) * x_rot - np.sin(self.ascending) * y_rot
        y = np.sin(self.ascending) * x_rot + np.cos(self.ascending) * y_rot
        z = z_rot
        return _convert_xyz_to_equatorial(x0=x, y0=y, z0=z)


class PointsModel(BaseModel):
    def __init__(self, t_list, xyz_list, interp_kind='linear'):
        if len(t_list) != len(xyz_list):
            raise ValueError('t_list & xyz_list not match')

        self.t_list = []  # expect (n)
        self.xyz_list = np.array(xyz_list)  # (n, 3)

        for ti in t_list:
            t = _convert_time(ti)
            if len(self.t_list) > 0:
                if t < self.t_list[-1]:
                    raise ValueError('t_list not in sequence')
            self.t_list.append(t)

        self.time_initial = self.t_list[0]

        numeric_t_list = []  # numeric, in days
        for ti in self.t_list:
            delta_ti = (ti - self.time_initial).total_seconds()
            numeric_t_list.append(delta_ti)  # in seconds
        numeric_t_list = np.array(numeric_t_list) / (24*3600)  # to days

        self.func_xyz = interp1d(x=numeric_t_list, y=self.xyz_list, axis=0,
                                 kind=interp_kind, bounds_error=False,
                                 fill_value='extrapolate')

    def _xyz(self, ti):
        delta_t = (ti - self.time_initial).total_seconds() / (24*3600)  # days
        return self.func_xyz(delta_t)


class PreciseModel(BaseModel):
    def __init__(self, name, ephemeris=None):
        self.target_name = name
        self.ephemeris = ephemeris

    def _xyz(self, ti):
        ti_utc = ti.astimezone(pytz.timezone('Asia/Shanghai'))  # compensate
        ti_astropy = Time(ti_utc, format='datetime', scale='utc')
        coord = get_body_barycentric(self.target_name,
                                     ti_astropy,
                                     ephemeris=self.ephemeris)
        return coord.xyz.to('AU').value

