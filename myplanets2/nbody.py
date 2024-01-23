import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from typing import List

import datetime
from .models import BaseModel, _convert_time


# to add get precise value

class NbodyModel(BaseModel):
    def __init__(self, m, xyz, v_xyz, time_initial=None):
        self.m = m  # in solar mass
        self.pos = np.array(xyz)  # AU, shape 3
        self.vel = np.array(v_xyz)  # AU/yr, earth value = 2pi, shape 3
        if time_initial is None:
            self.time_initial = datetime.datetime(year=2022, month=3, day=21)
        else:
            self.time_initial = _convert_time(time_initial)

        self.func_xyz = None

    def set_interp_func(self, t_list, xyz_list):
        """
        set the interp function
        :param t_list: shape n, numeric, in days
        :param xyz_list: shape (n, 3), in AU
        :return:
        """
        assert len(t_list) == len(xyz_list)
        self.func_xyz = interp1d(t_list, xyz_list, axis=0, kind='linear')

    def _xyz(self, ti):
        delta_day = (ti - self.time_initial) / datetime.timedelta(days=1)  # float
        return self.func_xyz(delta_day)


class NbodyCal:
    def __init__(self, model_list: List[NbodyModel], softening=0.01, G=(2*np.pi)**2):
        self.model_list = model_list
        self._mlist = None  # shape (n_obj)
        self._xlist = None  # shape (n_obj, 2, 3D)
        self._G = G  # (2pi)^2 in AU^3 yr^-2 Msol
        self._softening = softening  # AU, softening length, 20230628
        self._res = None
        # initialization here
        self._check_initial()
        self._load_initials()

    def _check_initial(self):
        di = self.model_list[0].time_initial
        for obj in self.model_list:
            if di != obj.time_initial:
                raise ValueError('inconsistent initial time!')

    def _load_initials(self):
        mlist = []
        xlist = []
        for obj in self.model_list:
            mlist.append(obj.m)
            xlist.append([obj.pos, obj.vel])
        self._mlist = np.array(mlist)  # shape (n,)
        self._xlist = np.array(xlist)  # shape (n, 2, 3)

    def _gravity(self, W):
        # create dw/dt:
        dwdt = np.zeros(shape=(len(W), 2, 3))  # same (n, 2, 3) shape

        # calculate dx/dt: (read from W directly)
        dwdt[:, 0] = W[:, 1]  # same as W[:,1,:], the velocity

        # caclculate dv/dt:
        xi = W[:, 0, np.newaxis]
        xj = W[:, 0]
        deltax = xi - xj  # shape (n, n, 3), relative position from one to the other, symmetric
        deltax_norm = np.linalg.norm(deltax, axis=2, keepdims=True)  # shape (n, n, 1) norm of dx
        deltax_norm_soft = np.array(deltax_norm)

        # 20230628
        # try map the x too small -> larger x value; r/(r*r_soft^2)
        if np.min(deltax_norm) < self._softening:
            # rx/rs = (1/3)*(rs/r) + (2/3)*(r/rs)**2
            # rx = (1/3)*(rs**2)/r + (2/3)*r**2/rs
            r = deltax_norm_soft[deltax_norm < self._softening]
            rx = (1 / 3) * (1 / r) * self._softening ** 2 + (2 / 3) * (1 / self._softening) * r ** 2
            deltax_norm_soft[deltax_norm < self._softening] = rx

        with np.errstate(divide='ignore', invalid='ignore'):  # ignore error in calculation
            # before_sum = self._G * self._mlist[:,np.newaxis] * deltax / deltax_norm**3  # GMx/|x|^3
            # soft
            before_sum = self._G * self._mlist[:, np.newaxis] * deltax / (
                        deltax_norm * deltax_norm_soft ** 2)  # GMx/|x|^3
            before_sum = np.where(np.isfinite(before_sum), before_sum, 0)  # replace inf -> 0
            # before_sum[i, j, :] is the acceleration on i from the gravity of j
            dwdt[:, 1] = -np.sum(before_sum, axis=1)

        return dwdt

    def run(self, tmax=36525, max_step=1,
            atol=1e-6, rtol=3e-3, method='RK45'):
        """

        :param tmax: max time, in days
        :param max_step: time step, in days
        :param atol:
        :param rtol:
        :param method:
        :return:
        """
        # other methods: RK45, RK23, LSODA, DOP853
        x0 = self._xlist.flatten()

        def my_g(t, y):
            return self._gravity(W=y.reshape(len(self.model_list), 2, 3)).flatten()

        # calculation in year unit
        r = solve_ivp(fun=my_g, t_span=(0, tmax/365.25), y0=x0, method=method,
                      max_step=max_step/365.25, atol=atol, rtol=rtol)

        # check if r success here! important
        if r.success:
            self._res = r
            self._assign_values()
        else:
            raise RuntimeError(f'Nbody calculation failed with:\n{r.message}')

    def _assign_values(self):
        n_obj = len(self.model_list)

        for i_obj, obj in enumerate(self.model_list):
            plist = []  # shape (t_step, 3d)
            tlist = self._res.t
            for i in range(len(tlist)):
                plist.append(self._res.y[:, i].reshape(n_obj, 2, 3)[i_obj, 0, :])
            plist = np.array(plist)

            # make interp function in each model
            obj.set_interp_func(t_list=tlist*365.25,  # (n), year to day
                                xyz_list=plist)  # (n, 3), AU
