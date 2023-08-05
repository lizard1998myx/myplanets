# via 20230628 3-body test.ipynb

from .planets import OrbitalModel
from scipy.integrate import solve_ivp
from astropy.coordinates import get_body_barycentric_posvel
from astropy import units as u
from astropy.time import Time

import datetime
from scipy.interpolate import interp1d
import numpy as np

def get_xv(target, day_initial):
    # day_initial = datetime.datetime(2023, 8, 3)
    x, v = get_body_barycentric_posvel(target, Time(day_initial.strftime('%Y-%m-%d %H:%M')))
    x, v = x.get_xyz().value, v.get_xyz().to(u.AU/u.yr).value
    return x.tolist(), v.tolist()

def get_models(mlist, xlist, vlist, day_initial, recenter=False):
    assert len(mlist) == len(xlist), 'list not match'
    assert len(mlist) == len(vlist), 'list not match'
    
    mlist = np.array(mlist, dtype=float)
    xlist = np.array(xlist, dtype=float)
    vlist = np.array(vlist, dtype=float)

    if recenter:
        # one more step for changing center of mass & velocity frame
        m_arr = np.vstack([mlist]*3).T  # shape (n, 3D)
        m_tot = np.sum(mlist)
        xlist -= np.sum(m_arr*xlist, axis=0)/m_tot
        vlist -= np.sum(m_arr*vlist, axis=0)/m_tot

    obj_list = []
    for m, x, v in zip(mlist, xlist, vlist):
        obj_list.append(NbodyModel(mass=m, x=x, v=v, day_initial=day_initial))
    return obj_list

class NbodyModel(OrbitalModel):
    def __init__(self, mass, x, v, day_initial=None):
        OrbitalModel.__init__(self)
        self.mass = mass  # Msun
        self.x = np.array(x)  # AU
        self.v = np.array(v)  # AU/yr, Vearth=2pi
        if day_initial is None:
            self.day_initial = datetime.datetime.fromisoformat('2022-03-21')
        else:
            self.day_initial = day_initial
        self._interp_funcs = None

    def _time_to_t_yr(self, time):
        # convert datetime.datetime object to float in yr
        dt = time - self.day_initial
        return dt.total_seconds() / (24*60*60*365.25)
    
    def _t_yr_to_xyz(self, t_yr):  # xyz in coord_1 (equatorial)
        assert self._interp_funcs is not None
        p = []
        for f in self._interp_funcs:
            p.append(f([t_yr])[0])  # result is an 1d array
        return p

    def set_interp_func(self, t, x, y, z):
        interp_funcs = []
        for p in [x, y, z]:
            interp_funcs.append(interp1d(t, p, kind='linear'))
        self._interp_funcs = interp_funcs

    def coord_1(self, time):
        return self._t_yr_to_xyz(t_yr=self._time_to_t_yr(time=time))

    # via PreciseModel
    def coord_0(self, time):
        alpha = 0.4095  # 黄赤交角 23.461 deg -> rad
        x1, y1, z1 = self.coord_1(time)
        x0 = x1
        y0 = np.cos(alpha) * y1 + np.sin(alpha) * z1
        z0 = - np.sin(alpha) * y1 + np.cos(alpha) * z1
        return [x0, y0, z0]
    
class NbodyCalculator:
    def __init__(self, object_list: list, softening=0.01, G=(2*np.pi)**2):
        self.object_list = object_list
        self._mlist = None  # shape (n_obj)
        self._xlist = None  # shape (n_obj, 2, 3D)
        self._G = G  # (2pi)^2 in AU^3 yr^-2 Msol
        self._softening = softening  # AU, softening length, 20230628
        self._res = None
        # initialization here
        self._check_initial()
        self._load_initials()

    def _check_initial(self):
        di = self.object_list[0].day_initial
        for obj in self.object_list:
            assert di == obj.day_initial

    def _load_initials(self):
        mlist = []
        xlist = []
        for obj in self.object_list:
            mlist.append(obj.mass)
            xlist.append([obj.x, obj.v])
        self._mlist = np.array(mlist)
        self._xlist = np.array(xlist)

    def _gravity(self, W):
        # create dw/dt:
        dwdt = np.zeros(shape=(len(W), 2, 3))  # same (n, 2, 3) shape
        
        # calculate dx/dt: (read from W directly)
        dwdt[:,0] = W[:,1]  # same as W[:,1,:], the velocity
        
        # caclculate dv/dt:
        xi = W[:,0,np.newaxis]
        xj = W[:,0]
        deltax = xi-xj  # shape (n, n, 3), relative position from one to the other, symmeteric
        deltax_norm = np.linalg.norm(deltax, axis=2, keepdims=True)  # shape (n, n, 1) norm of dx
        deltax_norm_soft = np.array(deltax_norm)
        
        # 20230628
        # try map the x too small -> larger x value; r/(r*r_soft^2)
        if np.min(deltax_norm) < self._softening:
            # rx/rs = (1/3)*(rs/r) + (2/3)*(r/rs)**2
            # rx = (1/3)*(rs**2)/r + (2/3)*r**2/rs
            r = deltax_norm_soft[deltax_norm < self._softening]
            rx = (1/3)*(1/r)*(self._softening)**2 + (2/3)*(1/self._softening)*r**2
            deltax_norm_soft[deltax_norm < self._softening] = rx
        
        with np.errstate(divide='ignore', invalid='ignore'):  # ignore error in calculation
            # before_sum = self._G * self._mlist[:,np.newaxis] * deltax / deltax_norm**3  # GMx/|x|^3
            # soft
            before_sum = self._G * self._mlist[:,np.newaxis] * deltax / (deltax_norm*deltax_norm_soft**2)  # GMx/|x|^3
            before_sum = np.where(np.isfinite(before_sum), before_sum, 0)  # replace inf -> 0
            # before_sum[i, j, :] is the acceleration on i from the gravity of j
            dwdt[:,1] = -np.sum(before_sum, axis=1)
        
        return dwdt

    def run(self, tmax=100, max_step=1/365.25, atol=1e-6, rtol=3e-3, method='RK45'):
        # other methods: RK45, RK23, LSODA, DOP853
        x0 = self._xlist.flatten()

        def my_g(t, y):
            return self._gravity(W=y.reshape(len(self.object_list), 2, 3)).flatten()

        r = solve_ivp(fun=my_g, t_span=(0, tmax), y0=x0, method=method,
                      max_step=max_step, atol=atol, rtol=rtol)
        
        # check if r success here! important
        if r.success:
            self._res = r
        else:
            raise RuntimeError(f'Nbody calculation failed with:\n{r.message}')

    def assign_values(self):
        n_obj = len(self.object_list)

        for i_obj, obj in enumerate(self.object_list):
            plist = []  # shape (t_step, 3d)
            tlist = self._res.t
            for i in range(len(tlist)):
                plist.append(self._res.y[:, i].reshape(n_obj, 2, 3)[i_obj, 0, :])
            plist = np.array(plist)

            # make interp function in each model
            obj.set_interp_func(t=tlist, x=plist[:, 0], 
                                y=plist[:, 1], z=plist[:, 2])