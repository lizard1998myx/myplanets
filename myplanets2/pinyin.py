# pinyin version of major

from .skymap import SkyMap
from .catalog import StarCatalog as Xingbiao

from .models import cal_radec as jisuan_radec
from .models import cal_period as jisuan_zhouqi
from .models import time_sequence as shijian_xulie

from .models import StaticModel as JingzhiMoxing
from .models import CircularModel
from .models import EllipticalModel
from .models import PointsModel as DianMoxing
from .models import PreciseModel as JingqueMoxing

from .nbody import NbodyModel as YinliMoxing
from .nbody import NbodyCal as YinliJisuan
from .nbody import get_model_real as zhenshi_moxing
from .nbody import get_models as moxing_xulie
from .nbody import center_models as xiuzheng_yuandian


class YuanMoxing(CircularModel):
    def __init__(self, banjing, zhouqi, xiangwei, time_initial=None):
        super().__init__(a=banjing, period=zhouqi, t0=xiangwei,
                         time_initial=time_initial)


class TuoyuanMoxing(EllipticalModel):
    def __init__(self, banchangzhou, zhouqi, xiangwei, pianxinlv,
                 shengjiaodian, jinridian, qingjiao, time_initial=None):
        super().__init__(a=banchangzhou, period=zhouqi, t0=xiangwei,
                         e=pianxinlv, ascending=shengjiaodian,
                         perihelion=jinridian, inclination=qingjiao,
                         time_initial=time_initial)


class Tiantu(SkyMap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def hua_xingzuo(self, *args, **kwargs):
        self.plot_constellation_easy(*args, **kwargs)

    def hua_xingxing(self, *args, **kwargs):
        self.plot_stars_easy(*args, **kwargs)

    def hua_moxing(self, *args, **kwargs):
        self.plot_model_easy(*args, **kwargs)

    def donghua_moxing(self, *args, **kwargs):
        self.animate_model_easy(*args, **kwargs)

    def donghua_shijian(self, *args, **kwargs):
        self.animate_text(*args, **kwargs)

    def hua_tuli(self, *args, **kwargs):
        self.legend(*args, **kwargs)

    def hua(self):
        self.show()

    def donghua(self, *args, **kwargs):
        self.animation_easy(*args, **kwargs)

