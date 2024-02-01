import os.path
import sys
import pandas as pd
import numpy as np
from .catalog_util import (get_BSCupdate_cat, get_RGB_list, get_pixel_size,
                           get_cons_ids, get_cons_names, filter_con_id,
                           combine_travel, move_cat_df)


class StarCatalog:
    def __init__(self, filename=None, teff_filename=None, cons_filename=None):
        pkg_dir = os.path.dirname(sys.modules[__package__].__file__)

        # T_eff configuration file
        if teff_filename is None:
            teff_filename = os.path.join(pkg_dir, 'resources', 'TCTable.csv')
        self.teff_filename = teff_filename

        # get catalogue dataframe & initialize
        # warning, df['RA'] & df['DE'] in radian, not degree
        if filename is None:
            filename = os.path.join(pkg_dir, 'resources', 'myBSCupdate1arcmin.csv')
        self.df = get_BSCupdate_cat(filename)
        # drop nan parallax rows
        self.df = pd.DataFrame(self.df.loc[~np.isnan(self.df['Parallax'])])
        self.df['RGB'] = get_RGB_list(bv_list=self.df['B-V'].values, teff_filename=self.teff_filename)
        self.df['pix_size'] = get_pixel_size(self.df['Vmag'].values)

        # constellation file
        if cons_filename is None:
            cons_filename = os.path.join(pkg_dir, 'resources', 'cons_lines.txt')
        # match with SAO, Name & Vmag
        self.con_list = get_cons_ids(filename=cons_filename, catalogue_df=self.df)
        self.con_name = get_cons_names(filename=cons_filename, full_name=False)

        # travel
        self.moved_df = None
        self.travel_time = []
        self.travel_ra = []
        self.travel_dec = []
        self.travel_ly = []

        # filter constellations
        self.fil_con_list = None

    # filter constellation, use brief name (e.g. UMi, Cep, Lyr), auto reset
    def filter_cons(self, include_list=None, exclude_list=None):
        self.fil_con_list = filter_con_id(full_con_id=self.con_list, con_names=self.con_name,
                                          include_list=include_list, exclude_list=exclude_list)

    # next move
    def move_to(self, time_year=0, ra=0, dec=0, travel_ly=0):
        self.travel_time.append(time_year)
        self.travel_ra.append(ra)
        self.travel_dec.append(dec)
        self.travel_ly.append(travel_ly)

        if self.moved_df is not None:  # more than one time
            combined = combine_travel(ra_deg_list=self.travel_ra, dec_deg_list=self.travel_dec,
                                      travel_ly_list=self.travel_ly)
            time_year = np.sum(self.travel_time)
            ra = combined['ra_deg']
            dec = combined['dec_deg']
            travel_ly = combined['travel_ly']
            # print(combined)

        moved_df = move_cat_df(self.df, time_year=time_year,
                               ra_deg=ra, dec_deg=dec, travel_ly=travel_ly,
                               default_distance=10, use_astropy=False,
                               add_sun=(travel_ly > 0), min_mag=-50)
        moved_df['RGB'] = get_RGB_list(bv_list=moved_df['B-V'].values, teff_filename=self.teff_filename)
        moved_df['pix_size'] = get_pixel_size(moved_df['Vmag'].values)
        self.moved_df = moved_df

    def move_reset(self):
        self.moved_df = None
        self.travel_time = []
        self.travel_ra = []
        self.travel_dec = []
        self.travel_ly = []

    def cons_reset(self):
        self.fil_con_list = None

    def get_stars(self):
        if self.moved_df is None:
            df = self.df
        else:
            df = self.moved_df

        return {'ra_deg': df['RAdeg'].values, 'dec_deg': df['DEdeg'].values,
                'size': df['pix_size'].values, 'color': df['RGB'].values}

    def get_cons(self):
        if self.fil_con_list is None:
            con_list = self.con_list
        else:
            con_list = self.fil_con_list

        if self.moved_df is None:
            df = self.df
        else:
            df = self.moved_df

        ra_line_list = []
        dec_line_list = []
        for con in con_list:
            for line in con:
                ra_line_list.append(df['RAdeg'].values[line])
                dec_line_list.append(df['DEdeg'].values[line])

        return {'lines_ra_deg': ra_line_list, 'lines_dec_deg': dec_line_list}

