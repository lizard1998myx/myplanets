# via 20230708 modular map.ipynb

import pandas as pd
import numpy as np
import re
import os

# star catalogue movement calculation
from astropy.coordinates import SkyCoord  # High-level coordinates
import astropy.units as u

# CAT_DIR = '/home/yuxi/catalog_data'
# CAT_DIR = r'C:\Users\Yuxi\OneDrive\代码\20230422 Nbody RK4\catalog_data'
CAT_DIR = ''


def get_BSC_cat(filename=os.path.join(CAT_DIR, 'catalog')):
    catalog_cols = [(1, 4, 'HR'), (5, 14, 'Name'), (15, 25, 'DM'), (26, 31, 'HD'), (32, 37, 'SAO'),
                    (38, 41, 'FK5'), (42, 42, 'IRflag'), (43, 43, 'r_IRflag'), (44, 44, 'Multiple'),
                    (45, 49, 'ADS'), (50, 51, 'ADScomp'), (52, 60, 'VarID'), 
                    (61, 62, 'RAh1900'), (63, 64, 'RAm1900'), (65, 68, 'RAs1900'), 
                    (69, 69, 'DE-1900'), (70, 71, 'DEd1900'), (72, 73, 'DEm1900'), (74, 75, 'DEs1900'),
                    (76, 77, 'RAh'), (78, 79, 'RAm'), (80, 83, 'RAs'),  # FK5 J2000
                    (84, 84, 'DE-'), (85, 86, 'DEd'), (87, 88, 'DEm'), (89, 90, 'DEs'),
                    (91, 96, 'GLON'), (97, 102, 'GLAT'),  # Galactic in deg
                    (103, 107, 'Vmag'), (108, 108, 'n_Vmag'), (109, 109, 'u_Vmag'),
                    (110, 114, 'B-V'), (115, 115, 'u_B-V'), (116, 120, 'U-B'), (121, 121, 'u_U-B'),
                    (122, 126, 'R-I'), (127, 127, 'n_R-I'), (128, 147, 'SpType'), (148, 148, 'n_SpType'),
                    (149, 154, 'pmRA'), (155, 160, 'pmDE'),  # arcsec/yr
                    (161, 161, 'n_Parallax'), (162, 166, 'Parallax'),  # arcsec
                    (167, 170, 'RadVel'), (171, 174, 'n_RadVel'),
                    (175, 176, 'l_RotVel'), (177, 179, 'RotVel'), (180, 180, 'u_RotVel'),
                    (181, 184, 'Dmag'), (185, 190, 'Sep'), (191, 194, 'MultID'), (195, 196, 'MultCnt'),
                    (197, 197, 'NoteFlag')]

    catalogue_raw = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            star = {}
            for col in catalog_cols:
                star[col[2]] = line[col[0] - 1: col[1]].strip()
            catalogue_raw.append(star)

    catalogue_raw = pd.DataFrame(catalogue_raw)

    # test and exclude

    def get_col(key, idx, use_nan=False, catalogue_raw=catalogue_raw):
        rlist = []
        for r in catalogue_raw[key].values[idx]:
            try:
                rlist.append(float(r))
            except ValueError:
                if use_nan:
                    rlist.append(np.nan)
                else:
                    pass
        return np.array(rlist)

    idx = ~np.isnan(get_col('Vmag', idx=np.ones(len(catalogue_raw), dtype=bool), use_nan=True))

    ra_list_rad = []
    de_list_rad = []

    for i in range(len(catalogue_raw)):
        try:
            ra = int(catalogue_raw['RAh'][i]) + int(catalogue_raw['RAm'][i]) / 60 + float(catalogue_raw['RAs'][i]) / 3600
            ra *= 15  # to deg
            ra *= np.pi / 180  # to rad
            dec = int(catalogue_raw['DEd'][i]) + int(catalogue_raw['DEm'][i]) / 60 + float(catalogue_raw['DEs'][i]) / 3600
            if catalogue_raw['DE-'][i] == '-':
                dec *= -1
            dec *= np.pi / 180  # to rad
        except ValueError:
            ra = np.nan
            dec = np.nan
        ra_list_rad.append(ra)
        de_list_rad.append(dec)

    # name, SAO, HD, HR
    cat_df = pd.DataFrame({
        'Name': catalogue_raw['Name'].values[idx],
        'HR': catalogue_raw['HR'].values[idx],
        'HD': catalogue_raw['HD'].values[idx],
        'SAO': catalogue_raw['SAO'].values[idx],
        'RA': np.array(ra_list_rad)[idx],  # radian
        'DE': np.array(de_list_rad)[idx],  # radian
        'RAdeg': np.array(ra_list_rad)[idx] * 180/np.pi,  # degree
        'DEdeg': np.array(de_list_rad)[idx] * 180/np.pi,  # degree
        'Vmag': get_col('Vmag', idx=idx, use_nan=True),
        'B-V': get_col('B-V', idx=idx, use_nan=True),
        'pmRA': get_col('pmRA', idx=idx, use_nan=True),  # arcsec/yr
        'pmDE': get_col('pmDE', idx=idx, use_nan=True),  # arcsec/yr
        'Parallax': get_col('Parallax', idx=idx, use_nan=True),  # arcsec
        'RadVel': get_col('RadVel', idx=idx, use_nan=True),  # km/s
    })

    return cat_df

def get_HIP_cat(filename=os.path.join(CAT_DIR, 'hip_main.dat')):
    catalog_cols = ['Catalog', 'HIP', 'Proxy', 'RAhms', 'DEdms',
                    'Vmag', 'VarFlag', 'r_Vmag', 'RAdeg', 'DEdeg',  # J1991.25
                    'AstroRef', 'Plx', 'pmRA', 'pmDE', # Plx in mas
                    'e_RAdeg', 'e_DEdeg', 'e_Plx', 'e_pmRA', 'e_pmDE', 
                    'DE:RA', 'Plx:RA', 'Plx:DE', 'pmRA:RA', 'pmRA:DE', 
                    'pmRA:Plx', 'pmDE:RA', 'pmDE:DE', 'pmDE:Plx', 'pmDE:pmRA',  # coefficients
                    'F1', 'F2', '---', # repeat HIP number
                    'BTmag', 'e_BTmag', 'VTmag', 'e_VTmag', 'm_BTmag', 
                    'B-V', 'e_B-V', 'r_B-V', 'V-I', 'e_V-I', 'r_V-I', 
                    'CombMag', 'Hpmag', 'e_Hpmag', 'Hpscat', 'o_Hpmag', 'm_Hpmag', 
                    'Hpmax', 'HPmin', 'Period', 'HvarType', 'moreVar', 'morePhoto', 
                    'CCDM', 'n_CCDM', 'Nsys', 'Ncomp', 'MultFlag', 'Source', 'Qual', 
                    'm_HIP', 'theta', 'rho', 'e_rho', 'dHp', 'e_dHp', 'Survey', 'Chart', 
                    'Notes', 'HD', 'BD', 'CoD', 'CPD', '(V-I)red', 'SpType', 'r_SpType']
    
    catalogue_raw = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            star = {}
            line_split = line.strip().split('|')
            if len(line_split) != len(catalog_cols):
                print(line)
                continue
            for i, col in enumerate(catalog_cols):
                star[col] = line_split[i].strip()
            catalogue_raw.append(star)

    catalogue_raw = pd.DataFrame(catalogue_raw)

    # test and exclude
    def get_col(key, idx, use_nan=False, catalogue_raw=catalogue_raw):
        rlist = []
        for r in catalogue_raw[key].values[idx]:
            try:
                rlist.append(float(r))
            except ValueError:
                if use_nan:
                    rlist.append(np.nan)
                else:
                    pass
        return np.array(rlist)

    idx = ~np.isnan(get_col('Vmag', idx=np.ones(len(catalogue_raw), dtype=bool), use_nan=True))

    # name, SAO, HD, HR
    cat_df = pd.DataFrame({
        'HIP': catalogue_raw['HIP'].values[idx],
        'RA': get_col('RAdeg', idx=idx, use_nan=True) * np.pi/180,  # radian
        'DE': get_col('RAdeg', idx=idx, use_nan=True) * np.pi/180,  # radian
        'RAdeg': get_col('RAdeg', idx=idx, use_nan=True),  # degree
        'DEdeg': get_col('DEdeg', idx=idx, use_nan=True),  # degree
        'Vmag': get_col('Vmag', idx=idx, use_nan=True),
        'B-V': get_col('B-V', idx=idx, use_nan=True),
        'pmRA': get_col('pmRA', idx=idx, use_nan=True)/1000,  # mas/yr -> as/yr
        'pmDE': get_col('pmDE', idx=idx, use_nan=True)/1000,  # mas/yr -> as/yr
        'Parallax': get_col('Plx', idx=idx, use_nan=True)/1000,  # mas -as
    })

    return cat_df

def get_Merge_cat(filename=os.path.join(CAT_DIR, 'myHIP+BSC.csv')):
    cat_df = pd.read_csv(filename)

    def get_number(key):
        ar = cat_df[key].values
        ar[np.isnan(ar)] = -1
        ar = ar.astype(int)
        return ar
    
    name_list = []
    for name in cat_df['Name']:
        if isinstance(name, str):
            name_list.append(name)
        elif np.isnan(name):
            name_list.append('')
        else:
            print('error')
            print(type(name))
            print(name)

    results = {'Name': name_list}
    for key in ['HR', 'HD', 'SAO', 'HIP']:
        results[key] = get_number(key)

    for key in ['RA', 'DE', 'RAdeg', 'DEdeg', 'Vmag',
                'B-V', 'pmRA', 'pmDE', 'Parallax']:
        results[key] = cat_df[f'{key}_2'].values

def get_BSCupdate_cat(filename=os.path.join(CAT_DIR, 'myBSCupdate.csv')):
    cat_df = pd.read_csv(filename)

    def get_number(key):
        ar = cat_df[key].values
        ar[np.isnan(ar)] = -1
        ar = ar.astype(int)
        return ar

    name_list = []
    for name in cat_df['Name']:
        if isinstance(name, str):
            name_list.append(name)
        elif np.isnan(name):
            name_list.append('')
        else:
            print('error')
            print(type(name))
            print(name)

    def get_2or1(key):
        ar1 = cat_df[f'{key}_1'].values
        ar2 = cat_df[f'{key}_2'].values
        idx = ~np.isnan(ar2)
        ar1[idx] = ar2[idx]
        return ar1

    results = {'Name': name_list}
    for key in ['HR', 'HD', 'SAO', 'HIP']:
        results[key] = get_number(key)

    for key in ['RA', 'DE', 'RAdeg', 'DEdeg',
                'Vmag', 'B-V']:
        results[key] = cat_df[f'{key}_1'].values

    for key in ['pmRA', 'pmDE', 'Parallax']:
        results[key] = get_2or1(key)

    # radial velocity
    rad_vel = cat_df['RadVel'].values
    rad_vel[np.isnan(rad_vel)] = 0  # take zero
    results['RadVel'] = rad_vel

    return pd.DataFrame(results)

def drop_parrallax(cat_df):
    new_df = pd.DataFrame(cat_df.loc[~np.isnan(cat_df['Parallax'])])
    print(f'{len(new_df)} out of {len(cat_df)} found')
    return new_df

def get_RGB_list(bv_list, default_teff=5700, teff_filename=os.path.join(CAT_DIR, 'TCTable.csv')):
    df_rgb = pd.read_csv(teff_filename)
    tlist = 4600 * ((1/(0.92*bv_list + 1.7)) + (1/(0.92*bv_list + 0.62)))
    tlist[np.isnan(tlist)] = default_teff  # white color
    tlist[tlist < min(df_rgb['T'])] = min(df_rgb['T'])
    tlist[tlist > max(df_rgb['T'])] = max(df_rgb['T'])
    rgb_list = []
    for t in tlist:
        i = np.max(np.where(df_rgb['T'].values - t <= 0))
        rgb_list.append([df_rgb[b].values[i] for b in 'RGB'])
    return rgb_list

def get_pixel_size(mag_list):
    radius = (5 - mag_list) * (0.7 / 5)  # mag 0 -> 1 deg, mag 5 -> 0.3 deg
    radius[radius > 1] = 1  # max 1.0
    radius[mag_list > 5] = 0.1  # other 0.1
    return radius

# obtain constellation data
def get_cons_ids(filename=os.path.join(CAT_DIR, 'cons_lines.txt'),
                 catalogue_df=get_BSC_cat(filename=os.path.join(CAT_DIR, 'catalog'))):

    greek_alphabet = ['alpha', 'beta', 'gamma', 'delta',
                    'epsilon', 'zeta', 'eta', 'theta',
                    'iota', 'kappa', 'lambda', 'mu',
                    'nu', 'xi', 'omicron', 'pi', 'rho',
                    'sigma', 'tau', 'upsilon', 'phi',
                    'chi', 'psi', 'omega']

    def get_star(s_raw, constellation):
        s_raw = s_raw.strip().lower()
        star = {'raw': s_raw, 'constellation': constellation}

        if s_raw[:4] == 'sao ':
            star['SAO'] = s_raw[4:]
            return star

        match = re.search(r'(.*) \((.*)\)', s_raw)

        if match is None:
            name_raw = s_raw
        else:
            name_raw = match[1]
            assert match[2][:4] == 'sao ', f'no SAO: {star}'
            star['SAO'] = match[2][4:]

        for letter in greek_alphabet:
            if letter == name_raw[:len(letter)]:
                formal_name = letter[:3].title().ljust(3)  # 'mu' -> 'Mu '
                star['name'] = formal_name + name_raw[len(letter):]
                break

        assert 'name' in star.keys() or 'SAO' in star.keys(), f'no matched: {star}'
        return star

    # get a list of stars
    with open(filename, 'r', encoding='utf-8') as f:
        list_of_lines = []

        constellation = None
        for s_line in f.readlines():
            if s_line == '\n':
                constellation = None
                list_of_lines.append([])  # time to plot
            elif constellation is None:
                assert len(re.split(r'-|–', s_line)) == 2
                constellation = s_line[:3]
            elif s_line[0] == '(':  # in constellation, inter constellation lines
                s_list = re.split(r'-|–', s_line.strip('()\n'))
                star_list = []
                for s_raw in s_list:
                    s_raw = s_raw.strip()
                    star_list.append(get_star(s_raw=s_raw[:-3],
                                            constellation=s_raw[-3:]))
                list_of_lines.append(star_list)
            else:  # in constellation, lines inside
                s_list = re.split(r'-|–', s_line)
                star_list = []
                for s_raw in s_list:
                    star_list.append(get_star(s_raw=s_raw,
                                              constellation=constellation))
                list_of_lines.append(star_list)

        # list_of_lines.append([])  # time to plot

    # match star with catalogue
    def match_id(star):
        matched_id = None
        if 'SAO' in star.keys():
            for i, sid in enumerate(catalogue_df['SAO'].values):
                # check sid in cat_df                
                if isinstance(sid, str):
                    pass
                elif isinstance(sid, (int, np.integer)):
                    sid = str(sid)
                else:
                    raise TypeError(f'SAO in cat_df should not be {type(sid)}')

                if star['SAO'] == sid:
                    if matched_id is not None:
                        v0 = catalogue_df['Vmag'].values[matched_id]
                        v1 = catalogue_df['Vmag'].values[i]
                        # print(f'warning: multiple SAO {matched_id}/{i}: {star}')
                        if v1 < v0:
                            matched_id = i  # replace
                            # print(f'use {matched_id} instead ({v1:.2f}<{v0:.2f})')
                    else:
                        matched_id = i
        elif 'name' in star.keys():  # SAO all matched, skip double searching
            for i, name in enumerate(catalogue_df['Name'].values):
                if name[-3:] == star['constellation']:
                    if star['name'] in name:
                        if matched_id is not None:
                            assert matched_id == i, f'conflict in name {matched_id}/{i}: {star}'
                        else:
                            matched_id = i
        if matched_id is None:
            print(f'star not found:\n{star}')
        return matched_id

    list_of_lines_id = []

    for line in list_of_lines:
        line_id = []
        for star in line:
            line_id.append(match_id(star))
        list_of_lines_id.append(line_id)

    # separate into list of lines
    # con, list, id (3D array)
    list_of_lines_id_by_cons = []

    list_of_lines_id_this_con = []
    for line_id in list_of_lines_id:
        if line_id == []:
            list_of_lines_id_by_cons.append(list_of_lines_id_this_con)
            list_of_lines_id_this_con = []
        else:
            list_of_lines_id_this_con.append(line_id)

    if len(list_of_lines_id_this_con) > 0:
        list_of_lines_id_by_cons.append(list_of_lines_id_this_con)

    return list_of_lines_id_by_cons

def get_cons_names(filename=os.path.join(CAT_DIR, 'cons_lines.txt'), full_name=False):
    list_of_names = []
    with open(filename, 'r', encoding='utf-8') as f:
        constellation = None
        for s_line in f.readlines():
            if s_line == '\n':
                constellation = None  # refresh
                continue
            elif constellation is None:
                assert len(re.split(r'-|–', s_line)) == 2
                if full_name:
                    constellation = re.split(r'-|–', s_line)[1].strip()
                else:
                    constellation = s_line[:3]
                list_of_names.append(constellation)
            else:
                continue
    return list_of_names

def filter_con_id(full_con_id, con_names, include_list=None, exclude_list=None):
    assert len(full_con_id) == 88, 'Full constellation needed'
    # default value
    if include_list is None:
        include_list = con_names
    if exclude_list is None:
        exclude_list = []

    results = []
    for n, con in zip(con_names, full_con_id):
        if n in include_list and n not in exclude_list:
            results.append(con)

    return results


# ra, dec, distance (rad, rad, parsec) to XYZ
def get_cartesian(ra, dec, distance):
    return [distance*np.cos(ra)*np.cos(dec),
            distance*np.sin(ra)*np.cos(dec),
            distance*np.sin(dec)]

# XYZ to ra, dec, distance (deg, deg, parsec)
def get_spherical_deg(x, y, z):
    ra = np.zeros(len(x))
    ra[x!=0] = np.arctan(y[x!=0]/x[x!=0]) * 180 / np.pi
    ra[x<0] += 180
    ra[(x==0)&(y>0)] = 90
    ra[(x==0)&(y<0)] = 270
    ra[(x==0)&(y==0)] = 0
    ra = ra % 360
    dec = np.arctan(z/(x**2 + y**2)**0.5) * 180 / np.pi
    distance = np.linalg.norm(np.array([x, y, z]), axis=0)
    return [ra, dec, distance]

# combine multiple movements
def combine_travel(ra_deg_list, dec_deg_list, travel_ly_list):
    xyz_list = get_cartesian(np.array(ra_deg_list)*np.pi/180,
                             np.array(dec_deg_list)*np.pi/180,
                             np.array(travel_ly_list))
    x, y, z = np.sum(np.array(xyz_list), axis=0)
    ra, dec, distance = get_spherical_deg(x, y, z)
    return {'ra_deg': ra, 'dec_deg': dec, 'travel_ly': distance}

def move_cat_df(cat_df, time_year=0, ra_deg=0, dec_deg=0, travel_ly=0, 
                default_distance=10, use_astropy=False, add_sun=True):
    distance = 1/cat_df['Parallax'].values
    distance[np.isinf(distance)] = default_distance  # parsec
    distance[np.isnan(distance)] = default_distance
    distance[distance <= 0] = default_distance

    if use_astropy:  # astropy for verification, problematic under specific circumstances
        c = SkyCoord(ra=cat_df['RA'].values*u.rad, 
                     dec=cat_df['DE'].values*u.rad,
                     distance=distance*u.parsec,
                     pm_ra_cosdec=cat_df['pmRA'].values*u.arcsec/u.yr,
                     pm_dec=cat_df['pmDE'].values*u.arcsec/u.yr,
                     radial_velocity=cat_df['RadVel'].values*u.km/u.s
                     )
        
        if time_year != 0:
            c = c.apply_space_motion(dt=time_year*u.yr)
            
        x0, y0, z0 = get_cartesian(ra=c.ra.radian, 
                                   dec=c.dec.radian,
                                   distance=c.distance.parsec)
        
    else:  # own coordinate conversion, recommended
        ra0 = cat_df['RA'].values
        de0 = cat_df['DE'].values

        x0, y0, z0 = get_cartesian(ra=ra0, 
                                   dec=de0,
                                   distance=distance)
        
        if time_year != 0:  # time travel
            v_rad = cat_df['RadVel'].values / ((3.086e13)/(365.25*24*3600))  # km/s -> pc/yr
            v_ra = cat_df['pmRA'].values*distance * np.pi/(180*3600)  # pc*as/yr -> pc/yr
            v_de = cat_df['pmDE'].values*distance * np.pi/(180*3600)  # pc*as/yr -> pc/yr

            # rotation on RA axis, fix DE
            v_z = v_de*np.cos(de0) + v_rad*np.sin(de0)
            v_xi = -v_de*np.sin(de0) + v_rad*np.cos(de0)
            # rotation on Spin axis, fix RA
            v_x = v_xi*np.cos(ra0) - v_ra*np.sin(ra0)
            v_y = v_xi*np.sin(ra0) + v_ra*np.cos(ra0)

            x0 += v_x*time_year
            y0 += v_y*time_year
            z0 += v_z*time_year
            
    x1, y1, z1 = get_cartesian(ra=ra_deg*np.pi/180, 
                               dec=dec_deg*np.pi/180,
                               distance=travel_ly/3.261563777)

    ra, dec, distance_new = get_spherical_deg(x=x0-x1, y=y0-y1, z=z0-z1)
    mag_new = cat_df['Vmag'] + 5*np.log10(distance_new/distance)
    c_new = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

    results = {}
    results['RA'] = c_new.ra.radian
    results['DE'] = c_new.dec.radian
    results['RAdeg'] = c_new.ra.degree
    results['DEdeg'] = c_new.dec.degree
    results['Vmag'] = mag_new
    results['B-V'] = cat_df['B-V'].values
    # new pmRA, pmDE, RadVel is calculatable theoratically
    # but not included in this version
    
    result_df = pd.DataFrame(results)
    
    # add sun
    if add_sun:
        ra_deg_sun = (ra_deg + 180) % 360
        dec_deg_sun = -1 * dec_deg
        sun = {'RA': ra_deg_sun*np.pi/180,
               'DE': dec_deg_sun*np.pi/180,
               'RAdeg': ra_deg_sun, 'DEdeg': dec_deg_sun,
               'Vmag': 4.8 + 5*np.log10(travel_ly/(3.261563777*10)),
               'B-V': 0.656}

        result_df.append(sun, ignore_index=True)

    return result_df