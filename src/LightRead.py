# -*- coding: utf-8 -*-
"""
MOSiP read input source code.
developped by Niels Fuchs (Universität Hamburg, 2021)
contact: niels.fuchs@uni-hamburg.de
"""

import numpy as np
import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.tri as tri
from matplotlib import cm
from scipy import interpolate
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import xarray as xr
from pandas.tseries.offsets import DateOffset
from io import StringIO

profile_short_names = ['T_deg', 'VIS_in_R', 'VIS_in_G', 'VIS_in_B', 'VIS_in_C',
                       'NIR_in_R', 'NIR_in_G', 'NIR_in_B', 'NIR_in_C',
                       'VIS_out_R', 'VIS_out_G', 'VIS_out_B', 'VIS_out_C',
                       'NIR_out_R', 'NIR_out_G', 'NIR_out_B', 'NIR_out_C']


def bytespdate2num(b):
    return mdates.date2num(dt.datetime.strptime(b, '%y%m%d%H%M%S'))


def bytespdate2date(b):
    return dt.datetime.strptime(b, '%y%m%d%H%M%S')


def date2num(date):
    return mdates.date2num(dt.datetime.strptime(date, '%Y%m%d%H%M%S'))


def bytespdate2num_dship(b):
    return mdates.date2num(dt.datetime.strptime(b, '%Y/%m/%d %H:%M:%S'))


def bytespdate2num_buoy(b):
    return mdates.date2num(dt.datetime.strptime(b, '%Y-%m-%dT%H:%M:%S'))


def bytespdate2num_lightchain(b):
    return mdates.date2num(dt.datetime.strptime(b, '%Y-%m-%d %H:%M:%S'))



class lightharp:

    def __init__(self, station_dict):
        self.station_dict = station_dict

    def read_lightharp(self):
        """ import lightharp data, correct gain, MOSAiC configuration """

        # configuration part:
        file_path = self.station_dict['filepath']
        gain_dict = {3: 60., 2: 16., 1: 4., 0: 1.}  # Amplification factors lightharp 3->60x, , 2- >16x,, 1->4x, 0->1x)

        column_names = ['datenum', 'T_deg', 'VIS_in_C', 'VIS_in_R', 'VIS_in_G', 'VIS_in_B', 'VIS_in_gain',
                        'NIR_in_C', 'NIR_in_R', 'NIR_in_G', 'NIR_in_B', 'NIR_in_gain',
                        'VIS_out_C', 'VIS_out_R', 'VIS_out_G', 'VIS_out_B', 'VIS_out_gain',
                        'NIR_out_C', 'NIR_out_R', 'NIR_out_G', 'NIR_out_B', 'NIR_out_gain']

        z_axis = (np.array([2., 3.5, 6.5, 12., 23., 48.5, 97., 142.]) / 100.) + float(
            self.station_dict['vertical_offset'])  # initial vertical position [m]

        # initialization part:

        module_dict = {}

        # read data

        for m1 in [0, 2, 4, 6]:
            m2 = m1 + 1
            for n, m in enumerate([m1, m2]):
                columns = [0] + [1 + n * 21 + c for c in range(21)]
                module_dict[m] = pd.read_csv(file_path + '20200115_20200813_Light_M' + str(m1) + str(m2) + '.txt',
                                             delimiter=' ', names=column_names, usecols=columns,
                                             converters={0: bytespdate2num})

        data_list = set(profile_short_names).intersection(
            column_names)  # returns all column names which are included in profile_short_names (https://www.geeksforgeeks.org/python-intersection-two-lists/)

        # apply gain correction and write to pandas dataframe

        for module in range(8):
            for variable in data_list:
                if variable != 'T_deg':
                    # ['VIS_in_gain', 'NIR_in_gain', 'VIS_out_gain', 'NIR_out_gain']
                    gain_column = variable.rsplit('_', 1)[0] + '_gain'
                    col_variable = module_dict[module].columns.get_loc(variable)
                    col_gain = module_dict[module].columns.get_loc(gain_column)
                    module_dict[module].iloc[:, col_variable] = \
                        module_dict[module].iloc[:, col_variable] / np.array(list(map(gain_dict.get,
                                                                                      module_dict[module].iloc[:,
                                                                                      col_gain].tolist())))
            if 'df' not in locals():
                df = pd.DataFrame({'datenum': module_dict[module].loc[:, 'datenum'],
                                   'depth': np.ones(len(module_dict[module].loc[:, 'datenum'])) *
                                            z_axis[module],
                                   **{variable: module_dict[module].loc[:, variable] for variable in data_list}})
            elif 'df' in locals():
                df = df.append(pd.DataFrame({'datenum': module_dict[module].loc[:, 'datenum'],
                                             'depth': np.ones(len(module_dict[module].loc[:, 'datenum'])) *
                                                      z_axis[module],
                                             **{variable: module_dict[module].loc[:, variable] for variable in
                                                data_list}}),
                               ignore_index=True)

        self.data = df

    def crop_timeline(self):
        """ crop data to all dates between start and end date """
        if 'start_date' in self.station_dict.keys():
            self.data.drop(np.where(self.data['datenum'] < date2num(self.station_dict['start_date']))[0], 0,
                           inplace=True)
            self.data.reset_index(inplace=True, drop=True)
        if 'end_date' in self.station_dict.keys():
            self.data.drop(np.where(self.data['datenum'] > date2num(self.station_dict['end_date']))[0], 0, inplace=True)
            self.data.reset_index(inplace=True, drop=True)

    def add_nominal_date(self):
        """
        Some sensors record time stamps for each individual depth measurement.
        Since the timedelta between profiles is >> than the timedelta between individual depths,
        the date of the uppermost available sensor is used as profile date
        """
        timeaxis_sort_index = np.argsort(self.data['datenum'])
        timeaxis_orig = np.array(self.data['datenum'])[timeaxis_sort_index]
        dt = timeaxis_orig[1:] - timeaxis_orig[:-1]
        dt_index = np.concatenate([[True], dt > 0.1])
        dt_cumsum = np.cumsum(dt_index) - 1
        timeaxis_out = timeaxis_orig[dt_index][dt_cumsum]
        self.data['nominal_date'] = np.empty_like(timeaxis_orig)
        self.data['nominal_date'][timeaxis_sort_index] = timeaxis_out


class SAMSIM:

    def __init__(self, station_dict):
        self.station_dict = station_dict

    def read_SAMSIM(self):
        """ import model results """

        # configuration part:
        file_path = self.station_dict['filepath']

        S = np.loadtxt(file_path + "dat_S_bu.dat")
        T = np.loadtxt(file_path + "dat_T.dat")
        psi_l = np.loadtxt(file_path + "dat_psi_l.dat")
        psi_s = np.loadtxt(file_path + "dat_psi_s.dat")
        psi_g = np.loadtxt(file_path + "dat_psi_g.dat")
        thick = np.loadtxt(file_path + "dat_thick.dat")
        snow = np.loadtxt(file_path + "dat_snow.dat")
        freeboard = np.loadtxt(file_path + "dat_freeboard.dat")

        # extract timespan between output from settings file
        with open(file_path + "dat_settings.dat", 'r') as configfile:
            while True:
                line = configfile.readline()
                if line.split(' ')[0] == 'time_out':
                    break
            dt_out = float(line.rsplit(' ', 1)[1])  # in seconds

        # generate date vector

        nominal_date = pd.date_range(start=bytespdate2date(self.station_dict['start_date']), periods=S.shape[0],
                                     freq=DateOffset(seconds=dt_out))

        # combine snow and ice layers

        xlen = len(thick[:, 0])

        T_snow = snow[:, 1]
        T_snow = T_snow.reshape(xlen, 1)
        psi_l_snow = snow[:, 2]
        psi_l_snow = psi_l_snow.reshape(xlen, 1)
        thick_snow = snow[:, 0]
        thick_snow = thick_snow.reshape(xlen, 1)
        S_snow = T_snow * 0.0

        thick = np.hstack((thick_snow, thick))
        T = np.hstack((T_snow, T))
        psi_l = np.hstack((psi_l_snow, psi_l))
        psi_s = np.hstack((1. - psi_l_snow, psi_s))
        psi_g = np.hstack((psi_l_snow - psi_l_snow, psi_g))
        S = np.hstack((S_snow, S))

        # map data on depth grid (from Philipps Plot script)

        ylen = len(thick[0, :])
        xlen = len(thick[:, 0])

        depth = thick * 1.

        i = 0
        j = 0
        while (i < xlen):
            while (j < ylen):
                depth[i, j] = sum(thick[i, 0:j]) + thick[i, j] / 2. - freeboard[i] - thick_snow[i]
                j = j + 1
            i = i + 1
            j = 0

        depth_axis = np.arange(depth.min(), depth.max() + 0.1, 0.01)

        S_mapped = np.ones((depth_axis.shape[0], nominal_date.shape[0]))*np.nan
        T_mapped = np.ones((depth_axis.shape[0], nominal_date.shape[0]))*np.nan
        psi_l_mapped = np.ones((depth_axis.shape[0], nominal_date.shape[0]))*np.nan
        psi_s_mapped = np.ones((depth_axis.shape[0], nominal_date.shape[0]))*np.nan
        psi_g_mapped = np.ones((depth_axis.shape[0], nominal_date.shape[0]))*np.nan

        for t in range(nominal_date.shape[0]):
            S_mapped[:, t] = interpolate.interp1d(depth[t, :], S[t, :], bounds_error=False, fill_value=np.nan)(depth_axis)
            T_mapped[:, t] = interpolate.interp1d(depth[t, :], T[t, :], bounds_error=False, fill_value=np.nan)(depth_axis)
            psi_l_mapped[:, t] = interpolate.interp1d(depth[t, :], psi_l[t, :], bounds_error=False, fill_value=np.nan)(depth_axis)
            psi_s_mapped[:, t] = interpolate.interp1d(depth[t, :], psi_s[t, :], bounds_error=False, fill_value=np.nan)(depth_axis)
            psi_g_mapped[:, t] = interpolate.interp1d(depth[t, :], psi_g[t, :], bounds_error=False, fill_value=np.nan)(depth_axis)

        da_list = []
        for (data, name) in [(S_mapped, 'S_bulk'), (T_mapped, 'T_deg'), (psi_s_mapped, 'phi_s_v'), (psi_g_mapped, 'phi_g_v')]:
            da_list.append(xr.DataArray(data=data.T,
                                        name=name,
                                        dims=['nominal_date', 'depth'],
                                        coords=dict(depth=depth_axis,
                                                    nominal_date=nominal_date,
                                                    freeboard=('nominal_date', freeboard),
                                                    snow_thickness=('nominal_date', thick_snow[:,0]))))

        # da_list.append(xr.DataArray(data=thick_snow[:,0],
        #                                 name='snow_thickness',
        #                                 dims=['nominal_date'],
        #                                 coords=dict(nominal_date=nominal_date)))

        ds = xr.merge(da_list)
        ds['sensor_id'] = xr.DataArray(self.station_dict['id'], dims=(), attrs={'cf_role': 'trajectory_id'})

        self.ds = ds

class SIMBA:
    def __init__(self, station_dict):
        self.station_dict = station_dict

    def get_start_line(self, file_path):
        f = open(file_path, "r")
        data = f.read()
        for i, line in enumerate(data.splitlines()):
            try:
                d=bytespdate2num_buoy(line.split('\t', 1)[0])
                return i
            except:
                continue

    def read_buoy(self, file_type, **kwargs):

        if file_type == 'SiMBA_temp':
            file_path = self.station_dict['filepath'] + \
                        self.station_dict['id'] + '/datasets/' + self.station_dict['id'] + \
                        '_temp.dat'
            column_names = ['nominal_date', 'lat', 'lon'] + \
                           [str(np.around(float(i) * 0.02 + float(self.station_dict['vertical_offset']), decimals=2))
                            for i in range(1, 242)]
            start_line = self.get_start_line(file_path)
            df = pd.read_csv(
                file_path,
                delimiter='\t',
                names=column_names,
                usecols=[0, 1, 2] + list(range(4, 245)),
                converters={0: bytespdate2num_buoy},
                skiprows=start_line,
                comment='#')    # on bad lines: most simple but possible erroneous way to skip header
            df = df.melt(id_vars=["nominal_date", "lat", "lon"], var_name="depth", value_name="T_deg")
            df['depth'] = df['depth'].astype('float64')
            self.data = df

        elif file_type == 'SiMBA_heat030':
            file_path = self.station_dict['filepath'] + \
                        self.station_dict['id'] + '/datasets/' + self.station_dict['id'] + \
                        '_temp_diff_030.dat'
            column_names = ['nominal_date', 'lat', 'lon'] + \
                           [str(np.around(float(i) * 0.02 + float(self.station_dict['vertical_offset']), decimals=2))
                            for i in range(1, 242)]
            start_line = self.get_start_line(file_path)
            df = pd.read_csv(
                file_path,
                delimiter='\t',
                names=column_names,
                usecols=[0, 1, 2] + list(range(4, 245)),
                converters={0: bytespdate2num_buoy},
                skiprows=start_line,
                comment='#')
            df = df.melt(id_vars=["nominal_date", "lat", "lon"], var_name="depth", value_name="H030")
            df['depth'] = df['depth'].astype('float64')
            self.data = df

        elif file_type == 'SiMBA_heat120':
            file_path = self.station_dict['filepath'] + \
                        self.station_dict['id'] + '/datasets/' + self.station_dict['id'] + \
                        '_temp_diff_120.dat'
            column_names = ['nominal_date', 'lat', 'lon'] + \
                           [str(np.around(float(i) * 0.02 + float(self.station_dict['vertical_offset']), decimals=2))
                            for i in range(1, 242)]
            start_line = self.get_start_line(file_path)
            df = pd.read_csv(
                file_path,
                delimiter='\t',
                names=column_names,
                usecols=[0, 1, 2] + list(range(4, 245)),
                converters={0: bytespdate2num_buoy},
                skiprows=start_line,
                comment='#')
            df = df.melt(id_vars=["nominal_date", "lat", "lon"], var_name="depth", value_name="H120")
            df['depth'] = df['depth'].astype('float64')
            self.data = df



    def add_snow_ice_interfaces(self, filepath):
        df = pd.read_excel(filepath,
                           sheet_name='Tem & Th', usecols=[0, 1, 2], skiprows=[1], parse_dates=[0],
                           names=['date', 'snow_thickness', 'ice_thickness']).dropna()
        df.reset_index(inplace=True)

        df['surface_melt'] = df.loc[df['snow_thickness'] < 0, 'snow_thickness']
        df.loc[df['snow_thickness'] < 0, 'snow_thickness'] = np.nan

        snow_thick = interpolate.interp1d(mdates.date2num(df['date'].astype("M8[ms]").tolist()),
                                          df['snow_thickness'] / 100., bounds_error=False)
        surface_melt = interpolate.interp1d(mdates.date2num(df['date'].astype("M8[ms]").tolist()),
                                            df['surface_melt'] / 100., bounds_error=False)
        ice_thick = interpolate.interp1d(mdates.date2num(df['date'].astype("M8[ms]").tolist()),
                                         df['ice_thickness'] / 100., bounds_error=False)

        self.data['snow_thickness'] = snow_thick(self.data['nominal_date'])
        self.data['ice_thickness'] = ice_thick(self.data['nominal_date'])
        self.data['surface_melt'] = surface_melt(self.data['nominal_date'])

    def outliers2nan(self):

        """ replace previously defined outliers by np.nan """

        variables = self.station_dict['variables']

        for var in variables.keys():
            if var in list(self.data.columns):
                for condition in variables[var].keys():
                    if condition == 'upper_bound':
                        self.data.loc[self.data[var] > float(variables[var][condition]), var] = np.nan
                    if condition == 'lower_bound':
                        self.data.loc[self.data[var] < float(variables[var][condition]), var] = np.nan
                    if condition == 'nan':
                        self.data.loc[self.data[var] == float(variables[var][condition]), var] = np.nan

    def crop_timeline(self):
        """ crop data to all dates between start and end date """
        if 'start_date' in self.station_dict.keys():
            self.data.drop(np.where(self.data['datenum'] < date2num(self.station_dict['start_date']))[0], 0,
                           inplace=True)
            self.data.reset_index(inplace=True, drop=True)
        if 'end_date' in self.station_dict.keys():
            self.data.drop(np.where(self.data['datenum'] > date2num(self.station_dict['end_date']))[0], 0, inplace=True)
            self.data.reset_index(inplace=True, drop=True)




class meereisportal:
    def __init__(self, station_dict):
        self.station_dict = station_dict

    def read_buoy(self, file_type, **kwargs):
        if file_type == 'lightchain':
            column_names = ['datenum', 'depth', 'VIS_side_R', 'VIS_side_G', 'VIS_side_B', 'VIS_side_C']
            file_path = self.station_dict['filepath'] + \
                        self.station_dict['id'] + '_' + self.station_dict['long_name'] + \
                        '_LIGHTCHAIN_proc.csv'
            try:
                df = pd.read_csv(
                file_path,
                delimiter=',',
                names=column_names,
                usecols=range(6),
                converters={0: bytespdate2num_buoy},
                skiprows=[0],
                comment='#')
            except:
                df = pd.read_csv(
                    file_path,
                    delimiter=',',
                    names=column_names,
                    usecols=range(6),
                    converters={0: bytespdate2num_lightchain},
                    skiprows=[0],
                    comment='#')
            df[df == 65535] = np.nan
            calib_data = xr.load_dataset(self.station_dict['filepath'].rsplit('/',3)[0] + \
                                        '/calib/' + \
                                         self.station_dict['id'] + '_' + self.station_dict['long_name'] + \
                                         '_LIGHTCHAIN_calibration_coeffs.nc')
            if self.station_dict['id'] != "2020R12":
                for depth in range(1, 65):
                    for var in ['VIS_side_R', 'VIS_side_G', 'VIS_side_B', 'VIS_side_C']:
                        df.loc[df['depth']==depth, var+'_cal'] = \
                            np.clip(
                                np.int64(
                                    df.loc[df['depth']==depth, var] * calib_data[var].sel(depth=depth).data
                                ),
                                0, 2 ** 16 - 1)
            df['depth'] = (df['depth'] - 1) * 0.05 + float(self.station_dict['vertical_offset'])  # 5cm spacing
            if np.all(np.equal(np.repeat(np.array(
                    df.loc[df['depth'] == float(self.station_dict['vertical_offset']), 'datenum'])
                    , 64), np.array(df['datenum']))):
                # check if timestamp is always equal to the uppermost measurement
                df['nominal_date'] = df['datenum']
            self.data = df

    def add_snow_ice_interfaces(self, filepath):
        df = pd.read_excel(filepath,
                           sheet_name='Tem & Th', usecols=[0, 1, 2], skiprows=[1], parse_dates=[0],
                           names=['date', 'snow_thickness', 'ice_thickness']).dropna()
        df.reset_index(inplace=True)

        df['surface_melt'] = df.loc[df['snow_thickness'] < 0, 'snow_thickness']
        df.loc[df['snow_thickness'] < 0, 'snow_thickness'] = np.nan

        snow_thick = interpolate.interp1d(mdates.date2num(df['date'].astype("M8[ms]").tolist()),
                                          df['snow_thickness'] / 100., bounds_error=False)
        surface_melt = interpolate.interp1d(mdates.date2num(df['date'].astype("M8[ms]").tolist()),
                                            df['surface_melt'] / 100., bounds_error=False)
        ice_thick = interpolate.interp1d(mdates.date2num(df['date'].astype("M8[ms]").tolist()),
                                         df['ice_thickness'] / 100., bounds_error=False)

        self.data['snow_thickness'] = snow_thick(self.data['nominal_date'])
        self.data['ice_thickness'] = ice_thick(self.data['nominal_date'])
        self.data['surface_melt'] = surface_melt(self.data['nominal_date'])

    def outliers2nan(self):

        """ replace previously defined outliers by np.nan """

        variables = self.station_dict['variables']

        for var in variables.keys():
            if var in list(self.data.columns):
                for condition in variables[var].keys():
                    if condition == 'upper_bound':
                        self.data.loc[self.data[var] > float(variables[var][condition]), var] = np.nan
                    if condition == 'lower_bound':
                        self.data.loc[self.data[var] < float(variables[var][condition]), var] = np.nan
                    if condition == 'nan':
                        self.data.loc[self.data[var] == float(variables[var][condition]), var] = np.nan

    def crop_timeline(self):
        """ crop data to all dates between start and end date """
        if 'start_date' in self.station_dict.keys():
            self.data.drop(np.where(self.data['datenum'] < date2num(self.station_dict['start_date']))[0], 0,
                           inplace=True)
            self.data.reset_index(inplace=True, drop=True)
        if 'end_date' in self.station_dict.keys():
            self.data.drop(np.where(self.data['datenum'] > date2num(self.station_dict['end_date']))[0], 0, inplace=True)
            self.data.reset_index(inplace=True, drop=True)

class auxiliary:
    def __init__(self, station_dict):
        self.station_dict = station_dict

    def read_data(self):
        self.ds = xr.load_dataset(self.station_dict['filepath'])

class RanTaoFiles:
    def __init__(self, station_dict):
        self.station_dict = station_dict

    def read_buoy(self, file_type, **kwargs):

        if file_type == 'awi_rad_ran':
            file_path = self.station_dict['filepath'] + \
                        self.station_dict['id'] + '_' + kwargs['name'] + '.txt'

            rad = kwargs['name']+'_Ran'
            column_names = ['datenum', 'Solar_'+rad] + [str(wavelength) for wavelength in range(320, 951)]
            df_temp = pd.read_csv(
                file_path,
                delimiter=';',
                names=column_names,
                converters={0: bytespdate2num_buoy},
                skiprows=[0],
                comment='#')
            df = df_temp[['datenum', 'Solar_'+rad]]

            PAR_F2Q_converter = np.arange(320, 951, 1, dtype=np.float32)
            PAR_F2Q_converter /= 1.196e2

            df.loc[:, 'PAR_Q_'+rad] = np.nansum(df_temp.iloc[:, 82:383].values / 1000. * PAR_F2Q_converter[82:383], axis=1)

            for (window, sensor) in [('VIS', 'filtered'), ('NIR', 'unfiltered')]:
                for band in ['R', 'G', 'B', 'C']:
                    df_rsr = pd.read_csv('data/processed/meta/RSR_' + sensor + '_' + band + '_orig_scale.txt',
                                         delimiter=' ',
                                         names=['wv', 'rsr'],
                                         comment='#')
                    df.loc[:, window + '_'+rad+'_' + band] = np.nansum(
                        df_temp.iloc[:, 2:] * (df_rsr.loc[np.isin(df_rsr['wv'], range(320, 951)), 'rsr']).values,
                        axis=1)/1000.

                    df_rsr = pd.read_csv('data/processed/meta/RSR_' + sensor + '_' + band + '_orig_scale.txt',
                                         delimiter=' ',
                                         names=['wv', 'rsr'],
                                         comment='#')
                    df.loc[:, window + '_' + rad + '_' + band + '_quant'] = np.nansum(
                        df_temp.iloc[:, 2:]/ 1000. * PAR_F2Q_converter *(df_rsr.loc[np.isin(df_rsr['wv'], range(320, 951)), 'rsr']).values,
                        axis=1)

            df_rsr = pd.read_csv('data/processed/meta/RSR_RGB_PAR_proxy_orig_scale.txt',
                                 delimiter=' ',
                                 names=['wv', 'rsr'],
                                 comment='#')
            df.loc[:, 'PAR_' + rad + '_RGB_proxy'] = np.nansum(
                df_temp.iloc[:, 2:] * (df_rsr.loc[np.isin(df_rsr['wv'], range(320, 951)), 'rsr']).values,
                axis=1) / 1000.

            df_rsr = pd.read_csv('data/processed/meta/RSR_RGB_PAR_proxy_orig_scale.txt',
                                 delimiter=' ',
                                 names=['wv', 'rsr'],
                                 comment='#')
            df.loc[:, 'PAR_' + rad + '_RGB_quant_proxy'] = np.nansum(
                df_temp.iloc[:, 2:] / 1000. * PAR_F2Q_converter * (df_rsr.loc[np.isin(df_rsr['wv'], range(320, 951)), 'rsr']).values,
                axis=1)

            df.loc[:, 'depth'] = -1.
            df.loc[:, 'nominal_date'] = df['datenum']

            df=df.groupby('nominal_date').mean().reset_index() # removes duplicated measurements (offsets?)

            self.data = df.copy()


    def add_full_spectrum(self, ds, **kwargs):
        file_path = self.station_dict['filepath'] + \
                    self.station_dict['id'] + '_' + kwargs['name'] + '.txt'

        rad = kwargs['name'] + '_Ran'
        column_names = ['datenum', 'Solar_' + rad] + [str(wavelength) for wavelength in range(320, 951)]
        df_temp = pd.read_csv(
            file_path,
            delimiter=';',
            names=column_names,
            converters={0: bytespdate2num_buoy},
            skiprows=[0],
            comment='#')

        df_temp = df_temp.groupby('datenum').mean().reset_index()

        nominal_date = np.array(mdates.num2date(df_temp['datenum'])).astype('datetime64[s]')
        df_temp = df_temp.set_index(nominal_date)
        #if np.any(np.not_equal(nominal_date,ds.nominal_date.data)):
        #    raise 'strange timeaxis in Ran files'

        ds['wavelength']=np.arange(320, 951)
        ds['spectral_'+rad] = (('nominal_date', 'wavelength'), df_temp.loc[ds.nominal_date].iloc[:,2:].values)

        return ds

    def add_snow_ice_interfaces(self, filepath):
        df = pd.read_excel(filepath,
                           sheet_name='Tem & Th', usecols=[0, 1, 2], skiprows=[1], parse_dates=[0],
                           names=['date', 'snow_thickness', 'ice_thickness']).dropna()
        df.reset_index(inplace=True)

        df['surface_melt'] = df.loc[df['snow_thickness'] < 0, 'snow_thickness']
        df.loc[df['snow_thickness'] < 0, 'snow_thickness'] = np.nan

        snow_thick = interpolate.interp1d(mdates.date2num(df['date'].astype("M8[ms]").tolist()),
                                          df['snow_thickness'] / 100., bounds_error=False)
        surface_melt = interpolate.interp1d(mdates.date2num(df['date'].astype("M8[ms]").tolist()),
                                            df['surface_melt'] / 100., bounds_error=False)
        ice_thick = interpolate.interp1d(mdates.date2num(df['date'].astype("M8[ms]").tolist()),
                                         df['ice_thickness'] / 100., bounds_error=False)

        self.data['snow_thickness'] = snow_thick(self.data['nominal_date'])
        self.data['ice_thickness'] = ice_thick(self.data['nominal_date'])
        self.data['surface_melt'] = surface_melt(self.data['nominal_date'])

    def outliers2nan(self):

        """ replace previously defined outliers by np.nan """

        variables = self.station_dict['variables']

        for var in variables.keys():
            if var in list(self.data.columns):
                for condition in variables[var].keys():
                    if condition == 'upper_bound':
                        self.data.loc[self.data[var] > float(variables[var][condition]), var] = np.nan
                    if condition == 'lower_bound':
                        self.data.loc[self.data[var] < float(variables[var][condition]), var] = np.nan
                    if condition == 'nan':
                        self.data.loc[self.data[var] == float(variables[var][condition]), var] = np.nan

    def crop_timeline(self):
        """ crop data to all dates between start and end date """
        if 'start_date' in self.station_dict.keys():
            self.data.drop(np.where(self.data['datenum'] < date2num(self.station_dict['start_date']))[0], 0,
                           inplace=True)
            self.data.reset_index(inplace=True, drop=True)
        if 'end_date' in self.station_dict.keys():
            self.data.drop(np.where(self.data['datenum'] > date2num(self.station_dict['end_date']))[0], 0, inplace=True)
            self.data.reset_index(inplace=True, drop=True)

