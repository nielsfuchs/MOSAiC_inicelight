# -*- coding: utf-8 -*-
"""
MOSiP read input source code.
developed by Niels Fuchs (Universität Hamburg, 2021)
contact: niels.fuchs@uni-hamburg.de
"""

import numpy as np
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
import os
import xml.etree.ElementTree as ET
import tqdm
from src import LightRead, LightProfiles

# read xml input file
tree = ET.parse('rawinput_template.xml')
rawinput = tree.getroot()

# contains all stations that should be loaded
station_list = rawinput.find('station_list')

print('Process buoy data from XML input')

for station in tqdm.tqdm(station_list.iter('station')):
    station_dict = station.attrib
    for element in station.iter():
        if element.tag not in ['variable', 'subfiles', 'file']:
            station_dict[element.tag] = element.text

    # load individual settings for variables as upper and lower bounds, nan
    station_dict['variables'] = {}
    for variable in station.iter('variable'):
        station_dict['variables'][variable.attrib['var']] = variable.attrib

    # station data is partially separated into different files, load specifications for each of these subfiles
    station_dict['subfiles'] = {}
    for file in station.iter('file'):
        station_dict['subfiles'][file.attrib['name']] = file.attrib

    auxdata = LightRead.auxiliary({'filepath':
                                           station_dict['filepath'].rsplit('raw', 1)[0]+
                                           'aux/'+
                                           LightProfiles.auxiliary_data_reference[station_dict['site']]})
    auxdata.read_data()



    if station_dict['buoy_type'] == 'lightharp':
        lightharp = LightRead.lightharp(station_dict)
        lightharp = LightRead.lightharp(station_dict)
        lightharp.read_lightharp()
        lightharp.crop_timeline()
        lightharp.add_nominal_date()
        ds = LightProfiles.convert2xarray(lightharp.data, 'lightharp1',
                aux=auxdata.ds)

    elif station_dict['buoy_type'] == 'radstation':

        for file in station_dict['subfiles'].keys():
            if file != 'AUX_proc':
                radstation = LightRead.meereisportal(station_dict)
                radstation.read_buoy(station_dict['subfiles'][file]['type'],
                                     name=station_dict['subfiles'][file]['name'])
                radstation.crop_timeline()
                ds = LightProfiles.convert2xarray(radstation.data, station_dict['id'],
                                                  aux=auxdata.ds)

    standard_profile = LightProfiles.IceProfile(data=ds, id=ds['sensor_id'].data,
                                                            meta=station_dict)
    standard_profile.correct_surface_melt()
    standard_profile.write_NetCDF()