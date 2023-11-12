# -*- coding: utf-8 -*-
"""
MOSiP database source code.
developped by Niels Fuchs (Universität Hamburg, 2021)
contact: niels.fuchs@uni-hamburg.de
"""



def set_variable_attributes(data):
    """ standard attributes sea ice profiles """
    band_dict={'R':'red', 'G':'green', 'B':'blue', 'C':'clear'}
    # loop data and coords variables
    for var in list(data.data_vars)+list(data.coords):

        # position
        if var == 'lat':
            data[var].attrs = {'units': 'degree_north', 'standard_name': 'latitude', 'long_name': 'latitude'}
        elif var == 'lon':
            data[var].attrs = {'units': 'degree_east', 'standard_name': 'longitude', 'long_name': 'longitude'}
        elif var == 'ice_depth':
            data[var].attrs = {'units': 'm', 'standard_name': 'sea ice depth',
                                       'long_name': 'momentary sensor depth relative to sea ice surface'}
        elif var == 'depth':
            data[var].attrs = {'units': 'm', 'long_name': 'deployment depth'}

        elif var == 'depth_center':
            data[var].attrs = {'units': 'm', 'long_name': 'layer center depth at deployment'}

        elif var == 'depth_top':
            data[var].attrs = {'units': 'm', 'long_name': 'layer top depth at deployment'}

        elif var == 'depth_bottom':
            data[var].attrs = {'units': 'm', 'long_name': 'layer bottom depth at deployment'}

        elif var == 'ice_depth_center':
            data[var].attrs = {'units': 'm', 'long_name': 'momentary layer center depth in the ice'}

        elif var == 'ice_depth_top':
            data[var].attrs = {'units': 'm', 'long_name': 'momentary layer top depth in the ice'}

        elif var == 'ice_depth_bottom':
            data[var].attrs = {'units': 'm', 'long_name': 'momentary layer bottom depth in the ice'}


        # time

        elif var == 'nominal_date':
            data[var].attrs = {'tz': 'UTC', 'long_name': 'date and time of uppermost valid measurement'}

        elif var == 'delay_from_nominal':
            data[var].attrs = {'units': 'seconds', 'long_name': 'time delay of measurement from nominal date of the profile'}

        # temperature measurements
        elif var == 'T_deg':
            data[var].attrs = {'units': 'degree_Celsius', 'long_name': 'temperature'}
        elif var == 'T_ice':
            data[var].attrs = {'units': 'degree_Celsius', 'standard_name': 'sea_ice_temperature',
                               'long_name': 'sea ice temperature'}
        elif var == 'T_ice_surface':
            data[var].attrs = {'units': 'degree_Celsius', 'standard_name': 'sea_ice_surface_temperature',
                               'long_name': 'temperature at the ice surface'}
        elif var == 'T_ice_bottom':
            data[var].attrs = {'units': 'degree_Celsius', 'standard_name': 'sea_ice_basal_temperature',
                               'long_name': 'temperature at the ice bottom'}
        elif var == 'T_ocean':
            data[var].attrs = {'units': 'degree_Celsius', 'standard_name': 'sea_water_temperature',
                               'long_name': 'sea water temperature'}
        elif var == 'T_air':
            data[var].attrs = {'units': 'degree_Celsius', 'standard_name': 'air_temperature',
                               'long_name': 'air temperature'}

        # ice properties
        elif var == 'S_bulk':
            data[var].attrs = {'units': 'g/kg', 'long_name': 'bulk salinity'}
        elif var == 'S_bulk_simple':
            data[var].attrs = {'units': 'g/kg', 'long_name': 'bulk salinity, without brine sensitivity'}
        elif var == 'S_brine':
            data[var].attrs = {'units': 'g/kg', 'long_name': 'brine salinity'}
        elif var == 'phi_l_m':
            data[var].attrs = {'units': '1', 'long_name': 'liquid mass fraction'}
        elif var == 'phi_l_m_simple':
            data[var].attrs = {'units': '1', 'long_name': 'liquid mass fraction, without brine sensitivity'}
        elif var == 'phi_s_m':
            data[var].attrs = {'units': '1', 'long_name': 'solid mass fraction'}
        elif var == 'phi_s_v':
            data[var].attrs = {'units': '1', 'long_name': 'solid volume fraction'}
        elif var == 'phi_g_v':
            data[var].attrs = {'units': '1', 'long_name': 'gas volume fraction'}
        elif var == 'ice_thickness':
            data[var].attrs.update({'units': 'm', 'long_name': 'ice thickness', 'standard_name': 'sea_ice_thickness'})
        elif var == 'surface_melt':
            data[var].attrs.update({'units': 'm', 'long_name': 'sea ice-surface melt'})


        # snow measurements

        elif var[:13] == 'snow_distance':
            data[var].attrs = {'units': 'meter', 'long_name': 'instruments distance to snow surface'}
        elif var == 'snow_thickness':
            data[var].attrs.update({'units': 'm', 'long_name': 'snow thickness', 'standard_name': 'surface snow thickness'})

        # electrical measurements
        elif var == 'Impedance':
            data[var].attrs = {'units': 'Ohm', 'long_name': 'impedance measured as Ohm resistance in the ice'}
        elif var == 'H030':
            data[var].attrs = {'units': 'degree_Celsius', 'long_name': 'temperature increase after 30s of heating'}
        elif var == 'H120':
            data[var].attrs = {'units': 'degree_Celsius', 'long_name': 'temperature increase after 120s of heating'}

        # optical measurements

        if var.split('_',1)[0] in ['VIS', 'NIR']:
            spec = var.split('_',1)[0]
            if var.rsplit('_',1)[1]=='cal':
                cal = ' calibrated'
                band = band_dict[var.split('_')[-2]]
            elif var.rsplit('_', 1)[1] == 'quant':
                # Ramses band approximations
                cal = ''
                band = band_dict[var.split('_')[-2]]
            else:
                cal = ''
                band = band_dict[var.split('_')[-1]]
            if var.split('_')[1] == 'in':
                data[var].attrs = {'units': 'counts', 'long_name': 'planar irradiance incoming '+spec+' '+band+' channel'+cal}
            elif var.split('_')[1] == 'out':
                data[var].attrs = {'units': 'counts', 'long_name': 'planar irradiance outgoing '+spec+' '+band+' channel'+cal}
            elif var.split('_')[1] == 'side':
                data[var].attrs = {'units': 'counts', 'long_name': 'planar irradiance sideways '+spec+' '+band+' channel'+cal}
            elif var.split('_')[1] in ['scalar','upwelling', 'downwelling']:
                if var.split('_')[2] == 'attenuation':
                    if var.split('_')[3] == '2lvl':
                        data[var].attrs = {'units': 'per m',
                                           'long_name': 'approximated diffuse attenuation coefficient of '+var.split('_')[1]+\
                                                        'irradiance, ' + spec + ' ' + band + ' channel' + cal +
                                           ', retrieved over two layers'}
                    else:
                        data[var].attrs = {'units': 'per m', 'long_name': 'approximated diffuse attenuation coefficient of '+var.split('_')[1]+\
                                                        'irradiance, ' + spec + ' ' + band + ' channel' + cal}
                else:
                    data[var].attrs = {'units': 'counts', 'long_name': 'approximated scalar irradiance '+spec+' '+band+' channel'+cal}
            elif var.split('_')[1] == 'incident':
                data[var].attrs = {'units': 'W per m2',
                               'long_name': 'incident ' + spec + ' ' + band + ' channel irradiance above the ice surface'}
            elif var.split('_')[1] == 'trans':
                data[var].attrs = {'units': 'W per m2',
                               'long_name': 'transmitted ' + spec + ' ' + band + ' channel irradiance above the ice surface'}
            elif var.split('_')[1] == 'reflectivity':
                data[var].attrs = {'units': 'ratio',
                                   'long_name': 'layer reflectivity ' + spec + ' ' + band + ' channel' + cal}
            elif var.split('_')[1] == 'transmissivity':
                data[var].attrs = {'units': 'ratio',
                                   'long_name': 'layer transmissivity ' + spec + ' ' + band + ' channel' + cal}
            elif var.split('_')[1] == 'absorptivity':
                data[var].attrs = {'units': 'ratio',
                                   'long_name': 'layer absorptivity ' + spec + ' ' + band + ' channel' + cal}
            elif var.split('_')[1] == 'slab':
                data[var].attrs = {'units': 'ratio',
                                   'long_name': 'irradiance reflectance, ' + spec + ' ' + band + ' channel' + cal}

            elif var.split('_')[1] == 'incoming':
                if var.rsplit('_', 1)[1] == 'quant':
                    data[var].attrs = {'units': 'ymol photons per m2 per s',
                                       'long_name': 'incoming ' + spec + ' ' + band + ' channel planar irradiance above the ice surface'}
                else:
                    data[var].attrs = {'units': 'W per m2',
                                   'long_name': 'incoming ' + spec + ' ' + band + ' channel planar irradiance above the ice surface'}
            elif var.split('_')[1] == 'transmitted':
                if var.rsplit('_', 1)[1] == 'quant':
                    data[var].attrs = {'units': 'ymol photons per m2 per s',
                                       'long_name': 'transmitted ' + spec + ' ' + band + ' channel planar irradiance below the ice'}
                else:
                    data[var].attrs = {'units': 'W per m2',
                                   'long_name': 'transmitted ' + spec + ' ' + band + ' channel planar irradiance below the ice'}
            elif var.split('_')[1] == 'reflected':
                if var.rsplit('_', 1)[1] == 'quant':
                    data[var].attrs = {'units': 'ymol photons per m2 per s',
                                       'long_name': 'transmitted ' + spec + ' ' + band + ' channel planar irradiance reflected on the ice surface'}
                else:
                    data[var].attrs = {'units': 'W per m2',
                                   'long_name': 'transmitted ' + spec + ' ' + band + ' channel planar irradiance reflected on the ice surface'}

        elif var.split('_', 1)[0] == 'PAR':
            if var.split('_')[2] == 'in':
                data[var].attrs = {'units': 'ymol m-2 s-1',
                                   'long_name': 'incoming planar PAR irradiance proxy'}
            elif var.split('_')[2] == 'out':
                data[var].attrs = {'units': 'ymol m-2 s-1',
                                   'long_name': 'outgoing planar PAR irradiance proxy'}
            elif var.split('_')[2] == 'scalar':
                data[var].attrs = {'units': 'ymol m-2 s-1',
                                   'long_name': 'scalar PAR irradiance proxy'}

    return data


def set_global_attributes(data, id, meta):
    # major settings
    data.attrs['expedition'] = 'MOSAiC expedition 2019 - 2020'
    data.attrs['database'] = 'MOSiP sea ice profile database (NiceLABpro project)'
    data.attrs['database created by'] = 'Niels Fuchs (Universitaet Hamburg, 2021)'
    data.attrs['database contact'] = 'Niels.Fuchs@uni-hamburg.de'
    data.attrs['buoy_ID'] = id
    if meta['buoy_type'] in ['radstation']:
        data.attrs['buoy_number'] = meta['long_name']
    data.attrs['buoy_type'] = meta['buoy_type']
    data.attrs['data_Pi'] = meta['pi']
    data.attrs['institute'] = meta['pi_institute']
    data.attrs['deployment_site'] = meta['site']
    data.attrs['date_first'] = meta['date_first']
    data.attrs['date_last'] = meta['date_last']

    if 'DOI' in meta.keys():
        data.attrs['DOI'] = meta['DOI']
    if 'data origin' in meta.keys():
        data.attrs['data_origin'] = meta['data origin']
    if 'involved scientist' in meta.keys():
        data.attrs['involved_scientist'] = meta['involved scientist']

    return data

def nc_meta_2_station_dict(meta):
    """ rename .nc file meta attributes to initial station dict keys, necessary when
    retrievals are derived subsequently to initial database build"""

    meta['site'] = meta.pop('deployment_site')
    meta['pi_institute'] = meta.pop('institute')
    meta['pi'] = meta.pop('data_Pi')
    if 'buoy_number' in meta.keys():
        meta['long_name'] = meta.pop('buoy_number')

    return meta
