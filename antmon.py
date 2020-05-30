import numpy as np
import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.transform import linear_cmap

import dsautils.dsa_syslog as dsl
logger = dsl.DsaSyslogger()
logger.subsystem('software')
logger.app('vis')

from dsautils import dsa_store
de = dsa_store.DsaStore() 

# parameters
antlist = list(range(1,111))
ignorelist = ['ant_num', 'index']

# min/max range for color coding
minmax = {'mp_age_seconds': [0, 1],
          'sim': [False, True],  # initialized directly
          'ant_el': [0., 145.],
          'ant_cmd_el': [0., 145.],
          'drv_cmd': [0, 2],
          'drv_act': [0, 2],
          'drv_state': [0, 2],
          'at_north_lim': [False, True],
          'at_south_lim': [False, True],
          'brake_on': [False, True],
          'motor_temp': [-10., 40.],
#          'focus_temp': [],
          'lna_current_a': [45., 85.],
          'lna_current_b': [45., 85.],
          'noise_a_on': [False, True],
          'noise_b_on': [False, True],
          'rf_pwr_a': [-80., -60.],
          'rf_pwr_b': [-80., -60.],
          'feb_current_a': [250., 300.],
          'feb_current_b': [250., 300.],
          'laser_volts_a': [2.5, 2.9],
          'laser_volts_b': [2.5, 2.9],
          'feb_temp_a': [-10., 60.],
          'feb_temp_b': [-10., 60.],
#          'psu_volt': [],
#          'lj_temp': [],
          'fan_err': [False, True],
#          'emergency_off': [False, True]  # what is good/bad here?
          'pd_current_a': [0.6, 2],
          'pd_current_b': [0.6, 2],
          'if_pwr_a': [-69, -60],
          'if_pwr_b': [-69, -60],
          'lo_pwr': [4.5, 5],
          'beb_current_a': [200, 300],
          'beb_current_b': [200, 300],
          'beb_temp': [20, 35]
          }

# set up data
def makedf():
    """ Makes a dataframe from data store (etcd).
    returns latest measurement time in mjd.
    column mp_age_seconds gives offset from that value.
    """
    
    logger.info('Making dataframe for antenna monitor points')
    dfs = []
    for ant in antlist:
        dd = {}
        try:
            dd.update(de.get_dict("/mon/ant/{0}".format(ant)))  # ant mps
        except: # should be KeyDoesNotExistException
            pass

        try:
            dd2 = de.get_dict("/mon/beb/{0}".format(ant))  # beb mps
            _ = dd2.pop('time')  # TODO: ignore for now to avoid clobbering ant time
            dd.update(dd2)
        except:  # should be KeyDoesNotExistException
            pass

        # TODO: add snap? cal?
        # TODO: fill missing values somehow

        if len(dd):
            df = pd.DataFrame.from_dict(dd, orient='index')
            dfs.append(df)

    df = pd.concat(dfs, axis=1).transpose().reset_index()
    time_latest = df.time.max()
    df.time = 24*3600*(time_latest - df.time)
    df.rename(columns={'time': 'mp_age_seconds'}, inplace=True)

    df.ant_num = df.ant_num.astype(int).astype(str)
    df.set_index('ant_num', 0, inplace=True)
    df.columns.name = 'mp'
    df = pd.DataFrame(df[reversed(df.columns)].stack(), columns=['value']).reset_index()
          
    color = np.where(df['mp'] == 'sim', 1, 0) * np.where(df['value'] == True, 1, 0)

    # Define a color scheme:
    # false/true/in/out-of-range == black/white/green/yellow
    for key, value in minmax.items():
        if key == 'sim':
            continue

        if isinstance(minmax[key][0], bool):
            color += np.where(df['mp'] == key, 1, 0) * np.where(df['value'] == value[1], 1, 0)
#        elif isinstance(minmax[key][0], int):  # something special for int valued (enumerations?)
        else:
            color += np.where(df['mp'] == key, 1, 0) * np.where(pd.to_numeric(df['value']) > value[1], 3, 2)

    df['color'] = np.array(['black', 'white', 'green', 'yellow'])[color]
    return time_latest, df

doc = curdoc()
time_latest, df = makedf()
source = ColumnDataSource(df)

# set up plot
logger.info('Setting up plot')
TOOLTIPS = [("value", "@value"), ("(ant_num, mp)", "(@ant_num, @mp)")]
mplist = [mp for mp in list(minmax.keys()) if mp not in ignorelist]
p = figure(plot_width=700, plot_height=1000, x_range=[str(aa) for aa in np.unique(df.ant_num)],
           y_range=list(reversed(mplist)), y_axis_label='Monitor Point', x_axis_label='Antenna Number',
           tooltips=TOOLTIPS, toolbar_location=None, x_axis_location="above",
           title="Antenna Monitor Points (loaded at MJD {0})".format(time_latest))
p.rect(x='ant_num', y='mp', width=1, height=1, source=source,
       fill_color='color', alpha=0.5)
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

def update():
    time_latest, df = makedf()
    source.stream(df, rollover=len(df))  # updates each value

doc.add_periodic_callback(update, 5000)

doc.add_root(p)
