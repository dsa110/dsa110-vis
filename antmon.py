import numpy as np
import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.transform import linear_cmap

from dsautils import dsa_store
de = dsa_store.DsaStore() 

# parameters
antlist = [24] + list(range(80,100))
ignorelist = ['ant_num', 'time', 'index']

# collect data
dfs = []
for ant in antlist:
    dd = de.get_dict("/mon/ant/{0}".format(ant))
    df = pd.DataFrame.from_dict(dd, orient='index')
    dfs.append(df)
df = pd.concat(dfs, axis=1).transpose().reset_index()
df.ant_num = df.ant_num.astype(str)
df.set_index('ant_num', 0, inplace=True)
df.columns.name = 'mp'
df = pd.DataFrame(df[reversed(df.columns)].stack(), columns=['value']).reset_index()

# add column for color coding
minmax = {#'sim': [False, True],  # initialized directly
          'ant_el': [0., 145.],
#          'time': []
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
          }
          
color = np.where(df['mp'] == 'sim', 1, 0) * np.where(df['value'] == True, 1, 0)
for key, value in minmax.items():
    if isinstance(minmax[key][0], bool):
        color += np.where(df['mp'] == key, 1, 0) * np.where(df['value'] == value[1], 1, 0)
#    elif isinstance(minmax[key][0], int):  # something special for int valued (enumerations?)
    else:
        color += np.where(df['mp'] == key, 1, 0) * np.where(pd.to_numeric(df['value']) > value[1], 1, 0)

df['color'] = np.array(['green', 'red'])[color]  # color for in/out of range
source = ColumnDataSource(df)

# set up plot
TOOLTIPS = [("value", "@value"), ("(ant_num, mp)", "(@ant_num, @mp)")]
mplist = [mp for mp in list(df.mp.unique()[:-1]) if mp not in ignorelist]
p = figure(plot_width=700, plot_height=1000, x_range=list(df.ant_num.unique()),
           y_range=mplist, y_axis_label='Monitor Point', x_axis_label='Antenna Number',
           tooltips=TOOLTIPS, toolbar_location=None, x_axis_location="above",
           title="Antenna Monitor Points")
p.rect(x='ant_num', y='mp', width=1, height=1, source=source,
       fill_color='color', alpha=0.5)
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

curdoc().add_root(p)
