import numpy as np
import pandas as pd
from bokeh.plotting import figure 
from bokeh.io import show, output_file, save
from bokeh.models import ColumnDataSource
from bokeh.transform import linear_cmap

from dsautils import dsa_store
de = dsa_store.DsaStore() 

output_file('test.html')

dfs = []
for ant in [24] + list(range(80,90)):
    dd = de.get_dict("/mon/ant/{0}".format(ant))
    df = pd.DataFrame.from_dict(dd, orient='index')
    dfs.append(df)
df = pd.concat(dfs, axis=1).transpose().reset_index() 
df.ant_num = df.ant_num.astype(str)
df.set_index('ant_num', 0, inplace=True)
df.columns.name = 'mp'
df = pd.DataFrame(df.stack(), columns=['value']).reset_index()

# add column for color coding
color = np.where(df['mp'] == 'sim', 1, 0) * np.where(df['value'] == True, 1, 0)
color += np.where(df['mp'] == 'noise_a_on', 1, 0) * np.where(df['value'] == True, 1, 0)
color += np.where(df['mp'] == 'noise_b_on', 1, 0) * np.where(df['value'] == True, 1, 0)
color += np.where(df['mp'] == 'brake_on', 1, 0) * np.where(df['value'] == True, 1, 0)
df['color'] = color

source = ColumnDataSource(df)
TOOLTIPS = [("value", "@value"), ("(ant_num, mp)", "(@ant_num, @mp)")]

p = figure(plot_width=1300, plot_height=700, x_range=list(df.ant_num.unique()), y_range=list(df.mp.unique()), tooltips=TOOLTIPS)
p.rect(x='ant_num', y='mp', width=1, height=1, source=source, fill_color=linear_cmap('color', 'Viridis256', 0, 1))
save(p)
