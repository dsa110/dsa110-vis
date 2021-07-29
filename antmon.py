import numpy as np
import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.io import show, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.transform import linear_cmap
from astropy import time
import os

from slack_sdk import WebClient
try:
    client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])
except KeyError:
    print('No SLACK_BOT_TOKEN env var found. No slack posting.')

import logging
import dsautils.dsa_syslog as dsl
logger = dsl.DsaSyslogger('dsa', 'software', logging.INFO, 'antmon')
logger.app = 'bokeh'

from dsautils import dsa_store, cnf
de = dsa_store.DsaStore() 
my_cnf = cnf.Conf()

# parameters
antlist = list(range(1,120))
ignorelist = ['ant_num', 'index']
servicelist = ['calibration', 'bfweightcopy', 'calpreprocess'] + ['corr/'+str(i) for i in range(1,21)] + ['T2service', 'T2gulp']

# min/max range for color coding
minmax = my_cnf.get('minmax_ant')
# beb mps
minmax2 = my_cnf.get('minmax_beb')
minmax3 = my_cnf.get('minmax_service')

# set up data
def makedf():
    """ Makes dataframes (ant and beb mps) from data store (etcd).
    returns latest measurement time in mjd for both ant and beb mps.
    column mp*_age_seconds gives offset from that value.
    """
    
    logger.function = 'makedf'
    logger.info('Making dataframe for antenna monitor points')
    dfs = []
    dfs2 = []
    dfs3 = []

    for ant in antlist:

        # ant mps
        dd = de.get_dict("/mon/ant/{0}".format(ant))

        if dd is not None:
            if 'ant_num' in dd:
                df = pd.DataFrame.from_dict(dd, orient='index')
                dfs.append(df)
            else:
                logger.warning("get_dict returned nonstandard ant dict")
        else:
            continue

        # beb mps
        dd2 = de.get_dict("/mon/beb/{0}".format(ant))

        if dd2 is not None:
            if ('ant_num' in dd) and ('ant_num' in dd2) and 'pd_current_a' in dd2:  # only include ants in both lists
                df2 = pd.DataFrame.from_dict(dd2, orient='index')
                dfs2.append(df2)
            else:
                logger.warning("get_dict returned nonstandard ant dict")

    dd3 = {}
    print('Checking services:', servicelist)
    for service in servicelist:
        try:
            dd3[service] = de.get_dict("/mon/service/{0}".format(service))
        except: # should be KeyDoesNotExistException
            pass
    df3 = pd.DataFrame.from_dict(dd3, orient='index')

    # ant mps
    if not len(dfs):
        return (None, None, None)

    df = pd.concat(dfs, axis=1, sort=True).transpose().reset_index()
#    time_latest = df.time.max()  # most recent in data
    time_latest = time.Time.now().mjd  # actual current time
    df.time = 24*3600*(time_latest - df.time)
    df.rename(columns={'time': 'mp_age_seconds'}, inplace=True)

    df.ant_num = df.ant_num.astype(int).astype(str)
    df.set_index('ant_num', 0, inplace=True)
    df.columns.name = 'mp'
    df = pd.DataFrame(df[reversed(df.columns)].stack(), columns=['value']).reset_index()
    color = np.where(df['mp'] == 'sim', 1, 0) * np.where(df['value'] == True, 1, 0)
    
    # beb mps
    df2 = pd.concat(dfs2, axis=1, sort=True).transpose().reset_index()
#    time_latest = df2.time.max()
    df2.time = 24*3600*(time_latest - df2.time)
    df2.rename(columns={'time': 'mp_age_seconds'}, inplace=True)

    df2.ant_num = df2.ant_num.astype(int).astype(str)
    df2.set_index('ant_num', 0, inplace=True)
    df2.columns.name = 'mp'
    df2 = pd.DataFrame(df2[reversed(df2.columns)].stack(), columns=['value']).reset_index()
    color2 = np.zeros(len(df2))

    df3.time = 24*3600*(time_latest - df3.time)
    df3.rename(columns={'time': 'mp_age_seconds'}, inplace=True)
    color3 = df3.mp_age_seconds > minmax3['mp_age_seconds'][1]

    enums = {'drv_cmd': ['Off', 'North', 'South'],
             'drv_act': ['Off', 'North', 'South'],
             'drv_state': ['Halt', 'Seek', 'Acquired', 'Timeout', 'FwLimN', 'FwLimS']
             }

    
    # Define a color scheme:
    # false/true/in/out-of-range == black/white/green/yellow
    for key, value in minmax.items():
        if key == 'sim':
            continue

        if isinstance(minmax[key][0], bool):
            color += np.where(df['mp'] == key, 1, 0) * np.where(df['value'] == value[1], 1, 0)
        else:
            color += np.where(df['mp'] == key, 1, 0) * np.where( (pd.to_numeric(df['value']) > value[1]) | (pd.to_numeric(df['value']) < value[0]), 3, 2)
#            color += np.where(df['mp'] == key, 1, 0) * np.where( (pd.to_numeric(df['value']) > value[1]) & (pd.to_numeric(df['value']) < value[0]), 3, 2)

    # overload enums with their string meanings
    for key, value in minmax.items():
        if key == 'sim':
            continue

        if key in enums:
            ww = np.where(df['mp'] == key)[0]
            value = df['value'][ww].astype(int)
            df['value'][ww] = np.array(enums[key])[value]

    for key, value in minmax2.items():
        if key == 'sim':
            continue

        if isinstance(minmax2[key][0], bool):
            color2 += np.where(df2['mp'] == key, 1, 0) * np.where(df2['value'] == value[1], 1, 0)
        else:
            color2 += np.where(df2['mp'] == key, 1, 0) * np.where( (pd.to_numeric(df2['value']) > value[1]) | (pd.to_numeric(df2['value']) < value[0]), 3, 2)
#            color2 += np.where(df2['mp'] == key, 1, 0) * np.where( (pd.to_numeric(df2['value']) > value[1]) & (pd.to_numeric(df2['value']) < value[0]), 3, 2)

    df['color'] = np.array(['black', 'white', 'green', 'yellow'])[color.astype(int)]
    df2['color'] = np.array(['black', 'white', 'green', 'yellow'])[color2.astype(int)]
    df3['color'] = np.array(['green', 'yellow'])[color3.astype(int)]
    df3['y'] = np.ones(len(color3))

    return time_latest, df, df2, df3

doc = curdoc()

# how to combine callbacks?
#    wid = de.add_watch_prefix('/mon/ant/', makedf)
#    sleep(4)
#    de.cancel(wid)
# but actually this should be driven by etcd callback...

time_latest, df, df2, df3 = makedf()
if df is None:
    logger.warning("No data found")
else:
    source = ColumnDataSource(df)

    # set up plot
    logger.info('Setting up plot')
    TOOLTIPS = [("value", "@value"), ("(ant_num, mp)", "(@ant_num, @mp)")]
    TOOLTIPS3 = [("(service, age)", "(@index, @mp_age_seconds)")]

    mplist = [mp for mp in list(minmax.keys()) if mp not in ignorelist]
    p = figure(plot_width=1000, plot_height=1000, x_range=[str(aa) for aa in sorted(np.unique(df.ant_num).astype(int))],
               y_range=list(reversed(mplist)), y_axis_label='Monitor Point', x_axis_label='Antenna Number',
               tooltips=TOOLTIPS, toolbar_location=None, x_axis_location="above",
               title="Antenna Monitor Points")
    p.rect(x='ant_num', y='mp', width=1, height=1, source=source,
           fill_color='color', alpha=0.5)
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    source2 = ColumnDataSource(df2)
    mplist2 = [mp for mp in list(minmax2.keys()) if mp not in ignorelist]
    p2 = figure(plot_width=1000, plot_height=1000, x_range=[str(aa) for aa in sorted(np.unique(df2.ant_num).astype(int))],
                y_range=list(reversed(mplist2)), y_axis_label='Monitor Point', x_axis_label='Antenna Number',
                tooltips=TOOLTIPS, toolbar_location=None, x_axis_location="above",
                title="BEB Monitor Points")
    p2.rect(x='ant_num', y='mp', width=1, height=1, source=source2,
            fill_color='color', alpha=0.5)
    p2.xgrid.grid_line_color = None
    p2.ygrid.grid_line_color = None

    source3 = ColumnDataSource(df3)
    mplist3 = [mp for mp in list(minmax3.keys()) if mp not in ignorelist]
#    print(df3)
    p3 = figure(plot_width=1000, plot_height=150, x_range=[str(aa) for aa in sorted(np.unique(df3.index))],
                y_range=mplist3,
                tooltips=TOOLTIPS3, toolbar_location=None, x_axis_location="above",
                title=f"Service age from MJD={time_latest}")
    p3.rect(x='index', y='y', width=1, height=2, source=source3,
            fill_color='color', alpha=0.5)
    p3.xgrid.grid_line_color = None
    p3.ygrid.grid_line_color = None
    p3.xaxis.major_label_orientation = 0.9
    pall = column(p3, p, p2)

    def update():
        logger.function = 'update'
    time_latest, df, df2, df3 = makedf()

    doc.add_periodic_callback(update, 5000)

    doc.add_root(pall)
    doc.title = "DSA-110 Monitor Point Summary"


def slack_alert(df, df2, df3):
    """ Parse dfs for bad mps and push alert to slack
    """
    
    source.stream(df, rollover=len(df))  # updates each ant value
    source2.stream(df2, rollover=len(df2))  # updates each beb value
    source3.stream(df3, rollover=len(df3))  # updates each beb value
    dfyellow = df[df['color'] == 'yellow']
    df2yellow = df2[df2['color'] == 'yellow']
    df3yellow = df3[df3['color'] == 'yellow']
    dfyc = pd.concat([dfyellow, df2yellow])
    try:
        print(dfyc.index)
        if not all(dfyc.index == dfyc0.index):
            dfdiff = dfyc[~dfyc.isin(dfyc0)].dropna()
            message = dfdiff.to_string()
            response = client.chat_postMessage(channel='#observing', text=message, icon_emoji=':robot_face:')
            assert response["ok"]
            if response["message"]["text"] != message:
                logger.warn("Response from Slack API differs from message sent. "
                            "Maybe just broken into multiple updates: {0}".format(response))
            dfyc0 = dfyc
    except UnboundLocalError:
        dfyellow0 = df[df['color'] == 'yellow']
        df2yellow0 = df2[df2['color'] == 'yellow']
        df3yellow0 = df3[df3['color'] == 'yellow']
        dfyc0 = pd.concat([dfyellow0, df2yellow0])
