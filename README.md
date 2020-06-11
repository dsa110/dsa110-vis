# dsa110-vis
A Bokeh application to display monitor and control information.
Bokeh uses javascript to enhance data visualizations (e.g., mouse-over effects)
and includes features for real-time updates.

## Visualizations
* antmon.py -- A good/bad dashboard for all monitor points.
Any monitor point with a valid range is displayed in green (good) or yellow (bad).
Boolean monitor points are white (true) or black (bad).

## Requirements
* python 3.6
* bokeh
* numpy
* pandas
* dsa110-pyutils

## Use
A bokeh application is effectively a python web server. It needs to be run in a dedicated
process. For now, we use `nohup` like this:
```
nohup bokeh serve antmon.py --allow-websocket-origin=localhost:5005 &
```

And visitors can then set up an ssh tunnel to port 5005 to see the bokeh application.


