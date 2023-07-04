import numpy as np
from matplotlib import pyplot as plt
from dsapol import dsapol
from scipy.signal import correlate
from scipy.signal import savgol_filter as sf
from scipy.signal import convolve
from scipy.signal import fftconvolve
from scipy.ndimage import convolve1d
from scipy.signal import peak_widths
from scipy.stats import chi
from scipy.stats import norm
import panel as pn
pn.extension()
#from numpy.ma import masked_array as ma
import numpy.ma as ma
from scipy.stats import kstest
from scipy.optimize import curve_fit
import time
#import numpy.ma as ma 

from scipy.signal import find_peaks
from scipy.signal import peak_widths
import copy
import numpy as np
import os
import param

from sigpyproc import FilReader
from sigpyproc.Filterbank import FilterbankBlock
from sigpyproc.Header import Header
from matplotlib import pyplot as plt
import pylab
import pickle as pkl
import json
from scipy.interpolate import interp1d
from scipy.stats import chi2
from scipy.stats import chi
from scipy.signal import savgol_filter as sf
from scipy.signal import convolve
from scipy.ndimage import convolve1d
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from RMtools_1D.do_RMclean_1D import run_rmclean
from RMtools_1D.do_QUfit_1D_mnest import run_qufit
import csv

import os
from astropy import units as u
from astropy.coordinates import SkyCoord
import subprocess

plt.rcParams.update({
                    'font.size': 16,
                    'font.family': 'serif',
                    'axes.labelsize': 16,
                    'axes.titlesize': 16,
                    'xtick.labelsize': 12,
                    'ytick.labelsize': 12,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'lines.linewidth': 1,
                    'lines.markersize': 5,
                    'legend.fontsize': 14,
                    'legend.borderaxespad': 0,
                    'legend.frameon': False,
                    'legend.loc': 'lower right'})


from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u
from syshealth.status_mon import get_rm
from astropy.coordinates import SkyCoord

#testing with RM table: https://github.com/CIRADA-Tools/RMTable
from rmtable import RMTable
#testing with pol Table: https://github.com/CIRADA-Tools/PolSpectra/tree/master
import polspectra

def gauss_scint(x,bw,amp,off):
    return off + amp*np.exp(-np.log(2)*((x/bw)**2))


def lorentz_scint(x,bw,amp,off):
    return off + amp*(bw/(x**2 + (0.5*bw**2)))

fsize=30
fsize2=20
plt.rcParams.update({
                    'font.size': fsize,
                    'font.family': 'sans-serif',
                    'axes.labelsize': fsize,
                    'axes.titlesize': fsize,
                    'xtick.labelsize': fsize,
                    'ytick.labelsize': fsize,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'lines.linewidth': 1,
                    'lines.markersize': 5,
                    'legend.fontsize': fsize2,
                    'legend.borderaxespad': 0,
                    'legend.frameon': False,
                    'legend.loc': 'lower right'})

import matplotlib.ticker as ticker


from matplotlib.widgets import Slider, Button
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox

tmp_file_dir = "/media/ubuntu/ssd/sherman/code/RM_tmp_files/"
tmp_files = ["1D_spectra.pkl","comp_dict.pkl","dyn_spectra.pkl","fullburst_dict.pkl","parameters.pkl","rm_vals.pkl"]


#RM Synthesis code
def fit_parabola(x,a,b,c):
    return -a*((x-c)**2) + b

#New significance estimate
def L_sigma(Q,U,timestart,timestop,plot=False,weighted=False,I_w_t_filt=None):


    L0_t = np.sqrt(np.mean(Q,axis=0)**2 + np.mean(U,axis=0)**2)

    if weighted:
        L0_t_w = L0_t*I_w_t_filt
        L_trial_binned = convolve(L0_t,I_w_t_filt)
        sigbin = np.argmax(L_trial_binned)
        noise = np.std(np.concatenate([L_trial_binned[:sigbin],L_trial_binned[sigbin+1:]]))
        print("weighted: " + str(noise))

    else:
        L_trial_cut1 = L0_t[timestart%(timestop-timestart):]
        L_trial_cut = L_trial_cut1[:(len(L_trial_cut1)-(len(L_trial_cut1)%(timestop-timestart)))]
        L_trial_binned = L_trial_cut.reshape(len(L_trial_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(L_trial_binned)
        noise = (np.std(np.concatenate([L_trial_cut[:sigbin],L_trial_cut[sigbin+1:]])))
        print("not weighted: " + str(noise))
    return noise



def command_ionRM(RA,DEC,MJD,datadir,Lat=37.23,Lon=-118.2951):

    #get coordinates
    c = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree)
    idx = c.to_string('hmsdms').index(" ")
    RAstr = c.to_string('hmsdms')[:idx]
    DECstr = c.to_string('hmsdms')[idx+1:]

    if Lon < 0:
        c = SkyCoord(ra=-Lon*u.degree, dec=Lat*u.degree)
        key = "w"
    else:
        c = SkyCoord(ra=Lon*u.degree, dec=Lat*u.degree)
        key = "e"
    idx = c.to_string('dms').index(" ")
    Lonstr = c.to_string('dms')[:idx] + "w"
    Latstr = c.to_string('dms')[idx+1:] + "n"

    date = Time(MJD,format='mjd').isot
    day = Time(MJD,format='mjd').iso
    day = day[:day.index(" ")]
    timeobs = Time(MJD,format='mjd').to_datetime().hour + Time(MJD,format='mjd').to_datetime().minute/60 + Time(MJD,format='mjd').to_datetime().second/3600
    print(timeobs)

    #file_needed = os.system("python ionFR-master2/url_download.py -d " + day + " -t codg")

    output = subprocess.Popen(["python", "ionFR-master2/url_download.py", "-d" , str(day), "-t", "c1pg"], stdout=subprocess.PIPE ).communicate()[0]
    p = output.decode()
    site = p[14:len(p)-p[::-1][1:].index('\n')-2]
    file_needed = p[len(p)-p[::-1][1:].index('\n')-1:-3] 
    #construct command to get file needed
    command = "/media/ubuntu/ssd/sherman/code/ionFR-master2/ionFRM.py" + RAstr + DECstr + " " + Latstr + " " + Lonstr + " " + date + " " + datadir + " " + file_needed
    #print(command)
    #command = ["/media/ubuntu/ssd/sherman/code/ionFR-master2/ionFRM.py", RAstr, DECstr, Latstr, Lonstr, date, self.datadir , file_needed]

    return str(site),str(file_needed),str(command),timeobs

def get_ion_rm(timeobs):
    IonFR_out = np.loadtxt("/media/ubuntu/ssd/sherman/code/IonRM.txt")
    UT_hr = np.array(IonFR_out[:,0],dtype=float)
    ionRMs = np.array(IonFR_out[:,3],dtype=float)
    RMerrs = np.array(IonFR_out[:,4],dtype=float)
    f = interp1d(UT_hr[-24:],ionRMs[-24:],kind="linear",fill_value="extrapolate")
    RM_ion = f(timeobs)
    print(timeobs)
    ferr = interp1d(UT_hr[-24:],RMerrs[-24:],kind="linear",fill_value="extrapolate")
    #print(UT_hr)
    RM_ionerr = ferr(timeobs)
    return RM_ion,RM_ionerr

def RM_plot(RMsnrs,trial_RM,RMsnrstools,trial_RM_tools,RMsnrszoom,trial_RMzoom,RMsnrstoolszoom,trial_RM_tools_zoom,RMsnrs2zoom,init_RM,fine_RM,done_RM,RM,RMerror,threshold=9):
    fig = plt.figure(figsize=(20,24))
    ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 0),colspan=2)
    ax2 = plt.subplot2grid(shape=(2, 2), loc=(1, 0),colspan=2)
    ax2_1 = ax2.twinx()

    ax1.set_xlabel(r'RM ($rad/m^2$)')
    ax1.set_ylabel(r'F($\phi$)')
    ax2.set_xlabel(r'RM ($rad/m^2$)')
    ax2.set_ylabel(r'S/N')
    ax2_1.set_ylabel(r'F($\phi$)')


    #INITIAL SYNTHESIS
    if len(trial_RM) == len(RMsnrs):
        ax1.plot(trial_RM,RMsnrs,label="RM synthesis",color="black")
    if len(trial_RM_tools) == len(RMsnrstools):
        ax1.plot(trial_RM_tools,RMsnrstools,alpha=0.5,label="RM Tools",color="blue")
    ax1.legend(loc="upper right")


    #FINE SYNTHESIS
    if fine_RM or done_RM:
        lns = []
        labs = []
        if len(trial_RMzoom) == len(RMsnrszoom):
            l=ax2_1.plot(trial_RMzoom,RMsnrszoom,label="RM synthesis",color="black")
            lns.append(l[0])
            labs.append(l[0].get_label())
        if len(trial_RM_tools_zoom) == len(RMsnrstoolszoom):
            l=ax2_1.plot(trial_RM_tools_zoom,RMsnrstoolszoom,label="RM Tools",color="blue",alpha=0.5)
            lns.append(l[0])
            labs.append(l[0].get_label())
        if len(trial_RMzoom) == len(RMsnrs2zoom):
            l=ax2.plot(trial_RMzoom,RMsnrs2zoom,label="S/N Method",color="orange",linewidth=2)
            lns.append(l[0])
            labs.append(l[0].get_label())
            
            l=ax2.axvline(RM+RMerror,color="red",label="RM Error",linewidth=2)
            ax2.axvline(RM-RMerror,color="red",linewidth=2)
            lns.append(l)
            labs.append(l.get_label())


        l=ax2.axhline(threshold,color="purple",linestyle="--",label=r'${t}\sigma$ threshold'.format(t=threshold),linewidth=2)
        lns.append(l)
        labs.append(l.get_label())
        ax2_1.legend(lns,labs,loc="upper right")
    return fig



class RM_panel(param.Parameterized):

    error = param.String(default="",label="output/errors")
    I = np.zeros((20480,6144))
    Q = np.zeros((20480,6144))
    U = np.zeros((20480,6144))
    V = np.zeros((20480,6144))

    I_f = np.nan*np.ones(6144)
    Q_f = np.nan*np.ones(6144)
    U_f = np.nan*np.ones(6144)
    V_f = np.nan*np.ones(6144)

    freq_test = [np.zeros(6144)]*4
    n_t = 1
    n_f = 1
    tsamp = -1
    fobj = None
    curr_weights = np.nan*np.ones(20480)
    timestarts = []
    timestops = []
    timestart_in = 0
    timestop_in = 0
    comp_dict = dict()
    fullburst_dict = dict()
    curr_comp = 0

    RA = 0#np.nan
    DEC = 0#np.nan


    #***Initial RM synthesis + Rm tools***#
    init_RM = True
    fine_RM = False
    done_RM = False
    trial_RM = np.linspace(-1e6,1e6,1000)
    trial_RM_tools = copy.deepcopy(trial_RM)
    trial_phi = [0]
    RM1_str = param.String(default="",label=r'Initial RM (rad/m^2)')
    RMerr1_str = param.String(default="",label=r'error (rad/m^2)')
    RM1 = 0.0
    RMerr1 = 0.0
    RM1tools_str = param.String(default="",label=r'Initial RM-Tools RM (rad/m^2)')
    RMerr1tools_str = param.String(default="",label=r'error (rad/m^2)')
    RM1tools = 0.0
    RMerr1tools = 0.0
    RMsnrs1 = np.nan*np.ones(len(trial_RM))
    RMsnrs1tools = np.nan*np.ones(np.min([len(trial_RM),int(1e4)]))
    RMmin = param.String(default="-1000000",label=r'Minimum Trial RM (rad/m^2)')
    RMmax = param.String(default="1000000",label=r'Maximum Trial RM (rad/m^2)')
    numRMtrials = param.String(default="1000",label=r'Number of Trial RMs')


    #***Fine RM synthesis + RM tools + S/N method***#
    numRMtrials_zoom = param.String(default="5000",label=r'Number of Trial RMs (Fine)')
    zoom_window = param.String(default="1000",label=r'RM range above/below initial result (rad/m^2)')
    trial_RM2 = np.linspace(0-1000,0+1000,5000)
    trial_RM_tools_zoom = copy.deepcopy(trial_RM2)
    RMsnrs1zoom = np.nan*np.ones(len(trial_RM2))
    RM1zoom_str = param.String(default="",label=r'Fine RM Synthesis RM (rad/m^2)')
    RMerr1zoom_str = param.String(default="",label=r'error (rad/m^2)')
    RM1zoom = 0.0
    RMerr1zoom = 0.0
    RMtools_zoom_flag = True
    RM1tools_zoom_str = param.String(default="",label=r'Fine RM-Tools RM (rad/m^2)')
    RMerr1tools_zoom_str = param.String(default="",label=r'error (rad/m^2)')
    RM1tools_zoom = 0.0
    RMerr1tools_zoom = 0.0
    RMsnrs1tools_zoom = np.nan*np.ones(np.min([len(trial_RM2),int(1e4)]))
    RMsnrs2zoom = np.nan*np.ones(len(trial_RM2))
    RM2zoom_str = param.String(default="",label=r'Fine S/N Method RM (rad/m^2)')
    RMerr2zoom_str = param.String(default="",label=r'error (rad/m^2)')
    RM2zoom = 0.0
    RMerr2zoom = 0.0

    frb_name = ""
    ids = ""
    nickname = ""
    datadir = ""
    MJD = 0

    RM_FWHM = 0.0

    loaded = False
    def clicked_run(self):
        try:
            #check if there's only one component
            if self.curr_comp == -1 and len(self.comp_dict.keys()) == 1:
                self.error = "Only one component, copying results..."
                t1 = time.time()

                if self.init_RM:
                    self.RM1_str = self.RM1_str + ") " + str(np.around(self.RM1,2))
                    self.RMerr1_str = self.RMerr1_str + ") " + str(np.around(self.RMerr1,2))

                    self.RM1tools_str = self.RM1tools_str + ") " + str(np.around(self.RM1tools,2))
                    self.RMerr1tools_str = self.RMerr1tools_str + ") " + str(np.around(self.RMerr1tools,2))
                
                    self.fullburst_dict["RM1"] = self.RM1
                    self.fullburst_dict["RMerr1"] = self.RMerr1
                    self.fullburst_dict["RMsnrs1"] = copy.deepcopy(self.RMsnrs1)
                    self.fullburst_dict["trial_RM"] = copy.deepcopy(self.trial_RM)

                    self.fullburst_dict["RM1tools"] = self.RM1tools
                    self.fullburst_dict["RMerr1tools"] = self.RMerr1tools
                    self.fullburst_dict["RMsnrs1tools"] = copy.deepcopy(self.RMsnrs1tools)
                    self.fullburst_dict["trial_RM_tools"] = copy.deepcopy(self.trial_RM_tools)

                    self.init_RM = False
                    self.fine_RM = True
                elif self.fine_RM:
                    self.RM1zoom_str = self.RM1zoom_str + ") " + str(np.around(self.RM1zoom,2))
                    self.RMerr1zoom_str = self.RMerr1zoom_str + ") " + str(np.around(self.RMerr1zoom,2))

                    self.fullburst_dict["RM1zoom"] = self.RM1zoom
                    self.fullburst_dict["RMerr1zoom"] = self.RMerr1zoom
                    self.fullburst_dict["RMsnrs1zoom"] = copy.deepcopy(self.RMsnrs1zoom)
                    self.fullburst_dict["trial_RM2"] = copy.deepcopy(self.trial_RM2)


                    if self.RMtools_zoom_flag:
                        self.RM1tools_zoom_str = self.RM1tools_zoom_str + ") " + str(np.around(self.RM1tools_zoom,2))
                        self.RMerr1tools_zoom_str = self.RMerr1tools_zoom_str + ") " + str(np.around(self.RMerr1tools_zoom,2))

                        self.fullburst_dict["RM1tools_zoom"] = self.RM1tools_zoom
                        self.fullburst_dict["RMerr1tools_zoom"] = self.RMerr1tools_zoom
                        self.fullburst_dict["RMsnrs1tools_zoom"] = copy.deepcopy(self.RMsnrs1tools_zoom)
                        self.fullburst_dict["trial_RM_tools_zoom"] = copy.deepcopy(self.trial_RM_tools_zoom)

                    self.RM2zoom_str = self.RM2zoom_str + ") " + str(np.around(self.RM2zoom,2))
                    self.RMerr2zoom_str = self.RMerr2zoom_str + ") " + str(np.around(self.RMerr2zoom,2))

                    self.fullburst_dict["RM2zoom"] = self.RM2zoom
                    self.fullburst_dict["RMerr2zoom"] = self.RMerr2zoom
                    self.fullburst_dict["RMsnrs2zoom"] = copy.deepcopy(self.RMsnrs2zoom)
                    self.fullburst_dict["trial_RM2"] = copy.deepcopy(self.trial_RM2)
                    self.fullburst_dict["RM_FWHM"] = self.RM_FWHM

                    self.fine_RM = False
                    self.done_RM = True

                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to copy RM results"

            elif self.init_RM:
                self.error = "Running initial RM synthesis..."
                t1 = time.time()
                self.trial_RM = np.linspace(float(self.RMmin),float(self.RMmax),int(self.numRMtrials))
                self.RM1,phi1,self.RMsnrs1,self.RMerr1 = dsapol.faradaycal(self.I_f,self.Q_f,self.U_f,self.V_f,self.freq_test,self.trial_RM,self.trial_phi,plot=False,show=False,fit_window=100,err=True)
                
                if (self.RM1_str == "") and (self.curr_comp != -1):
                    self.RM1_str ="(" + str(np.around(self.RM1,2))
                    self.RMerr1_str = "(" + str(np.around(self.RMerr1,2))
                elif self.curr_comp != -1:
                    self.RM1_str = self.RM1_str + " ; " + str(np.around(self.RM1,2))
                    self.RMerr1_str = self.RMerr1_str + " ; " + str(np.around(self.RMerr1,2))
                elif self.curr_comp == -1:
                    self.RM1_str = self.RM1_str + ") " + str(np.around(self.RM1,2))
                    self.RMerr1_str = self.RMerr1_str + ") " + str(np.around(self.RMerr1,2))

                if self.curr_comp != -1:
                    self.comp_dict[self.curr_comp]["RM1"] = self.RM1
                    self.comp_dict[self.curr_comp]["RMerr1"] = self.RMerr1
                    self.comp_dict[self.curr_comp]["RMsnrs1"] = copy.deepcopy(self.RMsnrs1)
                    self.comp_dict[self.curr_comp]["trial_RM"] = copy.deepcopy(self.trial_RM)
                else:
                    self.fullburst_dict["RM1"] = self.RM1
                    self.fullburst_dict["RMerr1"] = self.RMerr1
                    self.fullburst_dict["RMsnrs1"] = copy.deepcopy(self.RMsnrs1)
                    self.fullburst_dict["trial_RM"] = copy.deepcopy(self.trial_RM)


                self.error = "Running initial RM tools..."
                #trial_RM_tools = np.linspace(-1e6,1e6,int(1e4))


                n_off = int(12000/self.n_t)


                Ierr = np.std(self.I[:,:n_off],axis=1)
                Ierr[Ierr.mask] = np.nan
                Ierr = Ierr.data

                Qerr = np.std(self.Q[:,:n_off],axis=1)
                Qerr[Qerr.mask] = np.nan
                Qerr = Qerr.data

                Uerr = np.std(self.U[:,:n_off],axis=1)
                Uerr[Uerr.mask] = np.nan
                Uerr = Uerr.data

                I_fcal_rmtools = self.I_f.data
                I_fcal_rmtools[self.I_f.mask] = np.nan

                Q_fcal_rmtools = self.Q_f.data
                Q_fcal_rmtools[self.Q_f.mask] = np.nan

                U_fcal_rmtools = self.U_f.data
                U_fcal_rmtools[self.U_f.mask] = np.nan

                if len(self.trial_RM) <= 1e4:
                    self.trial_RM_tools = copy.deepcopy(self.trial_RM)
                else:
                    self.error = "Using maximum 1e4 trials for RM tools..."
                    self.trial_RM_tools = np.linspace(np.min(self.trial_RM),np.max(self.trial_RM),int(1e4))
                out=run_rmsynth([self.freq_test[0]*1e6,I_fcal_rmtools,Q_fcal_rmtools,U_fcal_rmtools,Ierr,Qerr,Uerr],phiMax_radm2=np.max(self.trial_RM_tools),dPhi_radm2=np.abs(self.trial_RM_tools[1]-self.trial_RM_tools[0]))

                self.error = "RM Cleaning..."
                out=run_rmclean(out[0],out[1],2)

                self.trial_RM_tools = out[1]["phiArr_radm2"]
                self.RMsnrs1tools = np.abs(out[1]["cleanFDF"])
                self.RM1tools = out[0]["phiPeakPIchan_rm2"]
                self.RMerr1tools = out[0]["dPhiPeakPIchan_rm2"]
                if (self.RM1tools_str == "") and (self.curr_comp != -1):
                    self.RM1tools_str ="(" + str(np.around(self.RM1tools,2))
                    self.RMerr1tools_str = "(" + str(np.around(self.RMerr1tools,2))
                elif self.curr_comp != -1:
                    self.RM1tools_str = self.RM1tools_str + " ; " + str(np.around(self.RM1tools,2))
                    self.RMerr1tools_str = self.RMerr1tools_str + " ; " + str(np.around(self.RMerr1tools,2))
                elif self.curr_comp == -1:
                    self.RM1tools_str = self.RM1tools_str + ") " + str(np.around(self.RM1tools,2))
                    self.RMerr1tools_str = self.RMerr1tools_str + ") " + str(np.around(self.RMerr1tools,2))

                if self.curr_comp != -1:
                    self.comp_dict[self.curr_comp]["RM1tools"] = self.RM1tools
                    self.comp_dict[self.curr_comp]["RMerr1tools"] = self.RMerr1tools
                    self.comp_dict[self.curr_comp]["RMsnrs1tools"] = copy.deepcopy(self.RMsnrs1tools)
                    self.comp_dict[self.curr_comp]["trial_RM_tools"] = copy.deepcopy(self.trial_RM_tools)
                else:
                    self.fullburst_dict["RM1tools"] = self.RM1tools
                    self.fullburst_dict["RMerr1tools"] = self.RMerr1tools
                    self.fullburst_dict["RMsnrs1tools"] = copy.deepcopy(self.RMsnrs1tools)
                    self.fullburst_dict["trial_RM_tools"] = copy.deepcopy(self.trial_RM_tools)

                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to run initial RM synthesis"

                self.init_RM = False
                self.fine_RM = True
            elif self.fine_RM:
                self.error = "Running fine RM synthesis..."
                t1 = time.time()
                
                for i in self.comp_dict.keys():
                    self.timestarts.append(self.comp_dict[i]["timestart"])
                    self.timestops.append(self.comp_dict[i]["timestop"])
                if self.curr_comp == -1:
                    self.timestart_in = np.min(self.timestarts)
                    self.timestop_in = np.max(self.timestops)
                else:
                    self.timestart_in = self.timestarts[-1]
                    self.timestop_in = self.timestops[-1]

                self.trial_RM2 = np.linspace(float(self.RM1)-float(self.zoom_window),float(self.RM1)+float(self.zoom_window),int(self.numRMtrials_zoom))

                tmpRM1zoom,phi1zoom,self.RMsnrs1zoom,tmpRMerr1zoom = dsapol.faradaycal(self.I_f,self.Q_f,self.U_f,self.V_f,self.freq_test,self.trial_RM2,self.trial_phi,plot=False,show=False,fit_window=100,err=True)

                fit_window=50
                oversamps = 5000
                poptpar,pcovpar = curve_fit(fit_parabola,self.trial_RM2[np.argmax(self.RMsnrs1zoom)-fit_window:np.argmax(self.RMsnrs1zoom)+fit_window],self.RMsnrs1zoom[np.argmax(self.RMsnrs1zoom)-fit_window:np.argmax(self.RMsnrs1zoom)+fit_window],p0=[1,1,float(self.RM1)],sigma=1/self.RMsnrs1zoom[np.argmax(self.RMsnrs1zoom)-fit_window:np.argmax(self.RMsnrs1zoom)+fit_window])
                FWHMRM1zoom,tmp,tmp,tmp = peak_widths(self.RMsnrs1zoom,[np.argmax(self.RMsnrs1zoom)])
                noisezoom = L_sigma(self.Q,self.U,self.timestart_in,self.timestop_in,plot=False,weighted=True,I_w_t_filt=self.curr_weights)
                self.RM1zoom = poptpar[2]
                self.RMerr1zoom = FWHMRM1zoom[0]*(self.trial_RM2[1]-self.trial_RM2[0])*noisezoom/(2*np.max(self.RMsnrs1zoom))

                if (self.RM1zoom_str == "") and (self.curr_comp != -1):
                    self.RM1zoom_str ="(" + str(np.around(self.RM1zoom,2))
                    self.RMerr1zoom_str = "(" + str(np.around(self.RMerr1zoom,2))
                elif self.curr_comp != -1:
                    self.RM1zoom_str = self.RM1zoom_str + " ; " + str(np.around(self.RM1zoom,2))
                    self.RMerr1zoom_str = self.RMerr1zoom_str + " ; " + str(np.around(self.RMerr1zoom,2))
                elif self.curr_comp == -1:
                    self.RM1zoom_str = self.RM1zoom_str + ") " + str(np.around(self.RM1zoom,2))
                    self.RMerr1zoom_str = self.RMerr1zoom_str + ") " + str(np.around(self.RMerr1zoom,2))


                if self.curr_comp != -1:
                    self.comp_dict[self.curr_comp]["RM1zoom"] = self.RM1zoom
                    self.comp_dict[self.curr_comp]["RMerr1zoom"] = self.RMerr1zoom
                    self.comp_dict[self.curr_comp]["RMsnrs1zoom"] = copy.deepcopy(self.RMsnrs1zoom)
                    self.comp_dict[self.curr_comp]["trial_RM2"] = copy.deepcopy(self.trial_RM2)
                else:
                    self.fullburst_dict["RM1zoom"] = self.RM1zoom
                    self.fullburst_dict["RMerr1zoom"] = self.RMerr1zoom
                    self.fullburst_dict["RMsnrs1zoom"] = copy.deepcopy(self.RMsnrs1zoom)
                    self.fullburst_dict["trial_RM2"] = copy.deepcopy(self.trial_RM2)


                #check if RM in range for RM tools
                self.RMtools_zoom_flag = np.abs(float(self.RM1zoom)) < 1000
                if self.curr_comp != -1:
                    self.comp_dict[self.curr_comp]["RMtools_zoom_flag"] = self.RMtools_zoom_flag
                else:
                    self.fullburst_dict["RMtools_zoom_flag"] = self.RMtools_zoom_flag
                

                if not self.RMtools_zoom_flag:
                    self.error = "RM out of range for RM tools...skipping"
                else:
                    self.error = "Running fine RM tools..."

                    n_off = int(12000/self.n_t)

                    Ierr = np.std(self.I[:,:n_off],axis=1)
                    Ierr[Ierr.mask] = np.nan
                    Ierr = Ierr.data

                    Qerr = np.std(self.Q[:,:n_off],axis=1)
                    Qerr[Qerr.mask] = np.nan
                    Qerr = Qerr.data

                    Uerr = np.std(self.U[:,:n_off],axis=1)
                    Uerr[Uerr.mask] = np.nan
                    Uerr = Uerr.data

                    I_fcal_rmtools = self.I_f.data
                    I_fcal_rmtools[self.I_f.mask] = np.nan

                    Q_fcal_rmtools = self.Q_f.data
                    Q_fcal_rmtools[self.Q_f.mask] = np.nan

                    U_fcal_rmtools = self.U_f.data
                    U_fcal_rmtools[self.U_f.mask] = np.nan

                    if len(self.trial_RM2) <= 1e4:
                        self.trial_RM_tools_zoom = copy.deepcopy(self.trial_RM2)
                    else:
                        self.error = "Using maximum 1e4 trials for RM tools..."
                        self.trial_RM_tools_zoom = np.linspace(np.min(self.trial_RM2),np.max(self.trial_RM2),int(1e4))
                    out=run_rmsynth([self.freq_test[0]*1e6,I_fcal_rmtools,Q_fcal_rmtools,U_fcal_rmtools,Ierr,Qerr,Uerr],phiMax_radm2=np.max(np.abs(self.trial_RM_tools_zoom)),dPhi_radm2=np.abs(self.trial_RM_tools_zoom[1]-self.trial_RM_tools_zoom[0]))

                    self.error = "RM Cleaning..."
                    out=run_rmclean(out[0],out[1],2)

                    self.trial_RM_tools_zoom = out[1]["phiArr_radm2"]
                    self.RMsnrs1tools_zoom = np.abs(out[1]["cleanFDF"])
                    self.RM1tools_zoom = out[0]["phiPeakPIchan_rm2"]
                    self.RMerr1tools_zoom = out[0]["dPhiPeakPIchan_rm2"]
                    if (self.RM1tools_zoom_str == "") and (self.curr_comp != -1):
                        self.RM1tools_zoom_str ="(" + str(np.around(self.RM1tools_zoom,2))
                        self.RMerr1tools_zoom_str = "(" + str(np.around(self.RMerr1tools_zoom,2))
                    elif self.curr_comp != -1:
                        self.RM1tools_zoom_str = self.RM1tools_zoom_str + " ; " + str(np.around(self.RM1tools_zoom,2))
                        self.RMerr1tools_zoom_str = self.RMerr1tools_zoom_str + " ; " + str(np.around(self.RMerr1tools_zoom,2))
                    elif self.curr_comp == -1:
                        self.RM1tools_zoom_str = self.RM1tools_zoom_str + ") " + str(np.around(self.RM1tools_zoom,2))
                        self.RMerr1tools_zoom_str = self.RMerr1tools_zoom_str + ") " + str(np.around(self.RMerr1tools_zoom,2))

                    if self.curr_comp != -1:
                        self.comp_dict[self.curr_comp]["RM1tools_zoom"] = self.RM1tools_zoom
                        self.comp_dict[self.curr_comp]["RMerr1tools_zoom"] = self.RMerr1tools_zoom
                        self.comp_dict[self.curr_comp]["RMsnrs1tools_zoom"] = copy.deepcopy(self.RMsnrs1tools_zoom)
                        self.comp_dict[self.curr_comp]["trial_RM_tools_zoom"] = copy.deepcopy(self.trial_RM_tools_zoom)
                    else:
                        self.fullburst_dict["RM1tools_zoom"] = self.RM1tools_zoom
                        self.fullburst_dict["RMerr1tools_zoom"] = self.RMerr1tools_zoom
                        self.fullburst_dict["RMsnrs1tools_zoom"] = copy.deepcopy(self.RMsnrs1tools_zoom)
                        self.fullburst_dict["trial_RM_tools_zoom"] = copy.deepcopy(self.trial_RM_tools_zoom)

                    

                self.error = "Running fine S/N method..."


                RM2,phi2,self.RMsnrs2zoom,RMerr2,upp,low,sig,QUnoise = dsapol.faradaycal_SNR(self.I,self.Q,self.U,self.V,self.freq_test,self.trial_RM2,self.trial_phi,self.ibox,self.tsamp,plot=False,n_f=self.n_f,n_t=self.n_t,show=False,err=True,weighted=True,n_off=int(12000/self.n_t),fobj=self.fobj,input_weights=np.trim_zeros(self.curr_weights),timestart_in=self.timestart_in,timestop_in=self.timestop_in)
                self.RM_FWHM = 2*RMerr2

                fit_window=50
                oversamps = 5000
                poptpar,pcovpar = curve_fit(fit_parabola,self.trial_RM2[np.argmax(self.RMsnrs2zoom)-fit_window:np.argmax(self.RMsnrs2zoom)+fit_window],self.RMsnrs2zoom[np.argmax(self.RMsnrs2zoom)-fit_window:np.argmax(self.RMsnrs2zoom)+fit_window],p0=[1,1,RM2],sigma=1/self.RMsnrs2zoom[np.argmax(self.RMsnrs2zoom)-fit_window:np.argmax(self.RMsnrs2zoom)+fit_window])
                self.RM2zoom = poptpar[2]
                self.RMerr2zoom = dsapol.RM_error_fit(np.max(self.RMsnrs2zoom))
                if (self.RM2zoom_str == "") and (self.curr_comp != -1):
                    self.RM2zoom_str ="(" + str(np.around(self.RM2zoom,2))
                    self.RMerr2zoom_str = "(" + str(np.around(self.RMerr2zoom,2))
                elif self.curr_comp != -1:
                    self.RM2zoom_str = self.RM2zoom_str + " ; " + str(np.around(self.RM2zoom,2))
                    self.RMerr2zoom_str = self.RMerr2zoom_str + " ; " + str(np.around(self.RMerr2zoom,2))
                elif self.curr_comp == -1:
                    self.RM2zoom_str = self.RM2zoom_str + ") " + str(np.around(self.RM2zoom,2))
                    self.RMerr2zoom_str = self.RMerr2zoom_str + ") " + str(np.around(self.RMerr2zoom,2))
                

                if self.curr_comp != -1:
                    self.comp_dict[self.curr_comp]["RM2zoom"] = self.RM2zoom
                    self.comp_dict[self.curr_comp]["RMerr2zoom"] = self.RMerr2zoom
                    self.comp_dict[self.curr_comp]["RMsnrs2zoom"] = copy.deepcopy(self.RMsnrs2zoom)
                    self.comp_dict[self.curr_comp]["trial_RM2"] = copy.deepcopy(self.trial_RM2)
                    self.comp_dict[self.curr_comp]["RM_FWHM"] = 2*RMerr2
                else:
                    self.fullburst_dict["RM2zoom"] = self.RM2zoom
                    self.fullburst_dict["RMerr2zoom"] = self.RMerr2zoom
                    self.fullburst_dict["RMsnrs2zoom"] = copy.deepcopy(self.RMsnrs2zoom)
                    self.fullburst_dict["trial_RM2"] = copy.deepcopy(self.trial_RM2)
                    self.fullburst_dict["RM_FWHM"] = 2*RMerr2

                self.fine_RM = False
                self.done_RM = True
    
                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to run fine RM synthesis " + str(self.init_RM) + " " + str(self.fine_RM) + " " + str(self.done_RM)
                
        except Exception as e:
            #self.error = "From clicked_run(): " + str(e) + " " + str(len(self.curr_weights)) + " " + str(len(np.trim_zeros(self.curr_weights))) + " " + str(self.timestop_in-self.timestart_in)
            self.error = "From clicked_run(): " + str(e) + " " + str(len(self.I_f)) + " " + str(len(self.freq_test))
        return

    #plotting
    def clicked_plot(self):
        try:
            suffix="_TESTING"

            #individual components
            if self.curr_comp != -1 and self.done_RM:
                i =  self.curr_comp

                self.error = "Exporting summary plot for Component " + str(i+1) + "..."
                t1 = time.time()

                RMsnrs = (self.RMsnrs1,self.RMsnrs1tools)
                if self.RMtools_zoom_flag:
                    RMzoomsnrs = (self.RMsnrs1zoom,self.RMsnrs1tools_zoom,self.RMsnrs2zoom)
                else:
                    RMzoomsnrs = (self.RMsnrs1zoom,self.RMsnrs2zoom)
                dsapol.RM_summary_plot(self.ids,self.nickname,RMsnrs,RMzoomsnrs,self.RM2zoom,self.RMerr2zoom,self.trial_RM,self.trial_RM2,self.trial_RM_tools,self.trial_RM_tools_zoom,threshold=9,suffix="_PEAK" + str(i+1) + suffix,show=False)

                self.error = "Completed: " + str(np.around(time.time()-t1,2)) + "s to export summary plot to " + self.datadir + self.ids + "_" + self.nickname + "_RMsummary_plot" + "_PEAK" + str(i+1) + suffix + ".pdf"

            #dict_keys(['timestart', 'timestop', 'comp_num', 'buff', 'n_t_weight', 'sf_window_weights', 'ibox', 'mask_start', 'mask_stop', 'weights', 'multipeaks', 'sigflag', 'intL', 'intR', 'I_f', 'Q_f', 'U_f', 'V_f', 'I_f_init', 'Q_f_init', 'U_f_init', 'V_f_init', 'PA_f', 'PA_f_errs', 'PA_f_init', 'PA_f_errs_init', 'PA_pre', 'PAerr_pre', 'T/I_pre', 'T/I_pre_err', 'T/I_pre_snr', 'L/I_pre', 'L/I_pre_err', 'L/I_pre_snr', 'absV/I_pre', 'absV/I_pre_err', 'V/I', 'V/I_err', 'V/I_snr', 'I_snr', 'RM1', 'RMerr1', 'RMsnrs1', 'trial_RM', 'RM1tools', 'RMerr1tools', 'RMsnrs1tools', 'trial_RM_tools', 'RM1zoom', 'RMerr1zoom', 'RMsnrs1zoom', 'trial_RM2', 'RMtools_zoom_flag', 'RM1tools_zoom', 'RMerr1tools_zoom', 'RMsnrs1tools_zoom', 'trial_RM_tools_zoom', 'RM2zoom', 'RMerr2zoom', 'RMsnrs2zoom'])
            #all components and full burst
            elif self.curr_comp == -1 and self.done_RM:
                for i in range(len(self.comp_dict.keys())):
                    self.error = "Exporting summary plot for Component " + str(i+1) + "..."
                    t1 = time.time()

                    RMsnrs = (self.comp_dict[i]["RMsnrs1"],self.comp_dict[i]["RMsnrs1tools"])
                    if self.comp_dict[i]["RMtools_zoom_flag"]:
                        RMzoomsnrs = (self.comp_dict[i]["RMsnrs1zoom"],self.comp_dict[i]["RMsnrs1tools_zoom"],self.comp_dict[i]["RMsnrs2zoom"])
                    else:
                        RMzoomsnrs = (self.comp_dict[i]["RMsnrs1zoom"],self.comp_dict[i]["RMsnrs2zoom"])
                    dsapol.RM_summary_plot(self.ids,self.nickname,RMsnrs,RMzoomsnrs,self.comp_dict[i]["RM2zoom"],self.comp_dict[i]["RMerr2zoom"],self.comp_dict[i]["trial_RM"],self.comp_dict[i]["trial_RM2"],self.comp_dict[i]["trial_RM_tools"],self.comp_dict[i]["trial_RM_tools_zoom"],threshold=9,suffix="_PEAK" + str(i+1) + suffix,show=False)

                    self.error = "Completed: " + str(np.around(time.time()-t1,2)) + "s to export summary plot to " + self.datadir + self.ids + "_" + self.nickname + "_RMsummary_plot" + "_PEAK" + str(i+1) + suffix + ".pdf"

                self.error = "Exporting summary plot..."
                t1 = time.time()

                RMsnrs = (self.RMsnrs1,self.RMsnrs1tools)
                if self.RMtools_zoom_flag:
                    RMzoomsnrs = (self.RMsnrs1zoom,self.RMsnrs1tools_zoom,self.RMsnrs2zoom)
                else:
                    RMzoomsnrs = (self.RMsnrs1zoom,self.RMsnrs2zoom)
                dsapol.RM_summary_plot(self.ids,self.nickname,RMsnrs,RMzoomsnrs,self.RM2zoom,self.RMerr2zoom,self.trial_RM,self.trial_RM2,self.trial_RM_tools,self.trial_RM_tools_zoom,threshold=9,suffix=suffix,show=False)

                self.error = "Completed: " + str(np.around(time.time()-t1,2)) + "s to export summary plot to " + self.datadir + self.ids + "_" + self.nickname + "_RMsummary_plot" + suffix + ".pdf"




        except Exception as e:
            self.error = "From clicked_plot(): " + str(e)
            self.error = str(self.comp_dict[0].keys())
        return

    def clicked_update(self):
        self.error = "Forcing Display Update..."
        self.param.trigger('update_display')
        return

    #get galactic RM
    def clicked_galrm(self):
        try:
            self.error = "Computing Galactic RM..."
            t1 = time.time()
            self.RM_gal,self.RM_galerr = get_rm(radec=(self.RA,self.DEC),filename="/home/ubuntu/faraday2020v2.hdf5")
            self.RM_gal_str = str(np.around(self.RM_gal,2))
            self.RM_galerr_str = str(np.around(self.RM_galerr,2))
            self.got_rm_gal = True
            self.error = "Completed: " + str(np.around(time.time()-t1,2)) + " s to compute galactic RM"
            self.param.trigger('galrm')
        except Exception as e:
            self.error = "From clicked_galrm(): " + str(e)
            
        return

    #get ionospheric RM
    def clicked_ionrm(self):
        try:
            if self.datadir == "":
                self.error = "Please load FRB data prior to computing ionospheric RM"

            else:
                self.error = "Getting ionospheric RM command..."
                t1 = time.time()
                site,ion_file,command,timeobs = command_ionRM(self.RA,self.DEC,self.MJD,self.datadir)
                dir_list = os.listdir(self.datadir)

                if ion_file not in dir_list:
                    self.error = " To get ionospheric RM, download " + site + ", unzip, and place in directory " + self.datadir
                else:
                    self.error = "Computing ionospheric RM..."

                    os.system(command)
                    self.RM_ion,self.RM_ionerr = get_ion_rm(timeobs)
                    self.RM_ion_str = str(np.around(self.RM_ion,2))
                    self.RM_ionerr_str = str(np.around(self.RM_ionerr,2))
                    self.got_rm_ion = True
                    self.error = "Completed: " + str(np.around(time.time()-t1,2)) + " s to compute ion RM"
            self.param.trigger('ionrm')

        except Exception as e:
            self.error = "From clicked_ionrm(): " + str(e)

        return




    update_display = param.Action(clicked_update,label="Force Display Update")
    run = param.Action(clicked_run,label="Run")
    exportplot = param.Action(clicked_plot,label='Export Summary Plot')
    
    #***Galactic and Ionospheric RM***#
    RM_gal_str = param.String(default="",label=r'Galactic RM (rad/m^2)')
    RM_galerr_str = param.String(default="",label=r'error (rad/m^2)')
    RM_gal = 0
    RM_galerr = 0
    got_rm_gal = False

    RM_ion_str = param.String(default="",label=r'Ionospheric RM (rad/m^2)')
    RM_ionerr_str = param.String(default="",label=r'error (rad/m^2)')
    RM_ion = 0
    RM_ionerr = 0
    got_rm_ion = False

    galrm = param.Action(clicked_galrm,label='Compute Galactic RM')
    ionrm = param.Action(clicked_ionrm,label='Compute Ionospheric RM')

    """
    def gal_rm_panel(self):
        #self.error = "Computing Galactic RM..." + str(self.got_rm_gal)
        t1 = time.time()
        self.RM_gal,self.RM_galerr = get_rm(radec=(self.RA,self.DEC),filename="/home/ubuntu/faraday2020v2.hdf5")
        self.RM_gal_str = str(np.around(self.RM_gal,2))
        self.RM_galerr_str = str(np.around(self.RM_galerr,2))
        #self.error = "Completed: " + str(np.around(time.time()-t1,2)) + " s to compute galactic RM"
        self.got_rm_gal = True
        #self.clicked_update()
        return
    def ion_rm_panel(self):
        site,ion_file,command,timeobs = command_ionRM(self.RA,self.DEC,self.fobj,self.datadir)
        dir_list = os.listdir(self.datadir)
        #self.error = site + " " +ion_file
               
        #if ion_file not in dir_list:
        #    self.error = "To get ionospheric RM, download " + site + ", unzip, and place in directory " + self.datadir
        #    #pass
        if ion_file in dir_list:
            #self.error = "Computing Ionospheric RM..." + str(self.got_rm_ion)
            t1 = time.time()
            os.system(command)
            self.RM_ion,self.RM_ionerr = get_ion_rm(timeobs)
            self.RM_ion_str = str(np.around(self.RM_ion,2))
            self.RM_ionerr_str = str(np.around(self.RM_ionerr,2))
            #self.error = "Completed: " + str(np.around(time.time()-t1,2)) + " s to compute ion RM " + str(self.got_rm_ion) + str(self.got_rm_gal)
            self.got_rm_ion = True
            #self.clicked_update()
        return

    """
    """
    #check flage in ready.pkl
    def check_ready_flag(self):
        try:
            f = open(tmp_file_dir + "ready.pkl","rb")
            x = pkl.load(f)
            f.close()
            return x["ready"]
        except Exception as e:
            self.error = "From check_ready_flag(): " + str(e)
            return
    """

    #setup data
    def clicked_pkl_load(self):
        try:
            self.error = "Loading Data from pkl files..."
            t1 = time.time()
            #get dynamic spectra
            f = open(tmp_file_dir + "dyn_spectra.pkl","rb")
            dyn_spectra_dict = pkl.load(f)
            f.close()
            self.I = dyn_spectra_dict["I"]
            self.Q = dyn_spectra_dict["Q"]
            self.U = dyn_spectra_dict["U"]
            #self.V = dyn_spectra_dict["V"]

            #get 1D spectra
            f = open(tmp_file_dir + "1D_spectra.pkl","rb")
            spectra_dict = pkl.load(f)
            f.close()
            self.I_f = spectra_dict["I_f"]
            self.Q_f = spectra_dict["Q_f"]
            self.U_f = spectra_dict["U_f"]
            #self.V_f = spectra_dict["V_f"]
            self.freq_test = spectra_dict["freq_test"]


            #get parameters
            f = open(tmp_file_dir + "parameters.pkl","rb")
            parameters_dict = pkl.load(f)
            f.close()
            self.curr_weights = parameters_dict["curr_weights"]            
            self.n_t = parameters_dict["n_t"]
            self.n_f = parameters_dict["n_f"]
            self.curr_comp = parameters_dict["curr_comp"]
            self.ibox = parameters_dict["ibox"]
            self.tsamp = parameters_dict["tsamp"]
            self.nickname = parameters_dict["nickname"]
            self.ids = parameters_dict["ids"]
            self.datadir = parameters_dict["datadir"]
            self.RA = parameters_dict["RA"]
            self.DEC = parameters_dict["DEC"]
            self.MJD = parameters_dict["MJD"]
            self.frb_name = parameters_dict["frb_name"]

            #get comp_dict
            f = open(tmp_file_dir + "comp_dict.pkl","rb")
            self.comp_dict = pkl.load(f)
            f.close()

            #get fullburst dict
            f = open(tmp_file_dir + "fullburst_dict.pkl","rb")
            self.fullburst_dict = pkl.load(f)
            f.close()

            self.init_RM = True
            self.fine_RM = False
            self.done_RM = False
            #self.loaded = True
            self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to load pkl data"
            self.param.trigger('RMdata_init')
        except Exception as e:
            self.error = "From clicked_pkl_load(): " + str(e)
        return

    RMdata_init = param.Action(clicked_pkl_load,label="Initialize Data")


    #function for saving rm data back to pickle files
    def clicked_pkl_save(self):
        try:
            self.error = "Saving rm data to pkl files..."
            t1 = time.time()
            #write galactic and ionospheric rm
            rm_dict = dict()
            rm_dict['RM_gal'] = self.RM_gal
            rm_dict['RM_galerr'] = self.RM_galerr
            rm_dict['RM_ion'] = self.RM_ion
            rm_dict['RM_ionerr'] = self.RM_ionerr
            f = open(tmp_file_dir + "rm_vals.pkl","wb")
            pkl.dump(rm_dict,f)
            f.close()

            #update comp dict
            f = open(tmp_file_dir + "comp_dict.pkl","wb")
            pkl.dump(self.comp_dict,f)
            f.close()

            #update full burst dict
            f = open(tmp_file_dir + "fullburst_dict.pkl","wb")
            pkl.dump(self.fullburst_dict,f)
            f.close()

            self.error = "Complete: " + str(np.around(time.time()-t1)) + " to save to pkl files"

            self.param.trigger('RMdata_out')
        except Exception as e:
            self.error = "From clicked_pkl_load(): " + str(e)
        return
    RMdata_out = param.Action(clicked_pkl_save,label="Return to Pol Analysis")

    #clear temp data
    def clicked_clear(self):
        try:
            self.error = "Clearing pkl cache files..."
            t1 = time.time()
            for fname in tmp_files:
                f = open(tmp_file_dir + fname,"wb")
                pkl.dump(dict(),f)
                f.close()
            self.error = "Complete: " + str(np.around(time.time()-t1)) + " s to clear pkl files"
            self.param.trigger('clear_data')
        except Exception as e:
            self.error = "From clicked_clear(): " + str(e)
        return
    clear_data = param.Action(clicked_clear,label="Clear Cached Pkl Data")

    #***VIEWING MODULE***#
    def view(self):
        try:
            #use data from tmp files if ready flag is set
            #if self.check_ready_flag() and (not self.loaded):
            #    self.load_pkl_data()



            #update trial RM
            self.trial_RM = np.linspace(float(self.RMmin),float(self.RMmax),int(self.numRMtrials))
            if self.RM1 != "":
                self.trial_RM2 = np.linspace(float(self.RM1)-float(self.zoom_window),float(self.RM1)+float(self.zoom_window),int(self.numRMtrials_zoom))
            if self.RM2zoom == "" or self.RMerr2zoom=="":
                rm = np.nan
                rmerr = np.nan
            else:
                rm = float(self.RM2zoom)
                rmerr = float(self.RMerr2zoom)
       
            if self.frb_name != "":
                self.ids = self.frb_name[:10]#"230307aaao"#"220207aabh"#"221029aado"
                self.nickname = self.frb_name[11:]#"phineas"#"zach"#"mifanshan"
                self.datadir = "/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ self.ids + "_" + self.nickname + "/"
            """
            #get galactic rm
            if self.calibrated_for_gal_ion_rm and (self.got_rm_gal==False):# and self.got_rm_ion):
                self.gal_rm_panel()
            if self.calibrated_for_gal_ion_rm and (self.got_rm_ion==False):
                self.ion_rm_panel()
            """
            """
            #get galactic rm
            if self.calibrated_for_gal_ion_rm and (self.got_rm_gal==False):# and self.got_rm_ion):
                
                self.error = "Computing Galactic RM..." + str(self.got_rm_gal)
                t1 = time.time()
                self.RM_gal,self.RM_galerr = get_rm(radec=(self.RA,self.DEC),filename="/home/ubuntu/faraday2020v2.hdf5")
                self.RM_gal_str = str(np.around(self.RM_gal,2))
                self.RM_galerr_str = str(np.around(self.RM_galerr,2))
                self.error = "Completed: " + str(np.around(time.time()-t1,2)) + " s to compute galactic RM"
                self.got_rm_gal = True
    
            #get ionospheric rm
            if self.calibrated_for_gal_ion_rm and (self.got_rm_ion==False):
                site,ion_file,command,timeobs = command_ionRM(self.RA,self.DEC,self.fobj,self.datadir)
                dir_list = os.listdir(self.datadir)
                #self.error = site + " " +ion_file
               
                if ion_file not in dir_list:
                    self.error = "To get ionospheric RM, download " + site + ", unzip, and place in directory " + self.datadir
                else:
                    self.error = "Computing Ionospheric RM..." + str(self.got_rm_ion)
                    t1 = time.time()
                    os.system(command)
                    self.RM_ion,self.RM_ionerr = get_ion_rm(timeobs)
                    self.RM_ion_str = str(np.around(self.RM_ion,2))
                    self.RM_ionerr_str = str(np.around(self.RM_ionerr,2))
                    self.error = "Completed: " + str(np.around(time.time()-t1,2)) + " s to compute ion RM " + str(self.got_rm_ion) + str(self.got_rm_gal)
                    self.got_rm_ion = True
            elif self.got_rm_gal and self.got_rm_ion:
                self.calibrated_for_gal_ion_rm = False
                self.error = "all done " + str(self.calibrated_for_gal_ion_rm) + str(self.got_rm_gal) + str(self.got_rm_ion)
            """
            return RM_plot(self.RMsnrs1,self.trial_RM,self.RMsnrs1tools,self.trial_RM_tools,self.RMsnrs1zoom,self.trial_RM2,self.RMsnrs1tools_zoom,self.trial_RM_tools_zoom,self.RMsnrs2zoom,self.init_RM,self.fine_RM,self.done_RM,rm,rmerr)
        except Exception as e:
            self.error = "From view(): " + str(e) + " " + str(len(self.trial_RM2)) + " " + str(len(self.RMsnrs2zoom))
            return
