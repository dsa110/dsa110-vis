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

import dsarp_interface

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

def pol_plot(I_t,Q_t,U_t,V_t,PA_t,PA_t_errs,I_f,Q_f,U_f,V_f,PA_f,PA_f_errs,comp_dict,freq_test,I_t_weights,timestart,timestop,n_t=1,n_f=1,buff_L=1,buff_R=1,n_t_weight=1,sf_window_weights=1,width_native=1,lo=1,comp_width=100,comp_choose_on=False,fixed_comps=[],filt_weights_on=False,comp_num=0,freq_samp_on=False,wait=False,multipeaks=False,height=5,intLs=[],intRs=[],maskPA=False,finished=False,maxcomps=4):
    
    fig = plt.figure(figsize=(20,24))
    top = fig.add_gridspec(13,2,hspace=0,top=0.98)
    axPA = fig.add_subplot(top[0,:])
    axPA.set_ylim(-1.1*180,1.1*180)
    ax = fig.add_subplot(top[1:3,:])
    axPA.xaxis.set_major_locator(ticker.NullLocator())
    """
    axPA = plt.subplot2grid(shape=(9, 2), loc=(0, 0),colspan=2)
    axPA.set_ylim(-1.1*180,1.1*180)
    ax = plt.subplot2grid(shape=(9, 2), loc=(1, 0),colspan=2,rowspan=2)
    #fig.subplots_adjust(hspace=0)
    axPA.xaxis.set_major_locator(ticker.NullLocator())
    #plt.subplots_adjust(hspace=0.0)
    """
    faxs = []
    paxs = []
    #bottom = fig.add_gridspec(9,2,hspace=1.5)
    #print("check1")
    for i in range(1,maxcomps+1):
        #if filt_weights_on or freq_samp_on:
        #    plt.text(0.1,0.1,comp_dict[i]["T/I_pre_"])
        #faxs.append(plt.subplot2grid(shape=(9, 2), loc=( (i-1)//2+ 3 + (i-1)//2, (i-1)%2),rowspan=2))
        
        row = (i-1)//2 + 3 + 1 + 2*((i-1)//2)#(i-1)//2+ 3 + (i-1)//2
        col= (i-1)%2
        bottom = fig.add_gridspec(13,2,hspace=0.0,top=0.89 - 0.07*((row-1)//3 -1),bottom=0.14-0.07*((row-1)//3 - 1),wspace=0.25)# - (0.5*((row-1)//3 - 1)))
        fax_i = fig.add_subplot(bottom[row:row+2, col])
        faxs.append(fax_i)

        pax_i = fig.add_subplot(bottom[row-1,col])
        paxs.append(pax_i)
        pax_i.set_ylim(-1.1*180,1.1*180)
        

        paxs[i-1].set_xlim(np.min(freq_test[0]),np.max(freq_test[0]))
        paxs[i-1].set_ylabel("deg.")
        faxs[i-1].set_xlim(np.min(freq_test[0]),np.max(freq_test[0]))
        faxs[i-1].set_ylabel("S/N")
        paxs[i-1].set_title("Component #" + str(i))
        faxs[-1].set_xlabel("Freq. (MHz)")
        #print("check1.5")
        paxs[i-1].xaxis.set_major_locator(ticker.NullLocator())
    #print("check2")
    bottom = fig.add_gridspec(13,2,hspace=0.0,top=0.76,bottom=0.0)
    #faxs.append(plt.subplot2grid(shape=(9, 1), loc=(7, 0),colspan=2,rowspan=2))
    pax_i = fig.add_subplot(bottom[9,:])
    fax_i = fig.add_subplot(bottom[10:12,:])
    paxs.append(pax_i)
    faxs.append(fax_i)
    pax_i.set_ylim(-1.1*180,1.1*180)

    paxs[-1].set_xlim(np.min(freq_test[0]),np.max(freq_test[0]))
    faxs[-1].set_xlim(np.min(freq_test[0]),np.max(freq_test[0]))
    paxs[-1].set_ylabel("deg.")
    faxs[-1].set_ylabel("S/N")
    paxs[-1].set_title("All Components")
    faxs[-1].set_xlabel("Frequency (MHz)")
    paxs[i-1].xaxis.set_major_locator(ticker.NullLocator())

    #plot filterweights always
    if (filt_weights_on or freq_samp_on or finished) and (not wait):
        #ax.text(0,20,str(wait))
        ax.plot((I_t_weights*np.max(I_t)/np.max(I_t_weights))[timestart:timestop],label="weights",linewidth=4,color="purple",alpha=0.75)
        peak = np.argmax(I_t_weights[timestart:timestop])
        ax.set_xlim(int(peak - (1e-3)/(32.7e-6)),int(peak + (1e-3)/(32.7e-6)))
        axPA.set_xlim(int(peak - (1e-3)/(32.7e-6)),int(peak + (1e-3)/(32.7e-6)))
        #ax1.set_xlim(int(peak - (1e-3)/(32.7e-6)),int(peak + (1e-3)/(32.7e-6)))
    else:
        peak = np.argmax(I_t[timestart:timestop])
        ax.set_xlim(int(peak - (1e-3)/(32.7e-6)),int(peak + (1e-3)/(32.7e-6)))
        axPA.set_xlim(int(peak - (1e-3)/(32.7e-6)),int(peak + (1e-3)/(32.7e-6)))

    if filt_weights_on:
        if multipeaks and height < np.max(I_t):
            pks,props = find_peaks((I_t_weights*np.max(I_t)/np.max(I_t_weights))[timestart:timestop],height=height)
            FWHM,heights,intL,intR = peak_widths((I_t_weights*np.max(I_t)/np.max(I_t_weights))[timestart:timestop],pks)
            intL = int(intL[0])
            intR = int(intR[-1])
            ax.axhline(height,color="red",linestyle="--")
            axPA.errorbar(np.arange(intL,intR),(180/np.pi)*PA_t[timestart:timestop][intL:intR],yerr=(180/np.pi)*PA_t_errs[timestart:timestop][intL:intR],fmt='o',label="PA",color="blue",markersize=6,linewidth=2)
        elif multipeaks:
            intL = np.nan
            intR = np.nan
            ax.axhline(height,color="red",linestyle="--")
            #axPA.errorbar(np.arange(intL,intR),(180/np.pi)*PA_t[intL:intR],yerr=(180/np.pi)*PA_t_errs[intL:intR],fmt='o',label="Intrinsic PPA",color="blue",markersize=6,linewidth=2)
        else:
            FWHM,heights,intL,intR = peak_widths(I_t_weights[timestart:timestop],[np.argmax(I_t_weights[timestart:timestop])])
            intL = int(intL)
            intR = int(intR)
            axPA.errorbar(np.arange(intL,intR),(180/np.pi)*PA_t[timestart:timestop][intL:intR],yerr=(180/np.pi)*PA_t_errs[timestart:timestop][intL:intR],fmt='o',label="PA",color="blue",markersize=6,linewidth=2)
        ax.axvline(intL,color="red",linestyle="--")
        ax.axvline(intR,color="red",linestyle="--")
        




    elif freq_samp_on or finished:
        intL = np.min(intLs)
        intR = np.max(intRs)
        
        if maskPA:
            tstweight = (I_t_weights*np.max(I_t)/np.max(I_t_weights))[timestart:timestop][intL:intR]
            ax.plot(np.arange(intL,intR),tstweight,color="black")
            axPA.errorbar(np.arange(intL,intR)[(I_t_weights[timestart:timestop])[intL:intR] > 0.005],(180/np.pi)*PA_t[timestart:timestop][intL:intR][(I_t_weights[timestart:timestop])[intL:intR] > 0.005],yerr=(180/np.pi)*PA_t_errs[timestart:timestop][intL:intR][(I_t_weights[timestart:timestop])[intL:intR] > 0.005],fmt='o',label="PA",color="blue",markersize=6,linewidth=2)
        else:
            axPA.errorbar(np.arange(intL,intR),(180/np.pi)*PA_t[timestart:timestop][intL:intR],yerr=(180/np.pi)*PA_t_errs[timestart:timestop][intL:intR],fmt='o',label="PA",color="blue",markersize=6,linewidth=2)
    #else:
    #    axPA.errorbar(np.arange(0,timestop-timestart),PA_t[timestart:timestop],yerr=PA_t_errs[timestart:timestop],fmt='o',label="Intrinsic PPA",color="blue",markersize=6,linewidth=2)
    
    #print("check3")
    if filt_weights_on:
        for i in range(len(comp_dict.keys())):
            if "I_f" in comp_dict[i].keys() and "PA_f" in comp_dict[i].keys() and "PA_f_errs" in comp_dict[i].keys():
                #ax.text(100,25,"test")
                #ax.text(100,25,str(i))
                #ax.text(100,25,str("I_f" in comp_dict[i].keys()))
                print(comp_dict[i].keys())
                print(i)

                faxs[i].plot(freq_test[0],comp_dict[i]["I_f"],label="I")
                faxs[i].plot(freq_test[0],comp_dict[i]["Q_f"],label="Q")
                faxs[i].plot(freq_test[0],comp_dict[i]["U_f"],label="U")
                faxs[i].plot(freq_test[0],comp_dict[i]["V_f"],label="V")

                paxs[i].errorbar(freq_test[0],(180/np.pi)*(comp_dict[i]["PA_f"]),(180/np.pi)*(comp_dict[i]["PA_f_errs"]),fmt='o',label="PA",color="blue",markersize=6,linewidth=2)

    else:
        ax.set_xlim(0,timestop-timestart)
        #ax1.set_xlim(0,timestop-timestart)
        axPA.set_xlim(0,timestop-timestart)

    if wait:
        ax.set_xlim(0,timestop-timestart)
        axPA.set_xlim(0,timestop-timestart)


    axPA.set_ylabel("degrees")
    ax.plot(I_t[timestart:timestop],label="I")
    ax.plot(Q_t[timestart:timestop],label="Q")
    ax.plot(U_t[timestart:timestop],label="U")
    ax.plot(V_t[timestart:timestop],label="V")
    ax.legend(loc="upper right")
    axPA.legend(loc="upper right")

    #print("check4")
    if comp_choose_on:
        ax.axvline(lo,color='red')
        ax.axvline(lo+comp_width,color='red')
        ax.axvspan(lo, lo+comp_width, alpha=0.5, color='red')

        for i in range(len(fixed_comps)):
            ax.axvline(fixed_comps[i][0],color="green")
            ax.axvline(fixed_comps[i][1],color="green")
            ax.axvspan(fixed_comps[i][0], fixed_comps[i][1], alpha=0.5, color='green')
        #print("check4.5")



    ax.set_xlabel("Time Sample ({a} $\mu s$)".format(a=np.around(n_t*32.7,1)))
    #axPA.set_xlabel("Time Sample ({a} $\mu s$)".format(a=np.around(n_t*32.7,1)))
    ax.set_ylabel("S/N")
    #plt.show()

    if freq_samp_on or finished:# and len(freq_test[0]) == len(PA_f):
        faxs[-1].plot(freq_test[0],I_f,label="I")
        faxs[-1].plot(freq_test[0],Q_f,label="Q")
        faxs[-1].plot(freq_test[0],U_f,label="U")
        faxs[-1].plot(freq_test[0],V_f,label="V")
        paxs[-1].errorbar(freq_test[0],(180/np.pi)*PA_f,yerr=(180/np.pi)*PA_f_errs,label="PA",color="blue",markersize=6,linewidth=2,fmt='o')


        for i in range(len(comp_dict.keys())):
            faxs[i].plot(freq_test[0],comp_dict[i]["I_f"],label="I")
            faxs[i].plot(freq_test[0],comp_dict[i]["Q_f"],label="Q")
            faxs[i].plot(freq_test[0],comp_dict[i]["U_f"],label="U")
            faxs[i].plot(freq_test[0],comp_dict[i]["V_f"],label="V")
            paxs[i].errorbar(freq_test[0],(180/np.pi)*(comp_dict[i]["PA_f"]),yerr=(180/np.pi)*(comp_dict[i]["PA_f_errs"]),fmt='o',label="PA",color="blue",markersize=6,linewidth=2)


    #print("check5")
    #fig.tight_layout()
    
    
    #axPA.xaxis.set_major_locator(ticker.NullLocator())
    plt.subplots_adjust(hspace=1.5)
    return fig




class pol_panel(param.Parameterized):



    #param linked to dropdown menu
    #frb_submitted = param.String(default="")#param.Integer(default=0,bounds=(0,8),label=r'clicks')
    #frb_loaded = False
    cal_name = param.String(default="")
    frb_name = param.String(default="")
    error = param.String(default="",label="output/errors")
    error2 = param.String(default="",label="tmp")

    I_init = np.zeros((20480,6144))
    Q_init = np.zeros((20480,6144))
    U_init = np.zeros((20480,6144))
    V_init = np.zeros((20480,6144))


    I = np.zeros((20480,6144))
    Q = np.zeros((20480,6144))
    U = np.zeros((20480,6144))
    V = np.zeros((20480,6144))

    I_RMcal = np.zeros((20480,6144))
    Q_RMcal = np.zeros((20480,6144))
    U_RMcal = np.zeros((20480,6144))
    V_RMcal = np.zeros((20480,6144))

    I_RMcal_init = np.zeros((20480,6144))
    Q_RMcal_init = np.zeros((20480,6144))
    U_RMcal_init = np.zeros((20480,6144))
    V_RMcal_init = np.zeros((20480,6144))


    I_t_init = np.zeros(20480)
    Q_t_init = np.zeros(20480)
    U_t_init = np.zeros(20480)
    V_t_init = np.zeros(20480)

    PA_t_init = np.zeros(20480)
    PA_t_errs_init = np.zeros(20480)

    fobj = None#fobj#None
    timeaxis = np.zeros(20480)#timeaxis#np.zeros(20480)
    freq_test_init = [np.zeros(6144)]*4
    gxx = np.zeros(6144)
    gyy = np.zeros(6144)
    ibeam = param.Integer(default=0,bounds=(0,250),label='ibeam')
    RA = 0
    DEC = 0
    ParA = -1
    wav_test = [np.zeros(6144)]*4

    ids = ""
    nickname = ""
    datadir = ""
    rm_threshold = 9 #sigma

    n_t = param.Integer(default=1,bounds=(1,128),label=r'n_t')
    n_t_prev = 1
    buff_L = param.Integer(default=1,bounds=(0,20),label=r'left buffer')
    buff_R = param.Integer(default=1,bounds=(0,20),label=r'right buffer')
    #n_t_weight = param.Integer(default=1,bounds=(1,10),label=r'n_tw')
    log_n_t_weight = param.Integer(default=0,bounds=(0,10),label=r'log(n_tw)')
    n_t_weight = 1
    sf_window_weights = param.Integer(default=1,bounds=(1,19),step=2,label=r'sf window')
    ibox = param.Integer(default=1,bounds=(1,128),label=r'ibox')
    comp_width = param.Integer(default=10,bounds=(1,128),label=r'mask width')
    log_n_f = param.Integer(default=0,bounds=(0,10),label=r'log2(n_f)')
    n_f = 1
    n_f_prev = 1
    n_f_root = 1
    n_t_root = 1
    loaded = False
    calibrated = False
    saved = False
    rmcalibrated_all = False

    MJD = 0
    RM_gal = np.nan
    RM_galerr = np.nan
    RM_ion = np.nan
    RM_ionerr = np.nan
    
    #testidx = param.Integer(default=0,bounds=(0,2),label="test")

    #polarization display

    STEP = 0

    I_t = np.nan*np.ones(20480)
    Q_t = np.nan*np.ones(20480)
    U_t = np.nan*np.ones(20480)
    V_t = np.nan*np.ones(20480)
    
    PA_t = np.nan*np.ones(20480)
    PA_t_errs = np.nan*np.ones(20480)

    I_f_init = np.nan*np.ones(6144)#I.mean(1)#np.zeros(I.shape[0])
    Q_f_init = np.nan*np.ones(6144)#Q.mean(1)#np.zeros(Q.shape[0])
    U_f_init = np.nan*np.ones(6144)#U.mean(1)#np.zeros(U.shape[0])
    V_f_init = np.nan*np.ones(6144)#V.mean(1)#np.zeros(V.shape[0])

    PA_f_init = np.nan*np.ones(6144)
    PA_f_errs_init = np.nan*np.ones(6144)

    I_f = np.nan*np.ones(len(I_f_init))
    Q_f = np.nan*np.ones(len(I_f_init))
    U_f = np.nan*np.ones(len(I_f_init))
    V_f = np.nan*np.ones(len(I_f_init))
    PA_f = np.nan*np.ones(len(I_f_init))
    PA_f_errs = np.nan*np.ones(len(I_f_init))
    freq_test = copy.deepcopy(freq_test_init)
    #@param.depends('frb_mid', watch=True

    load_RMcal = param.Boolean(False)
    load_polcal =param.Boolean(False)
    def load_FRB(self):
        try:
            #if self.error=="Loading FRB...":
            #self.error2 = str(self.I.shape)
            self.error = "Loading FRB predownsampled by " + str(self.n_t) + " in time, " + str(self.n_f) + " in frequency..."
            t1 = time.time()
            self.ids = self.frb_name[:10]#"230307aaao"#"220207aabh"#"221029aado"
            self.nickname = self.frb_name[11:]#"phineas"#"zach"#"mifanshan"
            self.datadir = "/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ self.ids + "_" + self.nickname + "/"
            #ibeam = 218
            #caldate="22-12-18"
            #self.frb_name = "Loading " + ids + "_" + nickname + " ..."
            #self.view()
            self.error2 = "getting 2D array"
            suff = ""
            if (not self.load_polcal) and (not self.load_RMcal):
                suff = "_dev"
            else:
                suff += "_" + self.nickname

            if self.load_polcal:
                suff += "_polcal"
            if self.load_RMcal:
                suff += "_RMcal"
            (self.I,self.Q,self.U,self.V,self.fobj,self.timeaxis,self.freq_test_init,self.wav_test) = dsapol.get_stokes_2D(self.datadir,self.ids + suff,20480,n_t=self.n_t,n_f=self.n_f,n_off=int(12000//self.n_t),sub_offpulse_mean=True,dtype=np.float16)
            self.error2 = "done, copying 2D array"
            self.I_init,self.Q_init,self.U_init,self.V_init = copy.deepcopy(self.I),copy.deepcopy(self.Q),copy.deepcopy(self.U),copy.deepcopy(self.V)
            self.freq_test = copy.deepcopy(self.freq_test_init)
            self.error2 = "done, getting time series"
            (self.I_t_init,self.Q_t_init,self.U_t_init,self.V_t_init) = dsapol.get_stokes_vs_time(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,n_off=int(12000//self.n_t),plot=False,show=True,normalize=True,buff=1,window=30)
            self.error2 = "done, getting PA"
            #self.frb_loaded = True
            self.PA_t_init,self.PA_t_errs_init = dsapol.get_pol_angle_vs_time((self.I_t_init,self.I_f),(self.Q_t_init,self.Q_f),(self.U_t_init,self.U_f),(self.V_t_init,self.V_f),self.ibox,self.fobj.header.tsamp,self.n_t,n_off=int(12000//self.n_t),plot=False,pre_calc_tf=True,show=False,normalize=True,buff=1,weighted=False,timeaxis=self.timeaxis,fobj=self.fobj)
            self.error2 = "done"

            #time.sleep(5)
            self.error =  "Complete: " + str(np.around(time.time()-t1,2)) + " s to load data"
            #self.frb_name = "Loaded " + ids + "_" + nickname + " ..."
            self.n_f_root = self.n_f
            self.n_t_root = self.n_t
            self.log_n_f = 0#param.Integer(default=0,bounds=(0,10),label=r'log2(n_f)')
            self.n_f = 1
            self.n_f_prev = 1
            self.n_t = 1

            self.I_RMcal =  np.zeros(self.I.shape)
            self.Q_RMcal =  np.zeros(self.I.shape)
            self.U_RMcal =  np.zeros(self.I.shape)
            self.V_RMcal =  np.zeros(self.I.shape)

            self.I_t = copy.deepcopy(self.I_t_init)#np.nan*np.ones(len(self.I_t_init))
            self.Q_t = copy.deepcopy(self.Q_t_init)#np.nan*np.ones(len(self.I_t_init))
            self.U_t = copy.deepcopy(self.U_t_init)#np.nan*np.ones(len(self.I_t_init))
            self.V_t = copy.deepcopy(self.V_t_init)#np.nan*np.ones(len(self.I_t_init))

            self.PA_t = np.nan*np.ones(len(self.PA_t_init))
            self.PA_t_errs = np.nan*np.ones(len(self.PA_t_errs_init))

            self.I_f_init = np.nan*np.ones(len(self.freq_test_init))#I.mean(1)#np.zeros(I.shape[0])
            self.Q_f_init = np.nan*np.ones(len(self.freq_test_init))#Q.mean(1)#np.zeros(Q.shape[0])
            self.U_f_init = np.nan*np.ones(len(self.freq_test_init))#U.mean(1)#np.zeros(U.shape[0])
            self.V_f_init = np.nan*np.ones(len(self.freq_test_init))#V.mean(1)#np.zeros(V.shape[0])

            self.I_f = np.nan*np.ones(len(self.I_f_init))
            self.Q_f = np.nan*np.ones(len(self.I_f_init))
            self.U_f = np.nan*np.ones(len(self.I_f_init))
            self.V_f = np.nan*np.ones(len(self.I_f_init))
            self.PA_f = np.nan*np.ones(len(self.I_f_init))
            self.PA_f_errs = np.nan*np.ones(len(self.I_f_init))
            self.freq_test = copy.deepcopy(self.freq_test_init)
    
            self.PA_f_init = np.nan*np.ones(len(self.freq_test_init[0]))
            self.PA_f_errs_init = np.nan*np.ones(len(self.freq_test_init[0]))
           
            self.fullburst_dict["I_t_init"] = self.I_t_init
            self.fullburst_dict["Q_t_init"] = self.Q_t_init
            self.fullburst_dict["U_t_init"] = self.U_t_init
            self.fullburst_dict["V_t_init"] = self.V_t_init
            self.fullburst_dict["PA_t_init"] = self.PA_t_init
            self.fullburst_dict["PA_t_errs_init"] = self.PA_t_errs_init


            self.loaded = True
        except Exception as e:
            self.error = "From load_FRB(): " + str(e)
        return


    def cal_FRB(self):
        try:
            if self.loaded:
                self.error = "Calibrating FRB...."
                t1 = time.time()
                with open("/media/ubuntu/ssd/sherman/code/" + self.cal_name,'r') as csvfile:
                    reader = csv.reader(csvfile,delimiter=",")
                    for row in reader:
                        if row[0] == "|gxx|/|gyy|":
                            tmp_ratio = np.array(row[1:],dtype="float")
                        elif row[0] == "|gxx|/|gyy| fit":
                            tmp_ratio_fit = np.array(row[1:],dtype="float")
                        if row[0] == "phixx-phiyy":
                            tmp_phase = np.array(row[1:],dtype="float")
                        if row[0] == "phixx-phiyy fit":
                            tmp_phase_fit = np.array(row[1:],dtype="float")
                        if row[0] == "|gyy|":
                            tmp_gainY = np.array(row[1:],dtype="float")
                        if row[0] == "|gyy| FIT":
                            tmp_gainY_fit = np.array(row[1:],dtype="float")
                        if row[0] == "gxx":
                            self.gxx = np.array(row[1:],dtype="complex")
                        if row[0] == "gyy":
                            self.gyy = np.array(row[1:],dtype="complex")
                        if row[0] == "freq_axis":
                            tmp_freq_axis = np.array(row[1:],dtype="float")

                self.gxx = self.gxx[len(self.gxx)%self.n_f_root:]
                self.gxx = self.gxx.reshape(len(self.gxx)//self.n_f_root,self.n_f_root).mean(1)

                self.gyy = self.gyy[len(self.gyy)%self.n_f_root:]
                self.gyy = self.gyy.reshape(len(self.gyy)//self.n_f_root,self.n_f_root).mean(1)

                self.I_init,self.Q_init,self.U_init,self.V_init = dsapol.calibrate(self.I_init,self.Q_init,self.U_init,self.V_init,(self.gxx,self.gyy),stokes=True)
                self.I_init,self.Q_init,self.U_init,self.V_init,self.ParA = dsapol.calibrate_angle(self.I_init,self.Q_init,self.U_init,self.V_init,self.fobj,self.ibeam,self.RA,self.DEC)

                self.I = dsapol.avg_freq(dsapol.avg_time(self.I_init,self.n_t),self.n_f)
                self.Q = dsapol.avg_freq(dsapol.avg_time(self.Q_init,self.n_t),self.n_f)
                self.U = dsapol.avg_freq(dsapol.avg_time(self.U_init,self.n_t),self.n_f) 
                self.V = dsapol.avg_freq(dsapol.avg_time(self.V_init,self.n_t),self.n_f)

                #self.I,self.Q,self.U,self.V = dsapol.calibrate(self.I,self.Q,self.U,self.V,(self.gxx,self.gyy),stokes=True)
                #self.I,self.Q,self.U,self.V,self.ParA = dsapol.calibrate_angle(self.I,self.Q,self.U,self.V,self.fobj,self.ibeam,self.RA,self.DEC)
                #self.I_init,self.Q_init,self.U_init,self.V_init = copy.deepcopy(self.I),copy.deepcopy(self.Q),copy.deepcopy(self.U),copy.deepcopy(self.V)

                (self.I_t_init,self.Q_t_init,self.U_t_init,self.V_t_init) = dsapol.get_stokes_vs_time(self.I_init,self.Q_init,self.U_init,self.V_init,self.ibox,self.fobj.header.tsamp,self.n_t_root,n_off=int(12000//self.n_t_root),plot=False,show=True,normalize=True,buff=1,window=30)

                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to calibrate data"
                self.calibrated = True
        except Exception as e:
            self.error = "From cal_FRB(): " + str(e)
        return

    def savefil(self):
        try:
            if self.loaded and self.calibrated:
                self.error = "Saving Calibrated Filterbanks..."
                t1 = time.time()

                #need to create new filterbank object if different sampling
                newfobj = copy.copy(self.fobj)
                newfobj.header = self.fobj.header.newHeader(update_dict={'nsamples':self.I_init.shape[1],'nsamples_list':[self.I_init.shape[1]], 'tsamp':self.n_t_root*self.fobj.header.tsamp, 'nchans':self.I_init.shape[0]})
                dsapol.put_stokes_2D(self.I_init,self.Q_init,self.U_init,self.V_init,newfobj,self.datadir,self.frb_name,suffix="polcal")
                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to save calibrated filterbank"
            else:
                self.error = "Please load and calibrate data before saving filterbanks"
        except Exception as e:
            self.error = "From savefil(): " + str(e)

    def savefilRM(self):
        try:
            if self.loaded and self.rmcalibrated_all:
                self.error = "Saving RM Calibrated Filterbanks..."
                t1 = time.time()

                #need to create new filterbank object if different sampling
                if self.calibrated:
                    suff = "polcal_RMcal"
                else:
                    suff = "RMcal"
                newfobj = copy.copy(self.fobj)
                newfobj.header = self.fobj.header.newHeader(update_dict={'nsamples':self.I_RMcal_init.shape[1],'nsamples_list':[self.I_RMcal_init.shape[1]], 'tsamp':self.n_t_root*self.fobj.header.tsamp, 'nchans':self.I_RMcal_init.shape[0]})
                dsapol.put_stokes_2D(self.I_RMcal_init,self.Q_RMcal_init,self.U_RMcal_init,self.V_RMcal_init,newfobj,self.datadir,self.frb_name,suffix=suff)
                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to save RM calibrated filterbank"
            else:
                self.error = "Please load and RM calibrate data before saving filterbanks"
        except Exception as e:
            self.error = "From savefil(): " + str(e)


    #helper to format txt file summary
    def format_summary(self):
        try:
            #header tells calibration status
            header = "### " +  self.frb_name + " ###\n"
            if self.calibrated:
                header += "\n## Polarization Calibrated ##\n"
                header += "Cal File: " + self.cal_name + "\n"
                header += "ibeam: " + str(self.ibeam) + "\n"
                header += "RA: " + str(self.RA) + " degrees \n"
                header += "Dec: " + str(self.DEC) + " degrees \n"
                header += "Parallactic Angle: " + str(self.ParA) + " degrees \n"

            if self.rmcalibrated_all:
                header += "\n## RM Calibrated ##\n"

            #downsampling properties
            dsamp =  "\nn_t = " + str(self.n_t) + "\n"
            dsamp += "\nn_f = " + str(self.n_f) + "\n"


            #galactic rm
            rmgalstr = ""
            if not np.isnan(self.RM_gal) and not np.isnan(self.RM_galerr):
                rmgalstr += "\nGalactic RM: " + str(self.RM_gal) + " +- " + str(self.RM_galerr) + " rad/m^2\n"

            #ionospheric rm
            rmionstr = ""
            if not np.isnan(self.RM_ion) and not np.isnan(self.RM_ionerr):
                rmionstr += "\nIonospheric RM: " + str(self.RM_ion) + " +- " + str(self.RM_ionerr) + " rad/m^2\n"

            #For multiple components
            datastr = ""
            if len(self.comp_dict.keys()) > 1:
                datastr += "\n## " + str(len(self.comp_dict.keys())) + " Components ##\n"

                for i in range(len(self.comp_dict.keys())):
                    datastr += "\n# Component " + str(i) + ":\n"
                    
                    #RM synthesis results
                    datastr += "RM Synthesis: " + str(self.comp_dict[i]["RM1"]) + " +- " + str(self.comp_dict[i]["RMerr1"]) + " rad/m^2\n"
                    datastr += "RM Synthesis, zoom: " + str(self.comp_dict[i]["RM1zoom"]) + " +- " + str(self.comp_dict[i]["RMerr1zoom"]) + " rad/m^2\n"
                    
                    #RM tools results
                    datastr += "RM Tools: " + str(self.comp_dict[i]["RM1tools"]) + " +- " + str(self.comp_dict[i]["RMerr1tools"]) + " rad/m^2\n"
                    datastr += "RM Tools, zoom: " + str(self.comp_dict[i]["RM1tools_zoom"]) + " +- " + str(self.comp_dict[i]["RMerr1tools_zoom"]) + " rad/m^2\n"

                    #RM synthesis results
                    datastr += "Fine Synthesis: " + str(self.comp_dict[i]["RM2zoom"]) + " +- " + str(self.comp_dict[i]["RMerr2zoom"]) + " rad/m^2\n"
                    datastr += "Linear S/N: " + str(np.max(self.comp_dict[i]["RMsnrs2zoom"])) + "\n"
                    
                    if self.comp_dict[i]["sigflag"]:#self.rm_threshold < np.max(self.comp_dict[i]["RMsnrs2zoom"]):
                        datastr += "Significant\n"
                    else:
                        datastr += "Not Significant\n"
                    
                    datastr += "\n"

                    #FWHM
                    datastr += "FWHM: " + str((self.comp_dict[i]["intR"]-self.comp_dict[i]["intL"])*32.7*1000*self.n_t*self.n_t_root) + " ms ; " + str((self.comp_dict[i]["intR"]-self.comp_dict[i]["intL"])) + " samples\n"
                    datastr += "\n"
                   
                    datastr += "--Before RM Calibration--\n"
                    #S/N
                    datastr += "S/N I: " + str(self.comp_dict[i]["I_snr"]) + "\n"
                    datastr += "S/N T: " + str(self.comp_dict[i]["T/I_pre_snr"]) + "\n"
                    datastr += "S/N L: " + str(self.comp_dict[i]["L/I_pre_snr"]) + "\n"
                    datastr += "S/N V: " + str(self.comp_dict[i]["V/I_snr"]) + "\n"
                    datastr += "\n"
                        
                    #Polarization

                    datastr += "Total Polarization: " + str(self.comp_dict[i]["T/I_pre"]) + " +- " + str(self.comp_dict[i]["T/I_pre_err"]) + "\n"
                    datastr += "Linear Polarization: " + str(self.comp_dict[i]["L/I_pre"]) + " +- " + str(self.comp_dict[i]["L/I_pre_err"]) + "\n"
                    datastr += "|Circular Polarization|: " + str(self.comp_dict[i]["absV/I"]) + " +- " + str(self.comp_dict[i]["absV/I_err"]) + "\n"
                    datastr += "Circular Polarization: " + str(self.comp_dict[i]["V/I"]) + " +- " + str(self.comp_dict[i]["V/I_err"]) + "\n"
                    datastr += "\n"

                    #PA
                    datastr += "Measured PA: " + str(self.comp_dict[i]["PA_pre"]*180/np.pi) + " +- " + str(self.comp_dict[i]["PAerr_pre"]*180/np.pi) + " degrees\n"

                    if self.comp_dict[i]["sigflag"]:
                        datastr += "\n--After RM Calibration--\n"
                        #S/N
                        datastr += "S/N I: " + str(self.comp_dict[i]["I_snr"]) + "\n"
                        datastr += "S/N T: " + str(self.comp_dict[i]["T/I_post_snr"]) + "\n"
                        datastr += "S/N L: " + str(self.comp_dict[i]["L/I_post_snr"]) + "\n"
                        datastr += "S/N V: " + str(self.comp_dict[i]["V/I_snr"]) + "\n"
                        datastr += "\n"
                            
                        #Polarization

                        datastr += "Total Polarization: " + str(self.comp_dict[i]["T/I_post"]) + " +- " + str(self.comp_dict[i]["T/I_post_err"]) + "\n"
                        datastr += "Linear Polarization: " + str(self.comp_dict[i]["L/I_post"]) + " +- " + str(self.comp_dict[i]["L/I_post_err"]) + "\n"
                        datastr += "|Circular Polarization|: " + str(self.comp_dict[i]["absV/I"]) + " +- " + str(self.comp_dict[i]["absV/I_err"]) + "\n"
                        datastr += "Circular Polarization: " + str(self.comp_dict[i]["V/I"]) + " +- " + str(self.comp_dict[i]["V/I_err"]) + "\n"
                        datastr += "\n"

                        #PA
                        datastr += "Intrinsic PPA: " + str(self.comp_dict[i]["PA_post"]*180/np.pi) + " +- " + str(self.comp_dict[i]["PAerr_post"]*180/np.pi) + " degrees\n"







            
            #full burst
            datastr += "\n# Full Burst:\n"
                
            #RM synthesis results
            datastr += "RM Synthesis: " + str(self.fullburst_dict["RM1"]) + " +- " + str(self.fullburst_dict["RMerr1"]) + " rad/m^2\n"
            datastr += "RM Synthesis, zoom: " + str(self.fullburst_dict["RM1zoom"]) + " +- " + str(self.fullburst_dict["RMerr1zoom"]) + " rad/m^2\n"
                    
            #RM tools results
            datastr += "RM Tools: " + str(self.fullburst_dict["RM1tools"]) + " +- " + str(self.fullburst_dict["RMerr1tools"]) + " rad/m^2\n"
            datastr += "RM Tools, zoom: " + str(self.fullburst_dict["RM1tools_zoom"]) + " +- " + str(self.fullburst_dict["RMerr1tools_zoom"]) + " rad/m^2\n"

            #RM synthesis results
            datastr += "Fine Synthesis: " + str(self.fullburst_dict["RM2zoom"]) + " +- " + str(self.fullburst_dict["RMerr2zoom"]) + " rad/m^2\n"
            datastr += "Linear S/N: " + str(np.max(self.fullburst_dict["RMsnrs2zoom"])) + "\n"
            if self.fullburst_dict["sigflag"]:#self.rm_threshold < np.max(self.fullburst_dict["RMsnrs2zoom"]):
                datastr += "Significant\n"
            else:
                datastr += "Not Signficant\n"


            datastr += "\n"


            #FWHM
            datastr += "FWHM: " + str((self.fullburst_dict["intR"]-self.fullburst_dict["intL"])*32.7*1000*self.n_t*self.n_t_root) + " ms ; " + str((self.fullburst_dict["intR"]-self.fullburst_dict["intL"])) + " samples\n"
            datastr += "\n"
            datastr += "--Before RM Calibration--\n"

            #S/N
            datastr += "S/N I: " + str(self.fullburst_dict["I_snr"]) + "\n"
            datastr += "S/N T: " + str(self.fullburst_dict["T/I_pre_snr"]) + "\n"
            datastr += "S/N L: " + str(self.fullburst_dict["L/I_pre_snr"]) + "\n"
            datastr += "S/N V: " + str(self.fullburst_dict["V/I_snr"]) + "\n"
            datastr += "\n"
                        
            #Polarization

            datastr += "Total Polarization: " + str(self.fullburst_dict["T/I_pre"]) + " +- " + str(self.fullburst_dict["T/I_pre_err"]) + "\n"
            datastr += "Linear Polarization: " + str(self.fullburst_dict["L/I_pre"]) + " +- " + str(self.fullburst_dict["L/I_pre_err"]) + "\n"
            datastr += "|Circular Polarization|: " + str(self.fullburst_dict["absV/I"]) + " +- " + str(self.fullburst_dict["absV/I_err"]) + "\n"
            datastr += "Circular Polarization: " + str(self.fullburst_dict["V/I"]) + " +- " + str(self.fullburst_dict["V/I_err"]) + "\n"
            datastr += "\n"

            #PA
            datastr += "Measured PA: " + str(self.fullburst_dict["PA_pre"]*180/np.pi) + " +- " + str(self.fullburst_dict["PAerr_pre"]*180/np.pi) + " degrees\n"
            

            if self.fullburst_dict["sigflag"]:
                datastr += "\n--After RM Calibration--\n"

                #S/N
                datastr += "S/N I: " + str(self.fullburst_dict["I_snr"]) + "\n"
                datastr += "S/N T: " + str(self.fullburst_dict["T/I_post_snr"]) + "\n"
                datastr += "S/N L: " + str(self.fullburst_dict["L/I_post_snr"]) + "\n"
                datastr += "S/N V: " + str(self.fullburst_dict["V/I_snr"]) + "\n"
                datastr += "\n"

                #Polarization

                datastr += "Total Polarization: " + str(self.fullburst_dict["T/I_post"]) + " +- " + str(self.fullburst_dict["T/I_post_err"]) + "\n"
                datastr += "Linear Polarization: " + str(self.fullburst_dict["L/I_post"]) + " +- " + str(self.fullburst_dict["L/I_post_err"]) + "\n"
                datastr += "|Circular Polarization|: " + str(self.fullburst_dict["absV/I"]) + " +- " + str(self.fullburst_dict["absV/I_err"]) + "\n"
                datastr += "Circular Polarization: " + str(self.fullburst_dict["V/I"]) + " +- " + str(self.fullburst_dict["V/I_err"]) + "\n"
                datastr += "\n"

                #PA
                datastr += "Intrinsic PPA: " + str(self.fullburst_dict["PA_post"]*180/np.pi) + " +- " + str(self.fullburst_dict["PAerr_post"]*180/np.pi) + " degrees\n"






            summary = header + dsamp + rmgalstr + rmionstr + datastr
            return summary


        except Exception as e:
            self.error = "From format_summary(): " + str(e)

    def savetxt(self):
        try:
            if not self.finished:
                self.error = "Please finish processing before exporting text file"

            else:

                self.error = "Exporting Summary to Text File..."
                t1 = time.time()

                txtname = self.datadir + self.frb_name + "_summary_TESTING.txt"
           
                self.summary = self.format_summary()
                tfile = open(txtname,"w")
                tfile.write(self.summary)
                tfile.close()

                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to export to text file " + str(txtname)

        except Exception as e:
            self.error = "From savetxt(): " + str(e)


    def savejson(self):
        try:
            if self.datadir == "":
                dir_list = []
                fname = ""
            else:
                dir_list = os.listdir(self.datadir)
                fname = "T2_" + self.ids + "_copy.json"
            
            
            #check if finished
            if not self.finished:
                self.error = "Please finish processing before exporting json file"

            elif fname not in dir_list:
                self.error = "No JSON found in data directory" 

            else:
                self.error = "Exporting Data to json File..."
                t1 = time.time()

                #open file
                f = open(self.datadir + fname,"r")
                data_dict = json.load(f)
                f.close()

                #write downsampling parameters
                data_dict[self.ids]["n_t"] = self.n_t
                data_dict[self.ids]["n_f"] = self.n_f

                #write number of components
                data_dict[self.ids]["n_comps"] = self.fullburst_dict["num_comps"]

                #write final RM and error if available
                if "RM2zoom" in self.fullburst_dict.keys():
                    data_dict[self.ids]["RM"] = self.fullburst_dict["RM2zoom"]
                    data_dict[self.ids]["RM_error"] = self.fullburst_dict["RMerr2zoom"]

                #write polarization fractions
                if "T/I_post" in self.fullburst_dict.keys():
                    #use post-RM polarization
                    data_dict[self.ids]["RMcalibrated"] = True
                    data_dict[self.ids]["Tpol"] = self.fullburst_dict["T/I_post"]
                    data_dict[self.ids]["Lpol"] = self.fullburst_dict["L/I_post"]
                    data_dict[self.ids]["Vpol"] = self.fullburst_dict["V/I"]
                    data_dict[self.ids]["absVpol"] = self.fullburst_dict["absV/I"]
                    data_dict[self.ids]["PA"] = self.fullburst_dict["PA_post"]

                    data_dict[self.ids]["Tpol_error"] = self.fullburst_dict["T/I_post_err"]
                    data_dict[self.ids]["Lpol_error"] = self.fullburst_dict["L/I_post_err"]
                    data_dict[self.ids]["Vpol_error"] = self.fullburst_dict["V/I_err"]
                    data_dict[self.ids]["absVpol_error"] = self.fullburst_dict["absV/I_err"]
                    data_dict[self.ids]["PA_error"] = self.fullburst_dict["PAerr_post"]

                else:
                    #use pre-RM polarization
                    data_dict[self.ids]["RMcalibrated"] = False
                    data_dict[self.ids]["Tpol"] = self.fullburst_dict["T/I_pre"]
                    data_dict[self.ids]["Lpol"] = self.fullburst_dict["L/I_pre"]
                    data_dict[self.ids]["Vpol"] = self.fullburst_dict["V/I"]
                    data_dict[self.ids]["absVpol"] = self.fullburst_dict["absV/I"]
                    data_dict[self.ids]["PA"] = self.fullburst_dict["PA_pre"]

                    data_dict[self.ids]["Tpol_error"] = self.fullburst_dict["T/I_pre_err"]
                    data_dict[self.ids]["Lpol_error"] = self.fullburst_dict["L/I_pre_err"]
                    data_dict[self.ids]["Vpol_error"] = self.fullburst_dict["V/I_err"]
                    data_dict[self.ids]["absVpol_error"] = self.fullburst_dict["absV/I_err"]
                    data_dict[self.ids]["PA_error"] = self.fullburst_dict["PAerr_pre"]

                #write back to file
                f = open(self.datadir + fname,"w")
                json.dump(data_dict,f)
                f.close()
        

                self.error = "Complete: "  + str(np.around(time.time()-t1,2)) + " s to export to json file " + str(fname)


        except Exception as e:
            self.error = "From savejson(): " + str(e)


    RMTable_name = '/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/DSA110_RMTable_V1.fits'
    PolSpectra_name = '/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/DSA110_PolTable_V1.fits'

    """
    def addtocatalog(self):
        try:
            #only do if finished
            if not self.finished:
                self.error = "Please finish before adding to DSA-110 catalog"
            
            else:
                self.error = "Adding FRB to DSA Catalogs..."
                t1 = time.time()
                ##write RMTable if RM was detected and applied
                rmtable = RMTable.read(self.RMTable_name)
                
                #if already in table, remove and replace
                if self.ids in rmtable["candname"]:
                    while self.ids in rmtable["candname"]:
                        rmtable.remove_row(list(rmtable["candname"]).index(self.ids))
                
                #add row for full burst
                rowidx = len(rmtable['candname'])
                rmtable.add_row()

                rmtable['ra'][rowidx] = self.RA
                rmtable['dec'][rowidx] = self.DEC
                c = SkyCoord(ra=self.RA*u.deg,dec=self.DEC*u.deg,frame='icrs')
                rmtable['l'][rowidx] = c.galactic.l.degree
                rmtable['b'][rowidx] = c.galactic.b.degree
                rmtable['ionosphere'][rowidx] = "ionFR"
                rmtable['Ncomp'][rowidx] = len(self.comp_dict.keys())
                rmtable['Npeaks'][rowidx] = len(self.comp_dict.keys())
                rmtable['PeakNum'][rowidx] = -1
                rmtable['pol_bias'][rowidx] = "1985A&A...142..100S"
                rmtable['Vfracpol'][rowidx] = self.fullburst_dict["V/I"]
                rmtable['Vfracpol_err'][rowidx] = self.fullburst_dict["V/I_err"]
                rmtable['absVfracpol'][rowidx] = self.fullburst_dict["absV/I"]
                rmtable['absVfracpol_err'][rowidx] = self.fullburst_dict["absV/I_err"]
                rmtable['polangle'][rowidx] = self.fullburst_dict["PA_pre"]*180/np.pi
                rmtable['polangle_err'][rowidx] = self.fullburst_dict["PAerr_pre"]*180/np.pi
                rmtable['minfreq'][rowidx] = np.min(self.freq_test_init[0])*1e6
                rmtable['maxfreq'][rowidx] = np.max(self.freq_test_init[0])*1e6
                rmtable['channelwidth'][rowidx] = np.abs(self.freq_test_init[0][0]-self.freq_test_init[0][1])*1e6
                rmtable['rmsf_fwhm'][rowidx] = 349.81635341775683 #pre-calculated RMSF in jupyter notebook
                rmtable['telescope'][rowidx] = 'DSA110'
                rmtable['int_time'][rowidx] = 20480*32.7e-6
                rmtable['epoch'][rowidx] = self.MJD
                rmtable['candname'][rowidx] = self.ids
                
                if self.fullburst_dict['sigflag']:
                    rmtable['rm'][rowidx] = self.fullburst_dict["RM2zoom"]
                    rmtable['rm_err'][rowidx] = self.fullburst_dict["RMerr2zoom"]
                    rmtable['rm_width'][rowidx] = self.fullburst_dict["RM_FWHM"]
                    rmtable['rm_method'][rowidx] = "RM Synthesis"
                    rmtable['fracpol'][rowidx] = self.fullburst_dict["T/I_post"]
                    rmtable['fracpol_err'][rowidx] = self.fullburst_dict["T/I_post_err"]
                    rmtable['Lfracpol'][rowidx] = self.fullburst_dict["L/I_post"]
                    rmtable['Lfracpol_err'][rowidx] = self.fullburst_dict["L/I_post_err"]                   
                    rmtable['derot_polangle'][rowidx] = self.fullburst_dict["PA_post"]*180/np.pi
                    rmtable['derot_polangle_err'][rowidx] = self.fullburst_dict["PAerr_post"]*180/np.pi
                else:
                    rmtable['fracpol'][rowidx] = self.fullburst_dict["T/I_pre"]
                    rmtable['fracpol_err'][rowidx] = self.fullburst_dict["T/I_pre_err"]
                    rmtable['Lfracpol'][rowidx] = self.fullburst_dict["L/I_pre"]
                    rmtable['Lfracpol_err'][rowidx] = self.fullburst_dict["L/I_pre_err"]

                #add rows for each component
                if len(self.comp_dict.keys()) > 1:
                    for i in range(len(self.comp_dict.keys())):
                        rowidx = len(rmtable['candname'])
                        rmtable.add_row()

                        rmtable['ra'][rowidx] = self.RA
                        rmtable['dec'][rowidx] = self.DEC
                        c = SkyCoord(ra=self.RA*u.deg,dec=self.DEC*u.deg,frame='icrs')
                        rmtable['l'][rowidx] = c.galactic.l.degree
                        rmtable['b'][rowidx] = c.galactic.b.degree
                        rmtable['ionosphere'][rowidx] = "ionFR"
                        rmtable['Ncomp'][rowidx] = len(self.comp_dict.keys())
                        rmtable['Npeaks'][rowidx] = len(self.comp_dict.keys())
                        rmtable['PeakNum'][rowidx] = i
                        rmtable['pol_bias'][rowidx] = "1985A&A...142..100S"
                        rmtable['Vfracpol'][rowidx] = self.comp_dict[i]["V/I"]
                        rmtable['Vfracpol_err'][rowidx] = self.comp_dict[i]["V/I_err"]
                        rmtable['absVfracpol'][rowidx] = self.comp_dict[i]["absV/I"]
                        rmtable['absVfracpol_err'][rowidx] = self.comp_dict[i]["absV/I_err"]
                        rmtable['polangle'][rowidx] = self.comp_dict[i]["PA_pre"]*180/np.pi
                        rmtable['polangle_err'][rowidx] = self.comp_dict[i]["PAerr_pre"]*180/np.pi
                        rmtable['minfreq'][rowidx] = np.min(self.freq_test_init[0])*1e6
                        rmtable['maxfreq'][rowidx] = np.max(self.freq_test_init[0])*1e6
                        rmtable['channelwidth'][rowidx] = np.abs(self.freq_test_init[0][0]-self.freq_test_init[0][1])*1e6
                        rmtable['rmsf_fwhm'][rowidx] = 349.81635341775683 #pre-calculated RMSF in jupyter notebook
                        rmtable['telescope'][rowidx] = 'DSA110'
                        rmtable['int_time'][rowidx] = 20480*32.7e-6
                        rmtable['epoch'][rowidx] = self.MJD
                        rmtable['candname'][rowidx] = self.ids
                
                    if self.comp_dict[i]['sigflag']:
                        rmtable['rm'][rowidx] = self.comp_dict[i]["RM2zoom"]
                        rmtable['rm_err'][rowidx] = self.comp_dict[i]["RMerr2zoom"]
                        rmtable['rm_width'][rowidx] = self.comp_dict[i]["RM_FWHM"]
                        rmtable['rm_method'][rowidx] = "RM Synthesis"
                        rmtable['fracpol'][rowidx] = self.comp_dict[i]["T/I_post"]
                        rmtable['fracpol_err'][rowidx] = self.comp_dict[i]["T/I_post_err"]
                        rmtable['Lfracpol'][rowidx] = self.comp_dict[i]["L/I_post"]
                        rmtable['Lfracpol_err'][rowidx] = self.comp_dict[i]["L/I_post_err"]
                        rmtable['derot_polangle'][rowidx] = self.comp_dict[i]["PA_post"]*180/np.pi
                        rmtable['derot_polangle_err'][rowidx] = self.comp_dict[i]["PAerr_post"]*180/np.pi
                    else:
                        rmtable['fracpol'][rowidx] = self.comp_dict[i]["T/I_pre"]
                        rmtable['fracpol_err'][rowidx] = self.comp_dict[i]["T/I_pre_err"]
                        rmtable['Lfracpol'][rowidx] = self.comp_dict[i]["L/I_pre"]
                        rmtable['Lfracpol_err'][rowidx] = self.comp_dict[i]["L/I_pre_err"]



                #write back to file
                rmtable.write(self.RMTable_name,overwrite=True)

                ##write polarized spectra for each component
                
                #open current table
                spectrumtable = polspectra.from_FITS(self.PolSpectra_name)
                size = np.max(spectrumtable['source_number'])#len(spectrumtable['source_number'])-1



                #if already in table, remove and replace
                if self.ids in spectrumtable["candname"]:
                    tmp_spectrumtable = copy.deepcopy(spectrumtable[0:0])
                    for i in range(len(spectrumtable["candname"])):
                        if not (self.ids == spectrumtable["candname"][i]):
                            tmp_spectrumtable.merge_tables(spectrumtable[i])
                    spectrumtable = copy.deepcopy(tmp_spectrumtable)

    

                ra_arr = []
                dec_arr = []
                I_f_arr = []
                Q_f_arr = []
                U_f_arr = []
                V_f_arr = []

                I_f_err_arr = [] #nan for now
                Q_f_err_arr = []
                U_f_err_arr = []
                V_f_err_arr = []

                source_number_array = []
                peak_number_array = []
                numpeaks_array = []

                freq_array = []
                fstep_array = []
                bmaj_array = []
                bmin_array = []
                bpa_array = []
                candname_array = []


                #create row for full burst
                ra_arr.append(self.RA)
                dec_arr.append(self.DEC)
                I_f_arr.append(self.fullburst_dict["I_f_init"])
                Q_f_arr.append(self.fullburst_dict["Q_f_init"])
                U_f_arr.append(self.fullburst_dict["U_f_init"])
                V_f_arr.append(self.fullburst_dict["V_f_init"])

                I_f_err_arr.append(np.nan*np.ones(len(self.I_f_init)))
                Q_f_err_arr.append(np.nan*np.ones(len(self.Q_f_init)))
                U_f_err_arr.append(np.nan*np.ones(len(self.U_f_init)))
                V_f_err_arr.append(np.nan*np.ones(len(self.V_f_init)))

                source_number_array.append(size)
                peak_number_array.append(-1)
                numpeaks_array.append(len(self.comp_dict.keys()))
                freq_array.append(self.freq_test_init[0])
                fstep_array.append(np.abs(self.freq_test_init[0][1]-self.freq_test_init[0][0]))

                bmaj_array.append(0.0)
                bmin_array.append(0.0)
                bpa_array.append(0)
                candname_array.append(self.ids)

                #create row for each component if more than 1
                if len(self.comp_dict.keys()) > 1:
                    for i in range(len(self.comp_dict.keys())):

                        ra_arr.append(self.RA)
                        dec_arr.append(self.DEC)
                        I_f_arr.append(self.comp_dict[i]["I_f_init"])
                        Q_f_arr.append(self.comp_dict[i]["Q_f_init"])
                        U_f_arr.append(self.comp_dict[i]["U_f_init"])
                        V_f_arr.append(self.comp_dict[i]["V_f_init"])

                        I_f_err_arr.append(np.nan*np.ones(len(self.I_f_init)))
                        Q_f_err_arr.append(np.nan*np.ones(len(self.Q_f_init)))
                        U_f_err_arr.append(np.nan*np.ones(len(self.U_f_init)))
                        V_f_err_arr.append(np.nan*np.ones(len(self.V_f_init)))

                        source_number_array.append(size+i+1)
                        peak_number_array.append(i)
                        numpeaks_array.append(len(self.comp_dict.keys()))
                        freq_array.append(self.freq_test_init[0])
                        fstep_array.append(np.abs(self.freq_test_init[0][1]-self.freq_test_init[0][0]))

                        bmaj_array.append(0.0)
                        bmin_array.append(0.0)
                        bpa_array.append(0)

                        candname_array.append(self.ids)

                newrow = polspectra.from_arrays(ra_arr,dec_arr,freq_array,
                                    I_f_arr,I_f_err_arr,
                                    Q_f_arr,Q_f_err_arr,
                                    U_f_arr,U_f_err_arr,
                                    stokesV=V_f_arr,stokesV_error=V_f_err_arr,
                                    source_number_array=source_number_array,
                                    beam_maj=bmaj_array,beam_min=bmin_array,beam_pa=bpa_array,
                                    coordinate_system='icrs',channel_width=fstep_array)
                                    
                                    
                newrow.add_column(values=candname_array,name='candname',description='candidate name (str)')
                newrow.add_column(values=numpeaks_array,name='Npeaks',description='Number of burst components')
                newrow.add_column(values=peak_number_array,name='PeakNum',description='Peak number within burst for which spectrum was computed')
                
                #add to table and write
                spectrumtable.merge_tables(newrow)
                spectrumtable.write_FITS(self.PolSpectra_name,overwrite=True)
                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to update catalog"

        except Exception as e:
            self.error = "From addtocatalog(): "  + str(e)
    """
    #***COMPONENT SELECTION MODULE***#
    peak = int(15280)
    timestart = int(peak - (5e-3)/(32.7e-6))
    timestop = int(peak + (5e-3)/(32.7e-6))
    lo_frac = param.Number(default=0,bounds=(0,1),label=r'lower bound',step=1/(timestop-timestart))#param.Number(default=0,bounds=(0,timestop-timestart),label=r'lower bound')
    lo = 0#lo_frac*(timestop-timestart)

    #lo = ((timestop-timestart)/2)
    comp_choose_on = False
    filt_weights_on = False
    freq_samp_on = False
    finished = False
    wait = False

    sigflag = False #RM sigflag
    multipeaks = param.Boolean(False)
    maskPA=param.Boolean(False)
    height = param.Number(default=5,bounds=(0,200),step=1e-2)
    intLs = []
    intRs = []

    fixed_comps = []
    complist_min = []
    complist_max = []
    def clicked_next(self):
        try:
            if self.comp_choose_on:
                self.complist_min.append(int(np.floor(self.lo)))
                self.complist_max.append(int(np.ceil(self.lo+self.comp_width)))
                self.fixed_comps.append([int(np.floor(self.lo)),int(np.ceil(self.lo+self.comp_width))])
                self.param.trigger('next_comp')
            elif self.filt_weights_on and ( self.curr_comp < len(self.fixed_comps)):
                #save filter weights for current component
                self.comp_dict[self.curr_comp] =dict()
                #get actual timestart, timestop
                peak,t1,t2 = dsapol.find_peak([self.I_t,self.I_t],self.ibox,self.fobj.header.tsamp,self.n_t,peak_range=None,pre_calc_tf=True,buff=[self.buff_L,self.buff_R])
                self.comp_dict[self.curr_comp]["timestart"] = t1
                self.comp_dict[self.curr_comp]["timestop"] = t2
                self.comp_dict[self.curr_comp]["comp_num"] = self.curr_comp
                self.comp_dict[self.curr_comp]["buff"] = [self.buff_L,self.buff_R]
                self.comp_dict[self.curr_comp]["n_t_weight"] = self.n_t_weight
                self.comp_dict[self.curr_comp]["sf_window_weights"] = self.sf_window_weights
                self.comp_dict[self.curr_comp]["ibox"] = self.ibox
                self.comp_dict[self.curr_comp]["mask_start"] = self.complist_min[self.curr_comp]
                self.comp_dict[self.curr_comp]["mask_stop"] = self.complist_max[self.curr_comp]
                self.comp_dict[self.curr_comp]["weights"] = self.curr_weights
                self.comp_dict[self.curr_comp]["multipeaks"] = self.multipeaks
                self.comp_dict[self.curr_comp]["sigflag"] = False
                self.comp_dict[self.curr_comp]["rm_applied"] = False
                if self.multipeaks:
                    self.comp_dict[self.curr_comp]["height"] = self.height
                    self.comp_dict[self.curr_comp]["scaled_height"] = self.height*np.max(self.curr_weights)/np.max(self.I_t)

                peak = int(15280/self.n_t)
                timestart = int(peak - (5e-3)/(self.n_t*32.7e-6))
                timestop = int(peak + (5e-3)/(self.n_t*32.7e-6))

                #get intL, intR
                if self.multipeaks and self.height < np.max(self.I_t):
                    pks,props = find_peaks((self.curr_weights*np.max(self.I_t)/np.max(self.curr_weights))[timestart:timestop],height=self.height)
                    FWHM,heights,intL,intR = peak_widths((self.curr_weights*np.max(self.I_t)/np.max(self.curr_weights))[timestart:timestop],pks)
                    intL = int(intL[0])
                    intR = int(intR[-1])
                elif self.multipeaks:
                    intL = np.nan
                    intR = np.nan
                else:
                    FWHM,heights,intL,intR = peak_widths(self.curr_weights[timestart:timestop],[np.argmax(self.curr_weights[timestart:timestop])])
                    intL = int(intL)
                    intR = int(intR)
                self.comp_dict[self.curr_comp]["intL"] = intL
                self.comp_dict[self.curr_comp]["intR"] = intR
                self.intLs.append(intL)
                self.intRs.append(intR)
                print("yolo")
                print(self.intLs)
                print(self.intRs)

                #get spectrum for current component
                self.error = "Computing Spectrum..."
                t1 = time.time()
                (I_fmasked,Q_fmasked,U_fmasked,V_fmasked) = dsapol.get_stokes_vs_freq(self.I,self.Q,self.U,self.V,1,self.fobj.header.tsamp,1,self.n_t,self.freq_test_init,n_off=int(12000/self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,input_weights=self.curr_weights)
                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute spectrum"

                self.comp_dict[self.curr_comp]["I_f"] = I_fmasked
                self.comp_dict[self.curr_comp]["Q_f"] = Q_fmasked
                self.comp_dict[self.curr_comp]["U_f"] = U_fmasked
                self.comp_dict[self.curr_comp]["V_f"] = V_fmasked

                self.comp_dict[self.curr_comp]["I_f_init"] = I_fmasked
                self.comp_dict[self.curr_comp]["Q_f_init"] = Q_fmasked
                self.comp_dict[self.curr_comp]["U_f_init"] = U_fmasked
                self.comp_dict[self.curr_comp]["V_f_init"] = V_fmasked

                #get PA vs frequency
                self.error = "Computing Position Angle..."
                t1 = time.time()
                #PA_fmasked,tmpPA_t_init,PA_f_errsmasked,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                PA_fmasked,tmpPA_t_init,PA_f_errsmasked,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle((self.I_t,I_fmasked),(self.Q_t,Q_fmasked),(self.U_t,U_fmasked),(self.V_t,V_fmasked),self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,pre_calc_tf=True,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                self.comp_dict[self.curr_comp]["PA_f"] = PA_fmasked
                self.comp_dict[self.curr_comp]["PA_f_errs"] = PA_f_errsmasked
                
                self.comp_dict[self.curr_comp]["PA_f_init"] = PA_fmasked
                self.comp_dict[self.curr_comp]["PA_f_errs_init"] = PA_f_errsmasked

                self.comp_dict[self.curr_comp]["PA_pre"] = avg_PA
                self.comp_dict[self.curr_comp]["PAerr_pre"] = sigma_PA
                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute position angle"    


                #get polarization fractions and position angles
                self.error = "Computing polarization..."
                t1 = time.time()
                #[(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000/self.n_t),normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                [(pol_t,avg_frac,sigma_frac,snr_frac),(L_t,avg_L,sigma_L,snr_L),(C_t,avg_C_abs,sigma_C_abs,snr_C),(C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction_vs_time(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,n_off=int(12000/self.n_t),normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                self.comp_dict[self.curr_comp]["T/I_pre"] = avg_frac
                self.comp_dict[self.curr_comp]["T/I_pre_err"] = sigma_frac
                self.comp_dict[self.curr_comp]["T/I_pre_snr"] = snr_frac
                self.comp_dict[self.curr_comp]["L/I_pre"] = avg_L
                self.comp_dict[self.curr_comp]["L/I_pre_err"] = sigma_L
                self.comp_dict[self.curr_comp]["L/I_pre_snr"] = snr_L
                self.comp_dict[self.curr_comp]["absV/I"] = avg_C_abs
                self.comp_dict[self.curr_comp]["absV/I_err"] = sigma_C_abs
                self.comp_dict[self.curr_comp]["V/I"] = avg_C
                self.comp_dict[self.curr_comp]["V/I_err"] = sigma_C
                self.comp_dict[self.curr_comp]["V/I_snr"] = snr_C
                self.comp_dict[self.curr_comp]["I_snr"] = snr


                #update displays
                if self.curr_comp == 0 and len(self.fixed_comps) == 1:
                    self.snr = '(' + r'{a}'.format(a=np.around(snr,2))+ ') '
                    self.Tsnr = '('+ r'{a}'.format(a=np.around(snr_frac,2))+ ') '
                    self.Lsnr = '(' + r'{a}'.format(a=np.around(snr_L,2))+ ') '
                    self.Csnr = '(' + r'{a}'.format(a=np.around(snr_C,2))+ ') '

                    self.Tpol = '(' + r'{a}%'.format(a=np.around(100*avg_frac,2))+ ') '
                    self.Lpol = '(' + r'{a}%'.format(a=np.around(100*avg_L,2))+ ') '
                    self.absCpol = '(' + r'{a}%'.format(a=np.around(100*avg_C_abs,2))+ ') '
                    self.Cpol = '('+ r'{a}%'.format(a=np.around(100*avg_C,2))+ ') '

                    self.Tpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_frac,2))+ ') '
                    self.Lpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_L,2))+ ') '
                    self.absCpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))+ ') '
                    self.Cpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_C,2))+ ') '                    

                    self.avgPA = '(' + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))+ ') '
                    self.avgPAerr = '(' + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))+ ') '

                elif self.curr_comp == 0 and len(self.fixed_comps) > 1:
                    self.snr = '(' + r'{a}'.format(a=np.around(snr,2))+ ' ; '
                    self.Tsnr = '('+ r'{a}'.format(a=np.around(snr_frac,2))+ ' ; '
                    self.Lsnr = '(' + r'{a}'.format(a=np.around(snr_L,2))+ ' ; '
                    self.Csnr = '(' + r'{a}'.format(a=np.around(snr_C,2))+ ' ; '

                    self.Tpol = '(' + r'{a}%'.format(a=np.around(100*avg_frac,2))+ ' ; '
                    self.Lpol = '(' + r'{a}%'.format(a=np.around(100*avg_L,2))+ ' ; '
                    self.absCpol = '(' + r'{a}%'.format(a=np.around(100*avg_C_abs,2))+ ' ; '
                    self.Cpol = '('+ r'{a}%'.format(a=np.around(100*avg_C,2))+ ' ; '

                    self.Tpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_frac,2))+ ' ; '
                    self.Lpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_L,2))+ ' ; '
                    self.absCpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))+ ' ; '
                    self.Cpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_C,2))+ ' ; '
                    
                    self.avgPA = '(' + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))+ ' ; '
                    self.avgPAerr = '(' + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))+ ' ; '

                    self.curr_comp += 1

                elif self.curr_comp <  len(self.fixed_comps) -1 :

                    self.snr = self.snr + r'{a}'.format(a=np.around(snr,2))+ ' ; '
                    self.Tsnr = self.Tsnr + r'{a}'.format(a=np.around(snr_frac,2))+ ' ; '
                    self.Lsnr = self.Lsnr + r'{a}'.format(a=np.around(snr_L,2))+ ' ; '
                    self.Csnr = self.Csnr + r'{a}'.format(a=np.around(snr_C,2))+ ' ; '

                    self.Tpol = self.Tpol + r'{a}%'.format(a=np.around(100*avg_frac,2))+ ' ; '
                    self.Lpol = self.Lpol + r'{a}%'.format(a=np.around(100*avg_L,2))+ ' ; '
                    self.absCpol = self.absCpol + r'{a}%'.format(a=np.around(100*avg_C_abs,2))+ ' ; '
                    self.Cpol =self.Cpol + r'{a}%'.format(a=np.around(100*avg_C,2))+ ' ; '

                    self.Tpolerr = self.Tpolerr + r'{a}%'.format(a=np.around(100*sigma_frac,2))+ ' ; '
                    self.Lpolerr = self.Lpolerr + r'{a}%'.format(a=np.around(100*sigma_L,2))+ ' ; '
                    self.absCpolerr = self.absCpolerr + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))+ ' ; '
                    self.Cpolerr = self.Cpolerr + r'{a}%'.format(a=np.around(100*sigma_C,2))+ ' ; '

                    self.avgPA = self.avgPA + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))+ ' ; '
                    self.avgPAerr = self.avgPAerr + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))+ ' ; '

                    self.curr_comp += 1
                elif self.curr_comp == len(self.fixed_comps) - 1:
                    self.snr = self.snr + r'{a}'.format(a=np.around(snr,2))+ ') '
                    self.Tsnr = self.Tsnr + r'{a}'.format(a=np.around(snr_frac,2))+ ') '
                    self.Lsnr = self.Lsnr + r'{a}'.format(a=np.around(snr_L,2))+ ') '
                    self.Csnr = self.Csnr + r'{a}'.format(a=np.around(snr_C,2))+ ') '

                    self.Tpol = self.Tpol + r'{a}%'.format(a=np.around(100*avg_frac,2))+ ') '
                    self.Lpol = self.Lpol + r'{a}%'.format(a=np.around(100*avg_L,2))+ ') '
                    self.absCpol = self.absCpol + r'{a}%'.format(a=np.around(100*avg_C_abs,2))+ ') '
                    self.Cpol =self.Cpol + r'{a}%'.format(a=np.around(100*avg_C,2))+ ') '

                    self.Tpolerr = self.Tpolerr + r'{a}%'.format(a=np.around(100*sigma_frac,2))+ ') '
                    self.Lpolerr = self.Lpolerr + r'{a}%'.format(a=np.around(100*sigma_L,2))+ ') '
                    self.absCpolerr = self.absCpolerr + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))+ ') '
                    self.Cpolerr = self.Cpolerr + r'{a}%'.format(a=np.around(100*sigma_C,2))+ ') '

                    self.avgPA = self.avgPA + r'{a}%'.format(a=np.around((180/np.pi)*avg_PA,2))+ ') '
                    self.avgPAerr = self.avgPAerr + r'{a}%'.format(a=np.around((180/np.pi)*sigma_PA,2))+ ') '

                    self.error = "No more components, click Done"
                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute polarization"
                #self.error2 = str(self.curr_comp)
                self.param.trigger('next_comp')
                #self.error = str(len(self.comp_dict[self.curr_comp-1]["I_f"]))

        except Exception as e:
            self.error = "From clicked_next(): " + str(e)

        return
    def clicked_get(self):
        try:
            self.comp_choose_on = not self.comp_choose_on
            self.height = np.around(np.nanmax(self.I_t)/2,2)
            self.param.trigger('get_comp')
        except Exception as e:
            self.error = "From clicked_get(): " + str(e)
        return
    def clicked_done(self):
        try:
            if self.comp_choose_on:
                self.comp_choose_on = False
                self.filt_weights_on = True

                self.error = "Downsampling dynamic spectrum..."
                t1 = time.time()
                self.I = dsapol.avg_time(self.I,self.n_t)
                self.Q = dsapol.avg_time(self.Q,self.n_t)
                self.U = dsapol.avg_time(self.U,self.n_t)
                self.V = dsapol.avg_time(self.V,self.n_t)
                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to downsample in time"

                self.param.trigger('done')
            elif self.filt_weights_on:
                if ( self.curr_comp < len(self.fixed_comps)-1):
                    self.error = "Still have " + str(len(self.fixed_comps) - self.curr_comp) + " components left"
                elif (self.curr_comp == len(self.fixed_comps)-1): #case where next was clicked before done
                    self.wait = True
                    self.curr_comp += 1
                    #self.error = "Finishing " + str(self.curr_comp) + " " + str(len(self.fixed_comps))
                    self.curr_weights = np.zeros(len(self.curr_weights))
                    for i in range(len(self.fixed_comps)):
                        print("ADDING WEIGHTS TOGETHER " + str(len(self.fixed_comps)))
                        self.curr_weights += self.comp_dict[i]["weights"]
                    self.curr_weights = self.curr_weights/np.sum(self.curr_weights)

                    #self.filt_weights_on = False
                    #self.freq_samp_on = True

                    self.error = "Computing Spectrum..."
                    t1 = time.time()
                    (self.I_f_init,self.Q_f_init,self.U_f_init,self.V_f_init) = dsapol.get_stokes_vs_freq(self.I,self.Q,self.U,self.V,1,self.fobj.header.tsamp,1,self.n_t,self.freq_test_init,n_off=int(12000/self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,input_weights=self.curr_weights)
                    self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute spectrum"

                    #unmask time series
                    self.I_t = self.fullburst_dict["I_t"]
                    self.Q_t = self.fullburst_dict["Q_t"]
                    self.U_t = self.fullburst_dict["U_t"]
                    self.V_t = self.fullburst_dict["V_t"]

                    #get total PA 
                    self.error = "Computing Position Angle..."
                    t1 = time.time()
                    #self.PA_f_init,tmpPA_t_init,self.PA_f_errs_init,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                    self.PA_f_init,tmpPA_t_init,self.PA_f_errs_init,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle((self.I_t,self.I_f_init),(self.Q_t,self.Q_f_init),(self.U_t,self.U_f_init),(self.V_t,self.V_f_init),self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,pre_calc_tf=True,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                    self.avgPA = self.avgPA + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))
                    self.avgPAerr = self.avgPAerr + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))
                    self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute position angle"



                    #get total polarization
                    self.error = "Computing polarization..."

                    multipeaks_all = (len(self.fixed_comps) > 1) or (self.comp_dict[0]["multipeaks"])

                    t1 = time.time()
                    #[(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000/self.n_t),normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=multipeaks_all,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                    [(pol_t,avg_frac,sigma_frac,snr_frac),(L_t,avg_L,sigma_L,snr_L),(C_t,avg_C_abs,sigma_C_abs,snr_C),(C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction_vs_time(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,n_off=int(12000/self.n_t),normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=multipeaks_all,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                    self.snr = self.snr + r'{a}'.format(a=np.around(snr,2))  
                    self.Tsnr = self.Tsnr + r'{a}'.format(a=np.around(snr_frac,2))
                    self.Lsnr = self.Lsnr + r'{a}'.format(a=np.around(snr_L,2))
                    self.Csnr = self.Csnr + r'{a}'.format(a=np.around(snr_C,2))

                    self.Tpol = self.Tpol + r'{a}%'.format(a=np.around(100*avg_frac,2))
                    self.Lpol = self.Lpol + r'{a}%'.format(a=np.around(100*avg_L,2))
                    self.absCpol = self.absCpol + r'{a}%'.format(a=np.around(100*avg_C_abs,2))
                    self.Cpol = self.Cpol + r'{a}%'.format(a=np.around(100*avg_C,2))

                    self.Tpolerr = self.Tpolerr + r'{a}%'.format(a=np.around(100*sigma_frac,2))
                    self.Lpolerr = self.Lpolerr + r'{a}%'.format(a=np.around(100*sigma_L,2))
                    self.absCpolerr = self.absCpolerr + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))
                    self.Cpolerr = self.Cpolerr + r'{a}%'.format(a=np.around(100*sigma_C,2))

                    #save full burst properties to dict
                    tmp1 = []
                    tmp2 = []
                    b1 = []
                    b2 = []
                    intL1 = []
                    intR2 = []
                    for k in range(len(self.comp_dict.keys())):
                        tmp1.append(self.comp_dict[k]["timestart"])
                        tmp2.append(self.comp_dict[k]["timestop"])
                        b1.append(self.comp_dict[k]["buff"][0])
                        b2.append(self.comp_dict[k]["buff"][1])
                        intL1.append(self.comp_dict[k]["intL"])
                        intR2.append(self.comp_dict[k]["intR"])

                    self.fullburst_dict["timestart"] = np.min(tmp1)
                    self.fullburst_dict["timestop"] = np.max(tmp2)
                    self.fullburst_dict["num_comps"] = len(self.comp_dict.keys())
                    self.fullburst_dict["buff"] = (np.min(b1),np.max(b2))
                    self.fullburst_dict["weights"] = self.curr_weights
                    self.fullburst_dict["multipeaks_all"] = multipeaks_all
                    self.fullburst_dict["sigflag"] = False
                    self.fullburst_dict["rm_applied"] = False
                    self.fullburst_dict["intL"] = np.min(intL1)
                    self.fullburst_dict["intR"] = np.max(intR2)


                    self.fullburst_dict["I_f"] = self.I_f_init
                    self.fullburst_dict["Q_f"] = self.Q_f_init
                    self.fullburst_dict["U_f"] = self.U_f_init
                    self.fullburst_dict["V_f"] = self.V_f_init

                    self.fullburst_dict["I_f_init"] = self.I_f_init
                    self.fullburst_dict["Q_f_init"] = self.Q_f_init
                    self.fullburst_dict["U_f_init"] = self.U_f_init
                    self.fullburst_dict["V_f_init"] = self.V_f_init

                    self.fullburst_dict["PA_f"] = self.PA_f_init
                    self.fullburst_dict["PA_f_errs"] = self.PA_f_errs_init

                    self.fullburst_dict["PA_f_init"] = self.PA_f_init
                    self.fullburst_dict["PA_f_errs_init"] = self.PA_f_errs_init

                    self.fullburst_dict["PA_pre"] = avg_PA
                    self.fullburst_dict["PAerr_pre"] = sigma_PA

                    self.fullburst_dict["T/I_pre"] = avg_frac
                    self.fullburst_dict["T/I_pre_err"] = sigma_frac
                    self.fullburst_dict["T/I_pre_snr"] = snr_frac
                    self.fullburst_dict["L/I_pre"] = avg_L
                    self.fullburst_dict["L/I_pre_err"] = sigma_L
                    self.fullburst_dict["L/I_pre_snr"] = snr_L
                    self.fullburst_dict["absV/I"] = avg_C_abs
                    self.fullburst_dict["absV/I_err"] = sigma_C_abs
                    self.fullburst_dict["V/I"] = avg_C
                    self.fullburst_dict["V/I_err"] = sigma_C
                    self.fullburst_dict["V/I_snr"] = snr_C
                    self.fullburst_dict["I_snr"] = snr


                    self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute polarization"

                    self.STEP = 1
                    
                    self.filt_weights_on = False
                    self.freq_samp_on = True
                    self.wait = False
                    #self.error2 = str(self.curr_comp) 
                else:
                    #save filter weights for current component
                    self.comp_dict[self.curr_comp] =dict()
                    #get actual timestart, timestop
                    peak,t1,t2 = dsapol.find_peak([self.I_t,self.I_t],self.ibox,self.fobj.header.tsamp,self.n_t,peak_range=None,pre_calc_tf=True,buff=[self.buff_L,self.buff_R])
                    self.comp_dict[self.curr_comp]["timestart"] = t1
                    self.comp_dict[self.curr_comp]["timestop"] = t2
                    self.comp_dict[self.curr_comp]["comp_num"] = self.curr_comp
                    self.comp_dict[self.curr_comp]["buff"] = [self.buff_L,self.buff_R]
                    self.comp_dict[self.curr_comp]["n_t_weight"] = self.n_t_weight
                    self.comp_dict[self.curr_comp]["sf_window_weights"] = self.sf_window_weights
                    self.comp_dict[self.curr_comp]["ibox"] = self.ibox
                    self.comp_dict[self.curr_comp]["mask_start"] = self.complist_min[self.curr_comp]
                    self.comp_dict[self.curr_comp]["mask_stop"] = self.complist_max[self.curr_comp]
                    self.comp_dict[self.curr_comp]["weights"] = self.curr_weights
                    self.comp_dict[self.curr_comp]["multipeaks"] = self.multipeaks
                    self.comp_dict[self.curr_comp]["sigflag"] = False
                    self.comp_dict[self.curr_comp]["rm_applied"] = False

                    if self.multipeaks:
                        self.comp_dict[self.curr_comp]["height"] = self.height
                        self.comp_dict[self.curr_comp]["scaled_height"] = self.height*np.max(self.curr_weights)/np.max(self.I_t)

                    #get intL, intR
                    if self.multipeaks and self.height < np.max(self.I_t):
                        pks,props = find_peaks((self.curr_weights*np.max(self.I_t)/np.max(self.curr_weights))[timestart:timestop],height=self.height)
                        FWHM,heights,intL,intR = peak_widths((self.curr_weights*np.max(self.I_t)/np.max(self.curr_weights))[timestart:timestop],pks)
                        intL = int(intL[0])
                        intR = int(intR[-1])
                    elif self.multipeaks:
                        intL = np.nan
                        intR = np.nan
                    else:
                        FWHM,heights,intL,intR = peak_widths(self.curr_weights[timestart:timestop],[np.argmax(self.curr_weights[timestart:timestop])])
                        intL = int(intL)
                        intR = int(intR)
                    self.comp_dict[self.curr_comp]["intL"] = intL
                    self.comp_dict[self.curr_comp]["intR"] = intR
                    self.intLs.append(intL)
                    self.intRs.append(intR)


                    self.error = "Computing Spectrum..."
                    t1 = time.time()
                    (I_fmasked,Q_fmasked,U_fmasked,V_fmasked) = dsapol.get_stokes_vs_freq(self.I,self.Q,self.U,self.V,1,self.fobj.header.tsamp,1,self.n_t,self.freq_test_init,n_off=int(12000/self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,input_weights=self.curr_weights)
                    self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute spectrum"

                    self.comp_dict[self.curr_comp]["I_f"] = I_fmasked
                    self.comp_dict[self.curr_comp]["Q_f"] = Q_fmasked
                    self.comp_dict[self.curr_comp]["U_f"] = U_fmasked
                    self.comp_dict[self.curr_comp]["V_f"] = V_fmasked

                    self.comp_dict[self.curr_comp]["I_f_init"] = I_fmasked
                    self.comp_dict[self.curr_comp]["Q_f_init"] = Q_fmasked
                    self.comp_dict[self.curr_comp]["U_f_init"] = U_fmasked
                    self.comp_dict[self.curr_comp]["V_f_init"] = V_fmasked


                    #get PA vs frequency
                    self.error = "Computing Position Angle..."
                    t1 = time.time()
                    #PA_fmasked,tmpPA_t_init,PA_f_errsmasked,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                    PA_fmasked,tmpPA_t_init,PA_f_errsmasked,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle((self.I_t,I_fmasked),(self.Q_t,Q_fmasked),(self.U_t,U_fmasked),(self.V_t,V_fmasked),self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,pre_calc_tf=True,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                    self.comp_dict[self.curr_comp]["PA_f"] = PA_fmasked
                    self.comp_dict[self.curr_comp]["PA_f_errs"] = PA_f_errsmasked

                    self.comp_dict[self.curr_comp]["PA_f_init"] = PA_fmasked
                    self.comp_dict[self.curr_comp]["PA_f_errs_init"] = PA_f_errsmasked

                    self.comp_dict[self.curr_comp]["PA_pre"] = avg_PA
                    self.comp_dict[self.curr_comp]["PAerr_pre"] = sigma_PA
                    self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute position angle"



                    #get polarization fractions
                    self.error = "Computing polarization..."
                    t1 = time.time()
                    #[(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000/self.n_t),normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                    [(pol_t,avg_frac,sigma_frac,snr_frac),(L_t,avg_L,sigma_L,snr_L),(C_t,avg_C_abs,sigma_C_abs,snr_C),(C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction_vs_time(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,n_off=int(12000/self.n_t),normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                    self.comp_dict[self.curr_comp]["T/I_pre"] = avg_frac
                    self.comp_dict[self.curr_comp]["T/I_pre_err"] = sigma_frac
                    self.comp_dict[self.curr_comp]["T/I_pre_snr"] = snr_frac
                    self.comp_dict[self.curr_comp]["L/I_pre"] = avg_L
                    self.comp_dict[self.curr_comp]["L/I_pre_err"] = sigma_L
                    self.comp_dict[self.curr_comp]["L/I_pre_snr"] = snr_L
                    self.comp_dict[self.curr_comp]["absV/I"] = avg_C_abs
                    self.comp_dict[self.curr_comp]["absV/I_err"] = sigma_C_abs
                    self.comp_dict[self.curr_comp]["V/I"] = avg_C
                    self.comp_dict[self.curr_comp]["V/I_err"] = sigma_C
                    self.comp_dict[self.curr_comp]["V/I_snr"] = snr_C
                    self.comp_dict[self.curr_comp]["I_snr"] = snr

                    #update displays
                    self.snr = self.snr + r'{a}'.format(a=np.around(snr,2))  + ') '
                    self.Tsnr = self.Tsnr + r'{a}'.format(a=np.around(snr_frac,2)) + ') '
                    self.Lsnr = self.Lsnr + r'{a}'.format(a=np.around(snr_L,2)) + ') '
                    self.Csnr = self.Csnr + r'{a}'.format(a=np.around(snr_C,2)) + ') '

                    self.Tpol = self.Tpol + r'{a}%'.format(a=np.around(100*avg_frac,2)) + ') '
                    self.Lpol = self.Lpol + r'{a}%'.format(a=np.around(100*avg_L,2)) + ') '
                    self.absCpol = self.absCpol + r'{a}%'.format(a=np.around(100*avg_C_abs,2)) + ') '
                    self.Cpol = self.Cpol + r'{a}%'.format(a=np.around(100*avg_C,2)) + ') '

                    self.Tpolerr = self.Tpolerr + r'{a}%'.format(a=np.around(100*sigma_frac,2)) + ') '
                    self.Lpolerr = self.Lpolerr + r'{a}%'.format(a=np.around(100*sigma_L,2)) + ') '
                    self.absCpolerr = self.absCpolerr + r'{a}%'.format(a=np.around(100*sigma_C_abs,2)) + ') '
                    self.Cpolerr = self.Cpolerr + r'{a}%'.format(a=np.around(100*sigma_C,2)) + ') '
                    
                    self.avgPA = self.avgPA + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2)) + ') '
                    self.avgPAerr = self.avgPAerr + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2)) + ') '
                    self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute polarization"

                    self.curr_comp += 1

                    self.curr_weights = np.zeros(len(self.curr_weights))
                    for i in range(len(self.fixed_comps)):
                        self.curr_weights += self.comp_dict[i]["weights"]
                    self.curr_weights = self.curr_weights/np.sum(self.curr_weights)
                    #self.filt_weights_on = False
                    #self.freq_samp_on = True

                    self.error = "Computing Full Spectrum..."
                    t1 = time.time()
                    (self.I_f_init,self.Q_f_init,self.U_f_init,self.V_f_init) = dsapol.get_stokes_vs_freq(self.I,self.Q,self.U,self.V,1,self.fobj.header.tsamp,1,self.n_t,self.freq_test_init,n_off=int(12000/self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,input_weights=self.curr_weights)
                    self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute full spectrum"


                    #unmask time series
                    self.I_t = self.fullburst_dict["I_t"]
                    self.Q_t = self.fullburst_dict["Q_t"]
                    self.U_t = self.fullburst_dict["U_t"]
                    self.V_t = self.fullburst_dict["V_t"]

                    #get PA vs frequency
                    self.error = "Computing Position Angle..."
                    t1 = time.time()
                    #self.PA_f_init,tmpPA_t_init,self.PA_f_errs_init,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                    self.PA_f_init,tmpPA_t_init,self.PA_f_errs_init,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle((self.I_t,self.I_f_init),(self.Q_t,self.Q_f_init),(self.U_t,self.U_f_init),(self.V_t,self.V_f_init),self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,pre_calc_tf=True,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                    self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute position angle"



                    #get total polarization
                    self.error = "Computing total polarization..."

                    multipeaks_all = (len(self.fixed_comps) > 1) or (self.comp_dict[0]["multipeaks"])

                    t1 = time.time()
                    [(pol_t,avg_frac,sigma_frac,snr_frac),(L_t,avg_L,sigma_L,snr_L),(C_t,avg_C_abs,sigma_C_abs,snr_C),(C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction_vs_time(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,n_off=int(12000/self.n_t),normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=multipeaks_all,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                    self.snr = self.snr + r'{a}'.format(a=np.around(snr,2)) 
                    self.Tsnr = self.Tsnr + r'{a}'.format(a=np.around(snr_frac,2)) 
                    self.Lsnr = self.Lsnr + r'{a}'.format(a=np.around(snr_L,2)) 
                    self.Csnr = self.Csnr + r'{a}'.format(a=np.around(snr_C,2)) 

                    self.Tpol = self.Tpol + r'{a}%'.format(a=np.around(100*avg_frac,2)) 
                    self.Lpol = self.Lpol + r'{a}%'.format(a=np.around(100*avg_L,2)) 
                    self.absCpol = self.absCpol + r'{a}%'.format(a=np.around(100*avg_C_abs,2)) 
                    self.Cpol = self.Cpol + r'{a}%'.format(a=np.around(100*avg_C,2)) 

                    self.Tpolerr = self.Tpolerr + r'{a}%'.format(a=np.around(100*sigma_frac,2)) 
                    self.Lpolerr = self.Lpolerr + r'{a}%'.format(a=np.around(100*sigma_L,2)) 
                    self.absCpolerr = self.absCpolerr + r'{a}%'.format(a=np.around(100*sigma_C_abs,2)) 
                    self.Cpolerr = self.Cpolerr + r'{a}%'.format(a=np.around(100*sigma_C,2))

                    self.avgPA = self.avgPA + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))
                    self.avgPAerr = self.avgPAerr + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))
                    self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute full polarization"



                    #save full burst properties to dict
                    tmp1 = []
                    tmp2 = []
                    b1 = []
                    b2 = []
                    intL1 = []
                    intR2 = []
                    for k in range(len(self.comp_dict.keys())):
                        tmp1.append(self.comp_dict[k]["timestart"])
                        tmp2.append(self.comp_dict[k]["timestop"])
                        b1.append(self.comp_dict[k]["buff"][0])
                        b2.append(self.comp_dict[k]["buff"][1])
                        intL1.append(self.comp_dict[k]["intL"])
                        intR2.append(self.comp_dict[k]["intR"])

                    self.fullburst_dict["timestart"] = np.min(tmp1)
                    self.fullburst_dict["timestop"] = np.max(tmp2)
                    self.fullburst_dict["num_comps"] = len(self.comp_dict.keys())
                    self.fullburst_dict["buff"] = (np.min(b1),np.max(b2))
                    self.fullburst_dict["weights"] = self.curr_weights
                    self.fullburst_dict["multipeaks_all"] = multipeaks_all
                    self.fullburst_dict["sigflag"] = False
                    self.fullburst_dict["intL"] = np.min(intL1)
                    self.fullburst_dict["intR"] = np.max(intR2)


                    self.fullburst_dict["I_f"] = self.I_f_init
                    self.fullburst_dict["Q_f"] = self.Q_f_init
                    self.fullburst_dict["U_f"] = self.U_f_init
                    self.fullburst_dict["V_f"] = self.V_f_init

                    self.fullburst_dict["I_f_init"] = self.I_f_init
                    self.fullburst_dict["Q_f_init"] = self.Q_f_init
                    self.fullburst_dict["U_f_init"] = self.U_f_init
                    self.fullburst_dict["V_f_init"] = self.V_f_init

                    self.fullburst_dict["PA_f"] = self.PA_f_init
                    self.fullburst_dict["PA_f_errs"] = self.PA_f_errs_init

                    self.fullburst_dict["PA_f_init"] = self.PA_f_init
                    self.fullburst_dict["PA_f_errs_init"] = self.PA_f_errs_init

                    self.fullburst_dict["PA_pre"] = avg_PA
                    self.fullburst_dict["PAerr_pre"] = sigma_PA

                    self.fullburst_dict["T/I_pre"] = avg_frac
                    self.fullburst_dict["T/I_pre_err"] = sigma_frac
                    self.fullburst_dict["T/I_pre_snr"] = snr_frac
                    self.fullburst_dict["L/I_pre"] = avg_L
                    self.fullburst_dict["L/I_pre_err"] = sigma_L
                    self.fullburst_dict["L/I_pre_snr"] = snr_L
                    self.fullburst_dict["absV/I"] = avg_C_abs
                    self.fullburst_dict["absV/I_err"] = sigma_C_abs
                    self.fullburst_dict["V/I"] = avg_C
                    self.fullburst_dict["V/I_err"] = sigma_C
                    self.fullburst_dict["V/I_snr"] = snr_C
                    self.fullburst_dict["I_snr"] = snr
                    self.fullburst_dict["rm_applied"] = False

                    self.filt_weights_on = False
                    self.freq_samp_on = True

                    self.STEP = 1
                self.param.trigger('done')
            elif self.freq_samp_on:
                self.I = dsapol.avg_freq(self.I,self.n_f)
                self.Q = dsapol.avg_freq(self.Q,self.n_f)
                self.U = dsapol.avg_freq(self.U,self.n_f)
                self.V = dsapol.avg_freq(self.V,self.n_f)

                self.finished = True
                self.freq_samp_on = False

        except Exception as e:
            self.error2 = str(len(self.fixed_comps)) + " " + "From clicked_done(): " + str(e) + " " + str(self.comp_dict.keys())
            #self.error2 = "From clicked_done(): " + str(len(self.fixed_comps)) + str(self.curr_comp)
        return

    #summary plot
    def clicked_plot(self):
        try:
            suffix="_TESTING"

            if self.filt_weights_on and len(self.comp_dict.keys()) > 0 :
                i = len(self.comp_dict.keys())-1

                self.error = "Exporting summary plot for Component " + str(i+1) + "..."
                t1 = time.time()

                if self.comp_dict[i]["sigflag"]:
                    I1 = copy.deepcopy(self.I_RMcal)
                    Q1 = copy.deepcopy(self.Q_RMcal)
                    U1 = copy.deepcopy(self.U_RMcal)
                    V1 = copy.deepcopy(self.V_RMcal)
                else:
                    I1 = copy.deepcopy(self.I)
                    Q1 = copy.deepcopy(self.Q)
                    U1 = copy.deepcopy(self.U)
                    V1 = copy.deepcopy(self.V)

                for j in range(len(self.fixed_comps)):
                    if i != j:
                        mask = np.zeros(self.I.shape)
                        TSTART = int(int(15280/self.n_t) - (5e-3)/(self.n_t*32.7e-6)) + int(self.complist_min[j])
                        TSTOP = int(int(15280/self.n_t) - (5e-3)/(self.n_t*32.7e-6)) + int(self.complist_max[j])
                        mask[:,int(TSTART):int(TSTOP)] = 1

                        I1 = ma.masked_array(I1,mask)
                        Q1 = ma.masked_array(Q1,mask)
                        U1 = ma.masked_array(U1,mask)
                        V1 = ma.masked_array(V1,mask)

                dsapol.pol_summary_plot(I1,Q1,U1,V1,self.ids,self.nickname,self.comp_dict[i]["ibox"],self.fobj.header.tsamp,self.n_t,self.n_f,self.freq_test,self.timeaxis,self.fobj,n_off=int(12000/self.n_t),buff=self.comp_dict[i]["buff"],weighted=True,n_t_weight=self.comp_dict[i]["n_t_weight"],sf_window_weights=self.comp_dict[i]["sf_window_weights"],show=False,input_weights=self.comp_dict[i]["weights"],intL=self.timestart + self.intLs[i],intR=self.timestart + self.intRs[i],multipeaks=self.comp_dict[i]["multipeaks"],wind=self.n_t,suffix="_PEAK" + str(i+1) + suffix,mask_flag=False,sigflag=self.comp_dict[i]["sigflag"],plot_weights=False,timestart=self.comp_dict[i]["timestart"],timestop=self.comp_dict[i]["timestop"])

                self.error = "Completed: " + str(np.around(time.time()-t1,2)) + " s to export summary plot to " + self.datadir + self.ids + "_" + self.nickname + "_pol_summary_plot"+ "_PEAK" + str(i+1) + suffix + ".pdf"




            elif self.finished:

                #first update previous component plots
                if len(self.comp_dict.keys()) > 1:
                    for i in range(len(self.comp_dict.keys())):

                        self.error = "Exporting summary plot for Component " + str(i+1) + "..."
                        t1 = time.time()

                        if self.comp_dict[i]["sigflag"]:
                            I1 = copy.deepcopy(self.I_RMcal)
                            Q1 = copy.deepcopy(self.Q_RMcal)
                            U1 = copy.deepcopy(self.U_RMcal)
                            V1 = copy.deepcopy(self.V_RMcal)
                        else:
                            I1 = copy.deepcopy(self.I)
                            Q1 = copy.deepcopy(self.Q)
                            U1 = copy.deepcopy(self.U)
                            V1 = copy.deepcopy(self.V)

                        for j in range(len(self.fixed_comps)):
                            if i != j:
                                mask = np.zeros(self.I.shape)
                                TSTART = int(int(15280/self.n_t) - (5e-3)/(self.n_t*32.7e-6)) + int(self.complist_min[j])
                                TSTOP = int(int(15280/self.n_t) - (5e-3)/(self.n_t*32.7e-6)) + int(self.complist_max[j])
                                mask[:,int(TSTART):int(TSTOP)] = 1

                                I1 = ma.masked_array(I1,mask)
                                Q1 = ma.masked_array(Q1,mask)
                                U1 = ma.masked_array(U1,mask)
                                V1 = ma.masked_array(V1,mask)

                        dsapol.pol_summary_plot(I1,Q1,U1,V1,self.ids,self.nickname,self.comp_dict[i]["ibox"],self.fobj.header.tsamp,self.n_t,self.n_f,self.freq_test,self.timeaxis,self.fobj,n_off=int(12000/self.n_t),buff=self.comp_dict[i]["buff"],weighted=True,n_t_weight=self.comp_dict[i]["n_t_weight"],sf_window_weights=self.comp_dict[i]["sf_window_weights"],show=False,input_weights=self.comp_dict[i]["weights"],intL=self.timestart + self.intLs[i],intR=self.timestart + self.intRs[i],multipeaks=self.comp_dict[i]["multipeaks"],wind=self.n_t,suffix="_PEAK" + str(i+1) + suffix,mask_flag=False,sigflag=self.comp_dict[i]["sigflag"],plot_weights=False,timestart=self.comp_dict[i]["timestart"],timestop=self.comp_dict[i]["timestop"])

                        self.error = "Completed: " + str(np.around(time.time()-t1,2)) + " s to export summary plot to " + self.datadir + self.ids + "_" + self.nickname + "_pol_summary_plot"+ "_PEAK" + str(i+1) + suffix + ".pdf"



                #final plot
                if self.sigflag:
                    I1 = copy.deepcopy(self.I_RMcal)
                    Q1 = copy.deepcopy(self.Q_RMcal)
                    U1 = copy.deepcopy(self.U_RMcal)
                    V1 = copy.deepcopy(self.V_RMcal)
                else:
                    I1 = copy.deepcopy(self.I)
                    Q1 = copy.deepcopy(self.Q)
                    U1 = copy.deepcopy(self.U)
                    V1 = copy.deepcopy(self.V)


                self.error = "Exporting summary plot..."
                t1 = time.time()

                buffLall = self.comp_dict[0]["buff"][0]
                buffRall = self.comp_dict[len(self.comp_dict.keys())-1]["buff"][1]
                buffall = [buffLall,buffRall]

                multipeaks_all = (len(self.fixed_comps) > 1) or (self.comp_dict[0]["multipeaks"])

                timestartall = self.comp_dict[0]["timestart"]
                timestopall = self.comp_dict[len(self.comp_dict.keys())-1]["timestop"]

                dsapol.pol_summary_plot(I1,Q1,U1,V1,self.ids,self.nickname,self.ibox,self.fobj.header.tsamp,self.n_t,self.n_f,self.freq_test,self.timeaxis,self.fobj,n_off=int(12000/self.n_t),buff=buffall,weighted=True,n_t_weight=self.n_t_weight,sf_window_weights=self.sf_window_weights,show=False,input_weights=self.curr_weights,intL=self.timestart + np.min(self.intLs),intR=self.timestart + np.max(self.intRs),multipeaks=multipeaks_all,wind=self.n_t,suffix=suffix,mask_flag=self.maskPA,sigflag=self.sigflag,plot_weights=False,timestart=timestartall,timestop=timestopall)
                #datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+self.ids + "_" + self.nickname + "/"
                #self.error = str(buffall) + " " +str(self.intLs) + " " + str(self.intRs)+ " " 
                self.error = "Completed: " + str(np.around(time.time()-t1,2)) + "s to export summary plot to " + self.datadir + self.ids + "_" + self.nickname + "_pol_summary_plot"+ suffix + ".pdf"
            else:
                self.error = "Not done yet"
            return
        except Exception as e:
            self.error = "From clicked_plot(): " + str(e)
            return
    #up = param.Action(clicked_up)#clicked)
    #down = param.Action(clicked_down)
    next_comp = param.Action(clicked_next,label="Next")
    get_comp = param.Action(clicked_get,label="Get Components")
    done = param.Action(clicked_done,label='Done')
    exportplot = param.Action(clicked_plot,label='Export Summary Plot')



    #initialization and calibration buttons
    load_button = param.Action(load_FRB,label="Load FRB")
    cal_button = param.Action(cal_FRB,label="Calibrate")
    savefil_button = param.Action(savefil,label="Save Calibrated Filterbanks")
    savefilRM_button = param.Action(savefilRM,label="Save RM Calibrated Filterbanks")
    savetxt_button = param.Action(savetxt,label="Export Text File Summary")
    savejson_button = param.Action(savejson,label="Save Data to JSON")
    #addtocatalog_button = param.Action(addtocatalog,label="Add to DSA-110 Catalog")


    #polarization
    snr = param.String(default="",label="S/N")
    Tsnr = param.String(default="",label="T S/N")
    Tpol = param.String(default="",label="T/I")
    Tpolerr = param.String(default="",label="error")
    Lsnr = param.String(default="",label="L S/N")
    Lpol = param.String(default="",label="L/I")
    Lpolerr = param.String(default="",label="error")
    Csnr = param.String(default="",label="V S/N")
    absCpol = param.String(default="",label="|V|/I")
    absCpolerr = param.String(default="",label="error")
    Cpol = param.String(default="",label="V/I")
    Cpolerr = param.String(default="",label="error")
    avgPA = param.String(default="",label="PA (degrees)")
    avgPAerr = param.String(default="",label="error")

    #***FILTER WEIGHTS MODULE***#
    comp_dict = dict()
    fullburst_dict = dict()
    curr_comp = 0
    curr_weights = []

    #click_data = param.Dict(doc="test")
    #print(click_data)



    #***SAVING RM INTERMEDIATE DATA TO PKL FILES***#
    #link IQUV from panel1 to panel 2
    def clicked_pkl_save(self):
        try:
            self.error = "Transferring data between panels..."
            t1 = time.time()

            
            #add to RM table
            if (len(self.comp_dict.keys()) <= len(self.fixed_comps)) and (len(self.comp_dict.keys()) > 0) and self.filt_weights_on:
                curr_comp = np.max(list(self.comp_dict.keys()))
                self.error = "Writing component " +str(curr_comp) + " to RMTable..."

                #check if in table
                tabidx,ncomps = dsarp_interface.dsarp_FRBinTable_RMTable(self.ids,curr_comp)

                #if not in table, add normally
                if tabidx == -1: 
                    dsarp_interface.dsarp_addFRBcomp_RMTable(self.ids,self.nickname,self.datadir,self.comp_dict,curr_comp,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps))
                #if in table, but wrong number of components, remove everything and add
                elif tabidx != -1 and ncomps != len(self.fixed_comps):
                    dsarp_interface.dsarp_removeFRBcomp_RMTable(self.ids)
                    dsarp_interface.dsarp_addFRBcomp_RMTable(self.ids,self.nickname,self.datadir,self.comp_dict,curr_comp,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),replace=True,replace_idx=tabidx)
                #if in table and correct number of components, update values in table
                elif tabidx != -1 and ncomps == len(self.fixed_comps):
                    dsarp_interface.dsarp_addFRBcomp_RMTable(self.ids,self.nickname,self.datadir,self.comp_dict,curr_comp,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),replace=True,replace_idx=tabidx)
                

            elif (len(self.comp_dict.keys()) == len(self.fixed_comps)) and (not self.filt_weights_on):
                curr_comp = -1
                self.error = "Writing full burst component to RMTable..."

                #check if in table
                tabidx,ncomps = dsarp_interface.dsarp_FRBinTable_RMTable(self.ids,curr_comp)

                #if not in table, add normally
                if tabidx == -1:
                    dsarp_interface.dsarp_addFRBfull_RMTable(self.ids,self.nickname,self.datadir,self.fullburst_dict,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps))
                #if in table, but wrong number of components, remove everything and add
                elif tabidx != -1 and ncomps != len(self.fixed_comps):
                    dsarp_interface.dsarp_removeFRBcomp_RMTable(self.ids)
                    dsarp_interface.dsarp_addFRBfull_RMTable(self.ids,self.nickname,self.datadir,self.fullburst_dict,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),replace=True,replace_idx=tabidx)
                #if in table and correct number of components, update values in table
                elif tabidx != -1 and ncomps == len(self.fixed_comps):
                    dsarp_interface.dsarp_addFRBfull_RMTable(self.ids,self.nickname,self.datadir,self.fullburst_dict,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),replace=True,replace_idx=tabidx)

            #add to pre pol table
            if (len(self.comp_dict.keys()) <= len(self.fixed_comps)) and (len(self.comp_dict.keys()) > 0) and self.filt_weights_on and not self.comp_dict[np.max(list(self.comp_dict.keys()))]["rm_applied"]:
                curr_comp = np.max(list(self.comp_dict.keys()))
                self.error = "Writing component " +str(curr_comp) + " to pre PolSpectra Table..."
                
                #check if in table
                tabidx,ncomps = dsarp_interface.dsarp_FRBinTable_PolSpectrum(self.ids,curr_comp)

                #if not in table, add normally
                if tabidx == -1:
                    self.error = "Not in PolSpectra Table, appending entry..."
                    dsarp_interface.dsarp_addFRBcomp_PolSpectrum(self.ids,self.nickname,self.datadir,self.comp_dict,curr_comp,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),self.n_t,self.n_f,self.fobj.header.tsamp,suff="pre")
                #if in table, but wrong number of components, remove everything and add
                elif tabidx != -1 and ncomps != len(self.fixed_comps):
                    self.error = "In PolSpectra Table but wrong number of components, replacing entry..."
                    dsarp_interface.dsarp_removeFRBcomp_PolSpectrum(self.ids,suff="pre")
                    dsarp_interface.dsarp_addFRBcomp_PolSpectrum(self.ids,self.nickname,self.datadir,self.comp_dict,curr_comp,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),self.n_t,self.n_f,self.fobj.header.tsamp,replace=True,replace_idx=tabidx,suff="pre")
                #if in table and correct number of components, update values in table
                elif tabidx != -1 and ncomps == len(self.fixed_comps):
                    self.error = "In PolSpectra Table, updatig entry..."
                    dsarp_interface.dsarp_addFRBcomp_PolSpectrum(self.ids,self.nickname,self.datadir,self.comp_dict,curr_comp,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),self.n_t,self.n_f,self.fobj.header.tsamp,replace=True,replace_idx=tabidx,suff="pre")
                self.error = "Done, proceeding..."
            elif (len(self.comp_dict.keys()) == len(self.fixed_comps)) and (not self.filt_weights_on) and not self.fullburst_dict["rm_applied"]:
                curr_comp = -1
                self.error = "Writing full burst component to pre PolSpectra Table..."


                #check if in table
                tabidx,ncomps = dsarp_interface.dsarp_FRBinTable_PolSpectrum(self.ids,curr_comp)

                #if not in table, add normally
                if tabidx == -1:
                    self.error = "Not in PolSpectra Table, appending entry..."
                    dsarp_interface.dsarp_addFRBfull_PolSpectrum(self.ids,self.nickname,self.datadir,self.fullburst_dict,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),self.n_t,self.n_f,self.fobj.header.tsamp,len(self.fixed_comps),suff="pre")
                #if in table, but wrong number of components, remove everything and add
                elif tabidx != -1 and ncomps != len(self.fixed_comps):
                    self.error = "In PolSpectra Table but wrong number of components, replacing entry..."
                    dsarp_interface.dsarp_removeFRBcomp_PolSpectrum(self.ids,suff="pre")
                    dsarp_interface.dsarp_addFRBfull_PolSpectrum(self.ids,self.nickname,self.datadir,self.fullburst_dict,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),self.n_t,self.n_f,self.fobj.header.tsamp,replace=True,replace_idx=tabidx,suff="pre")
                #if in table and correct number of components, update values in table
                elif tabidx != -1 and ncomps == len(self.fixed_comps):
                    self.error = "In PolSpectra Table, updatig entry..."
                    dsarp_interface.dsarp_addFRBfull_PolSpectrum(self.ids,self.nickname,self.datadir,self.fullburst_dict,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),self.n_t,self.n_f,self.fobj.header.tsamp,replace=True,replace_idx=tabidx,suff="pre")
                self.error = "Done, proceeding..."

            #add to post-pol table if applied RM
            if (len(self.comp_dict.keys()) <= len(self.fixed_comps)) and (len(self.comp_dict.keys()) > 0) and self.filt_weights_on and self.comp_dict[np.max(list(self.comp_dict.keys()))]["rm_applied"]:
                curr_comp = np.max(list(self.comp_dict.keys()))
                self.error = "Writing component " +str(curr_comp) + " to post PolSpectra Table..."

                #check if in table
                tabidx,ncomps = dsarp_interface.dsarp_FRBinTable_PolSpectrum(self.ids,curr_comp,suff="post")

                #if not in table, add normally
                if tabidx == -1:
                    self.error = "Not in PolSpectra Table, appending entry..."
                    dsarp_interface.dsarp_addFRBcomp_PolSpectrum(self.ids,self.nickname,self.datadir,self.comp_dict,curr_comp,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),self.n_t,self.n_f,self.fobj.header.tsamp,suff="post")
                #if in table, but wrong number of components, remove everything and add
                elif tabidx != -1 and ncomps != len(self.fixed_comps):
                    self.error = "In PolSpectra Table but wrong number of components, replacing entry..."
                    dsarp_interface.dsarp_removeFRBcomp_PolSpectrum(self.ids,suff="post")
                    dsarp_interface.dsarp_addFRBcomp_PolSpectrum(self.ids,self.nickname,self.datadir,self.comp_dict,curr_comp,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),self.n_t,self.n_f,self.fobj.header.tsamp,replace=True,replace_idx=tabidx,suff="post")
                #if in table and correct number of components, update values in table
                elif tabidx != -1 and ncomps == len(self.fixed_comps):
                    self.error = "In PolSpectra Table, updatig entry..."
                    dsarp_interface.dsarp_addFRBcomp_PolSpectrum(self.ids,self.nickname,self.datadir,self.comp_dict,curr_comp,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),self.n_t,self.n_f,self.fobj.header.tsamp,replace=True,replace_idx=tabidx,suff="post")
                self.error = "Done, proceeding..."


            elif (len(self.comp_dict.keys()) == len(self.fixed_comps)) and (not self.filt_weights_on) and self.fullburst_dict["rm_applied"]:
                curr_comp = -1
                self.error = "Writing full burst component to post PolSpectra Table..."


                #check if in table
                tabidx,ncomps = dsarp_interface.dsarp_FRBinTable_PolSpectrum(self.ids,curr_comp,suff="post")

                #if not in table, add normally
                if tabidx == -1:
                    self.error = "Not in PolSpectra Table, appending entry..."
                    dsarp_interface.dsarp_addFRBfull_PolSpectrum(self.ids,self.nickname,self.datadir,self.fullburst_dict,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),self.n_t,self.n_f,self.fobj.header.tsamp,len(self.fixed_comps),suff="post")
                #if in table, but wrong number of components, remove everything and add
                elif tabidx != -1 and ncomps != len(self.fixed_comps):
                    self.error = "In PolSpectra Table but wrong number of components, replacing entry..."
                    dsarp_interface.dsarp_removeFRBcomp_PolSpectrum(self.ids,suff="post")
                    dsarp_interface.dsarp_addFRBfull_PolSpectrum(self.ids,self.nickname,self.datadir,self.fullburst_dict,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),self.n_t,self.n_f,self.fobj.header.tsamp,replace=True,replace_idx=tabidx,suff="post")
                #if in table and correct number of components, update values in table
                elif tabidx != -1 and ncomps == len(self.fixed_comps):
                    self.error = "In PolSpectra Table, updatig entry..."
                    dsarp_interface.dsarp_addFRBfull_PolSpectrum(self.ids,self.nickname,self.datadir,self.fullburst_dict,self.RA,self.DEC,self.freq_test_init,self.MJD,len(self.fixed_comps),self.n_t,self.n_f,self.fobj.header.tsamp,replace=True,replace_idx=tabidx,suff="post")
                self.error = "Done, proceeding..."


            #write current data to pkl files
            dyn_spectra_dict = dict()
            dyn_spectra_dict["I"] = self.I
            dyn_spectra_dict["Q"] = self.Q
            dyn_spectra_dict["U"] = self.U
            f = open(tmp_file_dir + "dyn_spectra.pkl","wb")
            pkl.dump(dyn_spectra_dict,f)
            f.close()
            
            """
            parameters_dict = dict()
            if (len(self.comp_dict.keys()) <= len(self.fixed_comps)) and (len(self.comp_dict.keys()) > 0) and self.filt_weights_on:
                curr_comp = np.max(list(self.comp_dict.keys()))
                self.error = "Multiple Components, using component " + str(curr_comp) + " " + str(len(self.comp_dict[curr_comp]["I_f_init"]))
                spectra_dict = dict()
                spectra_dict["I_f"] = self.comp_dict[curr_comp]["I_f_init"]
                spectra_dict["Q_f"] = self.comp_dict[curr_comp]["Q_f_init"]
                spectra_dict["U_f"] = self.comp_dict[curr_comp]["U_f_init"]
                spectra_dict["freq_test"] = self.freq_test_init
                self.error = "yososo"
                f = open(tmp_file_dir + "1D_spectra.pkl","wb")
                pkl.dump(spectra_dict,f)
                f.close()
                self.error = "but here tho"
                parameters_dict["curr_comp"] = curr_comp
                parameters_dict["ibox"] = self.ibox
                parameters_dict["curr_weights"] = self.comp_dict[curr_comp]["weights"]
                self.error = "done here"
            elif (len(self.comp_dict.keys()) == len(self.fixed_comps)) and (not self.filt_weights_on):
                curr_comp = -1
                self.error = str(curr_comp) + "All Components " + str(len(self.I_f_init))
                spectra_dict = dict()
                spectra_dict["I_f"] = self.I_f_init#pan1.comp_dict[curr_comp]["I_f_init"]
                spectra_dict["Q_f"] = self.Q_f_init#pan1.comp_dict[curr_comp]["Q_f_init"]
                spectra_dict["U_f"] = self.U_f_init#pan1.comp_dict[curr_comp]["U_f_init"]
                spectra_dict["freq_test"] = self.freq_test_init
                f = open(tmp_file_dir + "1D_spectra.pkl","wb")
                pkl.dump(spectra_dict,f)
                f.close()

                parameters_dict["curr_comp"] = curr_comp
                parameters_dict["ibox"] = self.ibox
                parameters_dict["curr_weights"] = self.curr_weights#comp_dict[curr_comp]["weights"]



            parameters_dict["datadir"] = self.datadir
            self.error = "1"
            parameters_dict["ids"] = self.ids
            self.error = "2"
            parameters_dict["nickname"] = self.nickname
            self.error = "3"
            parameters_dict["n_t"] = self.n_t
            self.error = "4"
            parameters_dict["n_f"] = self.n_f
            self.error = "5"
            parameters_dict["tsamp"] = self.fobj.header.tsamp
            self.error = "6"
            parameters_dict["ibox"] = self.ibox
            self.error = "7"
            parameters_dict["RA"] = self.RA
            self.error = "8"
            parameters_dict["DEC"] = self.DEC
            self.error = "9"
            parameters_dict["MJD"] = self.MJD
            self.error = "10"
            parameters_dict["frb_name"] = self.frb_name
            



            f = open(tmp_file_dir + "parameters.pkl","wb")
            pkl.dump(parameters_dict,f)
            f.close()

            self.error = "wrote parameters"
            f = open(tmp_file_dir + "comp_dict.pkl","wb")
            self.error = "is here yes"
            pkl.dump(self.comp_dict,f)
            self.error = "or here yes"
            f.close()

            self.error = "wrote comp dict"
            f = open(tmp_file_dir + "fullburst_dict.pkl","wb")
            pkl.dump(self.fullburst_dict,f)
            f.close()

            ready_dict = dict()
            ready_dict["ready"] = True
            f = open(tmp_file_dir + "ready.pkl","wb")
            pkl.dump(ready_dict,f)
            f.close()
            """
            self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to transfer data"
            self.param.trigger('link_button')
            return
        except Exception as e:
            self.error = "From clicked_link(): " + str(e)
            return
    
    link_button = param.Action(clicked_pkl_save,label="Update DSA-110 Catalog")

    #function for recovering rm data from pickle files
    def clicked_pkl_load(self):
        try:
            self.error = "Retrieving rm data to pkl files..."
            t1 = time.time()
            

            #setup comp dict and fullburst dict with data from catalog
            if self.freq_samp_on: #note, only getting full burst dict in this case...
                self.fullburst_dict,tmp,parameters_dict,spectra_dict = dsarp_interface.dsarp_getFRBfull(self.ids)
            if self.filt_weights_on and ( self.curr_comp < len(self.fixed_comps)):
                self.comp_dict,parameters_dict,spectra_dict = dsarp_interface.dsarp_getFRBcomp(self.ids,self.curr_comp,np.arange(self.curr_comp+1))


            #get dynamic spectra
            #f = open(tmp_file_dir + "dyn_spectra.pkl","rb")
            #dyn_spectra_dict = pkl.load(f)
            #f.close()
            #self.I = dyn_spectra_dict["I"]
            #self.Q = dyn_spectra_dict["Q"]
            #self.U = dyn_spectra_dict["U"]
            #self.V = dyn_spectra_dict["V"]

            #get 1D spectra
            #f = open(tmp_file_dir + "1D_spectra.pkl","rb")
            #spectra_dict = pkl.load(f)
            #f.close()
            #self.I_f = spectra_dict["I_f"]
            #self.Q_f = spectra_dict["Q_f"]
            #self.U_f = spectra_dict["U_f"]
            #self.V_f = spectra_dict["V_f"]
            #self.freq_test = spectra_dict["freq_test"]


            #get parameters
            #f = open(tmp_file_dir + "parameters.pkl","rb")
            #parameters_dict = pkl.load(f)
            #f.close()
            """
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
            """
            """
            #write galactic and ionospheric rm
            f = open(tmp_file_dir + "rm_vals.pkl","rb")
            rm_dict = pkl.load(f)
            f.close()
            self.RM_gal = rm_dict['RM_gal']
            self.RM_galerr = rm_dict['RM_galerr']
            self.RM_ion = rm_dict['RM_ion']
            self.RM_ionerr = rm_dict['RM_ionerr']

            #update comp dict
            f = open(tmp_file_dir + "comp_dict.pkl","rb")
            self.comp_dict = pkl.load(f)
            f.close()

            #update full burst dict
            f = open(tmp_file_dir + "fullburst_dict.pkl","rb")
            self.fullburst_dict = pkl.load(f)
            f.close()
            """
            self.error = "Complete: " + str(np.around(time.time()-t1)) + " s to get RM from pkl files"

            self.param.trigger('RMdata_init')
        except Exception as e:
            self.error = "From clicked_pkl_load(): " + str(e)
        return
    RMdata_init = param.Action(clicked_pkl_load,label="Load RM from DSA-110 Catalog")



    #RM calibrate using data retrieved from pickle files
    def clicked_RMcal(self):
    #get most recently estimated RM

        try:
            #get current component
            """
            f = open(tmp_file_dir + "parameters.pkl","rb")
            parameters_dict = pkl.load(f)
            f.close()
            if "curr_comp" not in parameters_dict.keys():
                return
            curr_comp = parameters_dict["curr_comp"]
            """
            if self.filt_weights_on and ( self.curr_comp <= len(self.fixed_comps)-1):
                curr_comp = self.curr_comp
            elif self.freq_samp_on:
                curr_comp = -1


            if curr_comp == -1:
                if "RM2zoom" in self.fullburst_dict.keys():
                    rmcal = self.fullburst_dict["RM2zoom"]
                else:   
                    rmcal = self.fullburst_dict["RM1"]
                self.error = "Derotating full burst to RM = " + str(np.around(rmcal,2)) + " rad/m^2..."
                t1 = time.time()

                self.sigflag = True
                self.fullburst_dict["sigflag"] = True
                self.I_RMcal_init,self.Q_RMcal_init,self.U_RMcal_init,self.V_RMcal_init = dsapol.calibrate_RM(self.I_init,self.Q_init,self.U_init,self.V_init,rmcal,0,self.freq_test_init,stokes=True)
                self.I_RMcal,self.Q_RMcal,self.U_RMcal,self.V_RMcal = dsapol.calibrate_RM(self.I,self.Q,self.U,self.V,rmcal,0,self.freq_test,stokes=True)

                self.error = "step 1"
                (self.I_f_init,self.Q_f_init,self.U_f_init,self.V_f_init) = dsapol.get_stokes_vs_freq(self.I_RMcal,self.Q_RMcal,self.U_RMcal,self.V_RMcal,1,self.fobj.header.tsamp,1,self.n_t,self.freq_test_init,n_off=int(12000/self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,input_weights=self.curr_weights)
                self.error = "step 2"
                self.I_f, self.Q_f, self.U_f, self.V_f = self.I_f_init,self.Q_f_init,self.U_f_init,self.V_f_init
                (self.I_t_init,self.Q_t_init,self.U_t_init,self.V_t_init) = dsapol.get_stokes_vs_time(self.I_RMcal_init,self.Q_RMcal_init,self.U_RMcal_init,self.V_RMcal_init,1,self.fobj.header.tsamp,1,n_off=int(12000/1),plot=False,show=False,normalize=True)


                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " to derotate"

                #recompute polarization and PA
                self.error = "Re-computing polarization and PA..."
                t1 = time.time()
                multipeaks_all = (len(self.fixed_comps) > 1) or (self.comp_dict[0]["multipeaks"])
                [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction(self.I_RMcal,self.Q_RMcal,self.U_RMcal,self.V_RMcal,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000/self.n_t),normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=multipeaks_all,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)

                self.PA_f_init,tmpPA_t_init,self.PA_f_errs_init,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle(self.I_RMcal,self.Q_RMcal,self.U_RMcal,self.V_RMcal,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)

                self.fullburst_dict["PA_f_init"] = self.PA_f_init#PA_fmasked
                self.fullburst_dict["PA_f_errs_init"] = self.PA_f_errs_init#PA_f_errsmasked

                self.fullburst_dict["PA_f"] = self.PA_f_init#PA_fmasked
                self.fullburst_dict["PA_f_errs"] = self.PA_f_errs_init#PA_f_errsmasked

                self.fullburst_dict["PA_post"] = avg_PA
                self.fullburst_dict["PAerr_post"] = sigma_PA

                self.fullburst_dict["T/I_post"] = avg_frac
                self.fullburst_dict["T/I_post_err"] = sigma_frac
                self.fullburst_dict["T/I_post_snr"] = snr_frac
                self.fullburst_dict["L/I_post"] = avg_L
                self.fullburst_dict["L/I_post_err"] = sigma_L
                self.fullburst_dict["L/I_post_snr"] = snr_L
                self.fullburst_dict["absV/I"] = avg_C_abs
                self.fullburst_dict["absV/I_err"] = sigma_C_abs
                self.fullburst_dict["V/I"] = avg_C
                self.fullburst_dict["V/I_err"] = sigma_C
                self.fullburst_dict["V/I_snr"] = snr_C
                self.fullburst_dict["I_snr"] = snr
                self.fullburst_dict["rm_applied"] = True



                self.snr = self.snr[:self.snr.index(")") + 2] + r'{a}'.format(a=np.around(snr,2))
                self.Tsnr = self.Tsnr[:self.Tsnr.index(")") + 2] + r'{a}'.format(a=np.around(snr_frac,2))
                self.Lsnr = self.Lsnr[:self.Lsnr.index(")") + 2] + r'{a}'.format(a=np.around(snr_L,2))
                self.Csnr = self.Csnr[:self.Csnr.index(")") + 2] + r'{a}'.format(a=np.around(snr_C,2))

                self.Tpol = self.Tpol[:self.Tpol.index(")") + 2] + r'{a}%'.format(a=np.around(100*avg_frac,2))
                self.Lpol = self.Lpol[:self.Lpol.index(")") + 2] + r'{a}%'.format(a=np.around(100*avg_L,2))
                self.absCpol = self.absCpol[:self.absCpol.index(")") + 2] + r'{a}%'.format(a=np.around(100*avg_C_abs,2))
                self.Cpol = self.Cpol[:self.Cpol.index(")") + 2] + r'{a}%'.format(a=np.around(100*avg_C,2))

                self.Tpolerr = self.Tpolerr[:self.Tpolerr.index(")") + 2] + r'{a}%'.format(a=np.around(100*sigma_frac,2))
                self.Lpolerr = self.Lpolerr[:self.Lpolerr.index(")") + 2] + r'{a}%'.format(a=np.around(100*sigma_L,2))
                self.absCpolerr = self.absCpolerr[:self.absCpolerr.index(")") + 2] + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))
                self.Cpolerr = self.Cpolerr[:self.Cpolerr.index(")") + 2] + r'{a}%'.format(a=np.around(100*sigma_C,2))

                self.avgPA = self.avgPA[:self.avgPA.index(")") + 2] + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))
                self.avgPAerr = self.avgPAerr[:self.avgPAerr.index(")") + 2] + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))

                self.rmcalibrated_all = True
                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute polarization"


            #Case 2: calibrate current component
            elif curr_comp != -1:
                if "RM2zoom" in self.comp_dict[curr_comp].keys():
                    rmcal = self.comp_dict[curr_comp]["RM2zoom"]
                else:
                    rmcal = self.comp_dict[curr_comp]["RM1"]
                self.error = "Derotating component " + str(curr_comp) + " to RM = " + str(np.around(rmcal,2)) + " rad/m^2..."
                t1 = time.time()
                self.comp_dict[curr_comp]["sigflag"] = True
                #pan2.comp_dict[curr_comp]["sigflag"] = True
                self.I_RMcal_init,self.Q_RMcal_init,self.U_RMcal_init,self.V_RMcal_init = dsapol.calibrate_RM(self.I_init,self.Q_init,self.U_init,self.V_init,rmcal,0,self.freq_test_init,stokes=True)
                self.I_RMcal,self.Q_RMcal,self.U_RMcal,self.V_RMcal = dsapol.calibrate_RM(self.I,self.Q,self.U,self.V,rmcal,0,self.freq_test,stokes=True)



                self.comp_dict[curr_comp]["I_f_init"], self.comp_dict[curr_comp]["Q_f_init"], self.comp_dict[curr_comp]["U_f_init"], self.comp_dict[curr_comp]["V_f_init"] = dsapol.get_stokes_vs_freq(self.I_RMcal,self.Q_RMcal,self.U_RMcal,self.V_RMcal,1,self.fobj.header.tsamp,1,self.n_t,self.freq_test_init,n_off=int(12000/self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,input_weights=self.comp_dict[curr_comp]["weights"])
                self.comp_dict[curr_comp]["I_f"], self.comp_dict[curr_comp]["Q_f"], self.comp_dict[curr_comp]["U_f"], self.comp_dict[curr_comp]["V_f"] = self.comp_dict[curr_comp]["I_f_init"], self.comp_dict[curr_comp]["Q_f_init"], self.comp_dict[curr_comp]["U_f_init"], self.comp_dict[curr_comp]["V_f_init"]

                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " to derotate"

                #recompute polarization and PA
                self.error = "Re-computing polarization and PA..."
                t1 = time.time()
                if self.comp_dict[curr_comp]["multipeaks"]:
                    h = self.comp_dict[curr_comp]["scaled_height"]
                else:
                    h =-1
                [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction(self.I_RMcal,self.Q_RMcal,self.U_RMcal,self.V_RMcal,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000/self.n_t),normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.comp_dict[curr_comp]["multipeaks"],height=h,input_weights=self.comp_dict[curr_comp]["weights"])


                PA_fmasked,tmpPA_t_init,PA_f_errsmasked,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle(self.I_RMcal,self.Q_RMcal,self.U_RMcal,self.V_RMcal,self.comp_dict[curr_comp]["ibox"],self.fobj.header.tsamp,self.n_t,self.n_f,self.freq_test,n_off=int(12000//self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=h,input_weights=self.comp_dict[curr_comp]["weights"])
                self.comp_dict[curr_comp]["PA_f_init"] = PA_fmasked
                self.comp_dict[curr_comp]["PA_f_errs_init"] = PA_f_errsmasked

                self.comp_dict[curr_comp]["PA_f"] = PA_fmasked
                self.comp_dict[curr_comp]["PA_f_errs"] = PA_f_errsmasked

                self.comp_dict[curr_comp]["PA_post"] = avg_PA
                self.comp_dict[curr_comp]["PAerr_post"] = sigma_PA

                self.comp_dict[curr_comp]["T/I_post"] = avg_frac
                self.comp_dict[curr_comp]["T/I_post_err"] = sigma_frac
                self.comp_dict[curr_comp]["T/I_post_snr"] = snr_frac
                self.comp_dict[curr_comp]["L/I_post"] = avg_L
                self.comp_dict[curr_comp]["L/I_post_err"] = sigma_L
                self.comp_dict[curr_comp]["L/I_post_snr"] = snr_L
                self.comp_dict[curr_comp]["absV/I"] = avg_C_abs
                self.comp_dict[curr_comp]["absV/I_err"] = sigma_C_abs
                self.comp_dict[curr_comp]["V/I"] = avg_C
                self.comp_dict[curr_comp]["V/I_err"] = sigma_C
                self.comp_dict[curr_comp]["V/I_snr"] = snr_C
                self.comp_dict[curr_comp]["I_snr"] = snr
                self.comp_dict[curr_comp]["rm_applied"] = True

                #update displays
                if curr_comp == 0 and len(self.fixed_comps) == 1:
                    self.snr = '(' + r'{a}'.format(a=np.around(snr,2))+ ') '
                    self.Tsnr = '('+ r'{a}'.format(a=np.around(snr_frac,2))+ ') '
                    self.Lsnr = '(' + r'{a}'.format(a=np.around(snr_L,2))+ ') '
                    self.Csnr = '(' + r'{a}'.format(a=np.around(snr_C,2))+ ') '

                    self.Tpol = '(' + r'{a}%'.format(a=np.around(100*avg_frac,2))+ ') '
                    self.Lpol = '(' + r'{a}%'.format(a=np.around(100*avg_L,2))+ ') '
                    self.absCpol = '(' + r'{a}%'.format(a=np.around(100*avg_C_abs,2))+ ') '
                    self.Cpol = '('+ r'{a}%'.format(a=np.around(100*avg_C,2))+ ') '

                    self.Tpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_frac,2))+ ') '
                    self.Lpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_L,2))+ ') '
                    self.absCpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))+ ') '
                    self.Cpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_C,2))+ ') '

                    self.avgPA = '(' + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))+ ') '
                    self.avgPAerr = '(' + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))+ ') '

                elif curr_comp == 0 and len(self.fixed_comps) > 1:
                    self.snr = '(' + r'{a}'.format(a=np.around(snr,2))+ ' ; '
                    self.Tsnr = '('+ r'{a}'.format(a=np.around(snr_frac,2))+ ' ; '
                    self.Lsnr = '(' + r'{a}'.format(a=np.around(snr_L,2))+ ' ; '
                    self.Csnr = '(' + r'{a}'.format(a=np.around(snr_C,2))+ ' ; '

                    self.Tpol = '(' + r'{a}%'.format(a=np.around(100*avg_frac,2))+ ' ; '
                    self.Lpol = '(' + r'{a}%'.format(a=np.around(100*avg_L,2))+ ' ; '
                    self.absCpol = '(' + r'{a}%'.format(a=np.around(100*avg_C_abs,2))+ ' ; '
                    self.Cpol = '('+ r'{a}%'.format(a=np.around(100*avg_C,2))+ ' ; '

                    self.Tpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_frac,2))+ ' ; '
                    self.Lpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_L,2))+ ' ; '
                    self.absCpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))+ ' ; '
                    self.Cpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_C,2))+ ' ; '

                    self.avgPA = '(' + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))+ ' ; '
                    self.avgPAerr = '(' + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))+ ' ; '

                elif curr_comp <  len(self.fixed_comps) -1 :

                    lastsplit = ((len(self.snr[:-1])-1)-(self.snr[:-1][:-1])[::-1].index(";"))
                    self.snr = self.snr[:lastsplit+1] + r'{a}'.format(a=np.around(snr,2))+ ' ; '
                    lastsplit = ((len(self.Tsnr[:-1])-1)-(self.Tsnr[:-1][:-1])[::-1].index(";"))
                    self.Tsnr = self.Tsnr[:lastsplit+1] + r'{a}'.format(a=np.around(snr_frac,2))+ ' ; '
                    lastsplit = ((len(self.Lsnr[:-1])-1)-(self.Lsnr[:-1][:-1])[::-1].index(";"))
                    self.Lsnr = self.Lsnr[:lastsplit+1] + r'{a}'.format(a=np.around(snr_L,2))+ ' ; '
                    lastsplit = ((len(self.Csnr[:-1])-1)-(self.Csnr[:-1][:-1])[::-1].index(";"))
                    self.Csnr = self.Csnr[:lastsplit+1] + r'{a}'.format(a=np.around(snr_C,2))+ ' ; '

                    lastsplit = ((len(self.Tpol[:-1])-1)-(self.Tpol[:-1][:-1])[::-1].index(";"))
                    self.Tpol = self.Tpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_frac,2))+ ' ; '
                    lastsplit = ((len(self.Lpol[:-1])-1)-(self.Lpol[:-1][:-1])[::-1].index(";"))
                    self.Lpol = self.Lpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_L,2))+ ' ; '
                    lastsplit = ((len(self.absCpol[:-1])-1)-(self.absCpol[:-1][:-1])[::-1].index(";"))
                    self.absCpol = self.absCpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_C_abs,2))+ ' ; '
                    lastsplit = ((len(self.Cpol[:-1])-1)-(self.Cpol[:-1][:-1])[::-1].index(";"))
                    self.Cpol = self.Cpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_C,2))+ ' ; '

                    lastsplit = ((len(self.Tpolerr[:-1])-1)-(self.Tpolerr[:-1][:-1])[::-1].index(";"))
                    self.Tpolerr = self.Tpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_frac,2))+ ' ; '
                    lastsplit = ((len(self.Lpolerr[:-1])-1)-(self.Lpolerr[:-1][:-1])[::-1].index(";"))
                    self.Lpolerr = self.Lpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_L,2))+ ' ; '
                    lastsplit = ((len(self.absCpolerr[:-1])-1)-(self.absCpolerr[:-1][:-1])[::-1].index(";"))
                    self.absCpolerr = self.absCpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))+ ' ; '
                    lastsplit = ((len(self.Cpolerr[:-1])-1)-(self.Cpolerr[:-1][:-1])[::-1].index(";"))
                    self.Cpolerr = self.Cpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_C,2))+ ' ; '

                    lastsplit = ((len(self.avgPA[:-1])-1)-(self.avgPA[:-1][:-1])[::-1].index(";"))
                    self.avgPA = self.avgPA[:lastsplit+1] + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))+ ' ; '
                    lastsplit = ((len(self.avgPAerr[:-1])-1)-(self.avgPAerr[:-1][:-1])[::-1].index(";"))
                    self.avgPAerr = self.avgPAerr[:lastsplit+1] + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))+ ' ; '


                elif curr_comp == len(self.fixed_comps) - 1:

                    lastsplit = ((len(self.snr[:-1])-1)-(self.snr[:-1][:-1])[::-1].index(";"))
                    self.snr = self.snr[:lastsplit+1] + r'{a}'.format(a=np.around(snr,2))+ ' ) '
                    lastsplit = ((len(self.Tsnr[:-1])-1)-(self.Tsnr[:-1][:-1])[::-1].index(";"))
                    self.Tsnr = self.Tsnr[:lastsplit+1] + r'{a}'.format(a=np.around(snr_frac,2))+ ' ) '
                    lastsplit = ((len(self.Lsnr[:-1])-1)-(self.Lsnr[:-1][:-1])[::-1].index(";"))
                    self.Lsnr = self.Lsnr[:lastsplit+1] + r'{a}'.format(a=np.around(snr_L,2))+ ' ) '
                    lastsplit = ((len(self.Csnr[:-1])-1)-(self.Csnr[:-1][:-1])[::-1].index(";"))
                    self.Csnr = self.Csnr[:lastsplit+1] + r'{a}'.format(a=np.around(snr_C,2))+ ' ) '

                    lastsplit = ((len(self.Tpol[:-1])-1)-(self.Tpol[:-1][:-1])[::-1].index(";"))
                    self.Tpol = self.Tpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_frac,2))+ ' ) '
                    lastsplit = ((len(self.Lpol[:-1])-1)-(self.Lpol[:-1][:-1])[::-1].index(";"))
                    self.Lpol = self.Lpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_L,2))+ ' ) '
                    lastsplit = ((len(self.absCpol[:-1])-1)-(self.absCpol[:-1][:-1])[::-1].index(";"))
                    self.absCpol = self.absCpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_C_abs,2))+ ' ) '
                    lastsplit = ((len(self.Cpol[:-1])-1)-(self.Cpol[:-1][:-1])[::-1].index(";"))
                    self.Cpol = self.Cpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_C,2))+ ' ) '

                    lastsplit = ((len(self.Tpolerr[:-1])-1)-(self.Tpolerr[:-1][:-1])[::-1].index(";"))
                    self.Tpolerr = self.Tpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_frac,2))+ ' ) '
                    lastsplit = ((len(self.Lpolerr[:-1])-1)-(self.Lpolerr[:-1][:-1])[::-1].index(";"))
                    self.Lpolerr = self.Lpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_L,2))+ ' ) '
                    lastsplit = ((len(self.absCpolerr[:-1])-1)-(self.absCpolerr[:-1][:-1])[::-1].index(";"))
                    self.absCpolerr = self.absCpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))+ ' ) '
                    lastsplit = ((len(self.Cpolerr[:-1])-1)-(self.Cpolerr[:-1][:-1])[::-1].index(";"))
                    self.Cpolerr = self.Cpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_C,2))+ ' ) '

                    lastsplit = ((len(self.avgPA[:-1])-1)-(self.avgPA[:-1][:-1])[::-1].index(";"))
                    self.avgPA = self.avgPA[:lastsplit+1] + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))+ ' ) '
                    lastsplit = ((len(self.avgPAerr[:-1])-1)-(self.avgPAerr[:-1][:-1])[::-1].index(";"))
                    self.avgPAerr = self.avgPAerr[:lastsplit+1] + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))+ ' ) '

                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute full polarization"
                self.param.trigger('RMcal_button')
        except Exception as e:
            self.error2 = "From callback_RMcal(): " + str(e)
        return
    RMcal_button = param.Action(clicked_RMcal,label="RM Calibrate")

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
    #@param.depends('frb_submitted', watch=True)
    #@param.depends('frb_loaded', watch=True)
    def view(self):
        try:
            #self.error2 = str(self.intLs) + " " + str(self.intRs)
            """
            if self.error == "Loading FRB...":
                #self.error = "Loading FRB..."
                self.load_FRB()

            if self.error == "Calibrating FRB...":
                self.cal_FRB()

            if self.error == "Saving Calibrated Filterbanks...":
                self.savefil()

            if self.error == "Saving RM Calibrated Filterbanks...":
                self.savefilRM()
            
            if self.error == "Exporting Summary to Text File...":
                self.savetxt()

            if self.error == "Writing Data to JSON File...":
                self.savejson()

            if self.error == "Adding to DSA110 RMTable and PolSpectra Catalogs...":
                self.addtocatalog()
            """
            #self.load_FRB()
            #self.frb_submitted = self.frb_submitted
            #self.error = str(self.frb_submitted)
            #if self.frb_loaded: self.error = "loaded"
            #self.error = self.frb_name + " " + str(self.frb_loaded) + str(self.frb_submitted)
            #self.hi = self.lo + self.comp_width
            #self.lo =(self.lo* (self.n_t_prev/self.n_t))
            #self.hi *= (self.n_t_prev/self.n_t)
            #self.comp_width = self.comp_width*(self.n_t_prev/self.n_t)
            #if self.n_t != self.n_t_prev or self.n_f != self.n_f_prev:
                #self.dyn_spec = dsapol.avg_freq(dsapol.avg_time(I2,self.n_t),self.n_f)

            #self.n_t_prev = self.n_t
            self.n_f = 2**self.log_n_f
            print("DEBUGGING: " + str(self.n_f) + " " + str(self.n_f_prev) + " " + str(self.log_n_f) + " " + str(self.freq_samp_on))
            self.n_t_weight = 2**self.log_n_t_weight

            #compute timestart, timestop around approx peak
            peak = int(15280/self.n_t)
            timestart = int(peak - (5e-3)/(self.n_t*32.7e-6))
            timestop = int(peak + (5e-3)/(self.n_t*32.7e-6))
            self.lo = self.lo_frac*(timestop-timestart)

            #downsample
            self.I_t = self.I_t_init[len(self.I_t_init)%self.n_t:]
            self.I_t = self.I_t.reshape(len(self.I_t)//self.n_t,self.n_t).mean(1)
            self.Q_t = self.Q_t_init[len(self.Q_t_init)%self.n_t:]
            self.Q_t = self.Q_t.reshape(len(self.Q_t)//self.n_t,self.n_t).mean(1)
            self.U_t = self.U_t_init[len(self.U_t_init)%self.n_t:]
            self.U_t = self.U_t.reshape(len(self.U_t)//self.n_t,self.n_t).mean(1)
            self.V_t = self.V_t_init[len(self.V_t_init)%self.n_t:]
            self.V_t = self.V_t.reshape(len(self.V_t)//self.n_t,self.n_t).mean(1)
            self.PA_t = self.PA_t_init[len(self.PA_t_init)%self.n_t:]
            self.PA_t = self.PA_t.reshape(len(self.PA_t)//self.n_t,self.n_t).mean(1)


            if self.loaded:
                if self.n_t == 1:
                    self.PA_t_errs = self.PA_t_errs_init
                elif self.n_t != self.n_t_prev:
                    unbias_factor = 1
                    L_t = np.sqrt(self.Q_t**2 + self.U_t**2)#*I_w_t_filt
                    L_t[L_t**2 <= (unbias_factor*np.std(self.I_t[:int(12000/self.n_t)]))**2] = np.std(self.I_t[:int(12000/self.n_t)])
                    L_t = np.sqrt(L_t**2 - np.std(self.I_t[:int(12000/self.n_t)])**2)
                    self.PA_t_errs = dsapol.PA_error_NKC_array(self.PA_t,L_t,np.std(self.I_t[:int(12000/self.n_t)]))

            self.fullburst_dict["I_t"] = self.I_t
            self.fullburst_dict["Q_t"] = self.Q_t
            self.fullburst_dict["U_t"] = self.U_t
            self.fullburst_dict["V_t"] = self.V_t
            self.fullburst_dict["PA_t"] = self.PA_t
            self.fullburst_dict["PA_t_errs"] = self.PA_t_errs

            self.n_t_prev = self.n_t


            #compute weights
            if self.filt_weights_on and self.curr_comp < len(self.fixed_comps):
                for i in range(len(self.fixed_comps)):
                    if i != self.curr_comp:
                        mask = np.zeros(len(self.I_t))
                        TSTART = int(timestart + self.complist_min[i])#int(int(15280/self.n_t) - (5e-3)/(self.n_t*32.7e-6)) + int(self.complist_min[i])
                        TSTOP = int(timestart + self.complist_max[i])#int(int(15280/self.n_t) - (5e-3)/(self.n_t*32.7e-6)) + int(self.complist_max[i])
                        mask[int(TSTART):int(TSTOP)] = 1
                        self.I_t = ma.masked_array(self.I_t,mask)
                        self.Q_t = ma.masked_array(self.Q_t,mask)
                        self.U_t = ma.masked_array(self.U_t,mask)
                        self.V_t = ma.masked_array(self.V_t,mask)

                        #get actual timestart, timestop
                        #peak,t1,t2 = dsapol.find_peak([self.I_t,self.I_t],self.ibox,self.fobj.header.tsamp,self.n_t,peak_range=None,pre_calc_tf=True,buff=[self.buff_L,self.buff_R])
                        #self.comp_dict[self.curr_comp]["timestart"] = t1
                        #self.comp_dict[self.curr_comp]["timestop"] = t2
                self.curr_weights = dsapol.get_weights_1D(self.I_t,self.Q_t,self.U_t,self.V_t,-1,-1,self.ibox,self.fobj.header.tsamp,self.n_f,self.n_t,self.freq_test_init,n_off=int(12000/self.n_t),buff=[self.buff_L,self.buff_R],n_t_weight=self.n_t_weight,sf_window_weights=self.sf_window_weights,padded=True,norm=False,timeaxis=self.timeaxis,fobj=self.fobj)
            #get frequency spectrum
            if self.freq_samp_on:
                self.freq_test = (self.freq_test_init[0])[len(self.freq_test_init[0])%self.n_f:]
                self.freq_test = self.freq_test.reshape(len(self.freq_test)//self.n_f,self.n_f).mean(1)
                self.freq_test = [self.freq_test]*4

                self.I_f = self.I_f_init[len(self.I_f_init)%self.n_f:]
                self.I_f = self.I_f.reshape(len(self.I_f)//self.n_f,self.n_f).mean(1)
                self.Q_f = self.Q_f_init[len(self.Q_f_init)%self.n_f:]
                self.Q_f = self.Q_f.reshape(len(self.Q_f)//self.n_f,self.n_f).mean(1)
                self.U_f = self.U_f_init[len(self.U_f_init)%self.n_f:]
                self.U_f = self.U_f.reshape(len(self.U_f)//self.n_f,self.n_f).mean(1)
                self.V_f = self.V_f_init[len(self.V_f_init)%self.n_f:]
                self.V_f = self.V_f.reshape(len(self.V_f)//self.n_f,self.n_f).mean(1)
                self.PA_f = self.PA_f_init[len(self.PA_f_init)%self.n_f:]
                self.PA_f = self.PA_f.reshape(len(self.PA_f)//self.n_f,self.n_f).mean(1)

                if self.loaded:
                    if self.n_f == 1:
                        self.PA_f_errs = self.PA_f_errs_init
                    elif self.n_f != self.n_f_prev:
                        unbias_factor = 1
                        L_f = np.sqrt(self.Q_f**2 + self.U_f**2)#*I_w_t_filt
                        L_f[L_f**2 <= (unbias_factor*np.std(self.I_t[:int(12000/self.n_t)]))**2] = np.std(self.I_t[:int(12000/self.n_t)])
                        L_f = np.sqrt(L_f**2 - np.std(self.I_t[:int(12000/self.n_t)])**2)
                        self.PA_f_errs = dsapol.PA_error_NKC_array(self.PA_f,L_f,np.std(self.I_t[:int(12000/self.n_t)]))

                self.fullburst_dict["I_f"] = self.I_f
                self.fullburst_dict["Q_f"] = self.Q_f
                self.fullburst_dict["U_f"] = self.U_f
                self.fullburst_dict["V_f"] = self.V_f
                self.fullburst_dict["PA_f"] = self.PA_f
                self.fullburst_dict["PA_f_errs"] = self.PA_f_errs


                for i in range(len(self.comp_dict.keys())):
                    self.comp_dict[i]["I_f"] = self.comp_dict[i]["I_f_init"][len(self.comp_dict[i]["I_f_init"])%self.n_f:]
                    self.comp_dict[i]["I_f"] = self.comp_dict[i]["I_f"].reshape(len(self.comp_dict[i]["I_f"])//self.n_f,self.n_f).mean(1)
                    self.comp_dict[i]["Q_f"] = self.comp_dict[i]["Q_f_init"][len(self.comp_dict[i]["Q_f_init"])%self.n_f:]
                    self.comp_dict[i]["Q_f"] = self.comp_dict[i]["Q_f"].reshape(len(self.comp_dict[i]["Q_f"])//self.n_f,self.n_f).mean(1)
                    self.comp_dict[i]["U_f"] = self.comp_dict[i]["U_f_init"][len(self.comp_dict[i]["U_f_init"])%self.n_f:]
                    self.comp_dict[i]["U_f"] = self.comp_dict[i]["U_f"].reshape(len(self.comp_dict[i]["U_f"])//self.n_f,self.n_f).mean(1)
                    self.comp_dict[i]["V_f"] = self.comp_dict[i]["V_f_init"][len(self.comp_dict[i]["V_f_init"])%self.n_f:]
                    self.comp_dict[i]["V_f"] = self.comp_dict[i]["V_f"].reshape(len(self.comp_dict[i]["V_f"])//self.n_f,self.n_f).mean(1)
                    self.comp_dict[i]["PA_f"] = self.comp_dict[i]["PA_f_init"][len(self.comp_dict[i]["PA_f_init"])%self.n_f:]
                    self.comp_dict[i]["PA_f"] = self.comp_dict[i]["PA_f"].reshape(len(self.comp_dict[i]["PA_f"])//self.n_f,self.n_f).mean(1)

                    if self.loaded:
                        if self.n_f == 1:
                            self.comp_dict[i]["PA_f_errs"] = self.comp_dict[i]["PA_f_errs_init"]
                        elif self.n_f != self.n_f_prev:
                            unbias_factor = 1
                            L_f = np.sqrt(self.comp_dict[i]["Q_f"]**2 + self.comp_dict[i]["U_f"]**2)#*I_w_t_filt
                            L_f[L_f**2 <= (unbias_factor*np.std(self.I_t[:int(12000/self.n_t)]))**2] = np.std(self.I_t[:int(12000/self.n_t)])
                            L_f = np.sqrt(L_f**2 - np.std(self.I_t[:int(12000/self.n_t)])**2)
                            self.comp_dict[i]["PA_f_errs"] = dsapol.PA_error_NKC_array(self.comp_dict[i]["PA_f"],L_f,np.std(self.I_t[:int(12000/self.n_t)]))

                
                self.n_f_prev = self.n_f
            """ 
            if self.freq_samp_on or self.finished:
                self.curr_weights = np.zeros(len(self.curr_weights))
                for i in range(len(self.comp_dict.keys())):
                    #self.curr_comp = np.zeros(len(self.curr_comp))
                    self.curr_weights += self.comp_dict[i]["weights"]
            """
            #except Exception as e:
            #    self.error = "From view2(): " + str(e)
            #try:
            """
            if self.finished:
                if self.testidx == 2:
                    self.curr_weights = (self.comp_dict[0]["weights"] + self.comp_dict[1]["weights"])/np.sum(self.comp_dict[0]["weights"] + self.comp_dict[1]["weights"])
                else:
                    self.curr_weights = self.comp_dict[self.testidx]["weights"]
            """
            return pol_plot(self.I_t,self.Q_t,self.U_t,self.V_t,self.PA_t,self.PA_t_errs,self.I_f,self.Q_f,self.U_f,self.V_f,self.PA_f,self.PA_f_errs,self.comp_dict,self.freq_test,self.curr_weights,timestart,timestop,self.n_t,self.n_f,self.buff_L,self.buff_R,self.n_t_weight,self.sf_window_weights,self.ibox,self.lo,self.comp_width,self.comp_choose_on,self.fixed_comps,self.filt_weights_on,self.curr_comp,self.freq_samp_on,self.wait,self.multipeaks,self.height,self.intLs,self.intRs,self.maskPA,self.finished)
        except Exception as e:
            print("HERE I AM")
            print(str(e))
            self.error = "From view3(): " + str(e) + " " + str(len(self.I_f_init)) + " " + str(len(self.I_f)) + " " + str(len(self.freq_test[0]))
            self.error2 = "ping"
        return


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



