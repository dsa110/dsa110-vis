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
import pickle
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




def pol_plot(I_t,Q_t,U_t,V_t,PA_t,PA_t_errs,I_f,Q_f,U_f,V_f,PA_f,PA_f_errs,comp_dict,freq_test,I_t_weights,timestart,timestop,n_t=1,n_f=1,buff_L=1,buff_R=1,n_t_weight=1,sf_window_weights=1,width_native=1,lo=1,comp_width=100,comp_choose_on=False,fixed_comps=[],filt_weights_on=False,comp_num=0,freq_samp_on=False,wait=False,multipeaks=False,height=5,intLs=[],intRs=[],maskPA=False,maxcomps=4):
    
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
    if (filt_weights_on or freq_samp_on) and (not wait):
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
            axPA.errorbar(np.arange(intL,intR),(180/np.pi)*PA_t[timestart:timestop][intL:intR],yerr=(180/np.pi)*PA_t_errs[timestart:timestop][intL:intR],fmt='o',label="Intrinsic PPA",color="blue",markersize=6,linewidth=2)
        elif multipeaks:
            intL = np.nan
            intR = np.nan
            ax.axhline(height,color="red",linestyle="--")
            #axPA.errorbar(np.arange(intL,intR),(180/np.pi)*PA_t[intL:intR],yerr=(180/np.pi)*PA_t_errs[intL:intR],fmt='o',label="Intrinsic PPA",color="blue",markersize=6,linewidth=2)
        else:
            FWHM,heights,intL,intR = peak_widths(I_t_weights[timestart:timestop],[np.argmax(I_t_weights[timestart:timestop])])
            intL = int(intL)
            intR = int(intR)
            axPA.errorbar(np.arange(intL,intR),(180/np.pi)*PA_t[timestart:timestop][intL:intR],yerr=(180/np.pi)*PA_t_errs[timestart:timestop][intL:intR],fmt='o',label="Intrinsic PPA",color="blue",markersize=6,linewidth=2)
        ax.axvline(intL,color="red",linestyle="--")
        ax.axvline(intR,color="red",linestyle="--")
        




    elif freq_samp_on:
        intL = np.min(intLs)
        intR = np.max(intRs)
        
        if maskPA:
            tstweight = (I_t_weights*np.max(I_t)/np.max(I_t_weights))[timestart:timestop][intL:intR]
            ax.plot(np.arange(intL,intR),tstweight,color="black")
            axPA.errorbar(np.arange(intL,intR)[(I_t_weights[timestart:timestop])[intL:intR] > 0.005],(180/np.pi)*PA_t[timestart:timestop][intL:intR][(I_t_weights[timestart:timestop])[intL:intR] > 0.005],yerr=(180/np.pi)*PA_t_errs[timestart:timestop][intL:intR][(I_t_weights[timestart:timestop])[intL:intR] > 0.005],fmt='o',label="Intrinsic PPA",color="blue",markersize=6,linewidth=2)
        else:
            axPA.errorbar(np.arange(intL,intR),(180/np.pi)*PA_t[timestart:timestop][intL:intR],yerr=(180/np.pi)*PA_t_errs[timestart:timestop][intL:intR],fmt='o',label="Intrinsic PPA",color="blue",markersize=6,linewidth=2)
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

                paxs[i].errorbar(freq_test[0],(180/np.pi)*(comp_dict[i]["PA_f"]),(180/np.pi)*(comp_dict[i]["PA_f_errs"]),fmt='o',label="Intrinsic PPA",color="blue",markersize=6,linewidth=2)

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

    if freq_samp_on:# and len(freq_test[0]) == len(PA_f):
        faxs[-1].plot(freq_test[0],I_f,label="I")
        faxs[-1].plot(freq_test[0],Q_f,label="Q")
        faxs[-1].plot(freq_test[0],U_f,label="U")
        faxs[-1].plot(freq_test[0],V_f,label="V")
        paxs[-1].errorbar(freq_test[0],(180/np.pi)*PA_f,yerr=(180/np.pi)*PA_f_errs,label="Intrinsic PPA",color="blue",markersize=6,linewidth=2,fmt='o')


        for i in range(len(comp_dict.keys())):
            faxs[i].plot(freq_test[0],comp_dict[i]["I_f"],label="I")
            faxs[i].plot(freq_test[0],comp_dict[i]["Q_f"],label="Q")
            faxs[i].plot(freq_test[0],comp_dict[i]["U_f"],label="U")
            faxs[i].plot(freq_test[0],comp_dict[i]["V_f"],label="V")
            paxs[i].errorbar(freq_test[0],(180/np.pi)*(comp_dict[i]["PA_f"]),yerr=(180/np.pi)*(comp_dict[i]["PA_f_errs"]),fmt='o',label="Intrinsic PPA",color="blue",markersize=6,linewidth=2)


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
    ibeam = 0
    RA = 0
    DEC = 0
    ParA = -1
    wav_test = [np.zeros(6144)]*4


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
    loaded = False

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

    def load_FRB(self):
        try:
            if self.error=="Loading FRB...":
                #self.error2 = str(self.I.shape)
                self.error = "Loading FRB predownsampled by " + str(self.n_t) + " in time, " + str(self.n_f) + " in frequency..."
                t1 = time.time()
                ids = self.frb_name[:10]#"230307aaao"#"220207aabh"#"221029aado"
                nickname = self.frb_name[11:]#"phineas"#"zach"#"mifanshan"
                datadir = "/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids + "_" + nickname + "/"
                #ibeam = 218
                #caldate="22-12-18"
                #self.frb_name = "Loading " + ids + "_" + nickname + " ..."
                #self.view()
                (self.I,self.Q,self.U,self.V,self.fobj,self.timeaxis,self.freq_test_init,self.wav_test) = dsapol.get_stokes_2D(datadir,ids + "_dev",20480,n_t=self.n_t,n_f=self.n_f,n_off=int(12000//self.n_t),sub_offpulse_mean=True)
                self.I_init,self.Q_init,self.U_init,self.V_init = copy.deepcopy(self.I),copy.deepcopy(self.Q),copy.deepcopy(self.U),copy.deepcopy(self.V)
                self.freq_test = copy.deepcopy(self.freq_test_init)
                (self.I_t_init,self.Q_t_init,self.U_t_init,self.V_t_init) = dsapol.get_stokes_vs_time(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,n_off=int(12000//self.n_t),plot=False,show=True,normalize=True,buff=1,window=30)
                #self.frb_loaded = True
                PA_f,self.PA_t_init,PA_f_errs,self.PA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,self.n_f,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,show=False,normalize=True,buff=1,weighted=False,timeaxis=self.timeaxis,fobj=self.fobj)

                #time.sleep(5)
                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to load data"
                #self.frb_name = "Loaded " + ids + "_" + nickname + " ..."
                self.n_f_root = self.n_f
                self.log_n_f = 0#param.Integer(default=0,bounds=(0,10),label=r'log2(n_f)')
                self.n_f = 1
                self.n_f_prev = 1
                self.n_t = 1

                self.I_RMcal =  np.zeros(self.I.shape)
                self.Q_RMcal =  np.zeros(self.I.shape)
                self.U_RMcal =  np.zeros(self.I.shape)
                self.V_RMcal =  np.zeros(self.I.shape)

                self.I_t = np.nan*np.ones(len(self.I_t_init))
                self.Q_t = np.nan*np.ones(len(self.I_t_init))
                self.U_t = np.nan*np.ones(len(self.I_t_init))
                self.V_t = np.nan*np.ones(len(self.I_t_init))

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
                


                self.loaded = True
        except Exception as e:
            self.error = "From load_FRB(): " + str(e)
        return


    def cal_FRB(self):
        if self.error=="Calibrating FRB...":
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

            self.I,self.Q,self.U,self.V = dsapol.calibrate(self.I,self.Q,self.U,self.V,(self.gxx,self.gyy),stokes=True)
            self.I,self.Q,self.U,self.V,self.ParA = dsapol.calibrate_angle(self.I,self.Q,self.U,self.V,self.fobj,self.ibeam,self.RA,self.DEC)
            self.I_init,self.Q_init,self.U_init,self.V_init = copy.deepcopy(self.I),copy.deepcopy(self.Q),copy.deepcopy(self.U),copy.deepcopy(self.V)

            (self.I_t_init,self.Q_t_init,self.U_t_init,self.V_t_init) = dsapol.get_stokes_vs_time(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,n_off=int(12000//self.n_t),plot=False,show=True,normalize=True,buff=1,window=30)

            self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to load data"

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
    wait = False
    multipeaks = param.Boolean(False)
    maskPA=param.Boolean(False)
    height = param.Number(default=5,bounds=(0,50),step=1e-2)
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
                PA_fmasked,tmpPA_t_init,PA_f_errsmasked,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
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
                [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000/self.n_t),normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                self.comp_dict[self.curr_comp]["T/I_pre"] = avg_frac
                self.comp_dict[self.curr_comp]["T/I_pre_err"] = sigma_frac
                self.comp_dict[self.curr_comp]["T/I_pre_snr"] = snr_frac
                self.comp_dict[self.curr_comp]["L/I_pre"] = avg_L
                self.comp_dict[self.curr_comp]["L/I_pre_err"] = sigma_L
                self.comp_dict[self.curr_comp]["L/I_pre_snr"] = snr_L
                self.comp_dict[self.curr_comp]["absV/I_pre"] = avg_C_abs
                self.comp_dict[self.curr_comp]["absV/I_pre_err"] = sigma_C_abs
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
                self.param.trigger('next_comp')
            #self.error = str(len(self.comp_dict[self.curr_comp-1]["I_f"]))
        except Exception as e:
            self.error = "From clicked_next(): " + str(e)

        return
    def clicked_get(self):
        try:
            self.comp_choose_on = not self.comp_choose_on
            self.height = np.around(np.max(self.I_t)/2,2)
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


                    #get total PA 
                    self.error = "Computing Position Angle..."
                    t1 = time.time()
                    self.PA_f_init,tmpPA_t_init,self.PA_f_errs_init,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)

                    self.avgPA = self.avgPA + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))
                    self.avgPAerr = self.avgPAerr + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))
                    self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute position angle"



                    #get total polarization
                    self.error = "Computing polarization..."

                    multipeaks_all = (len(self.fixed_comps) > 1) or (self.comp_dict[0]["multipeaks"])

                    t1 = time.time()
                    [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000/self.n_t),normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=multipeaks_all,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
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


                    self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute polarization"

                    self.STEP = 1
                    
                    self.filt_weights_on = False
                    self.freq_samp_on = True
                    self.wait = False
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
                    PA_fmasked,tmpPA_t_init,PA_f_errsmasked,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
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
                    [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000/self.n_t),normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                    self.comp_dict[self.curr_comp]["T/I_pre"] = avg_frac
                    self.comp_dict[self.curr_comp]["T/I_pre_err"] = sigma_frac
                    self.comp_dict[self.curr_comp]["T/I_pre_snr"] = snr_frac
                    self.comp_dict[self.curr_comp]["L/I_pre"] = avg_L
                    self.comp_dict[self.curr_comp]["L/I_pre_err"] = sigma_L
                    self.comp_dict[self.curr_comp]["L/I_pre_snr"] = snr_L
                    self.comp_dict[self.curr_comp]["absV/I_pre"] = avg_C_abs
                    self.comp_dict[self.curr_comp]["absV/I_pre_err"] = sigma_C_abs
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


                    #get PA vs frequency
                    self.error = "Computing Position Angle..."
                    t1 = time.time()
                    self.PA_f_init,tmpPA_t_init,self.PA_f_errs_init,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000//self.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=self.multipeaks,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
                    self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute position angle"



                    #get total polarization
                    self.error = "Computing total polarization..."

                    multipeaks_all = (len(self.fixed_comps) > 1) or (self.comp_dict[0]["multipeaks"])

                    t1 = time.time()
                    [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,1,self.freq_test_init,n_off=int(12000/self.n_t),normalize=True,weighted=True,timeaxis=self.timeaxis,fobj=self.fobj,multipeaks=multipeaks_all,height=self.height*np.max(self.curr_weights)/np.max(self.I_t),input_weights=self.curr_weights)
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

                    self.filt_weights_on = False
                    self.freq_samp_on = True

                    self.STEP = 1
                self.param.trigger('done')

        except Exception as e:
            self.error2 = str(len(self.fixed_comps)) + " " + "From clicked_done(): " + str(e) + " " + str(self.comp_dict.keys())
            #self.error2 = "From clicked_done(): " + str(len(self.fixed_comps)) + str(self.curr_comp)
        return
    #up = param.Action(clicked_up)#clicked)
    #down = param.Action(clicked_down)
    next_comp = param.Action(clicked_next,label="Next")
    get_comp = param.Action(clicked_get,label="Get Components")
    done = param.Action(clicked_done,label='Done')


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
    curr_comp = 0
    curr_weights = []

    #click_data = param.Dict(doc="test")
    #print(click_data)





    #***VIEWING MODULE***#
    #@param.depends('frb_submitted', watch=True)
    #@param.depends('frb_loaded', watch=True)
    def view(self):
        try:
            #self.error2 = str(self.intLs) + " " + str(self.intRs)
            if self.error == "Loading FRB...":
                #self.error = "Loading FRB..."
                self.load_FRB()

            if self.error == "Calibrating FRB...":
                self.cal_FRB()
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

            self.n_t_prev = self.n_t


            #compute weights
            if self.filt_weights_on and self.curr_comp < len(self.fixed_comps):
                for i in range(len(self.fixed_comps)):
                    if i != self.curr_comp:
                        mask = np.zeros(len(self.I_t))
                        TSTART = int(int(15280/self.n_t) - (5e-3)/(self.n_t*32.7e-6)) + int(self.complist_min[i])
                        TSTOP = int(int(15280/self.n_t) - (5e-3)/(self.n_t*32.7e-6)) + int(self.complist_max[i])
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
        #except Exception as e:
        #    self.error = "From view1(): " + str(e)
        #try:
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
        #except Exception as e:
        #    self.error = "From view2(): " + str(e)
        #try:
            return pol_plot(self.I_t,self.Q_t,self.U_t,self.V_t,self.PA_t,self.PA_t_errs,self.I_f,self.Q_f,self.U_f,self.V_f,self.PA_f,self.PA_f_errs,self.comp_dict,self.freq_test,self.curr_weights,timestart,timestop,self.n_t,self.n_f,self.buff_L,self.buff_R,self.n_t_weight,self.sf_window_weights,self.ibox,self.lo,self.comp_width,self.comp_choose_on,self.fixed_comps,self.filt_weights_on,self.curr_comp,self.freq_samp_on,self.wait,self.multipeaks,self.height,self.intLs,self.intRs,self.maskPA)
        except Exception as e:
            print("HERE I AM")
            print(str(e))
            self.error = "From view3(): " + str(e) + " " + str(len(self.PA_f)) + " " + str(len(self.PA_f_errs)) + " " + str(len(self.freq_test[0]))
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

def RM_plot(RMsnrs,trial_RM,RMsnrstools,trial_RM_tools,RMsnrszoom,trial_RMzoom,RMsnrstoolszoom,trial_RM_tools_zoom,RMsnrs2zoom,init_RM,fine_RM,RM,RMerror,threshold=9):
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
    if fine_RM:
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
    fobj = None
    curr_weights = np.nan*np.ones(20480)
    timestarts = []
    timestops = []
    timestart_in = 0
    timestop_in = 0
    comp_dict = dict()
    curr_comp = 0

    #***Initial RM synthesis + Rm tools***#
    init_RM = True
    fine_RM = False
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
                
            
                    self.init_RM = False
                    self.fine_RM = True
                elif self.fine_RM:
                    self.RM1zoom_str = self.RM1zoom_str + ") " + str(np.around(self.RM1zoom,2))
                    self.RMerr1zoom_str = self.RMerr1zoom_str + ") " + str(np.around(self.RMerr1zoom,2))
                    
                    self.RM1tools_zoom_str = self.RM1tools_zoom_str + ") " + str(np.around(self.RM1tools_zoom,2))
                    self.RMerr1tools_zoom_str = self.RMerr1tools_zoom_str + ") " + str(np.around(self.RMerr1tools_zoom,2))

                    self.RM2zoom_str = self.RM2zoom_str + ") " + str(np.around(self.RM2zoom,2))
                    self.RMerr2zoom_str = self.RMerr2zoom_str + ") " + str(np.around(self.RMerr2zoom,2))

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



                #check if RM in range for RM tools
                self.RMtools_zoom_flag = np.abs(float(self.RM1zoom)) < 1000
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



                self.error = "Running fine S/N method..."


                RM2,phi2,self.RMsnrs2zoom,RMerr2,upp,low,sig,QUnoise = dsapol.faradaycal_SNR(self.I,self.Q,self.U,self.V,self.freq_test,self.trial_RM2,self.trial_phi,self.ibox,self.fobj.header.tsamp,plot=False,n_f=self.n_f,n_t=self.n_t,show=False,err=True,weighted=True,n_off=int(12000/self.n_t),fobj=self.fobj,input_weights=np.trim_zeros(self.curr_weights),timestart_in=self.timestart_in,timestop_in=self.timestop_in)


                fit_window=50
                oversamps = 5000
                poptpar,pcovpar = curve_fit(fit_parabola,self.trial_RM2[np.argmax(self.RMsnrs2zoom)-fit_window:np.argmax(self.RMsnrs2zoom)+fit_window],self.RMsnrs2zoom[np.argmax(self.RMsnrs2zoom)-fit_window:np.argmax(self.RMsnrs2zoom)+fit_window],p0=[1,1,RM2],sigma=1/self.RMsnrs2zoom[np.argmax(self.RMsnrs2zoom)-fit_window:np.argmax(self.RMsnrs2zoom)+fit_window])
                self.RM2zoom = poptpar[2]
                self.RMerr2zoom = dsapol.RM_error_fit(np.max(self.RMsnrs2zoom))
                if (self.RM2zoom_str == "") and (self.curr_comp != -1):
                    self.RM2zoom_str ="(" + str(np.around(self.RM2zoom,2))
                    self.RMerr2zoom_str = "(" + str(np.around(self.RMerr2zoom,2))
                    self.comp_dict[self.curr_comp]["RM"] = self.RM2zoom
                    self.comp_dict[self.curr_comp]["RMerr"] = self.RMerr2zoom
                    self.comp_dict[self.curr_comp]["RMsnrs"] = copy.deepcopy(self.RMsnrs2zoom)
                    self.comp_dict[self.curr_comp]["trialRMs"] = copy.deepcopy(self.trial_RM2)
                elif self.curr_comp != -1:
                    self.RM2zoom_str = self.RM2zoom_str + " ; " + str(np.around(self.RM2zoom,2))
                    self.RMerr2zoom_str = self.RMerr2zoom_str + " ; " + str(np.around(self.RMerr2zoom,2))
                    self.comp_dict[self.curr_comp]["RM"] = self.RM2zoom
                    self.comp_dict[self.curr_comp]["RMerr"] = self.RMerr2zoom
                    self.comp_dict[self.curr_comp]["RMsnrs"] = copy.deepcopy(self.RMsnrs2zoom)
                    self.comp_dict[self.curr_comp]["trialRMs"] = copy.deepcopy(self.trial_RM2)
                elif self.curr_comp == -1:
                    self.RM2zoom_str = self.RM2zoom_str + ") " + str(np.around(self.RM2zoom,2))
                    self.RMerr2zoom_str = self.RMerr2zoom_str + ") " + str(np.around(self.RMerr2zoom,2))
                


    
                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to run fine RM synthesis"

        except Exception as e:
            #self.error = "From clicked_run(): " + str(e) + " " + str(len(self.curr_weights)) + " " + str(len(np.trim_zeros(self.curr_weights))) + " " + str(self.timestop_in-self.timestart_in)
            self.error = "From clicked_run(): " + str(e) + " " + str(len(self.I_f)) + " " + str(len(self.freq_test))
        return

    run = param.Action(clicked_run,label="Run")



    #***VIEWING MODULE***#
    def view(self):
        try:
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
            
            return RM_plot(self.RMsnrs1,self.trial_RM,self.RMsnrs1tools,self.trial_RM_tools,self.RMsnrs1zoom,self.trial_RM2,self.RMsnrs1tools_zoom,self.trial_RM_tools_zoom,self.RMsnrs2zoom,self.init_RM,self.fine_RM,rm,rmerr)
        except Exception as e:
            self.error = "From view(): " + str(e) + " " + str(len(self.trial_RM2)) + " " + str(len(self.RMsnrs2zoom))
            return
