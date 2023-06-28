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
import matplotlib.ticker as ticker

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


#  Jakob Faber
def incoherent_dedispersion(waterfall, dm, freqs, time_res, freq_res):
    """
    Perform incoherent dedispersion on aN FRB waterfall plot.

    Parameters:
    waterfall (numpy.ndarray): 2D array representing the waterfall plot (time x frequency).
    dm (float): Dispersion measure in pc cm^-3.
    freqs (numpy.ndarray): 1D array containing the frequency values from
                            HIGHEST to LOWEST (in MHz) of each channel.
    time_res (float): Time resolution in milliseconds.
    freq_res (float): Frequency resolution in MHz.

    Returns:
    numpy.ndarray: Dedispersed waterfall plot.
    """
    waterfall_dedispersed = np.copy(waterfall)

    k_dm = 4.148808e3  # MHz^2 pc^-1 cm^3 ms, constant for dispersion

    for i, freq in enumerate(freqs):
        #print(i, freq)

        # Calculate the time delay for each channel due to dispersion
        time_delay = -1 * dm * k_dm * (1/(freq ** 2) - 1/(freqs[0] ** 2))
        #print(time_delay)

        # Convert time delay to number of time bins to roll
        roll_bins = int(time_delay / time_res)
        #print(roll_bins)

        # Roll the time axis of the waterfall plot for the given channel
        waterfall_dedispersed[i, :] = np.roll(waterfall[i, :], roll_bins)

    return waterfall_dedispersed


def dedisp_plot(I,I_t,n_t,n_f,perc=90,window=5,n_t_root=1,n_f_root=1):
    peak = int(15280/n_t)
    timestart = int(peak - (window*1e-3)/(n_t*32.7e-6))
    timestop = int(peak + (window*1e-3)/(n_t*32.7e-6))


    fig = plt.figure(figsize=(24,24))
    ax0 = plt.subplot2grid(shape=(3, 1), loc=(0, 0),rowspan=1)
    ax1 = plt.subplot2grid(shape=(3, 1), loc=(1, 0),rowspan=2)

    #plot time series
    #ax0 = plt.subplot(311)
    ax0.plot(I_t[timestart:timestop])
    ax0.set_xlim(0,timestop-timestart)
    ax0.xaxis.set_major_locator(ticker.NullLocator())
    ax0.set_ylabel("S/N")

    #plot dynamic spectrum
    #ax1 = plt.subplot(312)
    c=ax1.imshow(I[:,timestart:timestop],aspect="auto",vmax=np.percentile(I,perc))
    ax1.set_xlim(0,timestop-timestart)
    ax1.set_xlabel("Time Sample ({a} $\mu s$)".format(a=np.around(n_t*n_t_root*32.7,1)))
    ax1.set_ylabel("Freqency Sample ({a} $kHz$)".format(a=np.around(n_f*n_f_root*30.5,1)))

    #fig.colorbar(c, ax=ax,label=r'S/N')
    plt.subplots_adjust(hspace=0)

    return fig


class dedisp_panel(param.Parameterized):



    #param linked to dropdown menu
    #frb_submitted = param.String(default="")#param.Integer(default=0,bounds=(0,8),label=r'clicks')
    #frb_loaded = False
    frb_name = param.String(default="")
    caldate = param.String(default="")
    error = param.String(default="",label="output/errors")
    ids = ""
    nickname = ""
    ibeam = 0
    mjd = ""


    I_init_dmzero = np.zeros((20480,6144))
    I_init = np.zeros((20480,6144))
    I = np.zeros((20480,6144))
    I_t_init = np.zeros(20480)
    I_t = np.nan*np.ones(20480)


    fobj = None#fobj#None
    timeaxis = np.zeros(20480)#timeaxis#np.zeros(20480)
    freq_axis_init = np.zeros(6144)
    freq_axis = np.zeros(6144)
    wav_axis_init = np.zeros(6144)
    wav_axis = np.zeros(6144)


    n_t_root = 1
    n_t = param.Integer(default=1,bounds=(1,128),label=r'n_t')
    n_t_prev = 1
    log_n_f = param.Integer(default=0,bounds=(0,10),label=r'log2(n_f)')
    n_f_root = 1
    n_f = 1
    n_f_prev = 1
    loaded = False

    #plot settings
    window = param.Integer(default=5,bounds=(1,128),label=r'window (ms)')
    perc = param.Number(default=90,bounds=(50,99.999),label=r'saturation')

    #DM settings
    ddm_str = param.String(default="0",label=r'Relative DM (pc/cc)')
    ddm = 0
    ddm_prev = 0
    DM = 0
    final_DM = 0
    final_DM_str = param.String(default="0",label="Final DM (pc/cc)")
    ddmdelay = 0
    ddmdelay_str = param.String(default="0",label=r'Relative DM Delay Across Band (ms)')

    STEP = 0


    #@param.depends('frb_mid', watch=True

    def load_FRB(self):
        try:
            if self.error=="Loading FRB...":
                self.ids = self.frb_name[:10]#"230307aaao"#"220207aabh"#"221029aado"
                self.nickname = self.frb_name[11:]#"phineas"#"zach"#"mifanshan"

                #first check that FRB already has a filterbank created
                x=os.listdir("/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us")
                FRB_list = []
                for i in range(len(x)):
                    if len(x[i]) >= 10 and x[i][10] == "_":
                        FRB_list.append(x[i])
                if self.frb_name not in FRB_list:
                    #create filterbank if nonexistent
                    self.error = "Creating initial filterbanks dedispersed to DM = " + str(self.DM) + " pc/cc..."
                    t1 = time.time()
                    command = "/media/ubuntu/ssd/sherman/code/run_beamformer_visibs_bfweightsupdate_sb.bash NA " + str(self.ids) + " "  + str(self.nickname) + " " + str(self.caldate) + " "  + str(self.ibeam) + " " + str(self.mjd) + " " + str(self.DM) #${datestrings[$i]} ${candnames[$i]} ${nicknames[$i]} ${dates[$i]} ${bms[$i]} ${mjds[$i]} ${dms[$i]}
                    self.error = command
                    os.system(command)
                    
                    self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to create initial filterbanks"

                #self.error2 = str(self.I.shape)
                self.error = "Loading FRB predownsampled by " + str(self.n_t) + " in time, " + str(self.n_f) + " in frequency..."
                t1 = time.time()
                #self.ids = self.frb_name[:10]#"230307aaao"#"220207aabh"#"221029aado"
                #self.nickname = self.frb_name[11:]#"phineas"#"zach"#"mifanshan"
                datadir = "/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+self.ids + "_" + self.nickname + "/"
                #ibeam = 218
                #caldate="22-12-18"
                #self.frb_name = "Loading " + ids + "_" + nickname + " ..."
                #self.view()
                (self.I,self.fobj,self.timeaxis,self.freq_axis_init,self.wav_axis_init) = dsapol.get_I_2D(datadir,self.ids + "_dev",20480,n_t=self.n_t,n_f=self.n_f,n_off=int(12000//self.n_t),sub_offpulse_mean=True,dtype=np.float16)
                self.I_init = copy.deepcopy(self.I)
                self.I_init_dmzero = copy.deepcopy(self.I)
                #(self.I_t_init,self.Q_t_init,self.U_t_init,self.V_t_init) = dsapol.get_stokes_vs_time(self.I,self.Q,self.U,self.V,self.ibox,self.fobj.header.tsamp,self.n_t,n_off=int(12000//self.n_t),plot=False,show=True,normalize=True,buff=1,window=30)

                self.I_t_init = (np.mean(self.I,axis=0) - np.mean(np.mean(self.I[:,:int(12000//self.n_t)],axis=0)))/np.std(np.mean(self.I[:,:int(12000//self.n_t)],axis=0))


                #time.sleep(5)
                self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to load data"
                #self.frb_name = "Loaded " + ids + "_" + nickname + " ..."
                #self.n_f_root = self.n_f
                self.n_t_root = self.n_t
                self.n_f_root = self.n_f



                self.log_n_f = 0#param.Integer(default=0,bounds=(0,10),label=r'log2(n_f)')
                self.n_f = 1
                self.n_f_prev = 1
                self.n_t = 1

                self.I_t = np.nan*np.ones(len(self.I_t_init))
                self.freq_axis = copy.deepcopy(self.freq_axis_init)
                self.wav_axis = copy.deepcopy(self.wav_axis_init)

                self.final_DM = self.DM
                self.final_DM_str = str(np.around(self.final_DM,2))




                self.loaded = True
        except Exception as e:
            self.error = "From load_FRB(): " + str(e)
        return


    #apply DM


    def clicked_apply(self):
        try:
            self.error = "De-dispersing to total DM = " + str(self.DM + self.ddm) + "..."
            t1 = time.time()
            self.I_init = incoherent_dedispersion(self.I_init_dmzero, dm=self.ddm, freqs=self.freq_axis, time_res=self.n_t*32.7/1000, freq_res=self.freq_axis[0]-self.freq_axis[1])
            #self.I = incoherent_dedispersion(self.I, dm=self.ddm, freqs=self.freq_axis, time_res=self.n_t*32.7/1000, freq_res=self.freq_axis[0]-self.freq_axis[1])
            self.I = dsapol.avg_time(self.I_init,self.n_t)
            self.I = dsapol.avg_freq(self.I,self.n_f)
            self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to dedisperse"


        except Exception as e:
            self.error = "From clicked_apply(): " + str(e)

    
    #after ideal DM determined, recompute filterbanks with toolkit script
    def clicked_recompute(self):
        try:
            t1 = time.time()
            if self.ddm != 0:
                self.error = "Running toolkit script for DM " + str(self.final_DM_str) + " pc/cc..."
                command = "/media/ubuntu/ssd/sherman/code/run_beamformer_visibs_bfweightsupdate_sb.bash NA " + str(self.ids) + " "  + str(self.nickname) + " " + str(self.caldate) + " "  + str(self.ibeam) + " " + str(self.mjd) + " " + str(self.final_DM) #${datestrings[$i]} ${candnames[$i]} ${nicknames[$i]} ${dates[$i]} ${bms[$i]} ${mjds[$i]} ${dms[$i]}
                os.system(command)
                self.DM = self.final_DM
                self.ddm = 0
                self.ddm_str = "0"

            self.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute filterbanks"

        except Exception as e:
            self.error = "From clicked_recompute(): " + str(e)



    applyb = param.Action(clicked_apply,label="Apply")
    recompute_filterbanks = param.Action(clicked_recompute,label="Re-Compute Filterbanks")



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

            self.ddm = float(self.ddm_str)
            k_dm = 4.148808e3  # MHz^2 pc^-1 cm^3 ms, constant for dispersion
            self.ddmdelay = -1 * self.ddm * k_dm * (1/(self.freq_axis[-1] ** 2) - 1/(self.freq_axis[0] ** 2)) #ms
            self.ddmdelay_str = str(np.around(self.ddmdelay,2))



            if self.loaded:
                self.final_DM = self.ddm + self.DM
                self.final_DM_str = str(np.around(self.final_DM,2))

            self.n_f = 2**self.log_n_f
            self.freq_axis = (self.freq_axis_init)[len(self.freq_axis_init)%self.n_f:]
            self.freq_axis = self.freq_axis.reshape(len(self.freq_axis)//self.n_f,self.n_f).mean(1)

            #compute timestart, timestop around approx peak
            peak = int(15280/self.n_t)
            timestart = int(peak - (5e-3)/(self.n_t*32.7e-6))
            timestop = int(peak + (5e-3)/(self.n_t*32.7e-6))

            #downsample
            if (self.n_t != self.n_t_prev) or (self.n_f != self.n_f_prev):
                self.I = dsapol.avg_time(self.I_init,self.n_t)
                self.I = dsapol.avg_freq(self.I,self.n_f)
            self.I_t = (np.mean(self.I,axis=0) - np.mean(np.mean(self.I[:,:int(12000//self.n_t)],axis=0)))/np.std(np.mean(self.I[:,:int(12000//self.n_t)],axis=0))


            self.n_t_prev = self.n_t
            self.n_f_prev = self.n_f

            return dedisp_plot(self.I,self.I_t,self.n_t,self.n_f,window=self.window,perc=self.perc,n_t_root=self.n_t_root,n_f_root=self.n_f_root)
        except Exception as e:
            self.error = "From view3(): " + str(e)
        return
