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




def callback(target, event):
    target.error = "Loading FRB..."
    #target.frb_submitted = True
    return
def callback_cal(target,event):
    target.error = "Calibrating FRB..."
    return
def callback_savefil(target,event):
    target.error = "Saving Calibrated Filterbanks..."
    return
def callback_savefilRM(target,event):
    target.error = "Saving RM Calibrated Filterbanks..."
    return

#link IQUV from panel1 to panel 2
def callback_link(target,event,pan1):
    target.error = "Transferring data between panels..."
    t1 = time.time()


    target.I = pan1.I
    target.Q = pan1.Q
    target.U = pan1.U
    target.V = pan1.V

    #get current component, or if all already done, get full thing
    if (len(pan1.comp_dict.keys()) <= len(pan1.fixed_comps)) and (len(pan1.comp_dict.keys()) > 0) and pan1.filt_weights_on:
        # get current component
        target.curr_comp = np.max(list(pan1.comp_dict.keys()))
        target.error = "Multiple Components, using component " + str(target.curr_comp) + " " + str(len(pan1.comp_dict[target.curr_comp]["I_f_init"]))


        target.I_f = pan1.comp_dict[target.curr_comp]["I_f_init"]#pan1.I_f_init
        target.Q_f = pan1.comp_dict[target.curr_comp]["Q_f_init"]#pan1.Q_f_init
        target.U_f = pan1.comp_dict[target.curr_comp]["U_f_init"]#pan1.U_f_init
        target.V_f = pan1.comp_dict[target.curr_comp]["V_f_init"]#pan1.V_f_init

        target.curr_weights = pan1.comp_dict[target.curr_comp]["weights"]
        target.ibox = pan1.comp_dict[target.curr_comp]["ibox"]
    elif (len(pan1.comp_dict.keys()) == len(pan1.fixed_comps)) and (not pan1.filt_weights_on):
        #get full component
        target.curr_comp = -1
        target.error = "All Components " + str(len(pan1.I_f_init))

        target.I_f = pan1.I_f_init
        target.Q_f = pan1.Q_f_init
        target.U_f = pan1.U_f_init
        target.V_f = pan1.V_f_init

        target.curr_weights = pan1.curr_weights
        target.ibox = pan1.ibox
    target.freq_test = pan1.freq_test_init
    target.n_t = pan1.n_t
    target.n_f = pan1.n_f
    #target.timestart = pan1.timestart
    #/itarget.timestop = pan1.timestop
    target.fobj = pan1.fobj
    target.comp_dict = pan1.comp_dict
    target.fullburst_dict = pan1.fullburst_dict

    #reset flags
    target.init_RM = True
    target.fine_RM = False
    target.done_RM = False

    target.ids = pan1.ids
    target.nickname = pan1.nickname
    target.datadir = pan1.datadir

    #target.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to transfer data"

def callback_linkback(target,event,pan2):
    target.comp_dict = pan2.comp_dict
    target.fullburst_dict = pan2.fullburst_dict

#RM calibration
def callback_RMcal(target,event,pan2):
    #get most recently estimated RM

    try:
        if not pan2.fine_RM:
            rmcal = pan2.RM1
        else:
            rmcal = pan2.RM2zoom

        pan2.error = str(pan2.curr_comp)
        #Case 1: calibrate full burst
        if pan2.curr_comp == -1:
            target.error = "Derotating full burst to RM = " + str(np.around(rmcal,2)) + " rad/m^2..."
            t1 = time.time()
            target.sigflag = True
            target.I_RMcal_init,target.Q_RMcal_init,target.U_RMcal_init,target.V_RMcal_init = dsapol.calibrate_RM(target.I_init,target.Q_init,target.U_init,target.V_init,rmcal,0,target.freq_test_init,stokes=True)
            target.I_RMcal,target.Q_RMcal,target.U_RMcal,target.V_RMcal = dsapol.calibrate_RM(target.I,target.Q,target.U,target.V,rmcal,0,target.freq_test,stokes=True)

            target.error = "step 1"
            (target.I_f_init,target.Q_f_init,target.U_f_init,target.V_f_init) = dsapol.get_stokes_vs_freq(target.I_RMcal,target.Q_RMcal,target.U_RMcal,target.V_RMcal,1,target.fobj.header.tsamp,1,target.n_t,target.freq_test_init,n_off=int(12000/target.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=target.timeaxis,fobj=target.fobj,input_weights=target.curr_weights)
            target.error = "step 2"
            target.I_f, target.Q_f, target.U_f, target.V_f = target.I_f_init,target.Q_f_init,target.U_f_init,target.V_f_init
            (target.I_t_init,target.Q_t_init,target.U_t_init,target.V_t_init) = dsapol.get_stokes_vs_time(target.I_RMcal_init,target.Q_RMcal_init,target.U_RMcal_init,target.V_RMcal_init,1,target.fobj.header.tsamp,1,n_off=int(12000/1),plot=False,show=False,normalize=True)


            target.error = "Complete: " + str(np.around(time.time()-t1,2)) + " to derotate"

            #recompute polarization and PA
            target.error = "Re-computing polarization and PA..."
            t1 = time.time()
            multipeaks_all = (len(target.fixed_comps) > 1) or (target.comp_dict[0]["multipeaks"])
            [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction(target.I_RMcal,target.Q_RMcal,target.U_RMcal,target.V_RMcal,target.ibox,target.fobj.header.tsamp,target.n_t,1,target.freq_test_init,n_off=int(12000/target.n_t),normalize=True,weighted=True,timeaxis=target.timeaxis,fobj=target.fobj,multipeaks=multipeaks_all,height=target.height*np.max(target.curr_weights)/np.max(target.I_t),input_weights=target.curr_weights)

            target.PA_f_init,tmpPA_t_init,target.PA_f_errs_init,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle(target.I_RMcal,target.Q_RMcal,target.U_RMcal,target.V_RMcal,target.ibox,target.fobj.header.tsamp,target.n_t,1,target.freq_test_init,n_off=int(12000//target.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=target.timeaxis,fobj=target.fobj,multipeaks=target.multipeaks,height=target.height*np.max(target.curr_weights)/np.max(target.I_t),input_weights=target.curr_weights)

            target.snr = target.snr[:target.snr.index(")") + 2] + r'{a}'.format(a=np.around(snr,2))
            target.Tsnr = target.Tsnr[:target.Tsnr.index(")") + 2] + r'{a}'.format(a=np.around(snr_frac,2))
            target.Lsnr = target.Lsnr[:target.Lsnr.index(")") + 2] + r'{a}'.format(a=np.around(snr_L,2))
            target.Csnr = target.Csnr[:target.Csnr.index(")") + 2] + r'{a}'.format(a=np.around(snr_C,2))

            target.Tpol = target.Tpol[:target.Tpol.index(")") + 2] + r'{a}%'.format(a=np.around(100*avg_frac,2))
            target.Lpol = target.Lpol[:target.Lpol.index(")") + 2] + r'{a}%'.format(a=np.around(100*avg_L,2))
            target.absCpol = target.absCpol[:target.absCpol.index(")") + 2] + r'{a}%'.format(a=np.around(100*avg_C_abs,2))
            target.Cpol = target.Cpol[:target.Cpol.index(")") + 2] + r'{a}%'.format(a=np.around(100*avg_C,2))

            target.Tpolerr = target.Tpolerr[:target.Tpolerr.index(")") + 2] + r'{a}%'.format(a=np.around(100*sigma_frac,2))
            target.Lpolerr = target.Lpolerr[:target.Lpolerr.index(")") + 2] + r'{a}%'.format(a=np.around(100*sigma_L,2))
            target.absCpolerr = target.absCpolerr[:target.absCpolerr.index(")") + 2] + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))
            target.Cpolerr = target.Cpolerr[:target.Cpolerr.index(")") + 2] + r'{a}%'.format(a=np.around(100*sigma_C,2))

            target.avgPA = target.avgPA[:target.avgPA.index(")") + 2] + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))
            target.avgPAerr = target.avgPAerr[:target.avgPAerr.index(")") + 2] + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))

            target.rmcalibrated_all = True
            target.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute polarization"


        #Case 2: calibrate current component
        elif pan2.curr_comp != -1:
            target.error = "Derotating component " + str(pan2.curr_comp) + " to RM = " + str(np.around(rmcal,2)) + " rad/m^2..."
            t1 = time.time()
            target.comp_dict[pan2.curr_comp]["sigflag"] = True
            pan2.comp_dict[pan2.curr_comp]["sigflag"] = True
            target.I_RMcal_init,target.Q_RMcal_init,target.U_RMcal_init,target.V_RMcal_init = dsapol.calibrate_RM(target.I_init,target.Q_init,target.U_init,target.V_init,rmcal,0,target.freq_test_init,stokes=True)
            target.I_RMcal,target.Q_RMcal,target.U_RMcal,target.V_RMcal = dsapol.calibrate_RM(target.I,target.Q,target.U,target.V,rmcal,0,target.freq_test,stokes=True)



            #get_stokes_vs_freq(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,n_off=3000,plot=False,datadir=DEFAULT_DATADIR,label='',calstr='',ext=ext,show=False,normalize=False,buff=0,weighted=False,n_t_weight=1,timeaxis=None,fobj=None,sf_window_weights=45,input_weights=[]

            #(target.I_f_init,target.Q_f_init,target.U_f_init,target.V_f_init) = dsapol.get_stokes_vs_freq(target.I_RMcal,target.Q_RMcal,target.U_RMcal,target.V_RMcal,1,target.fobj.header.tsamp,1,target.n_t,target.freq_test_init,n_off=int(12000/target.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=target.timeaxis,fobj=target.fobj,input_weights=target.curr_weights)
            target.comp_dict[pan2.curr_comp]["I_f_init"], target.comp_dict[pan2.curr_comp]["Q_f_init"], target.comp_dict[pan2.curr_comp]["U_f_init"], target.comp_dict[pan2.curr_comp]["V_f_init"] = dsapol.get_stokes_vs_freq(target.I_RMcal,target.Q_RMcal,target.U_RMcal,target.V_RMcal,1,target.fobj.header.tsamp,1,target.n_t,target.freq_test_init,n_off=int(12000/target.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=target.timeaxis,fobj=target.fobj,input_weights=target.comp_dict[pan2.curr_comp]["weights"])
            target.comp_dict[pan2.curr_comp]["I_f"], target.comp_dict[pan2.curr_comp]["Q_f"], target.comp_dict[pan2.curr_comp]["U_f"], target.comp_dict[pan2.curr_comp]["V_f"] = target.comp_dict[pan2.curr_comp]["I_f_init"], target.comp_dict[pan2.curr_comp]["Q_f_init"], target.comp_dict[pan2.curr_comp]["U_f_init"], target.comp_dict[pan2.curr_comp]["V_f_init"]
            #target.comp_dict[pan2.curr_comp]["I_f"], target.comp_dict[pan2.curr_comp]["Q_f"], target.comp_dict[pan2.curr_comp]["U_f"], target.comp_dict[pan2.curr_comp]["V_f"] = dsapol.get_stokes_vs_freq(target.I_RMcal,target.Q_RMcal,target.U_RMcal,target.V_RMcal,1,target.fobj.header.tsamp,target.n_f,target.n_t,target.freq_test,n_off=int(12000/target.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=target.timeaxis,fobj=target.fobj,input_weights=target.comp_dict[pan2.curr_comp]["weights"])
            #(target.I_t_init,target.Q_t_init,target.U_t_init,target.V_t_init) = dsapol.get_stokes_vs_time(I_RMcal_init,Q_RMcal_init,U_RMcal_init,V_RMcal_init,1,target.fobj.header.tsamp,1,n_off=int(12000/1),plot=False,show=False,normalize=True)

            target.error = "Complete: " + str(np.around(time.time()-t1,2)) + " to derotate"

            #recompute polarization and PA
            target.error = "Re-computing polarization and PA..."
            t1 = time.time()
            if target.comp_dict[pan2.curr_comp]["multipeaks"]:
                h = target.comp_dict[pan2.curr_comp]["scaled_height"]
            else:
                h =-1
            [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = dsapol.get_pol_fraction(target.I_RMcal,target.Q_RMcal,target.U_RMcal,target.V_RMcal,target.ibox,target.fobj.header.tsamp,target.n_t,1,target.freq_test_init,n_off=int(12000/target.n_t),normalize=True,weighted=True,timeaxis=target.timeaxis,fobj=target.fobj,multipeaks=target.comp_dict[pan2.curr_comp]["multipeaks"],height=h,input_weights=target.comp_dict[pan2.curr_comp]["weights"])


            PA_fmasked,tmpPA_t_init,PA_f_errsmasked,tmpPA_t_errs_init,avg_PA,sigma_PA = dsapol.get_pol_angle(target.I_RMcal,target.Q_RMcal,target.U_RMcal,target.V_RMcal,target.comp_dict[pan2.curr_comp]["ibox"],target.fobj.header.tsamp,target.n_t,target.n_f,target.freq_test,n_off=int(12000//target.n_t),plot=False,show=False,normalize=True,weighted=True,timeaxis=target.timeaxis,fobj=target.fobj,multipeaks=target.multipeaks,height=h,input_weights=target.comp_dict[pan2.curr_comp]["weights"])
            target.comp_dict[pan2.curr_comp]["PA_f_init"] = PA_fmasked
            target.comp_dict[pan2.curr_comp]["PA_f_errs_init"] = PA_f_errsmasked

            target.comp_dict[pan2.curr_comp]["PA_f"] = PA_fmasked
            target.comp_dict[pan2.curr_comp]["PA_f_errs"] = PA_f_errsmasked

            target.comp_dict[pan2.curr_comp]["PA_pre"] = avg_PA
            target.comp_dict[pan2.curr_comp]["PAerr_pre"] = sigma_PA

            target.comp_dict[pan2.curr_comp]["T/I_pre"] = avg_frac
            target.comp_dict[pan2.curr_comp]["T/I_pre_err"] = sigma_frac
            target.comp_dict[pan2.curr_comp]["T/I_pre_snr"] = snr_frac
            target.comp_dict[pan2.curr_comp]["L/I_pre"] = avg_L
            target.comp_dict[pan2.curr_comp]["L/I_pre_err"] = sigma_L
            target.comp_dict[pan2.curr_comp]["L/I_pre_snr"] = snr_L
            target.comp_dict[pan2.curr_comp]["absV/I_pre"] = avg_C_abs
            target.comp_dict[pan2.curr_comp]["absV/I_pre_err"] = sigma_C_abs
            target.comp_dict[pan2.curr_comp]["V/I"] = avg_C
            target.comp_dict[pan2.curr_comp]["V/I_err"] = sigma_C
            target.comp_dict[pan2.curr_comp]["V/I_snr"] = snr_C
            target.comp_dict[pan2.curr_comp]["I_snr"] = snr

            #update displays
            if pan2.curr_comp == 0 and len(target.fixed_comps) == 1:
                target.snr = '(' + r'{a}'.format(a=np.around(snr,2))+ ') '
                target.Tsnr = '('+ r'{a}'.format(a=np.around(snr_frac,2))+ ') '
                target.Lsnr = '(' + r'{a}'.format(a=np.around(snr_L,2))+ ') '
                target.Csnr = '(' + r'{a}'.format(a=np.around(snr_C,2))+ ') '

                target.Tpol = '(' + r'{a}%'.format(a=np.around(100*avg_frac,2))+ ') '
                target.Lpol = '(' + r'{a}%'.format(a=np.around(100*avg_L,2))+ ') '
                target.absCpol = '(' + r'{a}%'.format(a=np.around(100*avg_C_abs,2))+ ') '
                target.Cpol = '('+ r'{a}%'.format(a=np.around(100*avg_C,2))+ ') '

                target.Tpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_frac,2))+ ') '
                target.Lpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_L,2))+ ') '
                target.absCpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))+ ') '
                target.Cpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_C,2))+ ') '

                target.avgPA = '(' + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))+ ') '
                target.avgPAerr = '(' + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))+ ') '

            elif pan2.curr_comp == 0 and len(target.fixed_comps) > 1:
                target.snr = '(' + r'{a}'.format(a=np.around(snr,2))+ ' ; '
                target.Tsnr = '('+ r'{a}'.format(a=np.around(snr_frac,2))+ ' ; '
                target.Lsnr = '(' + r'{a}'.format(a=np.around(snr_L,2))+ ' ; '
                target.Csnr = '(' + r'{a}'.format(a=np.around(snr_C,2))+ ' ; '

                target.Tpol = '(' + r'{a}%'.format(a=np.around(100*avg_frac,2))+ ' ; '
                target.Lpol = '(' + r'{a}%'.format(a=np.around(100*avg_L,2))+ ' ; '
                target.absCpol = '(' + r'{a}%'.format(a=np.around(100*avg_C_abs,2))+ ' ; '
                target.Cpol = '('+ r'{a}%'.format(a=np.around(100*avg_C,2))+ ' ; '

                target.Tpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_frac,2))+ ' ; '
                target.Lpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_L,2))+ ' ; '
                target.absCpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))+ ' ; '
                target.Cpolerr = '(' + r'{a}%'.format(a=np.around(100*sigma_C,2))+ ' ; '

                target.avgPA = '(' + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))+ ' ; '
                target.avgPAerr = '(' + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))+ ' ; '

            elif pan2.curr_comp <  len(target.fixed_comps) -1 :

                lastsplit = ((len(target.snr[:-1])-1)-(target.snr[:-1][:-1])[::-1].index(";"))
                target.snr = target.snr[:lastsplit+1] + r'{a}'.format(a=np.around(snr,2))+ ' ; '
                lastsplit = ((len(target.Tsnr[:-1])-1)-(target.Tsnr[:-1][:-1])[::-1].index(";"))
                target.Tsnr = target.Tsnr[:lastsplit+1] + r'{a}'.format(a=np.around(snr_frac,2))+ ' ; '
                lastsplit = ((len(target.Lsnr[:-1])-1)-(target.Lsnr[:-1][:-1])[::-1].index(";"))
                target.Lsnr = target.Lsnr[:lastsplit+1] + r'{a}'.format(a=np.around(snr_L,2))+ ' ; '
                lastsplit = ((len(target.Csnr[:-1])-1)-(target.Csnr[:-1][:-1])[::-1].index(";"))
                target.Csnr = target.Csnr[:lastsplit+1] + r'{a}'.format(a=np.around(snr_C,2))+ ' ; '

                lastsplit = ((len(target.Tpol[:-1])-1)-(target.Tpol[:-1][:-1])[::-1].index(";"))
                target.Tpol = target.Tpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_frac,2))+ ' ; '
                lastsplit = ((len(target.Lpol[:-1])-1)-(target.Lpol[:-1][:-1])[::-1].index(";"))
                target.Lpol = target.Lpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_L,2))+ ' ; '
                lastsplit = ((len(target.absCpol[:-1])-1)-(target.absCpol[:-1][:-1])[::-1].index(";"))
                target.absCpol = target.absCpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_C_abs,2))+ ' ; '
                lastsplit = ((len(target.Cpol[:-1])-1)-(target.Cpol[:-1][:-1])[::-1].index(";"))
                target.Cpol = target.Cpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_C,2))+ ' ; '

                lastsplit = ((len(target.Tpolerr[:-1])-1)-(target.Tpolerr[:-1][:-1])[::-1].index(";"))
                target.Tpolerr = target.Tpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_frac,2))+ ' ; '
                lastsplit = ((len(target.Lpolerr[:-1])-1)-(target.Lpolerr[:-1][:-1])[::-1].index(";"))
                target.Lpolerr = target.Lpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_L,2))+ ' ; '
                lastsplit = ((len(target.absCpolerr[:-1])-1)-(target.absCpolerr[:-1][:-1])[::-1].index(";"))
                target.absCpolerr = target.absCpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))+ ' ; '
                lastsplit = ((len(target.Cpolerr[:-1])-1)-(target.Cpolerr[:-1][:-1])[::-1].index(";"))
                target.Cpolerr = target.Cpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_C,2))+ ' ; '

                lastsplit = ((len(target.avgPA[:-1])-1)-(target.avgPA[:-1][:-1])[::-1].index(";"))
                target.avgPA = target.avgPA[:lastsplit+1] + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))+ ' ; '
                lastsplit = ((len(target.avgPAerr[:-1])-1)-(target.avgPAerr[:-1][:-1])[::-1].index(";"))
                target.avgPAerr = target.avgPAerr[:lastsplit+1] + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))+ ' ; '


            elif pan2.curr_comp == len(target.fixed_comps) - 1:

                lastsplit = ((len(target.snr[:-1])-1)-(target.snr[:-1][:-1])[::-1].index(";"))
                target.snr = target.snr[:lastsplit+1] + r'{a}'.format(a=np.around(snr,2))+ ' ) '
                lastsplit = ((len(target.Tsnr[:-1])-1)-(target.Tsnr[:-1][:-1])[::-1].index(";"))
                target.Tsnr = target.Tsnr[:lastsplit+1] + r'{a}'.format(a=np.around(snr_frac,2))+ ' ) '
                lastsplit = ((len(target.Lsnr[:-1])-1)-(target.Lsnr[:-1][:-1])[::-1].index(";"))
                target.Lsnr = target.Lsnr[:lastsplit+1] + r'{a}'.format(a=np.around(snr_L,2))+ ' ) '
                lastsplit = ((len(target.Csnr[:-1])-1)-(target.Csnr[:-1][:-1])[::-1].index(";"))
                target.Csnr = target.Csnr[:lastsplit+1] + r'{a}'.format(a=np.around(snr_C,2))+ ' ) '

                lastsplit = ((len(target.Tpol[:-1])-1)-(target.Tpol[:-1][:-1])[::-1].index(";"))
                target.Tpol = target.Tpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_frac,2))+ ' ) '
                lastsplit = ((len(target.Lpol[:-1])-1)-(target.Lpol[:-1][:-1])[::-1].index(";"))
                target.Lpol = target.Lpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_L,2))+ ' ) '
                lastsplit = ((len(target.absCpol[:-1])-1)-(target.absCpol[:-1][:-1])[::-1].index(";"))
                target.absCpol = target.absCpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_C_abs,2))+ ' ) '
                lastsplit = ((len(target.Cpol[:-1])-1)-(target.Cpol[:-1][:-1])[::-1].index(";"))
                target.Cpol = target.Cpol[:lastsplit+1] + r'{a}%'.format(a=np.around(100*avg_C,2))+ ' ) '

                lastsplit = ((len(target.Tpolerr[:-1])-1)-(target.Tpolerr[:-1][:-1])[::-1].index(";"))
                target.Tpolerr = target.Tpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_frac,2))+ ' ) '
                lastsplit = ((len(target.Lpolerr[:-1])-1)-(target.Lpolerr[:-1][:-1])[::-1].index(";"))
                target.Lpolerr = target.Lpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_L,2))+ ' ) '
                lastsplit = ((len(target.absCpolerr[:-1])-1)-(target.absCpolerr[:-1][:-1])[::-1].index(";"))
                target.absCpolerr = target.absCpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_C_abs,2))+ ' ) '
                lastsplit = ((len(target.Cpolerr[:-1])-1)-(target.Cpolerr[:-1][:-1])[::-1].index(";"))
                target.Cpolerr = target.Cpolerr[:lastsplit+1] + r'{a}%'.format(a=np.around(100*sigma_C,2))+ ' ) '

                lastsplit = ((len(target.avgPA[:-1])-1)-(target.avgPA[:-1][:-1])[::-1].index(";"))
                target.avgPA = target.avgPA[:lastsplit+1] + r'{a}'.format(a=np.around((180/np.pi)*avg_PA,2))+ ' ) '
                lastsplit = ((len(target.avgPAerr[:-1])-1)-(target.avgPAerr[:-1][:-1])[::-1].index(";"))
                target.avgPAerr = target.avgPAerr[:lastsplit+1] + r'{a}'.format(a=np.around((180/np.pi)*sigma_PA,2))+ ' ) '

            target.error = "Complete: " + str(np.around(time.time()-t1,2)) + " s to compute full polarization"

    except Exception as e:
        target.error2 = "From callback_RMcal(): " + str(e)



