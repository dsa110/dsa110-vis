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
#from numpy.ma import masked_array as ma"
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
from astropy.coordinates import SkyCoord


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

#testing with RM table: https://github.com/CIRADA-Tools/RMTable
from rmtable import RMTable
#testing with pol Table: https://github.com/CIRADA-Tools/PolSpectra/tree/master
import polspectra



"""
This module contains functions for interfacing with the DSA-110 FRB catalogs, which are implemented using
RMTable and polspectra (links above)
"""

RMTable_name = '/media/ubuntu/ssd/sherman/code/RM_tmp_files/DSA110_RMTable.fits'
PolSpectra_postRM_name = '/media/ubuntu/ssd/sherman/code/RM_tmp_files/DSA110_PolTable_PostRMcal.fits'
PolSpectra_preRM_name = '/media/ubuntu/ssd/sherman/code/RM_tmp_files/DSA110_PolTable_PreRMcal.fits'

#Check if FRB in the table already
def dsarp_FRBinTable_RMTable(candname,comp_num=-1):

    #get tables
    rmtable = RMTable.read(RMTable_name)
    #polspectra_postRM = polspectra.from_FITS(PolSpectra_postRM_name)
    #polspectra_preRM = polspectra.from_FITS(PolSpectra_preRM_name)

    #get list of candnames
    rmtable_candnames = list(rmtable["candname"])
    #pspectra_candnames = list(polspectra_postRM["candname"])

    #see if in both lists, if so, return index
    if (candname in rmtable_candnames):# and (candname in pspectra_candnames):
        pidx = 0
        while candname in rmtable_candnames[pidx:]:
            if rmtable["PeakNum"][rmtable_candnames.index(candname,pidx)] == comp_num:
                return rmtable_candnames.index(candname,pidx),rmtable["Ncomp"][rmtable_candnames.index(candname,pidx)]
            elif candname in rmtable_candnames[pidx+1:]:
                pidx = rmtable_candnames.index(candname,pidx+1)
            else:
                return -1,np.nan
        #component not in list
        return -1,np.nan
    else:
        return -1,np.nan


#Check if FRB in the table already
def dsarp_FRBinTable_PolSpectrum(candname,comp_num=-1,suff="pre"):

    #get tables
    #rmtable = RMTable.read(RMTable_name)
    if suff == "pre":
        spectrumtable = polspectra.from_FITS(PolSpectra_preRM_name)
    elif suff == "post":
        spectrumtable = polspectra.from_FITS(PolSpectra_postRM_name)
    else:
        print("Please enter valid suff: 'pre' or 'post'")
        return

    #get list of candnames
    #rmtable_candnames = list(rmtable["candname"])
    pspectra_candnames = list(spectrumtable["candname"])

    #see if in both lists, if so, return index
    if (candname in pspectra_candnames):# and (candname in pspectra_candnames):
        pidx = 0
        while candname in pspectra_candnames[pidx:]:
            if spectrumtable["PeakNum"][pspectra_candnames.index(candname,pidx)] == comp_num:
                return pspectra_candnames.index(candname,pidx),spectrumtable["Npeaks"][pspectra_candnames.index(candname,pidx)]
            elif candname in pspectra_candnames[pidx+1:]:
                pidx = pspectra_candnames.index(candname,pidx+1)
            else:
                return -1,np.nan
        #component not in list
        return -1,np.nan
    else:
        return -1,np.nan


#Add single FRB component to the table
def dsarp_addFRBcomp_RMTable(candname,nickname,datadir,comp_dict,comp_num,RA,DEC,freq_test_init,MJD,ncomps,replace=False,replace_idx=-1):

    #get tables
    rmtable = RMTable.read(RMTable_name)
    polspectra_postRM = polspectra.from_FITS(PolSpectra_postRM_name)
    polspectra_preRM = polspectra.from_FITS(PolSpectra_preRM_name)

    #add new row to rm table
    if replace:
        rowidx = replace_idx
    else:
        rowidx = len(rmtable['candname'])
        rmtable.add_row()

    rmtable['ra'][rowidx] = RA
    rmtable['dec'][rowidx] = DEC
    rmtable['nickname'][rowidx] = nickname
    rmtable['datadir'][rowidx] = datadir
    c = SkyCoord(ra=RA*u.deg,dec=DEC*u.deg,frame='icrs')
    rmtable['l'][rowidx] = c.galactic.l.degree
    rmtable['b'][rowidx] = c.galactic.b.degree
    rmtable['ionosphere'][rowidx] = "ionFR"
    rmtable['Ncomp'][rowidx] = ncomps
    rmtable['Npeaks'][rowidx] = ncomps
    rmtable['PeakNum'][rowidx] = comp_num
    rmtable['pol_bias'][rowidx] = "1985A&A...142..100S"
    rmtable['Vfracpol'][rowidx] = comp_dict[comp_num]["V/I"]
    rmtable['Vfracpol_err'][rowidx] = comp_dict[comp_num]["V/I_err"]
    rmtable['absVfracpol'][rowidx] = comp_dict[comp_num]["absV/I"]
    rmtable['absVfracpol_err'][rowidx] = comp_dict[comp_num]["absV/I_err"]
    rmtable['polangle'][rowidx] = comp_dict[comp_num]["PA_pre"]*180/np.pi
    rmtable['polangle_err'][rowidx] = comp_dict[comp_num]["PAerr_pre"]*180/np.pi
    rmtable['minfreq'][rowidx] = np.min(freq_test_init[0])*1e6
    rmtable['maxfreq'][rowidx] = np.max(freq_test_init[0])*1e6
    rmtable['channelwidth'][rowidx] = np.abs(freq_test_init[0][0]-freq_test_init[0][1])*1e6
    rmtable['rmsf_fwhm'][rowidx] = 349.81635341775683 #pre-calculated RMSF in jupyter notebook
    rmtable['telescope'][rowidx] = 'DSA110'
    rmtable['int_time'][rowidx] = 20480*32.7e-6
    rmtable['epoch'][rowidx] = MJD
    rmtable['candname'][rowidx] = candname
    rmtable['snr'][rowidx] = comp_dict[comp_num]["I_snr"]
    rmtable['Vsnr'][rowidx] = comp_dict[comp_num]["V/I_snr"]

    if comp_dict[comp_num]['sigflag']:
        rmtable['rm'][rowidx] = comp_dict[comp_num]["RM2zoom"]
        rmtable['rm_err'][rowidx] = comp_dict[comp_num]["RMerr2zoom"]
        rmtable['rm_width'][rowidx] = comp_dict[comp_num]["RM_FWHM"]
        
        if comp_dict[comp_num]['rm_applied']:
            rmtable['rm_method'][rowidx] = "RM Synthesis"
            rmtable['fracpol'][rowidx] = comp_dict[comp_num]["T/I_post"]
            rmtable['fracpol_err'][rowidx] = comp_dict[comp_num]["T/I_post_err"]
            rmtable['Lfracpol'][rowidx] = comp_dict[comp_num]["L/I_post"]
            rmtable['Lfracpol_err'][rowidx] = comp_dict[comp_num]["L/I_post_err"]
            rmtable['derot_polangle'][rowidx] = comp_dict[comp_num]["PA_post"]*180/np.pi
            rmtable['derot_polangle_err'][rowidx] = comp_dict[comp_num]["PAerr_post"]*180/np.pi
            rmtable['Tsnr'][rowidx] = comp_dict[comp_num]["T/I_post_snr"]
            rmtable['Lsnr'][rowidx] = comp_dict[comp_num]["L/I_post_snr"]

        rmtable['RM1'][rowidx] = comp_dict[comp_num]['RM1']
        rmtable['RMerr1'][rowidx] = comp_dict[comp_num]['RMerr1']
        #rmtable['RMsnrs1'][rowidx] = comp_dict[comp_num]['RMsnrs1']
        #rmtable['trial_RM'][rowidx] = comp_dict[comp_num]['trial_RM']

        rmtable['RM1tools'][rowidx] = comp_dict[comp_num]['RM1tools']
        rmtable['RMerr1tools'][rowidx] = comp_dict[comp_num]['RMerr1tools']
        #rmtable['RMsnrs1tools'][rowidx] = comp_dict[comp_num]['RMsnrs1tools']
        #rmtable['trial_RM_tools'][rowidx] = comp_dict[comp_num]['trial_RM_tools']

        rmtable['RM1zoom'][rowidx] = comp_dict[comp_num]['RM1zoom']
        rmtable['RMerr1zoom'][rowidx] = comp_dict[comp_num]['RMerr1zoom']
        #rmtable['RMsnrs1zoom'][rowidx] = comp_dict[comp_num]['RMsnrs1zoom']
        #rmtable['trial_RM2'][rowidx] = comp_dict[comp_num]['trial_RM2']

        if "RM1tools_zoom" in comp_dict[comp_num].keys():
            rmtable['RM1tools_zoom'][rowidx] = comp_dict[comp_num]['RM1tools_zoom']
            rmtable['RMerr1tools_zoom'][rowidx] = comp_dict[comp_num]['RMerr1tools_zoom']
            #rmtable['RMsnrs1tools_zoom'][rowidx] = comp_dict[comp_num]['RMsnrs1tools_zoom']
            #rmtable['trial_RM_tools_zoom'][rowidx] = comp_dict[comp_num]['trial_RM_tools_zoom']

        rmtable['RM2zoom'][rowidx] = comp_dict[comp_num]['RM2zoom']
        rmtable['RM2errzoom'][rowidx] = comp_dict[comp_num]['RM2errzoom']
        #rmtable['RMsnrs2zoom'][rowidx] = comp_dict[comp_num]['RMsnrs2zoom']
        rmtable['RM_FWHM'][rowidx] = comp_dict[comp_num]['RM_FWHM']

        rmtable['sigflag'][rowidx] = comp_dict[comp_num]['sigflag']
        rmtable['rm_applied'][rowidx] = comp_dict[comp_num]['rm_applied']
        if 'RM_ion' in comp_dict[comp_num].keys():
            rmtable['RM_ion'][rowidx] = comp_dict[comp_num]['RM_ion']
            rmtable['RM_ionerr'][rowidx] = comp_dict[comp_num]['RM_ionerr']
        if 'RM_gal' in comp_dict[comp_num].keys():
            rmtable['RM_gal'][rowidx] = comp_dict[comp_num]['RM_gal']
            rmtable['RM_galerr'][rowidx] = comp_dict[comp_num]['RM_galerr']

    else:
        rmtable['fracpol'][rowidx] = comp_dict[comp_num]["T/I_pre"]
        rmtable['fracpol_err'][rowidx] = comp_dict[comp_num]["T/I_pre_err"]
        rmtable['Lfracpol'][rowidx] = comp_dict[comp_num]["L/I_pre"]
        rmtable['Lfracpol_err'][rowidx] = comp_dict[comp_num]["L/I_pre_err"]
        rmtable['Tsnr'][rowidx] = comp_dict[comp_num]["T/I_pre_snr"]
        rmtable['Lsnr'][rowidx] = comp_dict[comp_num]["L/I_pre_snr"]


    #write to table
    rmtable.write(RMTable_name,overwrite=True)

    return

#Add single FRB component to the table
def dsarp_addFRBcomp_PolSpectrum(candname,nickname,datadir,comp_dict,comp_num,RA,DEC,freq_test_init,MJD,ncomps,n_t,n_f,tsamp,replace=False,replace_idx=-1,suff="pre"):

    #get polspectra table --> pre RM
    if suff == "pre":
        spectrumtable = polspectra.from_FITS(PolSpectra_preRM_name)
    elif suff == "post":
        spectrumtable = polspectra.from_FITS(PolSpectra_postRM_name)
    else:
        print("Please enter valid suffix: 'pre' or 'post'")
        return -1

    #add new row to polspec table
    if replace:
        rowidx = replace_idx
    else:
        rowidx = len(spectrumtable['candname'])
        spectrumtable2 = spectrumtable[0].copy()
        spectrumtable.merge_tables(spectrumtable2)


    spectrumtable['Nchan'][rowidx] = len(freq_test_init[0])
    spectrumtable['source_number'][rowidx] = rowidx
    spectrumtable['nickname'][rowidx] = nickname
    spectrumtable['datadir'][rowidx] = datadir
    spectrumtable['ra'][rowidx] = RA
    spectrumtable['dec'][rowidx] = DEC
    c = SkyCoord(ra=RA*u.deg,dec=DEC*u.deg,frame='icrs')
    spectrumtable['l'][rowidx] = c.galactic.l.degree
    spectrumtable['b'][rowidx] = c.galactic.b.degree

    spectrumtable['freq'][rowidx] = freq_test_init[0]
    spectrumtable['stokesI'][rowidx] = comp_dict[comp_num]["I_f_init"].data
    spectrumtable['stokesQ'][rowidx] = comp_dict[comp_num]["Q_f_init"].data
    spectrumtable['stokesU'][rowidx] = comp_dict[comp_num]["U_f_init"].data
    spectrumtable['stokesV'][rowidx] = comp_dict[comp_num]["V_f_init"].data
    if ma.is_masked(comp_dict[comp_num]["I_f_init"]):
        spectrumtable['stokesI_mask'][rowidx] = comp_dict[comp_num]["I_f_init"].mask
    else:
        spectrumtable['stokesI_mask'][rowidx] = np.zeros(spectrumtable['Nchan'][rowidx],dtype=bool)
    if ma.is_masked(comp_dict[comp_num]["Q_f_init"]):
        spectrumtable['stokesQ_mask'][rowidx] = comp_dict[comp_num]["Q_f_init"].mask
    else:
        spectrumtable['stokesQ_mask'][rowidx] = np.zeros(spectrumtable['Nchan'][rowidx],dtype=bool)
    if ma.is_masked(comp_dict[comp_num]["U_f_init"]):
        spectrumtable['stokesU_mask'][rowidx] = comp_dict[comp_num]["U_f_init"].mask
    else:
        spectrumtable['stokesU_mask'][rowidx] = np.zeros(spectrumtable['Nchan'][rowidx],dtype=bool)
    if ma.is_masked(comp_dict[comp_num]["V_f_init"]):
        spectrumtable['stokesV_mask'][rowidx] = comp_dict[comp_num]["V_f_init"].mask    
    else:
        spectrumtable['stokesV_mask'][rowidx] = np.zeros(spectrumtable['Nchan'][rowidx],dtype=bool)

    spectrumtable['PA'][rowidx] = comp_dict[comp_num]["PA_f_init"].data
    if ma.is_masked(comp_dict[comp_num]["PA_f_init"]):
        spectrumtable['PA_mask'][rowidx] = comp_dict[comp_num]["PA_f_init"].mask
    else:
        spectrumtable['PA_mask'][rowidx] = np.zeros(spectrumtable['Nchan'][rowidx],dtype=bool)

    spectrumtable['stokesI_error'][rowidx] = np.nan*np.ones(len(freq_test_init[0]))#comp_dict[comp_num]["I_f_init"]
    spectrumtable['stokesQ_error'][rowidx] = np.nan*np.ones(len(freq_test_init[0]))#comp_dict[comp_num]["Q_f_init"]
    spectrumtable['stokesU_error'][rowidx] = np.nan*np.ones(len(freq_test_init[0]))#comp_dict[comp_num]["U_f_init"]
    spectrumtable['stokesV_error'][rowidx] = np.nan*np.ones(len(freq_test_init[0]))#comp_dict[comp_num]["V_f_init"]
    spectrumtable['PA_error'][rowidx] = comp_dict[comp_num]["PA_f_errs_init"]
    if ma.is_masked(comp_dict[comp_num]["PA_f_errs_init"]):
        spectrumtable['PA_error_mask'][rowidx] = comp_dict[comp_num]["PA_f_errs_init"].mask
    else:
        spectrumtable['PA_error_mask'][rowidx] = np.zeros(spectrumtable['Nchan'][rowidx],dtype=bool)

    spectrumtable['beam_maj'][rowidx] = [0.0]
    spectrumtable['beam_min'][rowidx] = [0.0]
    spectrumtable['beam_pa'][rowidx] = [0]
    spectrumtable['Nchan'][rowidx] = len(freq_test_init[0])
    spectrumtable['channel_width'][rowidx] = [np.abs(freq_test_init[0][1]-freq_test_init[0][0])]
    spectrumtable['Npeaks'][rowidx] = ncomps
    spectrumtable['PeakNum'][rowidx] = comp_num
    spectrumtable['candname'][rowidx] = candname
    

    spectrumtable['weights'][rowidx] = comp_dict[comp_num]["weights"]
    spectrumtable['buffL'][rowidx],spectrumtable['buffR'][rowidx] = comp_dict[comp_num]["buff"]
    spectrumtable['n_tw'][rowidx] = comp_dict[comp_num]["n_t_weight"]
    spectrumtable['sf_window'][rowidx] = comp_dict[comp_num]["sf_window_weights"]
    spectrumtable['ibox_rev'][rowidx] = comp_dict[comp_num]["ibox"]
    spectrumtable['maskL'][rowidx],spectrumtable['maskR'][rowidx] = comp_dict[comp_num]["mask_start"],comp_dict[comp_num]["mask_stop"]
    spectrumtable['multipeaks'][rowidx] = comp_dict[comp_num]["multipeaks"]
    if comp_dict[comp_num]["multipeaks"]:
        spectrumtable['height'][rowidx] = comp_dict[comp_num]["height"]
        spectrumtable['scaled_height'][rowidx] = comp_dict[comp_num]["scaled_height"]
    else:
        spectrumtable['height'][rowidx] = np.nan
        spectrumtable['scaled_height'][rowidx] = np.nan
    spectrumtable['intL'][rowidx] = comp_dict[comp_num]["intL"]
    spectrumtable['intR'][rowidx] = comp_dict[comp_num]["intR"]
    spectrumtable['timestart'][rowidx] = comp_dict[comp_num]["timestart"]
    spectrumtable['timestop'][rowidx] = comp_dict[comp_num]["timestop"]

    spectrumtable['n_t'] = n_t
    spectrumtable['n_f'] = n_f
    spectrumtable['dt'] = tsamp
    spectrumtable['mjd'] = MJD
    spectrumtable['sigflag'] = comp_dict[comp_num]["sigflag"]
    spectrumtable['rm_applied'] = comp_dict[comp_num]["rm_applied"]

    if comp_dict[comp_num]["sigflag"]:
        spectrumtable['RMsnrs1'][rowidx] = comp_dict[comp_num]['RMsnrs1']
        spectrumtable['trial_RM'][rowidx] = comp_dict[comp_num]['trial_RM']

        spectrumtable['RMsnrs1tools'][rowidx] = comp_dict[comp_num]['RMsnrs1tools']
        spectrumtable['trial_RM_tools'][rowidx] = comp_dict[comp_num]['trial_RM_tools']

        spectrumtable['RMsnrs1zoom'][rowidx] = comp_dict[comp_num]['RMsnrs1zoom']
        spectrumtable['trial_RM2'][rowidx] = comp_dict[comp_num]['trial_RM2']

        if "RM1tools_zoom" in comp_dict[comp_num].keys():
            spectrumtable['RMsnrs1tools_zoom'][rowidx] = comp_dict[comp_num]['RMsnrs1tools_zoom']
            spectrumtable['trial_RM_tools_zoom'][rowidx] = comp_dict[comp_num]['trial_RM_tools_zoom']

        spectrumtable['RMsnrs2zoom'][rowidx] = comp_dict[comp_num]['RMsnrs2zoom']




    #write to table
    if suff == "pre":
        spectrumtable.write_FITS(PolSpectra_preRM_name,overwrite=True)
    elif suff == "post":
        spectrumtable.write_FITS(PolSpectra_postRM_name,overwrite=True)
    
    return

#Add single full burst to the table
def dsarp_addFRBfull_PolSpectrum(candname,nickname,datadir,fullburst_dict,RA,DEC,freq_test_init,MJD,ncomps,n_t,n_f,tsamp,replace=False,replace_idx=-1,suff="pre"):

    #get polspectra table 
    if suff == "pre":
        spectrumtable = polspectra.from_FITS(PolSpectra_preRM_name)
    elif suff == "post":
        spectrumtable = polspectra.from_FITS(PolSpectra_postRM_name)
    else:
        print("Please enter valid suffix: 'pre' or 'post'")
        return -1

    #add new row to polspec table
    if replace:
        rowidx = replace_idx
    else:
        rowidx = len(spectrumtable['candname'])
        spectrumtable2 = spectrumtable[0].copy()
        spectrumtable.merge_tables(spectrumtable2)

    spectrumtable['source_number'][rowidx] = rowidx
    spectrumtable['nickname'][rowidx] = nickname
    spectrumtable['datadir'][rowidx] = datadir
    spectrumtable['ra'][rowidx] = RA
    spectrumtable['dec'][rowidx] = DEC
    c = SkyCoord(ra=RA*u.deg,dec=DEC*u.deg,frame='icrs')
    spectrumtable['l'][rowidx] = c.galactic.l.degree
    spectrumtable['b'][rowidx] = c.galactic.b.degree

    spectrumtable['freq'][rowidx] = freq_test_init[0]
    spectrumtable['stokesI'][rowidx] = fullburst_dict["I_f_init"].data 
    spectrumtable['stokesQ'][rowidx] = fullburst_dict["Q_f_init"].data
    spectrumtable['stokesU'][rowidx] = fullburst_dict["U_f_init"].data
    spectrumtable['stokesV'][rowidx] = fullburst_dict["V_f_init"].data
    spectrumtable['PA'][rowidx] = fullburst_dict["PA_f_init"].data
    if ma.is_masked(fullburst_dict["I_f_init"]):
        spectrumtable['stokesI_mask'][rowidx] = fullburst_dict["I_f_init"].mask
    else:
        spectrumtable['stokesI_mask'][rowidx] = np.zeros(spectrumtable['Nchan'][rowidx],dtype=bool)
    if ma.is_masked(fullburst_dict["Q_f_init"]):
        spectrumtable['stokesQ_mask'][rowidx] = fullburst_dict["Q_f_init"].mask
    else:
        spectrumtable['stokesQ_mask'][rowidx] = np.zeros(spectrumtable['Nchan'][rowidx],dtype=bool)
    if ma.is_masked(fullburst_dict["U_f_init"]):
        spectrumtable['stokesU_mask'][rowidx] = fullburst_dict["U_f_init"].mask
    else:
        spectrumtable['stokesU_mask'][rowidx] = np.zeros(spectrumtable['Nchan'][rowidx],dtype=bool)
    if ma.is_masked(fullburst_dict["V_f_init"]):
        spectrumtable['stokesV_mask'][rowidx] = fullburst_dict["V_f_init"].mask
    else:
        spectrumtable['stokesV_mask'][rowidx] = np.zeros(spectrumtable['Nchan'][rowidx],dtype=bool)
    if ma.is_masked(fullburst_dict["PA_f_init"]):
        spectrumtable['PA_mask'][rowidx] = fullburst_dict["PA_f_init"].mask
    else:
        spectrumtable['PA_mask'][rowidx] = np.zeros(spectrumtable['Nchan'][rowidx],dtype=bool)



    spectrumtable['stokesI_error'][rowidx] = np.nan*np.ones(len(freq_test_init[0]))#comp_dict[comp_num]["I_f_init"]
    spectrumtable['stokesQ_error'][rowidx] = np.nan*np.ones(len(freq_test_init[0]))#comp_dict[comp_num]["Q_f_init"]
    spectrumtable['stokesU_error'][rowidx] = np.nan*np.ones(len(freq_test_init[0]))#comp_dict[comp_num]["U_f_init"]
    spectrumtable['stokesV_error'][rowidx] = np.nan*np.ones(len(freq_test_init[0]))#comp_dict[comp_num]["V_f_init"]
    spectrumtable['PA_error'][rowidx] = fullburst_dict["PA_f_errs_init"]
    if ma.is_masked(fullburst_dict["PA_f_errs_init"]):
        spectrumtable['PA_error_mask'][rowidx] = fullburst_dict["PA_f_errs_init"].mask
    else:
        spectrumtable['PA_error_mask'][rowidx] = np.zeros(spectrumtable['Nchan'][rowidx],dtype=bool)

    spectrumtable['beam_maj'][rowidx] = [0.0]
    spectrumtable['beam_min'][rowidx] = [0.0]
    spectrumtable['beam_pa'][rowidx] = [0]
    spectrumtable['Nchan'][rowidx] = len(freq_test_init[0])
    spectrumtable['channel_width'][rowidx] = [np.abs(freq_test_init[0][1]-freq_test_init[0][0])]
    spectrumtable['Npeaks'][rowidx] = ncomps
    spectrumtable['PeakNum'][rowidx] = -1
    spectrumtable['candname'][rowidx] = candname

    spectrumtable['weights'][rowidx] = fullburst_dict["weights"]
    spectrumtable['buffL'][rowidx],spectrumtable['buffR'][rowidx] = fullburst_dict["buff"]
    spectrumtable['n_tw'][rowidx] = -1 
    spectrumtable['sf_window'][rowidx] = -1
    spectrumtable['ibox_rev'][rowidx] = -1
    spectrumtable['maskL'][rowidx],spectrumtable['maskR'][rowidx] = -1,-1#comp_dict[comp_num]["mask_start"],comp_dict[comp_num]["mask_stop"]
    spectrumtable['multipeaks'][rowidx] = fullburst_dict["multipeaks_all"]
    spectrumtable['height'][rowidx] = np.nan#comp_dict[comp_num]["height"]
    spectrumtable['scaled_height'][rowidx] = np.nan#comp_dict[comp_num]["scaled_height"]
    spectrumtable['intL'][rowidx] = fullburst_dict["intL"]
    spectrumtable['intR'][rowidx] = fullburst_dict["intR"]
    spectrumtable['timestart'][rowidx] = fullburst_dict["timestart"]
    spectrumtable['timestop'][rowidx] = fullburst_dict["timestop"]

    spectrumtable['n_t'] = n_t
    spectrumtable['n_f'] = n_f
    spectrumtable['dt'] = tsamp
    spectrumtable['mjd'] = MJD
    spectrumtable['sigflag'] = fullburst_dict["sigflag"]
    spectrumtable['rm_applied'] = fullburst_dict["rm_applied"]

    if fullburst_dict["sigflag"]:
        spectrumtable['RMsnrs1'][rowidx] = fullburst_dict['RMsnrs1']
        spectrumtable['trial_RM'][rowidx] = fullburst_dict['trial_RM']

        spectrumtable['RMsnrs1tools'][rowidx] = fullburst_dict['RMsnrs1tools']
        spectrumtable['trial_RM_tools'][rowidx] = fullburst_dict['trial_RM_tools']

        spectrumtable['RMsnrs1zoom'][rowidx] = fullburst_dict['RMsnrs1zoom']
        spectrumtable['trial_RM2'][rowidx] = fullburst_dict['trial_RM2']

        if "RM1tools_zoom" in fullburst_dict.keys():
            spectrumtable['RMsnrs1tools_zoom'][rowidx] = fullburst_dict['RMsnrs1tools_zoom']
            spectrumtable['trial_RM_tools_zoom'][rowidx] = fullburst_dict['trial_RM_tools_zoom']

        spectrumtable['RMsnrs2zoom'][rowidx] = fullburst_dict['RMsnrs2zoom']



    #write to table
    if suff == "pre":
        spectrumtable.write_FITS(PolSpectra_preRM_name,overwrite=True)
    elif suff == "post":
        spectrumtable.write_FITS(PolSpectra_postRM_name,overwrite=True)

    return



"""
    ['source_number',
 'ra',
 'dec',
 'l',
 'b',
 'freq',
 'stokesI',
 'stokesI_error',
 'stokesQ',
 'stokesQ_error',
 'stokesU',
 'stokesU_error',
 'beam_maj',
 'beam_min',
 'beam_pa',
 'Nchan',
 'stokesV',
 'stokesV_error',
 'channel_width',
 'candname',
 'Npeaks',
 'PeakNum',
 'PA',
 'PA_error',
 'weights',
 'buffL',
 'buffR',
 'n_tw',
 'sf_window',
 'maskL',
 'maskR',
 'multipeaks',
 'height',
 'scaled_height',
 'n_t',
 'n_f',
 'dt',
 'mjd',
 'timestart',
 'timestop',
 'intL',
 'intR',
 'sigflag']
"""



#Add full FRB component to the table
def dsarp_addFRBfull_RMTable(candname,nickname,datadir,fullburst_dict,RA,DEC,freq_test_init,MJD,ncomps,replace=False,replace_idx=-1):
    #get tables
    rmtable = RMTable.read(RMTable_name)

    #add new row to rm table
    if replace:
        rowidx = replace_idx
    else:
        rowidx = len(rmtable['candname'])
        rmtable.add_row()

    rmtable['ra'][rowidx] = RA
    rmtable['dec'][rowidx] = DEC
    rmtable['nickname'][rowidx] = nickname
    rmtable['datadir'][rowidx] = datadir
    c = SkyCoord(ra=RA*u.deg,dec=DEC*u.deg,frame='icrs')
    rmtable['l'][rowidx] = c.galactic.l.degree
    rmtable['b'][rowidx] = c.galactic.b.degree
    rmtable['ionosphere'][rowidx] = "ionFR"
    rmtable['Ncomp'][rowidx] = ncomps 
    rmtable['Npeaks'][rowidx] = ncomps
    rmtable['PeakNum'][rowidx] = -1
    rmtable['pol_bias'][rowidx] = "1985A&A...142..100S"
    rmtable['Vfracpol'][rowidx] = fullburst_dict["V/I"]
    rmtable['Vfracpol_err'][rowidx] = fullburst_dict["V/I_err"]
    rmtable['absVfracpol'][rowidx] = fullburst_dict["absV/I"]
    rmtable['absVfracpol_err'][rowidx] = fullburst_dict["absV/I_err"]
    rmtable['polangle'][rowidx] = fullburst_dict["PA_pre"]*180/np.pi
    rmtable['polangle_err'][rowidx] = fullburst_dict["PAerr_pre"]*180/np.pi
    rmtable['minfreq'][rowidx] = np.min(freq_test_init[0])*1e6
    rmtable['maxfreq'][rowidx] = np.max(freq_test_init[0])*1e6
    rmtable['channelwidth'][rowidx] = np.abs(freq_test_init[0][0]-freq_test_init[0][1])*1e6
    rmtable['rmsf_fwhm'][rowidx] = 349.81635341775683 #pre-calculated RMSF in jupyter notebook
    rmtable['telescope'][rowidx] = 'DSA110'
    rmtable['int_time'][rowidx] = 20480*32.7e-6
    rmtable['epoch'][rowidx] = MJD
    rmtable['candname'][rowidx] = candname

    if fullburst_dict['sigflag']:
        rmtable['rm'][rowidx] = fullburst_dict["RM2zoom"]            
        rmtable['rm_err'][rowidx] = fullburst_dict["RMerr2zoom"]            
        rmtable['rm_width'][rowidx] = fullburst_dict["RM_FWHM"]            
        rmtable['rm_method'][rowidx] = "RM Synthesis"            
        
        if fullburst_dict['rm_applied']:
            rmtable['fracpol'][rowidx] = fullburst_dict["T/I_post"]            
            rmtable['fracpol_err'][rowidx] = fullburst_dict["T/I_post_err"]            
            rmtable['Lfracpol'][rowidx] = fullburst_dict["L/I_post"]            
            rmtable['Lfracpol_err'][rowidx] = fullburst_dict["L/I_post_err"]            
            rmtable['derot_polangle'][rowidx] = fullburst_dict["PA_post"]*180/np.pi            
            rmtable['derot_polangle_err'][rowidx] = fullburst_dict["PAerr_post"]*180/np.pi        
            rmtable['Tsnr'][rowidx] = fullburst_dict["T/I_post_snr"]
            rmtable['Lsnr'][rowidx] = fullburst_dict["L/I_post_snr"]


        rmtable['RM1'][rowidx] = fullburst_dict['RM1']
        rmtable['RMerr1'][rowidx] = fullburst_dict['RMerr1']
        #rmtable['RMsnrs1'][rowidx] = fullburst_dict['RMsnrs1']
        #rmtable['trial_RM'][rowidx] = fullburst_dict['trial_RM']

        rmtable['RM1tools'][rowidx] = fullburst_dict['RM1tools']
        rmtable['RMerr1tools'][rowidx] = fullburst_dict['RMerr1tools']
        #rmtable['RMsnrs1tools'][rowidx] = fullburst_dict['RMsnrs1tools']
        #rmtable['trial_RM_tools'][rowidx] = fullburst_dict['trial_RM_tools']

        rmtable['RM1zoom'][rowidx] = fullburst_dict['RM1zoom']
        rmtable['RMerr1zoom'][rowidx] = fullburst_dict['RMerr1zoom']
        #rmtable['RMsnrs1zoom'][rowidx] = fullburst_dict['RMsnrs1zoom']
        #rmtable['trial_RM2'][rowidx] = fullburst_dict['trial_RM2']

        if "RM1tools_zoom" in fullburst_dict.keys():
            rmtable['RM1tools_zoom'][rowidx] = fullburst_dict['RM1tools_zoom']
            rmtable['RMerr1tools_zoom'][rowidx] = fullburst_dict['RMerr1tools_zoom']
            #rmtable['RMsnrs1tools_zoom'][rowidx] = fullburst_dict['RMsnrs1tools_zoom']
            #rmtable['trial_RM_tools_zoom'][rowidx] = fullburst_dict['trial_RM_tools_zoom']

        rmtable['RM2zoom'][rowidx] = fullburst_dict['RM2zoom']
        rmtable['RM2errzoom'][rowidx] = fullburst_dict['RM2errzoom']
        #rmtable['RMsnrs2zoom'][rowidx] = fullburst_dict['RMsnrs2zoom']
        rmtable['RM_FWHM'][rowidx] = fullburst_dict['RM_FWHM']

        rmtable['sigflag'][rowidx] = fullburst_dict['sigflag']
        rmtable['rm_applied'][rowidx] = fullburst_dict['rm_applied']
        if 'RM_ion' in fullburst_dict.keys():
            rmtable['RM_ion'][rowidx] = fullburst_dict['RM_ion']
            rmtable['RM_ionerr'][rowidx] = fullburst_dict['RM_ionerr']
        if 'RM_gal' in fullburst_dict.keys():
            rmtable['RM_gal'][rowidx] = fullburst_dict['RM_gal']
            rmtable['RM_galerr'][rowidx] = fullburst_dict['RM_galerr']

    else:                
        rmtable['fracpol'][rowidx] = fullburst_dict["T/I_pre"]            
        rmtable['fracpol_err'][rowidx] = fullburst_dict["T/I_pre_err"]            
        rmtable['Lfracpol'][rowidx] = fullburst_dict["L/I_pre"]
        rmtable['Lfracpol_err'][rowidx] = fullburst_dict["L/I_pre_err"]

    rmtable.write(RMTable_name,overwrite=True)

    return


#Update existing FRB entry
def dsarp_updateFB(frb_name,comp_dict,fullburst_dict):

    return

#Remove FRB from RM table
def dsarp_removeFRB_RMTable(candname):

    #get tables
    rmtable = RMTable.read(RMTable_name)

    #remove all components
    while candname in rmtable['candname']:
        rmtable.remove_row(list(rmtable["candname"]).index(candname))

    rmtable.write(RMTable_name,overwrite=True)

    return

#Remove FRB from pol table
def dsarp_removeFRB_PolSpectrum(candname,suff="pre"):

    #get polspectra table 
    if suff == "pre":
        spectrumtable = polspectra.from_FITS(PolSpectra_preRM_name)
    elif suff == "post":
        spectrumtable = polspectra.from_FITS(PolSpectra_postRM_name)
    else:
        print("Please enter valid suffix: 'pre' or 'post'")
        return -1

    #remove all components
    while candname in spectrumtable['candname']:
        ridx = list(spectrumtable['candname']).index(candname)
        spectrumtable_copy = spectrumtable[:ridx].copy()
        spectrumtable_copy.merge_tables(spectrumtable[ridx + 1:].copy())
        spectrumtable = spectrumtable_copy.copy()

    #write to table
    if suff == "pre":
        spectrumtable.write_FITS(PolSpectra_preRM_name,overwrite=True)
    elif suff == "post":
        spectrumtable.write_FITS(PolSpectra_postRM_name,overwrite=True)


    return


#pull FRB data from pol table
def dsarp_getFRBfull(candname,suff="pre"):
    #open tables
    rmtable = RMTable.read(RMTable_name)
    if suff == "pre":
        spectrumtable = polspectra.from_FITS(PolSpectra_preRM_name)
    elif suff == "post":
        spectrumtable = polspectra.from_FITS(PolSpectra_postRM_name)
    else:
        print("Please enter valid suffix: 'pre' or 'post'")
        return -1

    #find full burst index in table
    tabidx_rmtable,ncomps_rmtable = dsarp_FRBinTable_RMTable(candname)
    tabidx_pspectra,ncomps_pspectra = dsarp_FRBinTable_PolSpectrum(candname,suff=suff)
    if tabidx_rmtable == -1 or tabidx_pspectra == -1:
        if tabidx_rmtable == -1:
            print("FRB not in RMTable")
        if tabidx_pspectra == -1:
            print("FRB not in PolSpectrum")
        return


    #create empty fullburst and component dictionaries
    fullburst_dict = dict()
    comp_dict = dict()
    parameters_dict = dict()
    spectra_dict = dict()

    
    #set up full burst dict
    fullburst_dict["timestart"] = spectrumtable["timestart"][tabidx_pspectra]
    fullburst_dict["timestop"] = spectrumtable["timestop"][tabidx_pspectra]
    fullburst_dict["num_comps"] = spectrumtable["Npeaks"][tabidx_pspectra]
    fullburst_dict["buff"] = (spectrumtable["buffL"][tabidx_pspectra],spectrumtable["buffR"][tabidx_pspectra])
    fullburst_dict["weights"] = spectrumtable["weights"][tabidx_pspectra]
    fullburst_dict["multipeaks_all"] = spectrumtable["multipeaks"][tabidx_pspectra]
    fullburst_dict["sigflag"] = spectrumtable["sigflag"][tabidx_pspectra]
    fullburst_dict["rm_applied"] = spectrumtable["rm_applied"][tabidx_pspectra]
    fullburst_dict["intL"] = spectrumtable["intL"][tabidx_pspectra]
    fullburst_dict["intR"] = spectrumtable["intR"][tabidx_pspectra]

    fullburst_dict["I_f"] = ma.masked_array(spectrumtable["stokesI"][tabidx_pspectra],spectrumtable["stokesI_mask"][tabidx_pspectra])
    fullburst_dict["Q_f"] = ma.masked_array(spectrumtable["stokesQ"][tabidx_pspectra],spectrumtable["stokesQ_mask"][tabidx_pspectra]) 
    fullburst_dict["U_f"] = ma.masked_array(spectrumtable["stokesU"][tabidx_pspectra],spectrumtable["stokesU_mask"][tabidx_pspectra])
    fullburst_dict["V_f"] = ma.masked_array(spectrumtable["stokesV"][tabidx_pspectra],spectrumtable["stokesV_mask"][tabidx_pspectra])

    fullburst_dict["I_f_init"] = ma.masked_array(spectrumtable["stokesI"][tabidx_pspectra],spectrumtable["stokesI_mask"][tabidx_pspectra]) 
    fullburst_dict["Q_f_init"] = ma.masked_array(spectrumtable["stokesQ"][tabidx_pspectra],spectrumtable["stokesQ_mask"][tabidx_pspectra])
    fullburst_dict["U_f_init"] = ma.masked_array(spectrumtable["stokesU"][tabidx_pspectra],spectrumtable["stokesU_mask"][tabidx_pspectra])
    fullburst_dict["V_f_init"] = ma.masked_array(spectrumtable["stokesV"][tabidx_pspectra],spectrumtable["stokesV_mask"][tabidx_pspectra])

    fullburst_dict["PA_f"] = ma.masked_array(spectrumtable["PA"][tabidx_pspectra],spectrumtable["PA_mask"][tabidx_pspectra])
    fullburst_dict["PA_f_errs"] = ma.masked_array(spectrumtable["PA_error"][tabidx_pspectra],spectrumtable["PA_error_mask"][tabidx_pspectra])

    fullburst_dict["PA_f_init"] = ma.masked_array(spectrumtable["PA"][tabidx_pspectra],spectrumtable["PA_mask"][tabidx_pspectra])
    fullburst_dict["PA_f_errs_init"] = ma.masked_array(spectrumtable["PA_error"][tabidx_pspectra],spectrumtable["PA_error_mask"][tabidx_pspectra])

    if suff == "pre":
        fullburst_dict["PA_pre"] = rmtable["polangle"][tabidx_rmtable]
        fullburst_dict["PAerr_pre"] = rmtable["polangle_err"][tabidx_rmtable]

        fullburst_dict["T/I_pre"] = rmtable["fracpol"][tabidx_rmtable]
        fullburst_dict["T/I_pre_err"] = rmtable["fracpol_err"][tabidx_rmtable]
        fullburst_dict["T/I_pre_snr"] = rmtable["Tsnr"][tabidx_rmtable]
        fullburst_dict["L/I_pre"] = rmtable["Lfracpol"][tabidx_rmtable]
        fullburst_dict["L/I_pre_err"] = rmtable["Lfracpol_err"][tabidx_rmtable]
        fullburst_dict["L/I_pre_snr"] = rmtable["Lsnr"][tabidx_rmtable]
    
    elif suff == "post":
        fullburst_dict["PA_post"] = rmtable["derot_polangle"][tabidx_rmtable]
        fullburst_dict["PAerr_post"] = rmtable["derot_polangle_err"][tabidx_rmtable]

        fullburst_dict["T/I_post"] = rmtable["fracpol"][tabidx_rmtable]
        fullburst_dict["T/I_post_err"] = rmtable["fracpol_err"][tabidx_rmtable]
        fullburst_dict["T/I_post_snr"] = rmtable["Tsnr"][tabidx_rmtable]
        fullburst_dict["L/I_post"] = rmtable["Lfracpol"][tabidx_rmtable]
        fullburst_dict["L/I_post_err"] = rmtable["Lfracpol_err"][tabidx_rmtable]
        fullburst_dict["L/I_post_snr"] = rmtable["Lsnr"][tabidx_rmtable]

    fullburst_dict["RM1"] = rmtable["RM1"][tabidx_rmtable]
    fullburst_dict["RMerr1"] = rmtable["RMerr1"][tabidx_rmtable]
    #fullburst_dict["RMsnrs1"] = rmtable["RMsnrs1"][tabidx_rmtable]
    #fullburst_dict["trial_RM"] = rmtable["trial_RM"][tabidx_rmtable]

    fullburst_dict["RM1tools"] = rmtable["RM1tools"][tabidx_rmtable]
    fullburst_dict["RMerr1tools"] = rmtable["RMerr1tools"][tabidx_rmtable]
    #fullburst_dict["RMsnrs1tools"] = rmtable["RMsnrs1tools"][tabidx_rmtable]
    #fullburst_dict["trial_RM_tools"] = rmtable["trial_RM_tools"][tabidx_rmtable]


    fullburst_dict["RM1zoom"] = rmtable["RM1zoom"][tabidx_rmtable]
    fullburst_dict["RMerr1zoom"] = rmtable["RMerr1zoom"][tabidx_rmtable]
    #fullburst_dict["RMsnrs1zoom"] = rmtable["RMsnrs1zoom"][tabidx_rmtable]
    #fullburst_dict["trial_RM2"] = rmtable["trial_RM2"][tabidx_rmtable]


    fullburst_dict["RM1tools_zoom"] = rmtable["RM1tools_zoom"][tabidx_rmtable]
    fullburst_dict["RMerr1tools_zoom"] = rmtable["RMerr1tools_zoom"][tabidx_rmtable]
    #fullburst_dict["RMsnrs1tools_zoom"] = rmtable["RMsnrs1tools_zoom"][tabidx_rmtable]
    #fullburst_dict["trial_RM_tools_zoom"] = rmtable["trial_RM_tools_zoom"][tabidx_rmtable]

    fullburst_dict["RM2zoom"] = rmtable["RM2zoom"][tabidx_rmtable]
    fullburst_dict["RM2errzoom"] = rmtable["RM2errzoom"][tabidx_rmtable]
    #fullburst_dict["RMsnrs2zoom"] = rmtable["RMsnrs2zoom"][tabidx_rmtable]
    fullburst_dict["RM_FWHM"] = rmtable["RM_FWHM"][tabidx_rmtable]
    
    fullburst_dict["sigflag"] = rmtable["sigflag"][tabidx_rmtable]
    fullburst_dict["RM_ion"] = rmtable["RM_ion"][tabidx_rmtable]
    fullburst_dict["RM_ionerr"] = rmtable["RM_ionerr"][tabidx_rmtable]
    fullburst_dict["RM_gal"] = rmtable["RM_gal"][tabidx_rmtable]
    fullburst_dict["RM_galerr"] = rmtable["RM_galerr"][tabidx_rmtable]
    
    fullburst_dict["absV/I"] = rmtable["absVfracpol"][tabidx_rmtable]
    fullburst_dict["absV/I_err"] = rmtable["absVfracpol_err"][tabidx_rmtable]
    fullburst_dict["V/I"] = rmtable["Vfracpol"][tabidx_rmtable]
    fullburst_dict["V/I_err"] = rmtable["Vfracpol_err"][tabidx_rmtable]
    fullburst_dict["V/I_snr"] = rmtable["Vsnr"][tabidx_rmtable]
    fullburst_dict["I_snr"] = rmtable["snr"][tabidx_rmtable]

    #get component dict
    comp_dict,tmp2,tmp3 = dsarp_getFRBcomp(candname,-1,np.arange(fullburst_dict["num_comps"]),suff=suff)

    #get parameter dict
    parameters_dict["curr_weights"] = spectrumtable["weights"][tabidx_pspectra]
    parameters_dict["n_t"] = spectrumtable["n_t"][tabidx_pspectra]
    parameters_dict["n_f"] = spectrumtable["n_f"][tabidx_pspectra]
    parameters_dict["curr_comp"] = spectrumtable["PeakNum"][tabidx_pspectra]
    parameters_dict["ibox"] = spectrumtable["ibox_rev"][tabidx_pspectra]
    parameters_dict["tsamp"] = spectrumtable["dt"][tabidx_pspectra]
    parameters_dict["nickname"] = spectrumtable["nickname"][tabidx_pspectra]
    parameters_dict["ids"] = spectrumtable["candname"][tabidx_pspectra]
    parameters_dict["datadir"] = spectrumtable["datadir"][tabidx_pspectra]
    parameters_dict["RA"] = spectrumtable["dec"][tabidx_pspectra]
    parameters_dict["DEC"] = spectrumtable["ra"][tabidx_pspectra]
    parameters_dict["MJD"] = spectrumtable["mjd"][tabidx_pspectra]
    parameters_dict["frb_name"] = spectrumtable["candname"][tabidx_pspectra] + "_" + spectrumtable["nickname"][tabidx_pspectra]

    fullburst_dict["RMsnrs1"] = spectrumtable["RMsnrs1"][tabidx_pspectra]
    fullburst_dict["trial_RM"] = spectrumtable["trial_RM"][tabidx_pspectra]

    fullburst_dict["RMsnrs1tools"] = spectrumtable["RMsnrs1tools"][tabidx_pspectra]
    fullburst_dict["trial_RM_tools"] = spectrumtable["trial_RM_tools"][tabidx_pspectra]


    fullburst_dict["RMsnrs1zoom"] = spectrumtable["RMsnrs1zoom"][tabidx_pspectra]       
    fullburst_dict["trial_RM2"] = spectrumtable["trial_RM2"][tabidx_pspectra]


    fullburst_dict["RMsnrs1tools_zoom"] = spectrumtable["RMsnrs1tools_zoom"][tabidx_pspectra]      
    fullburst_dict["trial_RM_tools_zoom"] = spectrumtable["trial_RM_tools_zoom"][tabidx_pspectra]

    fullburst_dict["RMsnrs2zoom"] = spectrumtable["RMsnrs2zoom"][tabidx_pspectra]

    #get spectrum dict
    spectra_dict["I_f"] = ma.masked_array(spectrumtable["stokesI"][tabidx_pspectra],spectrumtable["stokesI_mask"][tabidx_pspectra]) 
    spectra_dict["Q_f"] = ma.masked_array(spectrumtable["stokesQ"][tabidx_pspectra],spectrumtable["stokesQ_mask"][tabidx_pspectra])
    spectra_dict["U_f"] = ma.masked_array(spectrumtable["stokesU"][tabidx_pspectra],spectrumtable["stokesU_mask"][tabidx_pspectra])
    spectra_dict["V_f"] = ma.masked_array(spectrumtable["stokesV"][tabidx_pspectra],spectrumtable["stokesV_mask"][tabidx_pspectra])
    spectra_dict["freq_test"] = [spectrumtable["freq"][tabidx_pspectra]]*4


    return fullburst_dict,comp_dict,parameters_dict,spectra_dict


#pull FRB data from pol table for individual components
def dsarp_getFRBcomp(candname,curr_comp,all_comp_nums,suff="pre"):
    #open tables
    rmtable = RMTable.read(RMTable_name)
    if suff == "pre":
        spectrumtable = polspectra.from_FITS(PolSpectra_preRM_name)
    elif suff == "post":
        spectrumtable = polspectra.from_FITS(PolSpectra_postRM_name)
    else:
        print("Please enter valid suffix: 'pre' or 'post'")
        return -1


    #make empty component dict
    comp_dict = dict()
    parameters_dict = dict()
    spectra_dict = dict()


    #get parameters and spectra for current component
    if curr_comp != -1:
    
        #find full burst index in table
        tabidx_rmtable,ncomps_rmtable = dsarp_FRBinTable_RMTable(candname,comp_num=curr_comp)
        tabidx_pspectra,ncomps_pspectra = dsarp_FRBinTable_PolSpectrum(candname,comp_num=curr_comp,suff=suff)
        if tabidx_rmtable == -1 or tabidx_pspectra == -1:
            if tabidx_rmtable == -1:
                print("FRB not in RMTable")
            if tabidx_pspectra == -1:
                print("FRB not in PolSpectrum")
            return

        #get parameter dict
        parameters_dict["curr_weights"] = spectrumtable["weights"][tabidx_pspectra]
        parameters_dict["n_t"] = spectrumtable["n_t"][tabidx_pspectra]
        parameters_dict["n_f"] = spectrumtable["n_f"][tabidx_pspectra]
        parameters_dict["curr_comp"] = spectrumtable["PeakNum"][tabidx_pspectra]
        parameters_dict["ibox"] = spectrumtable["ibox_rev"][tabidx_pspectra]
        parameters_dict["tsamp"] = spectrumtable["dt"][tabidx_pspectra]
        parameters_dict["nickname"] = spectrumtable["nickname"][tabidx_pspectra]
        parameters_dict["ids"] = spectrumtable["candname"][tabidx_pspectra]
        parameters_dict["datadir"] = spectrumtable["datadir"][tabidx_pspectra]
        parameters_dict["RA"] = spectrumtable["dec"][tabidx_pspectra]
        parameters_dict["DEC"] = spectrumtable["ra"][tabidx_pspectra]
        parameters_dict["MJD"] = spectrumtable["mjd"][tabidx_pspectra]
        parameters_dict["frb_name"] = spectrumtable["candname"][tabidx_pspectra] + "_" + spectrumtable["nickname"][tabidx_pspectra]

        #get spectrum dict
        spectra_dict["I_f"] = ma.masked_array(spectrumtable["stokesI"][tabidx_pspectra],spectrumtable["stokesI_mask"][tabidx_pspectra]) 
        spectra_dict["Q_f"] = ma.masked_array(spectrumtable["stokesQ"][tabidx_pspectra],spectrumtable["stokesQ_mask"][tabidx_pspectra])
        spectra_dict["U_f"] = ma.masked_array(spectrumtable["stokesU"][tabidx_pspectra],spectrumtable["stokesU_mask"][tabidx_pspectra])
        spectra_dict["V_f"] = ma.masked_array(spectrumtable["stokesV"][tabidx_pspectra],spectrumtable["stokesV_mask"][tabidx_pspectra])
        spectra_dict["freq_test"] = [spectrumtable["freq"][tabidx_pspectra]]*4

    for comp_num in all_comp_nums:
        comp_dict[comp_num] = dict()

        #find full burst index in table
        tabidx_rmtable,ncomps_rmtable = dsarp_FRBinTable_RMTable(candname,comp_num=comp_num)
        tabidx_pspectra,ncomps_pspectra = dsarp_FRBinTable_PolSpectrum(candname,comp_num=comp_num,suff=suff)
        if tabidx_rmtable == -1 or tabidx_pspectra == -1:
            if tabidx_rmtable == -1:
                print("FRB not in RMTable")
            if tabidx_pspectra == -1:
                print("FRB not in PolSpectrum")
            return

        #set up full burst dict
        comp_dict[comp_num]["timestart"] = spectrumtable["timestart"][tabidx_pspectra]
        comp_dict[comp_num]["timestop"] = spectrumtable["timestop"][tabidx_pspectra]
        comp_dict[comp_num]["num_comps"] = spectrumtable["Npeaks"][tabidx_pspectra]
        comp_dict[comp_num]["buff"] = (spectrumtable["buffL"][tabidx_pspectra],spectrumtable["buffR"][tabidx_pspectra])
        comp_dict[comp_num]["weights"] = spectrumtable["weights"][tabidx_pspectra]
        comp_dict[comp_num]["multipeaks_all"] = spectrumtable["multipeaks"][tabidx_pspectra]
        comp_dict[comp_num]["sigflag"] = spectrumtable["sigflag"][tabidx_pspectra]
        comp_dict[comp_num]["rm_applied"] = spectrumtable["rm_applied"][tabidx_pspectra]
        comp_dict[comp_num]["intL"] = spectrumtable["intL"][tabidx_pspectra]
        comp_dict[comp_num]["intR"] = spectrumtable["intR"][tabidx_pspectra]
        comp_dict[comp_num]["n_t_weight"] = spectrumtable['n_tw'][tabidx_pspectra]
        comp_dict[comp_num]["sf_window_weights"] = spectrumtable['sf_window'][tabidx_pspectra]
        comp_dict[comp_num]["ibox"] = spectrumtable['ibox_rev'][tabidx_pspectra]
        comp_dict[comp_num]["mask_start"],comp_dict[comp_num]["mask_stop"] = spectrumtable['maskL'][tabidx_pspectra],spectrumtable['maskR'][tabidx_pspectra]
        comp_dict[comp_num]["multipeaks"] = spectrumtable['multipeaks'][tabidx_pspectra]
        if spectrumtable['multipeaks'][tabidx_pspectra]:
            comp_dict[comp_num]["height"] = spectrumtable['height'][tabidx_pspectra]
            comp_dict[comp_num]["scaled_height"] = spectrumtable['scaled_height'][tabidx_pspectra]


        comp_dict[comp_num]["RMsnrs1"] = spectrumtable["RMsnrs1"][tabidx_pspectra]            
        comp_dict[comp_num]["trial_RM"] = spectrumtable["trial_RM"][tabidx_pspectra]

        comp_dict[comp_num]["RMsnrs1tools"] = spectrumtable["RMsnrs1tools"][tabidx_pspectra]
        comp_dict[comp_num]["trial_RM_tools"] = spectrumtable["trial_RM_tools"][tabidx_pspectra]


        comp_dict[comp_num]["RMsnrs1zoom"] = spectrumtable["RMsnrs1zoom"][tabidx_pspectra]
        comp_dict[comp_num]["trial_RM2"] = spectrumtable["trial_RM2"][tabidx_pspectra]            
        
        comp_dict[comp_num]["RMsnrs1tools_zoom"] = spectrumtable["RMsnrs1tools_zoom"][tabidx_pspectra]
        comp_dict[comp_num]["trial_RM_tools_zoom"] = spectrumtable["trial_RM_tools_zoom"][tabidx_pspectra]
            
        comp_dict[comp_num]["RMsnrs2zoom"] = spectrumtable["RMsnrs2zoom"][tabidx_pspectra] 


        comp_dict[comp_num]["I_f"] = ma.masked_array(spectrumtable["stokesI"][tabidx_pspectra],spectrumtable["stokesI_mask"][tabidx_pspectra])
        comp_dict[comp_num]["Q_f"] = ma.masked_array(spectrumtable["stokesQ"][tabidx_pspectra],spectrumtable["stokesQ_mask"][tabidx_pspectra])
        comp_dict[comp_num]["U_f"] = ma.masked_array(spectrumtable["stokesU"][tabidx_pspectra],spectrumtable["stokesU_mask"][tabidx_pspectra])
        comp_dict[comp_num]["V_f"] = ma.masked_array(spectrumtable["stokesV"][tabidx_pspectra],spectrumtable["stokesV_mask"][tabidx_pspectra])

        comp_dict[comp_num]["I_f_init"] = ma.masked_array(spectrumtable["stokesI"][tabidx_pspectra],spectrumtable["stokesI_mask"][tabidx_pspectra])
        comp_dict[comp_num]["Q_f_init"] = ma.masked_array(spectrumtable["stokesQ"][tabidx_pspectra],spectrumtable["stokesQ_mask"][tabidx_pspectra])
        comp_dict[comp_num]["U_f_init"] = ma.masked_array(spectrumtable["stokesU"][tabidx_pspectra],spectrumtable["stokesU_mask"][tabidx_pspectra])
        comp_dict[comp_num]["V_f_init"] = ma.masked_array(spectrumtable["stokesV"][tabidx_pspectra],spectrumtable["stokesV_mask"][tabidx_pspectra])

        comp_dict[comp_num]["PA_f"] = ma.masked_array(spectrumtable["PA"][tabidx_pspectra],spectrumtable["PA_mask"][tabidx_pspectra])
        comp_dict[comp_num]["PA_f_errs"] = ma.masked_array(spectrumtable["PA_error"][tabidx_pspectra],spectrumtable["PA_error_mask"][tabidx_pspectra]) 

        comp_dict[comp_num]["PA_f_init"] = ma.masked_array(spectrumtable["PA"][tabidx_pspectra],spectrumtable["PA_mask"][tabidx_pspectra])
        comp_dict[comp_num]["PA_f_errs_init"] = ma.masked_array(spectrumtable["PA_error"][tabidx_pspectra],spectrumtable["PA_error_mask"][tabidx_pspectra])

        if suff == "pre":
            comp_dict[comp_num]["PA_pre"] = rmtable["polangle"][tabidx_rmtable]
            comp_dict[comp_num]["PAerr_pre"] = rmtable["polangle_err"][tabidx_rmtable]

            comp_dict[comp_num]["T/I_pre"] = rmtable["fracpol"][tabidx_rmtable]
            comp_dict[comp_num]["T/I_pre_err"] = rmtable["fracpol_err"][tabidx_rmtable]
            comp_dict[comp_num]["T/I_pre_snr"] = rmtable["Tsnr"][tabidx_rmtable]
            comp_dict[comp_num]["L/I_pre"] = rmtable["Lfracpol"][tabidx_rmtable]
            comp_dict[comp_num]["L/I_pre_err"] = rmtable["Lfracpol_err"][tabidx_rmtable]
            comp_dict[comp_num]["L/I_pre_snr"] = rmtable["Lsnr"][tabidx_rmtable]


        elif suff == "post":
            comp_dict[comp_num]["PA_post"] = rmtable["derot_polangle"][tabidx_rmtable]
            comp_dict[comp_num]["PAerr_post"] = rmtable["derot_polangle_err"][tabidx_rmtable]

            comp_dict[comp_num]["T/I_post"] = rmtable["fracpol"][tabidx_rmtable]
            comp_dict[comp_num]["T/I_post_err"] = rmtable["fracpol_err"][tabidx_rmtable]
            comp_dict[comp_num]["T/I_post_snr"] = rmtable["Tsnr"][tabidx_rmtable]
            comp_dict[comp_num]["L/I_post"] = rmtable["Lfracpol"][tabidx_rmtable]
            comp_dict[comp_num]["L/I_post_err"] = rmtable["Lfracpol_err"][tabidx_rmtable]
            comp_dict[comp_num]["L/I_post_snr"] = rmtable["Lsnr"][tabidx_rmtable]

        comp_dict[comp_num]["RM1"] = rmtable["RM1"][tabidx_rmtable]
        comp_dict[comp_num]["RMerr1"] = rmtable["RMerr1"][tabidx_rmtable]
        #comp_dict[comp_num]["RMsnrs1"] = rmtable["RMsnrs1"][tabidx_rmtable]
        #comp_dict[comp_num]["trial_RM"] = rmtable["trial_RM"][tabidx_rmtable]

        comp_dict[comp_num]["RM1tools"] = rmtable["RM1tools"][tabidx_rmtable]
        comp_dict[comp_num]["RMerr1tools"] = rmtable["RMerr1tools"][tabidx_rmtable]
        #comp_dict[comp_num]["RMsnrs1tools"] = rmtable["RMsnrs1tools"][tabidx_rmtable]
        #comp_dict[comp_num]["trial_RM_tools"] = rmtable["trial_RM_tools"][tabidx_rmtable]


        comp_dict[comp_num]["RM1zoom"] = rmtable["RM1zoom"][tabidx_rmtable]
        comp_dict[comp_num]["RMerr1zoom"] = rmtable["RMerr1zoom"][tabidx_rmtable]
        #comp_dict[comp_num]["RMsnrs1zoom"] = rmtable["RMsnrs1zoom"][tabidx_rmtable]
        #comp_dict[comp_num]["trial_RM2"] = rmtable["trial_RM2"][tabidx_rmtable]
        
        comp_dict[comp_num]["RM1tools_zoom"] = rmtable["RM1tools_zoom"][tabidx_rmtable]
        comp_dict[comp_num]["RMerr1tools_zoom"] = rmtable["RMerr1tools_zoom"][tabidx_rmtable]
        #comp_dict[comp_num]["RMsnrs1tools_zoom"] = rmtable["RMsnrs1tools_zoom"][tabidx_rmtable]
        #comp_dict[comp_num]["trial_RM_tools_zoom"] = rmtable["trial_RM_tools_zoom"][tabidx_rmtable]

        comp_dict[comp_num]["RM2zoom"] = rmtable["RM2zoom"][tabidx_rmtable]
        comp_dict[comp_num]["RM2errzoom"] = rmtable["RM2errzoom"][tabidx_rmtable]
        #comp_dict[comp_num]["RMsnrs2zoom"] = rmtable["RMsnrs2zoom"][tabidx_rmtable]
        comp_dict[comp_num]["RM_FWHM"] = rmtable["RM_FWHM"][tabidx_rmtable]


        comp_dict[comp_num]["sigflag"] = rmtable["sigflag"][tabidx_rmtable]
        comp_dict[comp_num]["RM_ion"] = rmtable["RM_ion"][tabidx_rmtable]
        comp_dict[comp_num]["RM_ionerr"] = rmtable["RM_ionerr"][tabidx_rmtable]
        comp_dict[comp_num]["RM_gal"] = rmtable["RM_gal"][tabidx_rmtable]
        comp_dict[comp_num]["RM_galerr"] = rmtable["RM_galerr"][tabidx_rmtable]

        comp_dict[comp_num]["absV/I"] = rmtable["absVfracpol"][tabidx_rmtable]
        comp_dict[comp_num]["absV/I_err"] = rmtable["absVfracpol_err"][tabidx_rmtable]
        comp_dict[comp_num]["V/I"] = rmtable["Vfracpol"][tabidx_rmtable]
        comp_dict[comp_num]["V/I_err"] = rmtable["Vfracpol_err"][tabidx_rmtable]
        comp_dict[comp_num]["V/I_snr"] = rmtable["Vsnr"][tabidx_rmtable]
        comp_dict[comp_num]["I_snr"] = rmtable["snr"][tabidx_rmtable]

        return comp_dict,parameters_dict,spectra_dict
