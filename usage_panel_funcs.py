
dedisp_usage_str = """
    ### **Dedispersion Tab Usage**
    
    *Insert dedispersion tab usage here*
    """

pol_usage_str = """
    ### **Polarization Tab Usage**

    *Functionality*: The polarization tab is an interface for computing polarization fractions, position angles, and rotation measures. The user can calibrate data from full Stokes filterbanks using previously generated Jones matrix solutions. RMs can be calculated for each component of the burst and applied independently, as well as for the full burst.

    *Procedure*
    
    * **Load Data**: Click on the dropdown menu labelled 'Select FRB to Load' to choose the FRB filterbank set to load. The selected FRB's name is displayed in the 'Frb name' text box. Click 'Load FRB' to begin reading in the I,Q,U, and V filterbank data at native resolution (32.7 us, 30.5 kHz). Reading in data may take up to 10 minutes; to reduce latency, adjust the 'n_t' and 'log2(n_f)' sliders to downsample in time or frequency respectively. Downsampling at this stage will dictate the highest resolution that can be analyzed, and 'n_t' and 'log2(n_f)' will be reset to 1 and 0 respectively after 'Load FRB' is clicked. The FRB time series in a window of 5 ms around the peak will be displayed in the top plot.
    
   
    * **Calibrate Data**: Click on the dropdown menu labelled 'Select Calibration File to Load' to choose the Jones matrix calibration solution to apply. Each calibration file is labelled with the date (yy-mm-dd) it was generated; in general the most recent calibration set should be used. The 'ibeam' slider is used to select the beam the FRB was detected in. The RA and declination of the beam can be entered in the corresponding textboxes in degrees; these will be used for parallactic angle calibration. Click 'Calibrate'; the time series displayed will update with the polarization calibrated data.

    * **Downsample in Time**: The time series can be downsampled to maximize structure by adjusting the 'n_t' slider. 'n_t' corresponds to the factor by which the time axis is downsampled (e.g. n_t = 1 gives sampling rate 32.7 us, n_t = 2 gives sampling rate 32.7x2=65.4 us).

    * **Select Burst Components**: The dashboard allows the user to analyze individual components of the FRB sepearately. Click the 'Get Components' button located midway down the panel. A horizontal section of the time series will be highlighted in red. Use the 'lower bound' slider to shift the box to the location of the first burst component, and the 'mask width' slider to adjust its width to enclose the first burst component. This defines the mask used to cut out the burst for analysis of other bursts. When the desired component is highlighted click 'Next'; the component will be highlighted in green and the process can be repeated for any additional components. When all components have been selected, click 'Done'.

    * **Tune Ideal Filter Weights**: Ideal filter weights are downsampled and smoothed copies of the intensity time series which will be used to compute the burst spectrum, weighted polarization fraction, and weighted PA. Filter weights are computed for each burst component then concatenated together. Weights are plotted in purple on the time series plot. The following sliders and options can be used to tune filter weights:
        * ***left/right buffer***: number of samples to pad the burst window, outside of which filter weights will be zeroed.
        * ***log(n_tw)***: log base 2 of the factor by which time series is downsampled for smoothing
        * ***sf window***: width of the 3rd order Savgol filter used to smooth ideal filter weights after downsampling and reinterpolation. Note because the filter is 3rd order, the minimum window is 5; selecting 1 or 3 will result in non-filtered weights
        * ***ibox***: base width of the burst component in 256 us samples prior to addition of left/right buffers; in general this corresponds to the 'ibox' parameter reported by Heimdall, but can be adjusted as needed here. 
        * ***Multipeaks***: select this if the burst component has multiple peaks. Red dashed vertical lines indicate the Full-Width at Half-Max (FWHM) of the burst component. If the component has multiple peaks, selecting 'Multipeaks' allows the left and right bounds to be set at the leftmost and rightmost peaks. Use the 'height' slider to move the horizontal dashed line to give the minimum height of the peak. **The polarization fraction will be computed as an average within the FWHM indicated by the chosen bounds. The PA will also only be displayed in this range, and its average value computed in this range**
        
        Once weights have been tuned as desired, click 'Next'; the ideally weighted spectrum and PA for the first component will be displayed in the plot labelled 'Component #1', and its polarization fractions, signal-to-noise ratios, and average PA will be displayed in their respective output boxes on the left-hand side. If no RM analysis is desired, repeat the process for the remaining components.

    * **RM Analysis**: For a given burst component, the RM can be calculated by clicking the 'Proceed to RM Analysis' button in the top middle of the panel (this should be done *after* clicking 'Next'). The RM analysis panel computes the RM first on a coarse grid (Initial RM), then on a fine grid (Fine RM) of RM Trials. 
        * The initial RM is computed and displayed by clicking 'Run'; this computes both Manual 1D RM synthesis and RM-Tools with RM Cleaning (https://github.com/CIRADA-Tools/RM-Tools/tree/master), displaying the results in the 'Initial RM' and 'Initial RM-Tools RM' boxes respectively. The coarse grid of RM trials is tuned by entering the 'Minimum Trial RM', 'Maximum Trial RM', and 'Number of Trial RMs' in their respective boxes (the trial RM axis is computed as np.linspace('Minimum Trial RM', 'Maximum Trial RM', 'Number of Trial RMs'). RM spectra from each method are plotted in the upper RM plot.
        * The fin RM is computed and displayed by clicking 'Run' again; this uses Manual 1D RM synthesis, RM-Tools with RM Cleaning, and RM S/N Method to estimate the RM (see Sherman, et al. 2023 for details), displaying the results in the 'Fine RM Synthesis RM', 'Fine RM-Tools RM' and Fine S/N Method RM' boxes respectively. The fine grid of RM trials is tuned by entering the 'Number of Trial RMs (Fine)' and 'RM range above/below initial result' in their respective boxes (the trial RM axis is computed as np.linspace('Initial RM'-'RM range above/below initial result','Initial RM'+'RM range above/below initial result','Number of Trial RMs (Fine)')
        * Click 'Return to Pol Analysis' when complete. To RM calibrate using the final RM result ('Fine S/N Method RM'), click 'RM Calibrate' and the spectrum, polarization fractions, signal-to-noise, and PA will be recomputed and displayed. Repeat the **Tune Ideal Filter Weights** and **RM Analysis** steps for each component in turn. Note the polarization, PA, S/N, RMs, and errors will be displayed in their respective boxes in the format **('first component', 'second component',...,'last component') 'full burst'**

    * **Downsample in Frequency**: Once all burst components have been processed, click 'Done'. One can repeat the **Rm Analysis** step described above for the full burst using concatenated weights. Once all RM analysis is complete. The frequency spectra can be downsampled to maximized structure by adjusting the 'log2(n_f)' slider, defined as the log base 2 of the factor 'n_f' by which the frequency axis is downsampled (e.g. log2(n_f) = 0 gives channel bandwidth 30.5 kHz, log2(n_f) = 1 gives channel bandwidth 30.5x2 = 61.0 kHz). Click 'Done' when complete.

    *Exporting Plots*
    * After completing the **Downsample in Frequency** step, summary plots of the Stokes I dynamic spectrum, time dependent polarization, and polarization spectrum can be output by clicking 'Export Summary Plot'. This will output individual plots for each component, as well as a plot for the full burst. 

    """

burstfit_usage_str = """
    ### **Burstfit Tab Usage**
    
    *Insert burstfit tab usage here*
    """
