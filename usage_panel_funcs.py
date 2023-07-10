
dedisp_usage_str = """
    ### **Dedispersion Tab Usage**
    
    *Functionality*: The dedispersion tab is an interface for tuning to the ideal DM. The user can lead an FRB pre-dedispersed or at DM = 0, and after tuning the DM, can recompute the de-dispersed filterbanks from voltage files.

    *Procedure*

    * **Load Data (if initial filterbanks already computed)**: Click on the dropdown menu labelled 'Select FRB to Load' to choose the FRB filterbank set to load. The selected FRB's name is displayed in the 'Frb name' text box. Enter the DM that the current filterbank files are de-dispersed to in the 'Input DM (pc/cc)' box. Click 'Load FRB' to begin reading in the Stokes I filterbank data at native resolution (32.7 us, 30.5 kHz). Reading in data may take up to 10 minutes; to reduce latency, adjust the 'n_t' and 'log2(n_f)' sliders to downsample in time or frequency respectively. Downsampling at this stage will dictate the highest resolution that can be analyzed, and 'n_t' and 'log2(n_f)' will be reset to 1 and 0 respectively after 'Load FRB' is clicked. The FRB time series in a window of 5 ms around the peak will be displayed in the top plot, and the Stokes I dynamic spectrum is displayed on the bottom.

    * **Load Data (if no initial filterbanks computed)**: If no filterbank files have been generated, but voltage files exist for an FRB, the *toolkit* script can be called directly from the dedipsersion panel. First, enter the FRB name in the format "Candname_Nickname" (e.g. "220121aat_clare") directly in the textbox labelled 'Frb name'. Click the 'Select Beamformer Weights' dropdown menu and select the date of the beamformer calibration file to be used (in isot format). Use the 'ibeam' slider to enter the detected beam of the FRB, and enter the MJD in the text boxt labelled 'mjd'. If desired, enter a DM in the box labelled 'Input DM (pc/cc)' to dedisperse to an initial DM. Click 'Load FRB' to begin running the *toolkit* script and subsequently load the FRB at the selected time and frequency resolution. The FRB time series in a window of 5 ms around the peak will be displayed in the top plot, and the Stokes I dynamic spectrum is displayed on the bottom.

    * **Tune to Ideal DM**: The relative DM (desired change in DM relative to the input DM) can be entered into the box labelled 'Relative DM (pc/cc)'. This value can be positive or negative. The final DM (relative DM added to the input DM) is displayed in the box labelled 'Final DM (pc/cc)', while the additional DM delay across the full bandwidth is given in 'Relative DM Delay Across Band (ms)'. Click 'Apply' to incoherently dedisperse the displayed dynamic spectrum and time series to the Final DM. 

    * **Generate Dedispersed Filterbanks**: Once the desired DM has been achieved, click 'Re-Compute Filterbanks' to call the *toolkit* script to generate updated filterbanks coherently dedispersed to the final DM. The updated filterbanks will overwrite the current filterbanks, the Input DM will be updated to the Final DM, and the relative DM will be reset to 0 so that one can repeat the tuning process if necessary.

    * **Other Parameters**:
        * ***n_t***: factor by which time series and dynamic spectrum on display are downsampled in time
        * ***log2(n_f)***: log base 2 of the factor by which dynamic spectrum on display is downsampled in frequency
        * ***window***: range around peak sample displayed in the time series and dynamic spectrum in milliseconds
        * ***saturation***: maximum of colorscale in displayed dynamic spectrum as percentile

    

    """

pol_usage_str = """
    ### **Polarization Tab Usage**

    *Functionality*: The polarization tab is an interface for computing polarization fractions, position angles, polarized spectra, and polarized time series for FRBs. The user can calibrate data from full Stokes filterbanks using previously generated Jones matrix solutions. This analysis can be conducted concurrently with RM analysis by following the instructions in the **RM Tab Usage** section of this guide.

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
        
        Once weights have been tuned as desired, click 'Next'; the ideally weighted spectrum and PA for the first component will be displayed in the plot labelled 'Component #1', and its polarization fractions, signal-to-noise ratios, and average PA will be displayed in their respective output boxes on the left-hand side. If no RM analysis is desired, repeat the process for the remaining components. Otherwise, go to the section marked **RM Tab Usage** to begin RM analysis.


    * **Downsample in Frequency**: Once all burst components have been processed, click 'Done'. One can repeat the **RM Analysis** step described below for the full burst using concatenated weights. Once all RM analysis is complete. The frequency spectra can be downsampled to maximized structure by adjusting the 'log2(n_f)' slider, defined as the log base 2 of the factor 'n_f' by which the frequency axis is downsampled (e.g. log2(n_f) = 0 gives channel bandwidth 30.5 kHz, log2(n_f) = 1 gives channel bandwidth 30.5x2 = 61.0 kHz). Click 'Done' when complete.

    *Exporting Plots*

    * After completing the **Downsample in Frequency** step, summary plots of the Stokes I dynamic spectrum, time dependent polarization, and polarization spectrum can be output by clicking 'Export Summary Plot'. This will output individual plots for each component, as well as a plot for the full burst. 

    *Saving Filterbank Files*

    * **Polarization Calibrated Filterbanks**: After loading FRB data and polarization calibrating, click 'Save Calibrated Filterbanks' to save filterbanks at the **current resolution**. Calibrated filterbanks will be labelled "polcal" and be output in the same directory as the uncalibrated filterbanks.

    * **RM Calibrated Filterbanks**: After loading FRB data and RM calibrating **the full burst**, click 'Save RM Calibrated Filterbanks' to save filterbanks at he **current resolution**. RM calibrated filterbanks will be derotated to the RM derived for the full burst (i.e. not individual components), labelled "RMcal" and be output to the same directory as the uncalibrated filterbanks. RM calibrated filterbanks can be saved with or without polarization calibrating first, though it is recommended to polarization calibrate prior to running RM synthesis.

    *Saving to DSA-110 RMTable and PolSpectra Databases*

    * van Eck, et al. 2023 defines a standard table format for storing and sharing polarization and RM data. After all analysis (including RM analysis) is complete, an FRB can save to the DSA-110 database defined in this format by clicking "Update DSA-110 Catalog". The DSA110 catalogs are stored in 3 separate files located in the directory h23:/media/ubuntu/ssd/sherman/code/RM_tmp_files/:
        * **DSA110_PolTable_PreRMcal.fits**  
        * **DSA110_PolTable_PostRMcal.fits**  
        * **DSA110_RMTable.fits**
    * Data will be saved to each file depending on the processing stage at which "Update DSA-110 Catalog" is clicked. If clicked after computing the initial non-RM-calibrated polarization for a given component (or full burst), the weight parameters and spectra will be saved to **DSA110_RMTable.fits** and **DSA110_PolTable_PreRMcal.fits** respectively. If the data is RM-calibrated as described below, the spectra will instead be saved to **DSA110_PolTable_PostRMcal.fits** . 
    * To RM calibrate using previously computed rotation measures stored in the DSA-110 catalog, click "Load RM from DSA-110 Catalog" after computing the polarization instead of following the RM analysis steps. 
    * Further details on this format and accessing the data in these catalogs can be found at https://github.com/CIRADA-Tools/RMTable and https://github.com/CIRADA-Tools/PolSpectra .
    
    """

RM_usage_str = """
    ### **RM Tab Usage**

    *Functionality*: The RM tab is an interface for computing rotation measures, as well as galactic and ionospheric contributions to RM. RMs can be calculated for each component of the burst and applied independently, as well as for the full burst, and applied to derotate on the polarization tab.
   
    *Pre-Analysis*: The RM tab is developed to work in tandem with the polarization tab, and therefore an FRB must first be loaded by following the procedure laid out above in the *Load Data* section. After choosing components and computing the polarization of the first component, click "Update DSA-110 Catalog" to save the currently chosen component's data to the DSA catalog described above. The RM Tab will pull data from these catalogs for analysis. Then switch to the RM tab and follow the procedure below to carry out RM analysis.

    *Procedure*

    * **Initialize Data**: Enter the candidate name (e.g. 20230120aaab) for the FRB being analyzed in the box labelled "FRB candname". Enter the number of the desired component (starting at 0 for the first component) in the box labelled "Component Number". Click the *Initialize Data* button to load FRB data. To analyze the full burst rather than a single component, check the box labelled "Fullburst bool" and enter -1 as the "Component Number"

    * **Run Initial RM Synthesis**: Initial RM Synthesis estimates the RM following the procedure in Brentjens, et al. 2005 on the time averaged 1D spectrum of the chosen FRB. The following parameters are defined in the *Initial RM Synthesis Settings* section of the second column and can be adjusted to tailor the experiment to a specific RM range and resolution:
        * ***Minimum Trial RM (rad/m^2)***: sets lower limit on the range of RM trials 
        * ***Maximum Trial RM (rad/m^2)***: sets upper limit on the range of RM trials
        * ***Number of Trial RMs***: sets the number of RM trials within the specified range
        After setting these parameters, click *Run* to begin synthesis. RM synthesis will be conducted using the following two methods:
        * ***Custom RM Synthesis***: This is a manually implemented 1D RM synthesis algorithm which does not implement RM cleaning, but may use a higher resolution and arbitrarily centered range (i.e. the range of RM trials does not need to be centered on 0). The RM from this method will be displayed in the box labelled *Initial RM (rad/m^2)* with its error in the box below. The RM spectrum will be displayed in black in the upper plot.
        * ***RM-Tools (RM-Clean)***: This runs the RM synthesis algorithm in the *RM-Tools* library (Purcell, et. al. 2020), which uses RM Clean to lessen the affect of sidelobes. More information about this method can be found at https://github.com/CIRADA-Tools/RM-Tools. The RM from this method will be displayed in the box labelled *Initial RM-Tools RM (rad/m^2)* with its error in the box below. The RM spectrum will be displayed in blue in the upper plot. Note that the trial RMs for RM-Tools must be centered on 0 and the specified range will be clipped as necessary to meet this requirement.

    * **Run Fine RM Synthesis**: Fine RM synthesis further constrains the RM by conducting RM synthesis on a narrowed range of RM trials around the result estimated from Custom RM Synthesis in the previous step. The following parameters are defined in the *Fine RM Synthesis Settings* section of the second column and can be adjusted to tailor the experiment to a specific RM range and resolution:
        * ***Number of Trial RMs (Fine)***: sets the number of RM trials within the narrowed range
        * ***RM range above/below initial result (rad/m^2)***: sets the RM window around the initial estimate that will be 'zoomed in' on to do Fine Synthesis
        After setting these parameters, click *Run* to begin fine synthesis. This will be conducted using the following three methods:
        * ***Custom RM Synthesis***: Follows the same procedure as above on the narrowed RM trial region. The resulting RM will be displayed in the box labelled *Fine RM Synthesis RM (rad/m^2)* with its error in the box below. The RM spectrum will be displayed in blue in the bottom plot.
        * ***RM-Tools (RM-Clean)***: Follows the same procedure as above on the narrowed RM trial region clipped such that it is centered on 0. *Note that due to memory constraints within RM-Tools, Fine RM-Tools will be skipped if the initial RM estimate is greater than 10^4 rad/m^2*. The resulting RM will be displayed in the box labelled *Fine RM-Tools RM (rad/m^2)* with its error in the box below. The RM spectrum will be displayed in black in the bottom plot if computed.
        * ***S/N Method***: This is a manually implemented 2D RM synthesis algorithm which does not implement RM cleaning. This conducts the standard 1D RM synthesis at each time sample withing the FRB and maximizes the linear signal-to-noise (S/N) to estimate the RM. The resulting RM will be displayed in the box labelled *Fine S/N Method RM (rad/m^2)* with its error below. The RM spectrum displayed in gold in the bottom plot.


    * **Save and Apply RM**: To save the RM computed for the current component to the DSA-110 RMTable catalog, click *Update DSA-110 Catalog*. To derotate the FRB component the derived RM, return to the Polarization Tab and click *Load RM from DSA-110 Catalog*, then click *RM Calibrate*, which will derotate the spectrum of the current component and recompute polarization, S/N, and PA for the component. Make sure that the RM is applied before moving to the next component if it is considered significant.

    * **Repeat for Other Components and Full Burst**: The process above can be repeated after moving to the next component. After all components have been analyzed and *Done* has been clicked, RM synthesis can be performed for the full burst by repeating the steps above. Clicking *RM Calibrate* at this stage will apply the full RM to both the full burst spectrum and time series.

    * **Galactic RM Contribution**: Estimation of the Galactic contribution to RM along the FRB's line of sight is estimated using the Hutschenreuter, et al. 2020 galactic RM map. On the polarization tab, enter the FRB's RA and Declination in the boxes labelled *RA* and *DEC*. Click *Proceed to RM Synthesis* to cache these values, then switch to the RM tab. Click *Compute Galactic RM* in the second column, and the resulting galactic RM will be displayed in the box labelled *Galactic RM (rad/m^2)* with its error in the box below. Click *Return to Pol Analysis* to cache the data, and on the Polarization Tab, click *Retrieve RM Data* to save to the current dictionary (this will ensure that the galactic RM will be saved to any exported files).

    * **Ionospheric RM Contribution**: Estimation of the Ionospheric contribution to RM along the FRB's line of sight at the observation epoch is estimated using **ionFR**, described in Sotomayor-Beltran, et al., 2013 (further information can be found at https://github.com/csobey/ionFR). On the polarization tab, enter the FRB's RA and Declination in the boxes labelled *RA* and *DEC*. Enter the Mean Julian Date (MJD) in the box labelled *MJD*. Click *Proceed to RM Synthesis* to cache these values, then switch to the RM tab. Click *Compute Ionospheric RM* in the second column; the software requires files pulled from the NASA CDDIS Database (cddis.nasa.gov/archive/gps/products/ionex). If the necessary file is already downloaded, the ionospheric RM will be calculated and displayed in the box labelled *Ionospheric RM (rad/m^2)* with its error in the box below. If not, follow the link given in the error output box to download the necessary file, unzip it, and move it to the directory listed in the error output box. If following the link brings you to a page that asks for a password instead of downloading the file, you can find the username and password in the file: h23:/media/ubuntu/ssd/sherman/code/.netrc. (*Contact Myles Sherman at msherman@caltech.edu if further assistance is needed*). After the downloaded and unzipped file has been moved to the correct directory, click *Compute Ionospheric RM* again and the ionospheric RM will be calculated and displayed in the box labelled *Ionospheric RM (rad/m^2)* with its error in the box below. Click *Return to Pol Analysis* to cache the data, and on the Polarization Tab, click *Retrieve RM Data* to save to the current dictionary (this will ensure that the ionospheric RM will be saved to any exported files).

    *Exporting Plots*

    * After computing the initial and fine RM, click *Export Summary Plot* in the second column of the RM Tab to export a pdf plot containing the RM spectra currently displayed.
    """


burstfit_usage_str = """
    ### **Burstfit Tab Usage**
    
    *Insert burstfit tab usage here*
    """
