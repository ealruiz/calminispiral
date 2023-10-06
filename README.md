# calminispiral
Script to calibrate the SgrA* minispiral with CASA (the Common Astronomy Software Applications) software.

This document presents an approach for the intra-field Atacama Large Millimeter/submillimeter Array (ALMA) calibration and imaging of the radio source SgrA*, located at the center of our Milky Way galaxy.

This approach utilizes the extended structure of the source (the minispiral) to calibrate the flux variability of the compact core. The algorithm involves several steps: 
1. an initial CLEAN image is generated for the entire source;
2. the core is subtracted, leaving only the minispiral;
3. a two-component visibility model is constructed, comprising the minispiral and the core;
4. the model is fitted to the data, retrieving flux density parameters for each integration time;
5. the data are scaled and calibrated, resulting in nearly constant brightness for the minispiral and variable flux for the core.

The proposed approach offers a self-calibration method, using the extended structure to correct the amplitude gain calibration and produce accurate light curves for both SgrA* and the minispiral.

## Configuration of the script

The code begins by setting up various configuration parameters needed for each step. Here we summarize the relevant parameters for each step:

* Step 1:
  * **MSNAME**: Name of the measurement set file.
  * **IMSIZE**: Image size (number of pixels) for the CLEAN process.
  * **Ns**: Nyquist sampling parameter.
  * **Bmax**: The greatest projected baseline length, representing the highest resolution.
  * Cell: Cell size for imaging, calculated based on Nyquist sampling and Bmax.
* Step 2:
  * **REMOVE\_CENTER**: Boolean flag indicating whether to remove SgrA* from the extended model (recommended to be *True* to study the minispiral of SgrA*.
  * **REMOVE\_ALL\_CENTER\_BEAM**: Boolean flag indicating whether to set the central pixel to zero or the average in-beam extended brightness.
* Step 3:
  * **MINBAS**: Minimum baseline length in meters for flagging short baselines (classical approach).
  * **RELOAD\_MINISPIRAL\_MODEL**: Boolean flag indicating whether to reload the model of the extended component (recommended to be *True*).
  * **USE\_SELFCAL\_DATA**: Boolean flag indicating whether to use self-calibrated data; only *True* if repeating this step, *after step 5*.
* Step 4:
  * **SGRA\_MIN**, **SGRA\_MAX**: Minimum and maximum allowed SgrA* flux densities for flagging bad integrations.
  * **EXPORT\_MINISPIRAL**: Boolean flag indicating whether to export the calibrated minispiral visibilities, scaled to the flux-density average; needed if step 5 will be run.
* Other variables:
  * *mysteps*: List of steps to be executed; **required: steps 0-4**; optional: step 5 and repeat steps 3-4.
  * *thesteps*: Dictionary mapping step numbers to their corresponding descriptions.

## Data products

After executing all steps, we retrieve the calibrated light curves on different files. Here's a brief description:
* **'*.fit' file**: The **FIT_ARRAY**, obtained after the visibility model-fitting, is stored into a '*.fit' file with the following info., for each spectral window (spw): 'JDTime', 'I Extended', 'I Compact', 'I LongBas', 'Q', 'U', 'V', 'Error Ext.', 'Error Comp.', 'Covariance', 'Error Q', 'Error U', 'Error V'.
  * *Warning*: the stokes parameters are not scaled by the gains (i.e. AVERAGE/I_extended at each integration time, for each spw)
* **'Light_Curve_SPWi_*.dat'**: The calibrated light curves (scaled by gains) are saved in an ascii with the following info.: MJD, I(Jy), Ierr(Jy), P(Jy), Perr(Jy), EVPA(deg), EVPAerr(deg), V(Jy), Verr(Jy).
