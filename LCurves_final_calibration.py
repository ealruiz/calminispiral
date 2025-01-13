import glob, os, gc
import pickle as pk
import re
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import math as m
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
PATH0 = os.getcwd()

## LATEX FONTS:
if True:
  plt.rcParams['text.latex.preamble']= r"\usepackage{lmodern}"
  params = {'text.usetex' : True,
				'font.size' : 21,
				'font.family' : 'lmodern'
          }
  plt.rcParams.update(params)

"""
IMPORTANT: this code performs the LCurve analysis from data calibrated using the minispiral calibration method (see github), and as a result, the fitname read in this script follows the naming and structure used in that script. If you used that script, change the MSNAME to the name used there. If not, consider saving your Light curves following the same nomenclature, or edit this script to fit your needs.
"""

SOURCE = 'SGRA'
DATNAM = '' # common name of the ".ms" files
TRACKS = ['']
spw = [0,1,2,3] # ALMA B6 spectral windows
spw_frec=[213100,215100,227100,229100] # Frequency in MHz
Nspw = len(spw)  # number of spw
c = 299792458*10**9 # light speed in nm/s
spw_wl = [(c/(spw_frec[k]*10**6))**2 for k in range(Nspw)] # wavelenght in nm**2

FLAG_OUTLIERS = False #  True to flag specific outliers AFTER a manual inspection of the Stokes I light curves. The flag info should be stored in a "SGRA_flag_info_TRACK_spwN.dat" file (which must be in the main folder, together with the .fit files). Structure of the .dat: two columns: [time1,time2]		[min_flux,max_flux]
AUTO_FLAGGING = False # True to perform a flagging using a spline of the Stokes I light curves.
SAVE_OUTLIERS = False # True to save the identified outliers in the GOOD column. IMPORTANT: ONLY DO THIS IF YOU ARE SURE OF THE MANUAL OUTLIERS. AFTER FLAGGING MANUAL OUTLIERS, SET FLAG_OUTLIERS = False (i.e. ONLY DO THE MANUAL FLAGGING ONCE!)

### params for automatic flagging of outliers
### TODO: possible improvement (it may not work, this is just to remember the idea)
'''
	The High-Pass filter periodogram has a better spline fit to signal with gaps. Instead of the worse approach followed in this script, it would be better to use the spline curve implemented in the C++ code, and then flag outliers.
	Therefore, TIME_GAPS will not be needed, after implementing this changes.
'''



"""
How to flag the outliers?

0. First run the script without Flagging outliers.
	Set conservative maximum and minimum values of Stokes I,
		SGRA_MAXI = []
		SGRA_MINI = []
	and maximum value of Polarized Intensity,
		SGRA_MAXP = []
	for each track. This can be updated at the different steps.

1. If there are isolated data points (which can appear during the QA2+minispiral calibration)
	1.1. fill a "SGRA_flag_info_TRACK_spwN.dat" file (which must be in the main folder, together with the .fit files),
   ### Structure of the .dat: two columns: [time1,time2]		[min_flux,max_flux]
	1.2. set
		FLAG_OUTLIERS = True
		AUTO_FLAGGING = False
		SAVE_OUTLIERS = False
	to flag all data points not corresponding with our LCs.

2. Fine-tune the params for automatic flagging, i.e.
		TIME_GAPS = []
		FLAG_FACT = []
		FLAG_SCAN_FACT = []
	2.1. During the testing of the parameters, set
		FLAG_OUTLIERS = False
		AUTO_FLAGGING = True
		SAVE_OUTLIERS = False
	2.2. Once the parameters properly flag the significant outliers, save this flagging by setting:
		FLAG_OUTLIERS = False
		AUTO_FLAGGING = True
		SAVE_OUTLIERS = True

3. Finally, if there is a scan which has to be manually flagged, or some minor outliers which could be flagged:
	3.1. inspect the Light Curves of Stokes I from the .dat files created,
	3.2. fill the "SGRA_flag_info_TRACK_spwN.dat" file with the new outliers,
	3.3. set
		FLAG_OUTLIERS = True
		AUTO_FLAGGING = False
		SAVE_OUTLIERS = True
			and run the script to obtain the LCs without any outliers.
"""

### set duration of gaps in the signal (depends on the sampling). If the signal has more noise, then this can be larger to include more data and increase the SNR
TIME_GAPS = [] # in hours, approximate threshold for gaps duration; one different value per track, to optimize
### set multiplier to the average of the noise. Flag data if the noise at a given time is greater than FLAG_FACT times the noise avg.; recommended to begin testing: 5. The higher the value, the more data points will be flagged
FLAG_FACT = [] # one different value per track, to optimize
### set maximum percentage of data points vs. noise. If the noise is greater, then will flag the whole scan; recommended to begin testing: 0.5; if you want to MANUALLY select the scans, set to 0.1 (only the worst scans will be flagged, not common)
FLAG_SCAN_FACT = [] # one different value per track, to optimize
# Minimum and maximum allowed SgrA* flux densities, to flag bad integrations:
SGRA_MINI = [] # one different value per track
SGRA_MAXI = [] # one different value per track
SGRA_MAXP = [] # one different value per track

flag_path = os.path.join(PATH0,'SGRA_STKI_FLAGGED') # path to save inspection plots
if not os.path.exists(flag_path):
	os.makedirs(flag_path)	

SPLINE_FIT = False # True for spline fit, False for polyfit (to correct spw2)
S_FIT = 11. # Degree of the polyfit (or smootheness of the spline)
SplineOrder, Nknots = 5, 7 # params for spline fit

##########################################################
# Function to extract arrays from a string
def extract_arrays(line):
    arrays = re.findall(r'\[([\d.,]+)\]', line)
    result_array = [list(map(float, array.split(','))) for array in arrays]
    return result_array[0]

def find_scan_boundaries(times, gap_threshold):
	scan_boundaries = []
	current_start = times[0]
	for i in range(1, len(times)):
		time_diff = times[i] - times[i-1]
		if time_diff > gap_threshold:
			scan_boundaries.append((current_start, times[i-1]))
			current_start = times[i]
	scan_boundaries.append((current_start, times[-1])) # Add the last scan boundary
	return scan_boundaries

def fit_spline(times, fluxes, scan_boundaries, spline_order, nknots):
	fitted_fluxes = []
	for start, end in scan_boundaries:
		mask = (times >= start) & (times <= end)
		spline = UnivariateSpline(times[mask], fluxes[mask], k=spline_order, s=nknots)
		fitted_fluxes.extend(spline(times[mask]))
	return fitted_fluxes

def find_time_gaps(times,trackgaps=2):
	# Initialize variables to store the gaps
	gap_start_times = []
	gap_end_times = []

	# Iterate through the array to find gaps
	for i in range(1, len(times)):
		time_difference = times[i] - times[i-1]

		# Check if the gap is greater than trackgaps param
		if time_difference > trackgaps:
			gap_start_times.append(times[i-1]+1./3.) # start 20 min. after the start of the gap
			gap_end_times.append(times[i]-1./3.) # end 20 min. before the end of the gap

	return gap_start_times, gap_end_times

def flag_LCurve_outliers(Times, LCurve, fname, SplineOrder,Nknots,TimeGap, FLAG_NOISE_FACT = 5., FLAG_SCAN_FACT = 0.75):
	'''
		Denoise the light curve and compute the Structure Funtion
		Params:
			- fname: name of the ascii file with the Light Curve data
			- SplineOrder: order of the spline or polynomical fit to denoise the signal
			- Nknots: number of points to fill the spline light curve
			- TimeGap: duration of gaps in the signal
			- FLAG_NOISE_FACT: multiplier to the average of the noise. Flag data if the noise at a given time is greater than FLAG_NOISE_FACT times the noise avg
			- FLAG_SCAN_FACT: maximum percentage of data points vs. noise. If the noise is greater, then will flag the whole scan  
		Returns:
			- outliers: mask with the identified outliers to flag
	'''
	
	os.chdir(flag_path)
	
	scan_boundaries = find_scan_boundaries(Times,TimeGap)
	fitted_fluxes = fit_spline(Times, LCurve, scan_boundaries, SplineOrder, Nknots)
	
	Flux_diff = np.abs(LCurve - fitted_fluxes)
	Times_flagged = []
	LCurve_flagged = []
	outliers = []
	for start, end in scan_boundaries:
		time_mask = (Times >= start) & (Times <= end)
		data_mask = np.logical_and(Flux_diff < FLAG_NOISE_FACT*np.average(Flux_diff), time_mask)
		if data_mask.sum() > FLAG_SCAN_FACT*time_mask.sum():
			Times_flagged.extend(Times[data_mask])
			LCurve_flagged.extend(LCurve[data_mask])
			outliers.extend(np.logical_not(data_mask[time_mask])) ### Get outlier indx
		else:
			outliers.extend(np.full(len(time_mask[time_mask]),True)) ### Get outlier indx
	
	fig = pl.figure(figsize=(10,7))
	sub1 = fig.add_subplot(111)
	fig.subplots_adjust(wspace=0.01,hspace=0.01,right=0.98,left=0.125)
	fig.suptitle('	SGRA %s spw%s'%(track,i),fontsize=21)
	sub1.plot(Times_flagged,LCurve_flagged,'.k',label='Flagged Data')
	sub1.plot(Times[outliers],LCurve[outliers],'xr',label='Data Outliers')
#	sub1.plot(Times,fitted_fluxes,'.b',label='Spline')
	pl.sca(sub1)
	pl.legend(numpoints=1)
	sub1.set_xlabel('JDTime (h)')
	sub1.set_ylabel('Stk I (Jy)')
	pl.savefig('%s_spline_fit.png'%fname)
	pl.close()
	
#	data = np.column_stack((Times_flagged,LCurve_flagged))
#	np.savetxt('%s_spline_fit.dat'%fname, data, delimiter='\t', header='#\t JDTIME (h) \t LCurve (Jy)', comments='')
	gc.collect()
	os.chdir(PATH0)
	
	return outliers

### Function to compute the Spectral Index
def SpectralIndex(atimes,aflux,spw_frec,time_indx,fname='spectral_index.dat'):
	# atimes: array of times where none of the spws have outliers
	# aflux: fluxes measured for each spw, and each time
	# time_indx: time indexes where none of the spws have outliers
	# spw_frec: frequency of each spw
	SpectIndex = []
	SpectIndex_err = []
	
	with open(fname, "w+") as fpOut:
		for tindx,time in enumerate(atimes):
			# Perform linear regression using scipy function linregress
			spwfreq, yflux = [], []
			#for k in range(len(spw_frec)):
			for k in [0,3]:
				spw_time_indx = time_indx[k][tindx]
				spwfreq.append(spw_frec[k])
				yflux.append(aflux[k][spw_time_indx]) # Get flux values for the current time
			alpha, intercept, r_value, p_value, alpha_err = linregress(np.log10(spwfreq), np.log10(yflux))
			SpectIndex.append(alpha)
			SpectIndex_err.append(alpha_err)
			# write in the .dat file
			fpOut.write("{:.18e}   {:.7e}  {:.7e}\n".format(time,alpha,alpha_err))
	return SpectIndex,SpectIndex_err

### Function to compute the Depolarization Measure
def DepolarizationMeasure(atimes,polI,polI_err,time_indx, fname='depolarization_measure.dat'):
	DepolMeasure = []
	DepolMeasure_err = []
	with open(fname, "w+") as fpOut:
		for tindx,time in enumerate(atimes):
			spwi_time_indx = time_indx[0][tindx]
			spwf_time_indx = time_indx[Nspw-1][tindx]
			### compute Depolarization Measure
			polI0 = polI[0][spwi_time_indx]
			polI0_err = polI_err[0][spwi_time_indx]
			polIf = polI[Nspw-1][spwf_time_indx]
			polIf_err = polI_err[Nspw-1][spwf_time_indx]
			depol = (polIf - polI0) / (polIf + polI0)
			### compute Depolarization Measure error
			df_dx = 2 * polI0 / ((polIf + polI0) ** 2) # partial derivatives
			df_dy = 2 * polIf / ((polIf + polI0) ** 2)
			depol_err = np.sqrt((df_dx * polIf_err)**2 + (df_dy * polI0_err)**2)
			DepolMeasure.append(depol)
			DepolMeasure_err.append(depol_err)
			# write in the .dat file
			fpOut.write("{:.18e}   {:.7e}  {:.7e}\n".format(time,depol,depol_err))
	return DepolMeasure,DepolMeasure_err

########################################################3

### Compute median minispiral flux density for the whole experiment: perform the minispiral flux average over the whole campaign to derive the flux of the minispiral for the whole set of epochs
AVERAGE_ALL = []
for track in TRACKS:
	fitname = 'SGRA_FIT_%s_%s'%(DATNAM,track)
	if not os.path.exists('%s.fit'%fitname):
		print('\n'+track+' not in the current folder\n')
		continue
	print('Read pickled file (BEWARE if you use CASA 6.x or Python 3!)')
	INF = open('%s.fit'%fitname,'r')
	LCData = pk.load(INF)
	INF.close()

	TOFLAG = []
	GOODS = []
	for i in range(Nspw):
		FLAG = LCData[0][i]['I Extended']<0.0
		FLAG_GOOD = LCData[0][i]['Good'] == False
		TOFLAG.append(np.logical_or(FLAG,FLAG_GOOD))
		GOODS.append(np.logical_not(TOFLAG[i]))

	AVERAGE_ALL.append([np.median(LCData[0][i]['I Extended'][GOODS[i]]) for i in range(Nspw)])

AVERAGE = np.median(np.array(AVERAGE_ALL),axis=0)
print('All epoch minispiral flux density (per spw): ',AVERAGE)


### Main loop
for trindx,track in enumerate(TRACKS):
	track_path = os.path.join(PATH0,'LCurves_%s_%s'%(SOURCE,track))
	if not os.path.exists(track_path):
		os.makedirs(track_path)	
	
	fitname = 'SGRA_FIT_%s_%s'%(DATNAM,track)
	if not os.path.exists('%s.fit'%fitname):
		print('\n'+track+' not in the current folder\n')
		continue
	
	INF = open('%s.fit'%fitname,'r')
	LCData = pk.load(INF)
	INF.close()

	TOFLAG = []
	GOODS = []
	for i in range(Nspw):
		FLAG = LCData[0][i]['I Extended']<0.0
		FLAG_GOOD = LCData[0][i]['Good'] == False
		TOFLAG.append(np.logical_or(FLAG,FLAG_GOOD))
		GOODS.append(np.logical_not(TOFLAG[i]))
	
	# Compute gains (avg/extended) to scale the Stokes parameters
	GAINS = [AVERAGE[i]/LCData[0][i]['I Extended'] for i in range(Nspw)]
	SGRA_I = [LCData[0][i]['I Compact']*GAINS[i] for i in range(Nspw)]
	SGRA_I_QA2 = [LCData[0][i]['I Compact'] for i in range(Nspw)]
	SGRA_I_EXTEND = [LCData[0][i]['I Extended'] for i in range(Nspw)]
	SGRA_Q = [LCData[0][i]['Q']*GAINS[i] for i in range(Nspw)]
	SGRA_U = [LCData[0][i]['U']*GAINS[i] for i in range(Nspw)]
	SGRA_V = [LCData[0][i]['V']*GAINS[i] for i in range(Nspw)]
	TIMES = [LCData[0][i]['JDTime']/3600. for i in range(Nspw)]
	MINT = int(np.min(TIMES[0]))
	
	SGRA_P = [np.sqrt(np.array(SGRA_U[i])**2. + np.array(SGRA_Q[i])**2.) for i in range(Nspw)]
	
	Ierr = [LCData[0][i]['Error Comp.']*GAINS[i] for i in range(Nspw)]
	Ierr_extend = [LCData[0][i]['Error Ext.'] for i in range(Nspw)]
	Qerr = [LCData[0][i]['Error Q']*GAINS[i] for i in range(Nspw)]
	Uerr = [LCData[0][i]['Error U']*GAINS[i] for i in range(Nspw)]
	Verr = [LCData[0][i]['Error V']*GAINS[i] for i in range(Nspw)]
	
	TimeGap = TIME_GAPS[trindx]
	flag_fact = FLAG_FACT[trindx]
	flag_scan_fact = FLAG_SCAN_FACT[trindx]
	# Get outlier flags:
	for i in range(Nspw):
		TOFLAG[i][np.logical_or(SGRA_I[i] < SGRA_MINI[trindx], SGRA_I[i]>SGRA_MAXI[trindx])] = True
		TOFLAG[i][np.logical_or(SGRA_P[i] < 0., SGRA_P[i]>SGRA_MAXP[trindx])] = True
		flagmask = np.logical_not(TOFLAG[i])
		if AUTO_FLAGGING:
			fname = 'SGRA_%s_StkI_spw%i'%(track,i)
			times = np.array([ti - MINT for ti in TIMES[i][flagmask]])
			outliers = flag_LCurve_outliers(times,SGRA_I[i][flagmask], fname, SplineOrder,Nknots,TimeGap, flag_fact, flag_scan_fact)
			false_indices = [findx for findx, flag in enumerate(flagmask) if flag] # creates the list of indx where TOFLAG[i] is False
			for outindx,findx in enumerate(false_indices):
				if outliers[outindx]:
					TOFLAG[i][findx] = True # set outliers as True in TOFLAG array
			del times,outliers,false_indices
		if FLAG_OUTLIERS:
			print('Flagging manually selected outliers: %s spw%i'%(track,i))
			jdtime = np.array([ti - MINT for ti in TIMES[i]])
			# Flags based on time and flux limits
			timelim_flags, fluxlim_flags = [], []
			# Open file with flag info
			flaginfo_path = os.path.join(PATH0,'SGRA_flag_info_%s_spw%i.dat'%(track,i))
			if os.path.exists(flaginfo_path):
				with open(flaginfo_path,'r') as file:
					for line in file:
						column1, column2 = line.split('\t\t') # Split each line in two columns
						timelim_flags.append(extract_arrays(column1))
						fluxlim_flags.append(extract_arrays(column2))
			for indx,tlim_flag in enumerate(timelim_flags):
				fluxlim_flag = np.array(fluxlim_flags[indx])
				# Update flags based on time and flux conditions.
				timeFLAGS = np.logical_and(jdtime > tlim_flag[0], jdtime < tlim_flag[1])
				TOFLAG[i][ np.logical_or( np.logical_and(timeFLAGS, SGRA_I[i] < fluxlim_flag[0]), np.logical_and(timeFLAGS, SGRA_I[i] > fluxlim_flag[1]) ) ] = True
			del jdtime,timelim_flags,fluxlim_flags
	
	### Update '.fit' file with the flagged outliers (set 'Good' to False if outlier)
	if SAVE_OUTLIERS:
		GOODS = []
		for k in range(Nspw):
			GOODS.append(np.logical_not(TOFLAG[k]))
		for k in range(Nspw):
			LCData[0][k]['Good'] = GOODS[k]
		# Write back to the .fit file
		with open('%s.fit'%fitname, 'wb') as OUTF:
			pk.dump(LCData, OUTF)
	
	TIMES_FLAG = []
	SGRA_I_FLAG, SGRA_I_EXTEND_FLAG, SGRA_Q_FLAG,SGRA_U_FLAG,SGRA_V_FLAG = [],[],[],[],[]
	Ierr_FLAG, Ierr_extend_FLAG, Qerr_FLAG, Uerr_FLAG, Verr_FLAG = [],[],[],[],[]
	SGRA_I_QA2_FLAG = []
	for k in range(Nspw):
		FLAGMASK = np.logical_not(TOFLAG[k])
		TIMES_FLAG.append(TIMES[k][FLAGMASK])
		SGRA_I_FLAG.append(SGRA_I[k][FLAGMASK])
		SGRA_I_QA2_FLAG.append(SGRA_I_QA2[k][FLAGMASK])
		SGRA_I_EXTEND_FLAG.append(SGRA_I_EXTEND[k][FLAGMASK])
		SGRA_Q_FLAG.append(SGRA_Q[k][FLAGMASK])
		SGRA_U_FLAG.append(SGRA_U[k][FLAGMASK])
		SGRA_V_FLAG.append(SGRA_V[k][FLAGMASK])
		Ierr_FLAG.append(Ierr[k][FLAGMASK])
		Ierr_extend_FLAG.append(Ierr_extend[k][FLAGMASK])
		Qerr_FLAG.append(Qerr[k][FLAGMASK])
		Uerr_FLAG.append(Uerr[k][FLAGMASK])
		Verr_FLAG.append(Verr[k][FLAGMASK])
	del LCData, SGRA_I,SGRA_Q,SGRA_U,SGRA_V, Ierr,Qerr,Uerr,Verr, GAINS, GOODS,FLAG,TOFLAG,FLAGMASK, INF, SGRA_I_QA2, SGRA_P
	gc.collect()
	
	# Derive polarization intensity and EVPA:
	EVPA_FLAG = [180./np.pi*np.arctan2(SGRA_U_FLAG[i],SGRA_Q_FLAG[i])/2. for i in range(Nspw)]
	P_FLAG = [np.sqrt(np.array(SGRA_U_FLAG[i])**2. + np.array(SGRA_Q_FLAG[i])**2.) for i in range(Nspw)]

	Perr_FLAG = [np.sqrt((SGRA_Q_FLAG[i]/P_FLAG[i]*Qerr_FLAG[i])**2. + (SGRA_U_FLAG[i]/P_FLAG[i]*Uerr_FLAG[i])**2.) for i in range(Nspw)]
	EVPAerr_FLAG = [1./(P_FLAG[i]**2.)*np.sqrt((SGRA_Q_FLAG[i]*Uerr_FLAG[i])**2. + (SGRA_U_FLAG[i]*Qerr_FLAG[i])**2.)*180./np.pi for i in range(Nspw)]
	
	###################################################################################
	# We have FLAGGED different times with outliers for each spw
	# To compute the RM and the Spectral Index, 
	# we need to find the intersection of all times accross all spws
	# First, we create a dictionary to count the occurrence of each time
	time_count = {}
	# Then, we count all the occurrences of a time accross all spws
	for times in TIMES_FLAG:
		for time in times:
			if time in time_count:
				time_count[time] += 1
			else:
				time_count[time] = 1
	# And we find the times coincident for all spws
	times_coincident = [time for time, count in time_count.items() if count == len(TIMES_FLAG)]
	# Now, we get the times where we can compute the RM.
	RM_times = sorted(times_coincident)
	# Finally, we get the indexes of the coincident times accross all spws.
	time_indx = []
	for times in TIMES_FLAG:
		indexes = [i for i, time in enumerate(times) if time in RM_times]
		time_indx.append(indexes)
	del time_count, times_coincident, indexes, times
	gc.collect()
	
	#############################      Compute RM      ################################
	SGRA_RM = []
	RMerr = []
	SGRA_EVPA0 = []
	EVPA0err = []
	for tindx,time in enumerate(RM_times):
		# compute RM: linear regression EVPA (y) vs. wavelenght (x)
		Sxy, Sx, Sx2, Sy, sigma = 0., 0., 0., 0., 0.
		for k in range(Nspw):
			spw_time_indx = time_indx[k][tindx] # time indx with no outliers for all spws
			yEVPA = EVPA_FLAG[k][spw_time_indx]
			Sxy += spw_wl[k]*yEVPA
			Sx += spw_wl[k]
			Sy += yEVPA
			Sx2 += (spw_wl[k])**2
		
		RMF = ((Nspw*Sxy-Sx*Sy)/(Nspw*Sx2-Sx**2))*(np.pi/180)*(1*10**9)**2 # in rad/m2
		phi0 = (Sy*(np.pi/180)-RMF*Sx/(1*10**9)**2)/Nspw # in rad
		for k in range(Nspw):
			sigma += (yEVPA*(np.pi/180)-(spw_wl[k]/(1*10**9)**2)*RMF-phi0)**2
		RMFerr = m.sqrt(Nspw)*m.sqrt(sigma/(Nspw-2))*(1*10**9)**2/m.sqrt(Nspw*Sx2-Sx**2)
		phi0err = RMFerr*m.sqrt(Sx2/Nspw)/(1*10**9)**2
		
		SGRA_RM.append(RMF)
		RMerr.append(RMFerr)
		SGRA_EVPA0.append(phi0)
		EVPA0err.append(phi0err)
	del Sx,Sy,Sxy,Sx2,yEVPA,sigma,spw_time_indx,RMF,RMFerr,phi0,phi0err
	gc.collect()
	
	###################### Save all light curves! ######################
	print('Saving LCurves %s %s'%(SOURCE,track))
	os.chdir(track_path) # change directory to save all files per track!
	os.system('rm -rf *_LCurve_*.*')
	for k in range(Nspw):
		# save time, flux density and error:
		data = np.column_stack((TIMES_FLAG[k], SGRA_I_FLAG[k], Ierr_FLAG[k]))
		fname = '%s_%s_LCurve_StkI_spw%s.dat'%(SOURCE,track,k)
		np.savetxt(fname, data, delimiter='\t', header='#\t JDTIME (h) \t STK I (Jy) \t Error I (Jy)', comments='')
		data = np.column_stack((TIMES_FLAG[k], SGRA_I_QA2_FLAG[k], Ierr_FLAG[k]))
		fname = '%s_%s_LCurve_StkIQA2_spw%s.dat'%(SOURCE,track,k)
		np.savetxt(fname, data, delimiter='\t', header='#\t JDTIME (h) \t STK I (Jy) \t Error I (Jy)', comments='')
		data = np.column_stack((TIMES_FLAG[k], SGRA_I_EXTEND_FLAG_ORIGINAL[k], Ierr_extend_FLAG[k]))
		fname = '%s_%s_LCurve_StkI_minispiral_spw%s.dat'%(SOURCE,track,k)
		np.savetxt(fname, data, delimiter='\t', header='#\t JDTIME (h) \t STK I minispiral (Jy) \t Error I  minispiral (Jy)', comments='')
		data = np.column_stack((TIMES_FLAG[k], SGRA_I_EXTEND_FLAG[k], Ierr_extend_FLAG[k]))
		fname = '%s_%s_LCurve_StkI_minispiral_spw%s_corrected.dat'%(SOURCE,track,k)
		np.savetxt(fname, data, delimiter='\t', header='#\t JDTIME (h) \t STK I minispiral (Jy) \t Error I  minispiral (Jy)', comments='')
		data = np.column_stack((TIMES_FLAG[k], SGRA_Q_FLAG[k], Qerr_FLAG[k]))
		fname = '%s_%s_LCurve_StkQ_spw%s.dat'%(SOURCE,track,k)
		np.savetxt(fname, data, delimiter='\t', header='#\t JDTIME (h) \t STK Q (Jy) \t Error Q (Jy)', comments='')
		data = np.column_stack((TIMES_FLAG[k], SGRA_U_FLAG[k], Uerr_FLAG[k]))
		fname = '%s_%s_LCurve_StkU_spw%s.dat'%(SOURCE,track,k)
		np.savetxt(fname, data, delimiter='\t', header='#\t JDTIME (h) \t STK U (Jy) \t Error U (Jy)', comments='')
		data = np.column_stack((TIMES_FLAG[k], SGRA_V_FLAG[k], Verr_FLAG[k]))
		fname = '%s_%s_LCurve_StkV_spw%s.dat'%(SOURCE,track,k)
		np.savetxt(fname, data, delimiter='\t', header='#\t JDTIME (h) \t STK V (Jy) \t Error V (Jy)', comments='')
		data = np.column_stack((TIMES_FLAG[k], EVPA_FLAG[k], EVPAerr_FLAG[k]))
		fname = '%s_%s_LCurve_EVPA_spw%s.dat'%(SOURCE,track,k)
		np.savetxt(fname, data, delimiter='\t', header='#\t JDTIME (h) \t EVPA (deg.) \t Error EVPA (deg.)', comments='')
		data = np.column_stack((TIMES_FLAG[k], P_FLAG[k], Perr_FLAG[k]))
		fname = '%s_%s_LCurve_PolI_spw%s.dat'%(SOURCE,track,k)
		np.savetxt(fname, data, delimiter='\t', header='#\t JDTIME (h) \t Plin (Jy) \t Error Plin (Jy)', comments='')
	data = np.column_stack((RM_times, SGRA_RM, RMerr))
	fname = '%s_%s_LCurve_RM.dat'%(SOURCE,track)
	np.savetxt(fname, data, delimiter='\t', header='#\t JDTIME (h) \t RM (rad/m2) \t Error RM (rad/m2)', comments='')
	data = np.column_stack((RM_times, SGRA_EVPA0, EVPA0err))
	fname = '%s_%s_LCurve_EVPA0.dat'%(SOURCE,track)
	np.savetxt(fname, data, delimiter='\t', header='#\t JDTIME (h) \t EVPA0 (rad) \t Error EVPA0 (rad)', comments='')
	del data
	
	#########################    FIGURES    #########################
	# Fig. Stokes I
	fig = pl.figure(figsize=(10,7))
	sub1 = fig.add_subplot(111)
	fig.subplots_adjust(wspace=0.01,hspace=0.01,right=0.98,left=0.125)
	fig.suptitle('Stokes I LCurves for SGRA %s'%track,fontsize=21)
	for i in range(Nspw):
		TimePlot = [spwtime - MINT for spwtime in TIMES_FLAG[i]]
		sub1.plot(TimePlot,SGRA_I_FLAG[i],'.%s'%cols[i],label='spw%i'%i)
	pl.sca(sub1)
	pl.legend(numpoints=1)
	sub1.set_xlabel('JDTime (h)')
	sub1.set_ylabel('Stokes I (Jy)')
	pl.savefig('%s_LCurve_StokesI_%s.png'%(SOURCE,track))
	pl.close()
	
	# Fig. Stokes Q
	fig = pl.figure(figsize=(10,7))
	sub1 = fig.add_subplot(111)
	fig.subplots_adjust(wspace=0.01,hspace=0.01,right=0.98,left=0.125)
	fig.suptitle('Stokes Q LCurves for SGRA %s'%track,fontsize=21)
	for i in range(Nspw):
		TimePlot = [spwtime - MINT for spwtime in TIMES_FLAG[i]]
		sub1.plot(TimePlot,SGRA_Q_FLAG[i],'.%s'%cols[i],label='spw%i'%i)
	pl.sca(sub1)
	pl.legend(numpoints=1)
	sub1.set_xlabel('JDTime (h)')
	sub1.set_ylabel('Stokes Q (Jy)')
	pl.savefig('%s_LCurve_StokesQ_%s.png'%(SOURCE,track))
	pl.close()
	
	# Fig. Stokes U
	fig = pl.figure(figsize=(10,7))
	sub1 = fig.add_subplot(111)
	fig.subplots_adjust(wspace=0.01,hspace=0.01,right=0.98,left=0.125)
	fig.suptitle('Stokes U LCurves for SGRA %s'%track,fontsize=21)
	for i in range(Nspw):
		TimePlot = [spwtime - MINT for spwtime in TIMES_FLAG[i]]
		sub1.plot(TimePlot,SGRA_U_FLAG[i],'.%s'%cols[i],label='spw%i'%i)
	pl.sca(sub1)
	pl.legend(numpoints=1)
	sub1.set_xlabel('JDTime (h)')
	sub1.set_ylabel('Stokes U (Jy)')
	pl.savefig('%s_LCurve_StokesU_%s.png'%(SOURCE,track))
	pl.close()
	
	# Fig. Stokes V
	fig = pl.figure(figsize=(10,7))
	sub1 = fig.add_subplot(111)
	fig.subplots_adjust(wspace=0.01,hspace=0.01,right=0.98,left=0.125)
	fig.suptitle('Stokes V LCurves for SGRA %s'%track,fontsize=21)
	for i in range(Nspw):
		TimePlot = [spwtime - MINT for spwtime in TIMES_FLAG[i]]
		sub1.plot(TimePlot,SGRA_V_FLAG[i],'.%s'%cols[i],label='spw%i'%i)
	pl.sca(sub1)
	pl.legend(numpoints=1)
	sub1.set_xlabel('JDTime (h)')
	sub1.set_ylabel('Stokes V (Jy)')
	pl.savefig('%s_LCurve_StokesV_%s.png'%(SOURCE,track))
	pl.close()
	
	# Fig. EVPA
	fig = pl.figure(figsize=(10,7))
	sub1 = fig.add_subplot(111)
	fig.subplots_adjust(wspace=0.01,hspace=0.01,right=0.98,left=0.125)
	fig.suptitle('EVPA LCurves for SGRA %s'%track,fontsize=21)
	for i in range(Nspw):
		TimePlot = [spwtime - MINT for spwtime in TIMES_FLAG[i]]
		sub1.plot(TimePlot,EVPA_FLAG[i],'.%s'%cols[i],label='spw%i'%i)
	pl.sca(sub1)
	pl.legend(numpoints=1)
	sub1.set_xlabel('JDTime (h)')
	sub1.set_ylabel('EVPA (deg.)')
	pl.savefig('%s_LCurve_EVPA_%s.png'%(SOURCE,track))
	pl.close()
	
	# Fig. Pol. Intensity
	fig = pl.figure(figsize=(10,7))
	sub1 = fig.add_subplot(111)
	fig.subplots_adjust(wspace=0.01,hspace=0.01,right=0.98,left=0.125)
	fig.suptitle('PolI LCurves for SGRA %s'%track,fontsize=21)
	for i in range(Nspw):
		TimePlot = [spwtime - MINT for spwtime in TIMES_FLAG[i]]
		sub1.plot(TimePlot,P_FLAG[i],'.%s'%cols[i],label='spw%i'%i)
	pl.sca(sub1)
	pl.legend(numpoints=1)
	sub1.set_xlabel('JDTime (h)')
	sub1.set_ylabel(r'$P=\sqrt{Q^2+U^2}$ (Jy)')
	pl.savefig('%s_LCurve_PolI_%s.png'%(SOURCE,track))
	pl.close()
	
	print('Getting Spectral Index %s %s'%(SOURCE,track))
	os.system('rm -rf *_SpectralIndx_*.*')
	# Stokes I
	spectindx_fname = '%s_%s_LCurve_SpectralIndx_StokesI.dat'%(SOURCE,track)
	SPECTINDX_STKI, SPECTINDX_STKI_err = SpectralIndex(RM_times,SGRA_I_FLAG, spw_frec,time_indx,spectindx_fname)
	AVER_SPECTINDX_STKI.append(np.average(SPECTINDX_STKI))
	STD_SPECTINDX_STKI.append(np.std(SPECTINDX_STKI))
	### Depolaritation Measure
	os.system('rm -rf *_Depolarization.*')
	depol_fname = '%s_%s_LCurve_Depolaritation.dat'%(SOURCE,track)
	DEPOLARIZATION, DEPOLARIZATION_err = DepolarizationMeasure(RM_times,P_FLAG,Perr_FLAG,time_indx, depol_fname)
	AVER_DEPOLARIZATION.append(np.average(DEPOLARIZATION))
	STD_DEPOLARIZATION.append(np.std(DEPOLARIZATION))
	
	# SPECTRAL INDEX Stokes I
	fig = pl.figure(figsize=(10,7))
	sub1 = fig.add_subplot(111)
	fig.subplots_adjust(wspace=0.01,hspace=0.01,right=0.98,left=0.125)
	fig.suptitle('Stokes I Spectral Index for SGRA %s'%track,fontsize=21)
	TimePlot = [rmtime - MINT for rmtime in RM_times]
	sub1.plot(TimePlot,SPECTINDX_STKI,'.k')
	sub1.set_xlabel('JDTime (h)')
	sub1.set_ylabel('Stokes I Spectral Index')
	pl.savefig('%s_LCurve_SpectralIndx_StokesI_%s.png'%(SOURCE,track))
	pl.close()
	
	# Depolarization Measure Fig.
	fig = pl.figure(figsize=(10,7))
	sub1 = fig.add_subplot(111)
	fig.subplots_adjust(wspace=0.01,hspace=0.01,right=0.98,left=0.125)
	fig.suptitle('Depolarization Measure for SGRA %s'%track,fontsize=21)
	TimePlot = [rmtime - MINT for rmtime in RM_times]
	sub1.plot(TimePlot,DEPOLARIZATION,'.k')
	sub1.set_xlabel('JDTime (h)')
	sub1.set_ylabel('Depolarization Measure')
	pl.savefig('%s_LCurve_Depolaritation_%s.png'%(SOURCE,track))
	pl.close()
	
	gc.collect()
	os.chdir(PATH0)


