import pylab as pl
import numpy as np
import os,sys
import pickle as pk
import gc
import matplotlib.pyplot as plt


######################################
###########

########################
### READ AND SET UP  ###
### THESE PARAMETERS ###
### CAREFULLY:       ###
########################

##############
# MSNAME will be set as '%s_%s.ms'%(DATNAM,track).
# First, specify ALL_TRACKS:
#        True to combine the extended models of all tracks in a unique CLEAN MODEL
#        if False, calibration per track, i.e. only 1 track in the TRACKS list!
ALL_TRACKS = True
DATNAM = '' # common name of the ".ms" files
if ALL_TRACKS:
	TRACKS = ['']
else:
	TRACKS = [''] # only one track!

## CONFIGURATION OF STEPS 0-1:
IMSIZE = 512
Ns = 18 # Nyquist sampling
deg_to_arcsec= 3600
Bmax = 1.3e5 # greatest projected baseline lenght -> highest resolution (plotms UVwave vs amp)
Cell = deg_to_arcsec*180./np.pi/Ns/Bmax
#CELL = '0.15arcsec'
CELL = '%.4farcsec'%Cell
##############

##############
## CONFIGURATION OF STEP 2:
# This dictionary will help us compute everything always in degrees 
#                 (actual image units may depend on CASA version):
units = {'arcsec':1/3600., 'deg':1.0, 'rad':180./np.pi}

# Remove SgrA* from extended model?
REMOVE_CENTER = True 

REMOVE_ALL_CENTER_BEAM = True
# Make it equal to zero (i.e. we take all the flux at the center) [True] or 
#               to the average in-beam extended brightness [False]?
# should be set to True if ALL_TRACKS is set to True
##############

##############
## CONFIGURATION OF STEP 3:
# This is for comparison to the classical (i.e., Venki) approach:
MINBAS = 100. # in meters.

RELOAD_MINISPIRAL_MODEL=True
# You probably want to set this to TRUE, always.
# However, if you are "playing" with the fitting, while
# using the same minispiral model, it may be sensible to 
# set it to False, so it is not recomputed every time. 

# **IF REPEATING** this step, AFTER STEP 5, you should use the
# self-calibrated data, so this should be True. Otherwise,
# set it to False:
USE_SELFCAL_DATA = False ### False if FIRST ITER (i.e. NOT repeating steps)
##############

##############
## CONFIGURATION OF STEP 4:
# Minimum and maximum allowed SgrA* flux densities,
# to flag bad integrations: SET MANUALLY!!!
SGRA_MIN = 1.5 ; SGRA_MAX = 4.5

# Export calibrated minispiral visibilities??
# This is NEEDED **IF STEP 5** is going to be run!
EXPORT_MINISPIRAL = True
##############

##############
## After running STEP 5,
#    set ### USE_SELFCAL_DATA = True (STEP 3 config.)
#        ### DO NOT DO THIS! USE ONLY THE MINISPIRAL CLEAN COMPONENTS, i.e.
#        USE_SELFCAL_DATA = False (STEP 3 config.)
#        EXPORT_MINISPIRAL = False (STEP 4 config.) and
#    repeat steps [3,4]
##############

###########
######################################

mysteps = [0,1,2,3,4]

thesteps = {0: 'Clearcal and add model column',
            1: 'First CLEANing (SgrA* + minispiral)',
            2: 'Take SgrA* out of the CLEAN model',
            3: 'Visibility modelfitting',
            4: 'Calibrate lightcurves',
            5: 'OPTIONAL: Re-CLEAN minispiral + self-calibrate'}

if 0 in mysteps:
	print('\n\n  %s \n'%thesteps[0])
	# Clear any new calibration:
	for track in TRACKS:
		MSNAME = '%s_%s'%(DATNAM,track)
		clearcal(MSNAME,addmodel=True)

## CLEAN: The mask can be improved incrementally, as we loop over the spws:
if 1 in mysteps:
	print('\n\n  %s \n'%thesteps[1])
	for track in TRACKS:
		MSNAME = '%s_%s.ms'%(DATNAM,track)
		for i in range(4):
			os.system('rm -rf %s_FIRST-CLEAN_SPW%i.*'%(MSNAME[:-3],i))
			if i==0 and not os.path.exists('%s.mask'%MSNAME[:-3]):
				MASK = ''
			else:
				MASK = '%s.mask'%MSNAME[:-3]
			tclean(vis=MSNAME,
					imagename='%s_FIRST-CLEAN_SPW%i'%(MSNAME[:-3],i),
					cell=CELL, imsize=IMSIZE,
					stokes='I', specmode='mfs',nterms=1,
					mask=MASK, spw=str(i),
					niter=7500,gain=0.025,
					cyclefactor=1.0, interactive=True)
			os.system('rm -rf %s.mask'%MSNAME[:-3])
			os.system('cp -r %s_FIRST-CLEAN_SPW%i.mask %s.mask'%(MSNAME[:-3],i,MSNAME[:-3]))

## We make the average image among spws, after subtracting SgrA*.
if 2 in mysteps:
	print('\n\n  %s \n'%thesteps[2])
	
	### Compute the campaign model (if ALL_TRACKS set to True)
	if ALL_TRACKS:
		TOTAL_MODEL = []
		for i in range(4):
			ia.open('%s_FIRST-CLEAN_SPW%i.model'%('%s_%s.ms'%(DATNAM,TRACKS[0]),i))
			TOTAL_MODEL.append(np.copy(ia.getchunk()))
			ia.close()
			for track in TRACKS[1:]:
				MSNAME = '%s_%s.ms'%(DATNAM,track)
				ia.open('%s_FIRST-CLEAN_SPW%i.model'%(MSNAME[:-3],i))
				TOTAL_MODEL[-1] += ia.getchunk()
				ia.close()
			TOTAL_MODEL[-1] /= 4
	
	AVG_IMG = np.zeros((IMSIZE,IMSIZE)) 
	
	EXT_FIG = pl.figure()
	EXT_SUBP = EXT_FIG.add_subplot(111)
	
	CLEANS = np.zeros((IMSIZE,IMSIZE))
	# RR is an array that gives the distance of each image pixel to 
	# the CLEAN peak. This RR array will be used to make a mask that
	# only takes the CLEAN components closest to the CLEAN peak, in case
	# the user wants to estimate the extended contribution to the 
	# CLEAN peak:
	X1 = np.linspace(-float(IMSIZE)/2.,float(IMSIZE)/2.,IMSIZE)
	RX = np.outer(np.ones(IMSIZE),X1) ; RY = np.transpose(RX)
	RR = np.sqrt(RX*RX + RY*RY)
	
	mask = np.zeros(np.shape(CLEANS),dtype=bool)
	Beam = np.zeros(np.shape(CLEANS))
	X = np.zeros(np.shape(CLEANS)); Y = np.zeros(np.shape(CLEANS))
	ConvClean = np.zeros(np.shape(CLEANS))
	
	for track in TRACKS:
		MSNAME = '%s_%s.ms'%(DATNAM,track)
		for i in range(4):
			# Read the image metadata (e.g., beam size, pixel size, etc.):
			ia.open('%s_FIRST-CLEAN_SPW%i.image'%(MSNAME[:-3],i))
			summ = ia.summary()
			BM = summ['restoringbeam']['major']
			Bm = summ['restoringbeam']['minor']
			BPA = summ['restoringbeam']['positionangle']
			BMAJ = BM['value']*units[BM['unit']]
			BMIN = Bm['value']*units[Bm['unit']]
			PANG = BPA['value']*units[BPA['unit']]
			
			# Image coordinates of the intensity peak:
			PEAK = np.unravel_index(np.argmax(ia.getchunk()[:,:,0,0]),dims=summ['shape'][:2])
			ia.close()
			
			# Pixel size:
			DX = summ['incr'][0]*units[summ['axisunits'][0]]
			DY = summ['incr'][1]*units[summ['axisunits'][1]]
			
			# Beam area in pixel units:
			PIXBEAM = np.sqrt(BMAJ*BMIN)/np.abs(DX)
			
			# CLEAN beam (used for re-convolving the extended structure):
			beam = [BMAJ, BMIN, PANG+90.]
			NPIX = IMSIZE
			
			# Read the CLEAN components (from the "*.model" image):
			if ALL_TRACKS:
				CLEANS[:] = TOTAL_MODEL[i][:,:,0,0]
			else:
				ia.open('%s_FIRST-CLEAN_SPW%i.model'%(MSNAME[:-3],i))
				CLEANS[:] = ia.getchunk()[:,:,0,0]
				ia.close()
			
			# Coordinates of the CLEAN component with strongest flux density:
			CCPEAK = np.unravel_index(np.argmax(CLEANS),dims=summ['shape'][:2])
			
			# Actual flux-density value of the strongest CLEAN component:
			STARPEAK = float(CLEANS[CCPEAK])
			
			# Total CLEAN flux density:
			EXTFLUX = np.sum(CLEANS) - STARPEAK
			
			# Mask to take only the CLEANs closest to the peak (within the beam size):
			mask[:] = np.logical_and(CLEANS!=0.0,RR<PIXBEAM/2.)
			
			# Contribution from the extended source within the mask:
			EXTINT = np.average(CLEANS[mask]) - STARPEAK/np.sum(mask)
			
			print('COMPACT FLUX: ',STARPEAK)
			print('EXTENDED FLUX: ',EXTFLUX)
			print('EXTENDED INTENSITY AROUND SGRA*: ', EXTINT)
			
			if REMOVE_CENTER:
				# DO WE ZERO THE PEAK PIXEL COMPLETELY??
				if REMOVE_ALL_CENTER_BEAM:
					#CLEANS[CCPEAK] = 0.0 # Only the flux peak pixel
					CLEANS[mask] = 0.
				# OR DO WE MAKE IT EQUAL TO THE AVERAGE CLEAN FLUX WITHIN THE BEAM?
				else:
					#CLEANS[CCPEAK] = EXTINT # Only the flux peak pixel
					CLEANS[mask] = EXTINT
			
			# Prepare beam image (to be used as the convolving kernel):
			X[:] = np.outer(np.linspace(-NPIX/2.*DX,NPIX/2.*DX,NPIX),np.ones(NPIX))
			Y[:] = np.outer(np.ones(NPIX),np.linspace(-NPIX/2.*DX,NPIX/2.*DX,NPIX))
			
			# Equation of the elliptical Gaussian (CLEAN beam):
			Cos = np.cos(beam[2]*np.pi/180.)
			Sin = np.sin(beam[2]*np.pi/180.)
			CosS = np.cos(2.*beam[2]*np.pi/180.)
			SinS = np.sin(2.*beam[2]*np.pi/180.)
			
			Sx = beam[0] #/2.35482 ### This division depends on the casa version
			Sy = beam[1] #/2.35482
			a = Cos**2./(2.*Sx**2.) + Sin**2./(2.*Sy**2.)
			b = -SinS/(4.*Sx**2.) + SinS/(4.*Sy**2.)
			c = Sin**2./(2.*Sx**2.) + Cos**2./(2.*Sy**2.)
			
			Beam[:] = np.exp(-(a*X**2. + 2.*b*X*Y + c*Y**2.))
			
			# Convolve CLEAN beam with CLEAN model. We use the trick
			# of the Convolution Theorem (i.e., FFT of the convolution
			# is equal to the product of FFTs):
			FFTBeam = np.fft.fft2(np.fft.fftshift(Beam))
			FFTClean = np.zeros(np.shape(Beam),dtype=np.complex128)
			ConvClean[:] = (np.fft.ifft2(np.fft.fft2(CLEANS)*FFTBeam)).real
			
			# Plot image free of the central component:
			pl.sca(EXT_SUBP)
			EXT_SUBP.clear()
			EXT_SUBP.imshow(np.transpose(ConvClean),origin='lower',interpolation='nearest', cmap='Greys')
			EXT_SUBP.set_title('EXTENDED MODEL - %s SPW: %i'%(MSNAME[:-3],i))
			pl.savefig('%s_EXT-MODEL_SPW%i.png'%(MSNAME[:-3],i))
			
			# Add extended model to spw average image:
			AVG_IMG[:] += CLEANS
		
		# Plot average minispiral model (using the beam from spw3):
		AVG_IMG /= 4.0
		ConvClean[:] = (np.fft.ifft2(np.fft.fft2(AVG_IMG)*FFTBeam)).real

		pl.sca(EXT_SUBP)
		EXT_SUBP.clear()
		EXT_SUBP.imshow(np.transpose(ConvClean),origin='lower',interpolation='nearest', cmap='Greys')
		EXT_SUBP.set_title('EXTENDED MODEL %s'%(MSNAME[:-3]))
		pl.savefig('%s_EXT-MODEL.png'%(MSNAME[:-3]))
		
		# Export extended structure into a new CASA image:
		os.system('rm -rf %s_noPeak.image'%MSNAME[:-3])
		os.system('cp -r %s_FIRST-CLEAN_SPW%i.image %s_noPeak.image'%(MSNAME[:-3],i,MSNAME[:-3]))
		ia.open('%s_noPeak.image'%MSNAME[:-3])
		OUTPUT = ia.getchunk()
		OUTPUT[:,:,0,0] = ConvClean
		ia.putchunk(OUTPUT)
		ia.close()
		
		# Export extended structure to an ascii file:
		off = open('%s_CCs.dat'%MSNAME[:-3],'w')
		mask = np.where(AVG_IMG!=0.0)
		for i in range(len(mask[0])):
			Xoff = DX*(mask[0][i] - summ['refpix'][0])*3600. # coordinates, in arcsec.
			Yoff = DY*(mask[1][i] - summ['refpix'][1])*3600. # coordinates, in arcsec.
			S = AVG_IMG[ mask[0][i],mask[1][i] ] # flux density at each pixel
			print >> off, '%.4e  %.4e  %.4e'%(Xoff, Yoff, S)
		off.close()
		os.system('cp %s_CCs.dat %s_CCs_FIRST-CLEAN.dat'%(MSNAME[:-3],MSNAME[:-3]))
	
	del Beam,FFTBeam,FFTClean,ConvClean,AVG_IMG,X,Y,mask,CLEANS
	gc.collect()

## Visibility (two components) model fitting
if 3 in mysteps:
	print('\n\n  %s \n'%thesteps[3])
	
	for track in TRACKS:
		MSNAME = '%s_%s.ms'%(DATNAM,track)
		# Prefix name of output file:
		fitname = 'SGRA_FIT_%s'%MSNAME[:-3]
		
		# Read minispiral model (i.e. the CLEAN components of the minispiral):
		CCfile = open('%s_CCs.dat'%MSNAME[:-3],'r')
		cleans = []
		TotCLFlux = 0.0
		for line in CCfile.readlines():
			temp = map(float,line.split())
			cleans.append(temp)
			TotCLFlux += temp[2]
		CCfile.close()
		
		# Normalize extended model:
		for cli in cleans:
			## cli[0] and cli[1] are the RA and DEC offsets (in arcsec). 
			## cli[2] is the flux density.
			cli[2] = cli[2]/TotCLFlux
		
		
		# Read frequency for each spectral window (spw):
		tb.open(os.path.join(MSNAME,'SPECTRAL_WINDOW'))
		FREQS = np.average(tb.getcol('CHAN_FREQ'),axis=0) ## BEWARE OF MFS!!!
		tb.close()
		
		# wavelengths:
		LAMBDA = 299792458./FREQS
		
		tb.open(MSNAME,nomodify=False)
		
		T = tb.getcol('TIME')
		# List of unique (and sorted) observing times:
		UT = np.unique(T)
		
		# Read data (UV coordinates, spw, weights, flags, etc.):
		UVW = tb.getcol('UVW')
		MOD = tb.getcol('MODEL_DATA') # Here we will write the FT of the minispiral
		DAT = tb.getcol('DATA')
		COR = tb.getcol('CORRECTED_DATA')
		SPI = tb.getcol('DATA_DESC_ID')
		FG = tb.getcol('FLAG')
		WGT = tb.getcol('WEIGHT')
		A1 = tb.getcol('ANTENNA1')
		A2 = tb.getcol('ANTENNA2')
		FG[:] = False
		FG[:,:,A1==A2] = True
		
		# So we use the selfcal data for fitting?
		if USE_SELFCAL_DATA:
			DAT[:] = COR
		
		# Set weights of flagged data to zero:
		WGT[FG[:,0,:]] = 0.0
		WGT[WGT<0.0] = 0.0
		
		# Get distance in UV plane:
		Q = np.sqrt(UVW[0,:]**2.+UVW[1,:]**2.)
		
		# Mask for short baselines (Venki approach):
		QMask = Q>= MINBAS
		
		################################################
		##### LOAD THE MODEL OF THE EXTENDED COMPONENT:
		if RELOAD_MINISPIRAL_MODEL:
			# Set the model column to zero. We will fill 
			# this column with the Fourier transform of the
			# extended component:
			MOD[:] = 0.0
			
			# Phase factor used in the Fourier transform
			# RA and Dec offsets are assumed to be given in
			# arc-seconds, and UV coordinates in wavelengths:
			FOUFAC = np.pi/180./3600.
			
			# Array to store the phases of the Fourier transform.
			# We will reuse this array for each CLEAN component, 
			# to minimize memory leaks:
			PHASE = np.zeros(np.shape(T),dtype=np.complex128)
			
			# Fill in the MOD array with the Fourier transforms 
			# of all the CLEAN components of the extended source:
			for j,cli in enumerate(cleans):
				sys.stdout.write('\r LOADING CLEAN COMPONENT %i OF %i'%(j+1,len(cleans)))
				sys.stdout.flush()
				## cli[0] and cli[1] are the RA and DEC offsets (in arcsec). 
				## cli[2] is the flux density of the extended (CLEAN) component.
				PHASE[:] = np.exp(1.j*2.*np.pi*(UVW[0,:]*cli[0] + UVW[1,:]*cli[1])*FOUFAC/LAMBDA[SPI])
				MOD[0,0,:] += cli[2]*PHASE[:]
			
			## YY model is equal to XX model (i.e., the extended source is UNpolarized):
			MOD[3,0,:] = MOD[0,0,:]
			
			## Write the model column into the Measurement Set:
			tb.putcol('MODEL_DATA',MOD)

			del PHASE

		tb.close()
		################################################
		
		# Lists to store the results (one list per spw):
		I_FIT = [[] for i in range(4)]
		
		####################################
		# Our model only has 2 parameters:
		#
		# Par 1: integrated flux density of the extended source.
		# Par 2: flux density of the compact source.
		#
		# Furthermore, the model is linear for these parameters,
		# so the fit can be done algebraically, 
		#  just by inverting the 2x2 Hessian matrix!
		#  and the minimum is mathematical!
		
		# Hessian and gradient vector (will be reused for each integration
		# time, to minimize memory leaks):
		Hessian = np.zeros((2,2),dtype=np.float)
		ResVec = np.zeros((2),dtype=np.float)
		Inv = np.copy(Hessian) ## Array to store the inverse of the Hessian.
		
		# Loop over integration times:
		print('\n\n GOING TO FIT DATA\n\n')
		mask = np.zeros(np.shape(T),dtype=np.bool)
		mask2 = np.zeros(np.shape(T),dtype=np.bool)
		
		for j,ti in enumerate(UT):
			
			# Mask to select only the current integration time:
			mask[:] = T == ti
			sys.stdout.write('\r FITTING TIME %i OF %i'%(j+1,len(UT)))
			sys.stdout.flush()
			
			# Loop over spectral windows (i.e. we fit each integration time and spw independently):
			for k in range(4):
				
				# Mask to select the current integration time AND the current spectral window:
				mask2[:] = np.logical_and(mask,SPI==k)
				
				# Reset the Hessian and gradient vector:
				Hessian[:] = 0.0
				ResVec[:] = 0.0
				
				# Only proceed if there are unflagged visibilities in XX and YY
				# (we have plenty antennas and integration times):
				TWGT = WGT[0,mask2]
				TWGT2 = WGT[3,mask2]
				
				if np.sum(TWGT)>0.0 and np.sum(TWGT2)>0.0:
					###########################################################
					## Model (in I = (XX+YY)/2 ) of the extended source at this time and spw:
					TMOD = MOD[0,0,mask2]
					## Data (in I = (XX+YY)/2 ) at this time and spw:
					TDAT = (DAT[0,0,mask2] + DAT[3,0,mask2])*0.5
					
					##########
					# Hessian matrix (i.e., its elements are computed from the TMOD):
					## Contribution from XX+YY:
					Hessian[0,0] = 2.*np.sum((TMOD.real*TMOD.real + TMOD.imag*TMOD.imag)*TWGT)
					Hessian[0,1] = 2.*np.sum((TMOD.real)*TWGT)
					Hessian[1,1] = 2.*np.sum(TWGT)
					# The Hessian is symmetric. We know the value of [1,0] just from [0,1]:
					Hessian[1,0] = Hessian[0,1]
					
					##########
					# Gradient vector (contribution from XX+YY):
					ResVec[0] = 2.*np.sum((TMOD.real*TDAT.real + TMOD.imag*TDAT.imag)*TWGT)
					ResVec[1] = 2.*np.sum((TDAT.real)*TWGT)
				
				# If the Hessian is not singular, perform the fit:
				if Hessian[0,0] != 0.0 and Hessian[1,1] != 0.0:
					Inv[:] = np.linalg.inv(Hessian)
					
					# The two flux densities (extended and compact) are derived from:
					##    FLUXES = (HESSIAN)^(-1) x RESVEC (where "x" is a matrix-vector product).
					##                 EXTENDED FLUX                                     COMPACT FLUX
					EXTFLX = Inv[0,0]*ResVec[0]+Inv[0,1]*ResVec[1] ; CMPFLX = Inv[1,0]*ResVec[0]+Inv[1,1]*ResVec[1]
					
					# Modify the "corrected data" column:
					# Extended component with re-calibration assuming S_ext = 1 Jy:
					#  We subtract the compact flux from the data (only the extended component remains)
					#    and impose that the extended component has a constant 1 Jy flux (dividing by the extended flux)
					COR[0,:,mask2] = (DAT[0,:,mask2] - CMPFLX)/EXTFLX
					COR[3,:,mask2] = (DAT[3,:,mask2] - CMPFLX)/EXTFLX
					COR[1,:,mask2] = DAT[1,:,mask2]/EXTFLX
					COR[2,:,mask2] = DAT[2,:,mask2]/EXTFLX
					
					# Estimate flux from long baselines:
					VenkiMask = np.where(np.logical_and(QMask,mask2))[0]
					# Average of the Stokes I FT (XX+YY)
					VENKI = np.abs(np.average(DAT[0,:,VenkiMask]+DAT[3,:,VenkiMask])/2.)
					
					# Store results in the list:
					I_FIT[k].append([ti,EXTFLX,CMPFLX,VENKI])
					del VenkiMask
					
					# The uncertainties are taken from the Covariance matrix (i.e., the "Inv" matrix):
					RedChiSq = np.sum( TWGT*np.abs(TDAT - (TMOD*EXTFLX + CMPFLX))**2.)
					RedChiSq /= (2.*np.sum(mask2) - 2.)
					Error1 = np.sqrt(Inv[0,0]*RedChiSq)
					Error2 = np.sqrt(Inv[1,1]*RedChiSq)
					# Covariance between the two flux densities 
					#  (they are highly correlated in the fit):
					Cov = Inv[0,1]*Inv[1,0]/(Inv[0,0]*Inv[1,1])
					
					
					# Now, derive the other Stokes parameters 
					#  (for which one single point source is assumed):
					WGTP = np.sqrt(TWGT*TWGT2)
					Q = (np.sum((DAT[0,0,mask2]-DAT[3,0,mask2])*WGTP)/np.sum(WGTP)).real /2. # (XX-YY) average
					U = (np.sum((DAT[1,0,mask2]+DAT[2,0,mask2])*WGTP)/np.sum(WGTP)).real /2. # (XY+YX) average
					V = (np.sum((DAT[1,0,mask2]-DAT[2,0,mask2])*WGTP)/np.sum(WGTP)).imag /2. # (XY-YX) average
					Nsamp = np.sqrt(np.sum(mask2))
					ErrI = np.std(DAT[0,0,mask2]+DAT[3,0,mask2]).real/Nsamp /2.
					ErrQ = np.std(DAT[0,0,mask2]-DAT[3,0,mask2]).real/Nsamp /2.
					ErrU = np.std(DAT[1,0,mask2]+DAT[2,0,mask2]).real/Nsamp /2.
					ErrV = np.std(DAT[1,0,mask2]-DAT[2,0,mask2]).imag/Nsamp /2.
					# Store results in the list:
					I_FIT[k][-1] += [Q, U, V, Error1,Error2,Cov, ErrQ, ErrU, ErrV]
		
		# Update the corrected column of the measurement set.
		# It will now contain the minispiral data, normalized to 1Jy:
		tb.open(MSNAME,nomodify=False)
		tb.putcol('CORRECTED_DATA',COR)
		tb.close()
		
		# Convert the lists into numpy named arrays (for each spw):
		FIT_ARR = [np.array(np.zeros(len(I_FIT[k])), dtype=np.dtype([('JDTime', np.float), ('I Extended', np.float), ('I Compact',np.float),('I LongBas', np.float),('Q', np.float), ('U',np.float),('V',np.float), ('Error Ext.',np.float), ('Error Comp.',np.float), ('Covariance', np.float), ('Error Q',np.float),('Error U',np.float),('Error V',np.float),('Good',bool)])) for k in range(4)]
		
		for k in range(4):
			Itemp = np.array(I_FIT[k])
			FIT_ARR[k]['JDTime'][:] = Itemp[:,0]
			FIT_ARR[k]['I Extended'][:] = Itemp[:,1]
			FIT_ARR[k]['I Compact'][:] = Itemp[:,2]
			FIT_ARR[k]['I LongBas'][:] = Itemp[:,3] # Venki approach
			FIT_ARR[k]['Q'][:] = Itemp[:,4]
			FIT_ARR[k]['U'][:] = Itemp[:,5]
			FIT_ARR[k]['V'][:] = Itemp[:,6]
			FIT_ARR[k]['Error Ext.'][:] = Itemp[:,7]
			FIT_ARR[k]['Error Comp.'][:] = Itemp[:,8]
			FIT_ARR[k]['Covariance'][:] = Itemp[:,9]
			FIT_ARR[k]['Error Q'][:] = Itemp[:,10]
			FIT_ARR[k]['Error U'][:] = Itemp[:,11]
			FIT_ARR[k]['Error V'][:] = Itemp[:,12]
			FIT_ARR[k]['Good'][:] = True
		
		# Write in pickled file (BEWARE if you use CASA 6.x):
		OFF = open('%s.fit'%fitname,'w')
		pk.dump([FIT_ARR,FREQS],OFF)
		OFF.close()
		
		# Release memory:
		for i in range(3,-1,-1):
			del FIT_ARR[i]
		del mask,mask2,UVW,MOD,DAT,COR,SPI,WGT,A1,A2,Q
		gc.collect()

if 4 in mysteps:
	print('\n\n  %s \n'%thesteps[4])
	
	for track in TRACKS:
		MSNAME = '%s_%s.ms'%(DATNAM,track)
		
		fitname = 'SGRA_FIT_%s'%MSNAME[:-3]
		
		## Read pickled file (BEWARE if you use CASA 6.x):
		INF = open('%s.fit'%fitname,'r')
		LCData = pk.load(INF)
		INF.close()

		# Plot:
		#for k in range(4):
		fig = pl.figure()
		Tplot = LCData[0][0]['JDTime']/86400.
		T00 = np.floor(np.min(Tplot))
		Tplot -= T00
		Tplot *= 24.
		pl.plot(Tplot,LCData[0][0]['I Extended'],'ob',label='Extended')
		pl.plot(Tplot,LCData[0][0]['I Compact'],'or',label='IF selfcal')
		pl.plot(Tplot,LCData[0][0]['I LongBas'],'og',label='Long Bas.')
		pl.legend(numpoints=1)
		pl.xlabel('UT')
		pl.ylabel('SgrA* Flux Density (Jy)')
		pl.suptitle('REAL DATA',fontsize=25)
		pl.savefig('step4_LCURVES.png')
		
		TOFLAG = []
		GOODS = []
		for i in range(4):
			FLAG = LCData[0][i]['I Extended']<0.0
			times = LCData[0][i]['JDTime']
			#for t in range(len(times)-1):
			#	if times[t+1]-times[t]>100. and not FLAG[t]:
			#		FLAG[t+1] = True
			TOFLAG.append(FLAG)
			GOODS.append(np.logical_not(FLAG))
		
		AVERAGE = [np.average(LCData[0][i]['I Extended'][GOODS[i]]) for i in range(4)]
		print('Average minispiral flux density (per spw): ',AVERAGE)
		
		
		if EXPORT_MINISPIRAL:
			# Update the corrected column of the measurement set.
			# It will now contain the minispiral data, normalized to the flux-density average:
			print('Scaling minispiral data')
			tb.open(MSNAME,nomodify=False)
			SPI = tb.getcol('DATA_DESC_ID')
			COR = tb.getcol('CORRECTED_DATA')
			mask = np.zeros(np.shape(SPI),dtype=bool)
			for i in range(4):
				mask[:] = SPI==i
				COR[:,:,mask] *= AVERAGE[i]
			tb.putcol('CORRECTED_DATA',COR)
			tb.close()
			del COR,SPI
			
			# Split the minispiral-only calibrated data:
			print('Splitting minispiral data')
			os.system('rm -rf %s_ExtendedData'%MSNAME)
			split(vis=MSNAME,outputvis='%s_ExtendedData'%MSNAME,datacolumn='corrected')
		
		# Compute gains (avg/extended) to scale the Stokes parameters retrieved from step 3
		GAINS = [AVERAGE[i]/LCData[0][i]['I Extended'] for i in range(4)]
		
		SGRA_I = [LCData[0][i]['I Compact']*GAINS[i] for i in range(4)]
		SGRA_Q = [LCData[0][i]['Q']*GAINS[i] for i in range(4)]
		SGRA_U = [LCData[0][i]['U']*GAINS[i] for i in range(4)]
		SGRA_V = [LCData[0][i]['V']*GAINS[i] for i in range(4)]
		TIMES = [LCData[0][i]['JDTime']/86400. for i in range(4)]
		MINT = int(np.min(TIMES[0]))
		
		Ierr = [LCData[0][i]['Error Comp.']*GAINS[i] for i in range(4)]
		Qerr = [LCData[0][i]['Error Q']*GAINS[i] for i in range(4)]
		Uerr = [LCData[0][i]['Error U']*GAINS[i] for i in range(4)]
		Verr = [LCData[0][i]['Error V']*GAINS[i] for i in range(4)]
		
		# Compute outlier flags:
		for i in range(4):
			TOFLAG[i][np.logical_or(SGRA_I[i] < SGRA_MIN, SGRA_I[i]>SGRA_MAX)] = True
		
		# Derive polarization intensity and EVPA:
		EVPA = [180./np.pi*np.arctan2(SGRA_U[i],SGRA_Q[i])/2. for i in range(4)]
		P = [np.sqrt(SGRA_U[i]**2. + SGRA_Q[i]**2.) for i in range(4)]
		
		Perr = [np.sqrt((SGRA_Q[i]/P[i]*Qerr[i])**2. + (SGRA_U[i]/P[i]*Uerr[i])**2.) for i in range(4)]
		Phierr = [1./(P[i]**2.)*np.sqrt((SGRA_Q[i]*Uerr[i])**2. + (SGRA_U[i]*Qerr[i])**2.)*180./np.pi for i in range(4)]
		
		# Make plots:
		cols = ['r','g','b','k']
		fig = pl.figure(figsize=(14,7))
		sub1 = fig.add_subplot(411)
		sub2 = fig.add_subplot(412)
		sub3 = fig.add_subplot(413)
		sub4 = fig.add_subplot(414)
		fig.subplots_adjust(wspace=0.01,hspace=0.01,right=0.98,left=0.07)
		fig.suptitle('LCurve for %s'%MSNAME,fontsize=25)
		for i in range(4):
			PLOTMASK = np.logical_not(TOFLAG[i])
			TimePlot = TIMES[i][PLOTMASK]-MINT
			sub1.plot(TimePlot,SGRA_I[i][PLOTMASK],'o%s'%cols[i],label='spw%i'%i)
			sub2.plot(TimePlot,P[i][PLOTMASK],'o%s'%cols[i])
			sub3.plot(TimePlot,EVPA[i][PLOTMASK],'o%s'%cols[i])
			sub4.plot(TimePlot,SGRA_V[i][PLOTMASK],'o%s'%cols[i])
		
		pl.sca(sub1)
		pl.legend(numpoints=1)
		
		pl.setp(sub1.get_xticklabels(),'visible',False)
		pl.setp(sub2.get_xticklabels(),'visible',False)
		pl.setp(sub3.get_xticklabels(),'visible',False)
		
		sub4.set_xlabel('JD - %i'%MINT)
		sub1.set_ylabel('I (Jy)')
		sub2.set_ylabel('P (Jy)')
		sub3.set_ylabel('EVPA (deg.)')
		sub4.set_ylabel('V (Jy)')
		
		sub3.set_ylim((-210.,210.))
		#sub3.set_ylim((-181.,181.))
		sub1.set_ylim((0.01,np.max(SGRA_I[3][PLOTMASK])*1.1))
		sub2.set_ylim((0.01,np.max(P[3][PLOTMASK])*1.1))
		sub4.set_ylim((np.min(SGRA_V[3][PLOTMASK])*1.3,np.max(SGRA_V[3][PLOTMASK])*5.))
		
		pl.savefig('%s_LCurve.png'%MSNAME[:-3])
		pl.savefig('%s_LCurve.pdf'%MSNAME[:-3])
		pl.show()
		
		# Save calibrated light curves in ascii files:
		for i in range(4):
			OFF = open('Light_Curve_SPW%i_%s.dat'%(i,MSNAME[:-3]),'w')
			print >> OFF,'MJD   I(Jy) Ierr(Jy)    P(Jy) Perr(Jy)    EVPA(deg) EVPAerr(deg)   V(Jy) Verr(Jy)'
			for ent in np.where(np.logical_not(TOFLAG[i]))[0]:
				Iw = SGRA_I[i][ent]; Ie = Ierr[i][ent]
				Pw = P[i][ent]; Pe = Perr[i][ent]
				Phiw = EVPA[i][ent]; Phie = Phierr[i][ent]
				Vw = SGRA_V[i][ent]; Ve = Verr[i][ent]
				Tw = TIMES[i][ent]
				print >> OFF, '%.16e     %.4e %.4e      %.4e %.4e     %.4e %.4e     %.4e %.4e'%(Tw, Iw, Ie, Pw, Pe,Phiw,Phie,Vw,Ve)
			OFF.close()

if 5 in mysteps:
	print('\n\n  %s \n'%thesteps[5])
	
	for track in TRACKS:
		MSNAME = '%s_%s.ms'%(DATNAM,track)
		
		clearcal('%s_ExtendedData'%MSNAME, addmodel=True)
		
		# CLEAN just the minispiral model:
		for i in range(4):
			os.system('rm -rf %s_SECOND-CLEAN_SPW%i.*'%(MSNAME[:-3],i))
			if i==0 and not os.path.exists('%s.mask'%MSNAME[:-3]):
				MASK = ''
			else:
				MASK = '%s.mask'%MSNAME[:-3]
			tclean(vis='%s_ExtendedData'%MSNAME,
					imagename='%s_SECOND-CLEAN_SPW%i'%(MSNAME[:-3],i),
					cell=CELL, imsize=IMSIZE, spw=str(i),
					stokes='I', specmode='mfs',nterms=1,
					mask=MASK, savemodel='modelcolumn',
					niter=5000,gain=0.025,
					cyclefactor=1.0, interactive=True)
			os.system('rm -rf %s.mask'%MSNAME[:-3])
			os.system('cp -r %s_SECOND-CLEAN_SPW%i.mask %s.mask'%(MSNAME[:-3],i,MSNAME[:-3]))
		
		# Self-calibrate:
		gaincal(vis='%s_ExtendedData'%MSNAME,calmode='p',gaintype='T',solint='100s',combine='spw',caltable='%s_SELFCAL.tab'%MSNAME[:-3])
		#gaincal(vis='%s_ExtendedData'%MSNAME,calmode='ap',gaintype='T',solint='100s',combine='spw',caltable='%s_SELFCAL.tab'%MSNAME[:-3])
		applycal(vis='%s_ExtendedData'%MSNAME,spwmap=[0,0,0,0],applymode='calonly',gaintable='%s_SELFCAL.tab'%MSNAME[:-3])
		
		# Final CLEAN:
		FINAL_MINISP = np.zeros((IMSIZE,IMSIZE))
		for i in range(4):
			os.system('rm -rf %s_SECOND-CLEAN_SPW%i.*'%(MSNAME[:-3],i))
			MASK = '%s.mask'%MSNAME[:-3]
			tclean(vis='%s_ExtendedData'%MSNAME,
					imagename='%s_SECOND-CLEAN_SPW%i'%(MSNAME[:-3],i),
					cell=CELL, imsize=IMSIZE,spw=str(i),
					stokes='I', specmode='mfs',nterms=1,
					mask=MASK, savemodel='modelcolumn',
					niter=1500,gain=0.025,
					cyclefactor=1.0, interactive=False)

			############
			# Generate new CC model of minispiral:
			ia.open('%s_SECOND-CLEAN_SPW%i.model'%(MSNAME[:-3],i))
			FINAL_MINISP += ia.getchunk()[:,:,0,0]
			summ = ia.summary()
			DX = summ['incr'][0]*units[summ['axisunits'][0]]
			DY = summ['incr'][1]*units[summ['axisunits'][1]]
			ia.close()
		
		# Average minispiral image (normalized)
		FINAL_MINISP /= np.sum(FINAL_MINISP)
		
		# Write it to an ascii file:
		off = open('%s_CCs.dat'%MSNAME[:-3],'w')
		mask = np.where(FINAL_MINISP!=0.0)
		for i in range(len(mask[0])):
			Xoff = DX*(mask[0][i] - summ['refpix'][0])*3600. # in arcsec.
			Yoff = DY*(mask[1][i] - summ['refpix'][1])*3600. # in arcsec.
			S = FINAL_MINISP[ mask[0][i],mask[1][i] ]
			print >> off, '%.4e  %.4e  %.4e'%(Xoff, Yoff, S)
		off.close()
		############
		
		# Apply calibration to the original data:
		#applycal(vis=MSNAME,spwmap=[0,0,0,0],applymode='calonly',gaintable='%s_SELFCAL.tab'%MSNAME[:-3])


