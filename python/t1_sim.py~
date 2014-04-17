import numpy as np
from scipy.optimize import leastsq
import numpy.random as rand
def sim_SPGRs_single(R1, noiseLevel, CoilGain=1000, PD=1, B1=1,
                     flipAngles=np.array([4,10,20,30]).astype(float), tr=20):
    M0=PD*CoilGain
    
    if np.shape(CoilGain):    # If CoilGain has more than one entry, it has a shape
        nVoxels = len(CoilGain)
    else:
        nVoxels = 1
    
    Sig = np.zeros((nVoxels,len(flipAngles)))   # Sig Noise free
    SigN = np.zeros((nVoxels,len(flipAngles)))   # Sig plus gaussian noise
    SNR  = np.zeros((1,len(flipAngles)))         # Save out the SNR, just one coil
    
    #Simulate:
    # the SPGR equation
    for jj in np.arange(len(flipAngles)):    # for jj in an array with flipAngle indices
        #the effective flip Angles is also depend on B1
        fa=flipAngles[jj]*B1
        fa = fa/180.*np.pi
        
        Sig[:,jj] =M0*(1-np.exp(-tr*R1))*np.sin(fa)/(1-np.exp(-tr*R1)*np.cos(fa))    
        
        SNR = 20*np.log10(np.mean(Sig[:,jj])/noiseLevel)  #Coil SNR
        # Simulate M0 plus noise: M0SN= M0S + Noise
        SigN[:,jj]=  Sig[:,jj] + rand.randn(nVoxels,1)*noiseLevel
    
    x_fit = np.zeros((nVoxels, 2))   #Number of voxels x number of parameters (M0 & t1)
    resnorm = np.zeros(nVoxels)
    for ii in np.arange(nVoxels):
        initial = np.zeros(2) #One for M0 and R1
        t1t = linRelaxFitT1(SigN[ii],flipAngles,tr,B1,nVoxels)
        M0t = np.mean(SigN[ii]/((1-np.exp(-tr/t1t))*np.sin((flipAngles/180*np.pi*B1))/
                             (1-np.exp(-tr/t1t)*np.cos((flipAngles/180*np.pi*B1)))))
        
        #Fit
        fa = ((flipAngles*B1)/180.)*np.pi    
        initial[0]=t1t
        initial[1]=M0t
        [x_fit[ii], resnorm[ii]] = leastsq(errT1PD, initial,
                                           args=(fa,tr,np.squeeze(SigN),B1), xtol=1e-12)
    
    M0Fit = x_fit[:, 0]
    R1Fit  = 1./x_fit[:, 1]
    
    outSim = outputSim(M0, M0Fit, SNR, PD, R1, R1Fit, SigN, Sig, tr, flipAngles)
    return outSim
    
def linRelaxFitT1(SigN, flipAngles, tr, B1, nVoxels):
    
    #Linear for one voxel
    y = np.squeeze(SigN)/np.sin((flipAngles/180.)*np.pi*B1).T
    x = np.squeeze(SigN)/np.tan((flipAngles/180.)*np.pi*B1).T
    test = np.polyfit(x, y, 1)
    slope = test[0]
    t1t = np.squeeze(np.array(abs(-tr/np.log(slope))))
    
    if nVoxels == 1:
        t1t = t1t[..., None] #Add an extra dimension for indexing purposes
    #fit each voxel with lsq non linear fit on the full single equation
    
    return t1t
    
def errT1PD(x, fa, tr, S, B1):
    # Estimate the fit of x(1) PD and x(2) T1 to fit the SPGR T1 data with
    # different flip angles.
    
    #Input:
    # x - the fitted parameters
    # flipAngles - the scans' flip angles 
    # tr - the scans' tr
    # S - the mesured SPGRs images
    # Gain - the coil gain (can be also set to be one and fitted later
    # B1 - the exsite inhomogenity (the error in nominal flipangles
    # lsq - the kind of error calculation
    # SD - a way to normalize the error by the data std.
    
    #OutPut:
    # err -the error between the estimation and the data.
    M0 = x[0]
    t1 = x[1]
    S_fit = M0*(1-np.exp(-tr/t1))*np.sin(fa)/(1-np.exp(-tr/t1)*np.cos(fa))
    err=np.sqrt(abs((S-S_fit))) #let fit the median and not the mean that will give less weight to outliers

    return err
    
class outputSim:
    def __init__(self, M0, M0Fit, SNR, PD, R1, R1Fit, SigN, Sig, tr, flipAngles):
        self.M0=M0
        self.M0Fit=M0Fit
        self.SNR=SNR
        self.PD=PD
        self.R1=R1
        self.R1Fit=R1Fit
        self.SigN=SigN
        self.Sig=Sig
        self.tr=tr
        self.flipAngles=flipAngles