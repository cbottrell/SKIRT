#import texfig as tf
import illustris_python as il
import numpy as np
import matplotlib as mpl
import h5py
import scipy.integrate as integrate
import astropy as ap
from astropy.cosmology import Planck15 as cosmo

#convert units, depending on the input units
def convertmass(x):
    a = x * 1e10/(cosmo.H0.value/100)
    return a

def convertlength(x):
    a = x /(cosmo.H0.value/100)
    return a

def convertdensity(x):
    a = x * 1e10 * (cosmo.H0.value/100)**2
    return a

#calculate age in years from cosmology using astropy, a1 is array of birth scale factors, a2 is scale factor today
def age(a1,a2):
    age = (cosmo.age((1/a2)-1).value-cosmo.age((1/a1)-1).value)*1e9
    return(age)

#calculate temperature of gas
def temp(u,xe):
    #molecular weight in cgs units
    mu = 4/(1+3*0.76+4*0.76*xe)*ap.constants.m_p.cgs
    #temperature from k_B in cgs units, internal energy in cgs units and molecular weight in cgs
    T = (5/3-1)*u/ap.constants.k_B.cgs.value*1e10*mu.value
    return T

#make periodic fix, which returns fixed alues, and boolean, true if the data was fixed, false if not                                                              
# #This is hard coded for TNG50
# def periodicfix(x,center,boxsize = 35000):    
#     if (np.min(x) < boxsize/10) & (np.max(x) > boxsize - boxsize/10):
#         x = x + boxsize/2
#         for j in range(3):
#             for i in range(len(x)):
#                 if x[i,j] > boxsize:
#                     x[i,j] = x[i,j]- boxsize
#         x = x - boxsize/2

#         center = center + boxsize/2
#         for i in range(3):
#             if center[i] > boxsize:
#                 center[i] = center[i] - boxsize
#         center = center - boxsize/2
#     return(x,center)

def periodicfix(coords,center,boxSize):
    '''Periodic coordinate [coords] fix relative to some point of origin [center] for a box of size [boxSize]. New coordinate system origin is [center].'''
    rel_coords = coords-center
    rel_coords[rel_coords>boxSize/2] -= boxSize
    rel_coords[rel_coords<-boxSize/2] += boxSize
    return rel_coords


def getCentreOfVelocity(vel,mass):
    """Calculate mass weighted average velocity"""
    mtot = np.sum(mass)
    cvx = np.dot(vel[:,0],mass)/mtot
    cvy = np.dot(vel[:,1],mass)/mtot
    cvz = np.dot(vel[:,2],mass)/mtot
    return np.asarray([cvx,cvy,cvz])


def DTM(Zc):
    """
    Return the dust fracion of the all metals (Mdust / (Mdust + Mmetal)
    Based on Remy-Ruyer 2014, variable XCO
    """
    logOH  = np.log10(Zc)  + 8.69
    logGTD = 2.21 + (8.69 - logOH)
    logGTD[logOH < 8.1] = 0.96 + 3.1 * (8.69 - logOH[logOH < 8.1])
    GTD = 10**logGTD
    DTG = 1./GTD
    DTM = DTG / (Zc * 0.014)
    #Convert to density fraction rather than Mdust/Mmetal
    DTM = DTM/(1. + DTM)
    return DTM

