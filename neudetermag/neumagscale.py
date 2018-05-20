import os
import sys
import math
import glob
import json
import random
import warnings
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import optimize
from numpy.linalg import inv
from obspy import UTCDateTime
from mpl_toolkits.basemap import Basemap
from obspy.geodetics.base import locations2degrees

def leasquares(amplitudes,distances,magnitudes):
    """ numpy matrices are confusing so to get into the correct order we need to
    initially transpose, at initialization G and d are Nx2 and 1x2 matrices,
    resp. We need to solve the equation m = (GTG)^-1GTd where GT is G tranpose

    For the equation Mr = log(A) + Blog(D) + C => (B,C) = (m[0],m[1])

    :type amplitudes: array
    :param amplitudes: amplitude values for mag scale
    :type distances: array
    :param distances: epicentral distances in degrees
    :type magnitudes: array
    :param magnitudes: magnitudes in Ms or Mw
    :rtype m: float
    :param m: coefficients in magnitude equation
    """
    G = [np.log10(distances),np.ones(len(distances))]
    G = (np.asmatrix(G)).transpose()

    d_hold = []
    for i4 in range(0,len(magnitudes)):
        d_hold.append(magnitudes[i4]-np.log10(amplitudes[i4]))
    d = (np.asmatrix(d_hold)).transpose()

    GTG = np.dot(G.transpose(),G)
    GTGi = inv(GTG)
    GTGiGT = np.dot(GTGi,G.transpose())
    m = np.dot(GTGiGT,d)

    return float(m[0]),float(m[1]),GTGi,G.shape


def confidence(data,mags,dists,GTGi,nxp,m0,m1):
    """calculate the confidence interval for the least squares regression.
    an estimator for the confidence interval is given for estimator m, and
    for an n x p matrix G. We use a c value of 1.96 for 95 percent confidence.
    need to check as our sample size is quite small

            (GTG)^-1 * (sum(residuals))/(n-p)
    """
    c = 1.96
    n = nxp[0]
    p = nxp[1]
    resi, resi_sq = [],[]
    y = lambda x,B,C,Mr: 10**(Mr-B*np.log10(x)-C)

    for j in range(len(data)):
        resi_sq.append((np.log10(data[j])-
                            np.log10(y(dists[j],m0,m1,mags[j])))**2)
        resi.append((np.log10(data[j])-np.log10(y(dists[j],m0,m1,mags[j]))))

    noise_varia = (1/(n-p)) * sum(resi_sq)
    sum_resi_sq = sum(resi_sq)
    # print('Sum of squared residuals: ',sum_resi_sq)
    var_b1 = GTGi.item(0) * noise_varia
    var_b2 = GTGi.item(3) * noise_varia

    m0_nci = m0 - c*np.sqrt(var_b1)
    m0_pci = m0 + c*np.sqrt(var_b1)
    m1_nci = m1 - c*np.sqrt(var_b2)
    m1_pci = m1 + c*np.sqrt(var_b2)

    return m0_nci,m0_pci,m1_nci,m1_pci,resi,sum_resi_sq
    
def run_through():
    npzlistpath = '/Users/chowbr/Documents/magscale_paper/neudetermag/WETlist.npz'
    npzlist = np.load(npzlistpath)
    starttimes = npzlist['starttimes']
    magnitudes = npzlist['magnitudes']
    amplitudes = npzlist['amplitudes']
    distances = npzlist['distances']
    
    amplitudes *= 1E6
    import ipdb;ipdb.set_trace()
    m0,m1,GTGi,nxp = leasquares(amplitudes,distances,magnitudes)
    print(m0,m1)
    
if __name__ == '__main__':
    run_through()

    
    
    
    
    
    