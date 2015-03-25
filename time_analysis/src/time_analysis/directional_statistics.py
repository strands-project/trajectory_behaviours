#!/usr/bin/env python

import numpy as np

def circ_mean(samples, low=0, high=2*np.pi):
    """Compute the circular mean for samples assumed to be in the range [low to high]
    """
    ang = (np.array(samples) - low)*2*np.pi / (high-low)
    mu = np.angle(np.mean(np.exp(1j*ang)))
    if (mu < 0):
        mu = mu + 2*np.pi
    return mu * (high-low)/(2.0*np.pi) + low

def weighted_circ_mean(samples, weights, low=0, high=2*np.pi):
    ang = np.array(samples - low)*2*np.pi / (high-low)
    sumcos, sumsin = 0, 0
    for i in range(len(ang)):
        sumcos += np.cos(ang(i)) * weights(i)
        sumsin += np.sin(ang(i)) * weights(i)
    mu = np.arctan2(sumsin/sumcos)
    if (mu < 0):
        mu = mu + 2*np.pi
    return mu * (high-low)/(2.0*np.pi) + low
    
def circ_std(samples, low=0, high=2*np.pi):
    """Compute the circular standard deviation for samples assumed to be in the range [low to high]
    """
    ang = (np.array(samples) - low)*2*np.pi / (high-low)
    R = np.mean(np.exp(1j*ang))
    V = 1-abs(R)
    return np.sqrt(V) * (high-low)/(2.0*np.pi)
    
def circ_var(samples, low=0, high=2*np.pi):
    """Compute the circular variance for samples assumed to be in the range [low to high]
    """
    ang = (np.array(samples) - low)*2*np.pi / (high-low)
    R = np.mean(np.exp(1j*ang))
    V = 1-abs(R)
    return V * (high-low)/(2.0*np.pi)

def kappa(samples, low=0, high=2*np.pi):
    
#    mu = circ_mean(samples, low, high)
#    R = np.mean(np.cos(np.array(samples)-mu))
    ang = (np.array(samples) - low)*2*np.pi / (high-low)
    R = abs(np.mean(np.exp(1j*ang)))
    
    # implementation of A1 inverse function on estimated R: see pag.85-86 of Mardia's directional statistics
    if R >= 0 and R < 0.53:
        k = 2 * R + R**3 + (5 * R**5)/6
    elif R < 0.85:
        k = -0.4 + 1.39 * R + 0.43/(1 - R)
    else:
        k = 1/(R**3 - 4 * R**2 + 3 * R)

    return k
