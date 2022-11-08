#!/usr/bin/env python
# coding: utf-8

# # Combined Implementation of our three models
# This notebook aims to test the likelihood functions and serve as unit test for any change on the model.
# Modifications: Johnny Esteves <br>
# Author: Allen Pinjic - Created on October 29th, 2022

from __future__ import print_function, division
from IPython.core.display import display, HTML
from astropy.io.fits import getdata
from astropy.table import Table
from astropy.cosmology import WMAP9 as cosmo
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy import interpolate
from multiprocessing import Pool
cosmology.setCosmology('WMAP9')

import numpy as np
import pandas as pd
import emcee
import pandas as pd
import time
import os
import sys
from scipy.optimize import minimize
import scipy.stats
import math
import matplotlib.pyplot as plt
import pylab as plt
import corner

from colossus.cosmology import cosmology
cosmology.setCosmology('WMAP9')


########### MODEL INTIALIZATION #################
##################################################

### Timer ###
from datetime import datetime
time_for_now = datetime.now()

def header():
    print('\n')
    print(10*'-----')
    print('\t Scaling Relation MCMC Fitting Code')
    print('\n')
    print('Today is:', time_for_now)
    print(10*'-----')

header()

##### Prior Setup #####
SZ_Priors = {'A_sze':[5.24, 0.85], 'B_sze':[1.534, 0.100],'C_sze':[0.465, 0.407],
             'scatter_sze':[0.161, 0.080]}

sz_theta_values = ['A_sze', 'B_sze', 'C_sze', 'scatter_sze']

## gaussian priors on lambda with 3 sigma from the true params
Lambda_Priors = {'A_lambda':[76.9, 3*8.2], 'B_lambda':[1.020, 5*0.080],'C_lambda':[0.23, 5*0.16],
             'scatter_lambda':[0.23, 3.0*0.16]}

lambda_theta_values = ['A_lambda', 'B_lambda', 'C_lambda', 'scatter_lambda']


##### MCMC Setup #########
theta_true = [5.24, 1.534, 0.465, 0.161, 76.9, 1.02, 0.29, 0.16, 0.8]
Nburnin = 1000 # number of burn-in samples
Nsamples = 50000 # number of final posterior samples (ORIGINALLY 5000)
walkers = 64 #(ORIGINALLY 32)
ndims = len(theta_true)
guess = (np.array(theta_true)[:, np.newaxis]*(1.+0.01*np.random.normal(size=(ndims,walkers)))).T

### Parameters ############
run_mcmc = True
quick_fit = True
debug = False

is_real_data = False

very_simple_model = False
simple_model = True
spt_model = False

###############################################################################
###################### Model-Specific Functions ###############################
###############################################################################

if very_simple_model:
    ### Likelihood Functions ###
    def log_likelihood(theta, indices, eps=1e-9):
        probs = []
        for ix in indices:
            probs.append(log_likelihood_additional(theta, ix))

        p = np.array(probs)
        log_p = np.log(p)
        log_p = np.where(np.isnan(log_p), -np.inf, log_p)
        return np.sum(log_p)
    
    def log_likelihood_additional(theta):
        # unfolding theta
        A_lambda, B_lambda, C_lambda, scatter_lambda = theta[4:8]
        A_sze, B_sze, C_sze, scatter_sze = theta[:4]
        rho = theta[-1]
            
        # calling predictions;
        ln_lbd_pred = ln_lbd_given_M([A_lambda, B_lambda, C_lambda, scatter_lambda], mass, redshift)
        ln_zeta_pred= ln_zeta_given_M([A_sze, B_sze, C_sze, scatter_sze], mass, redshift)
    
            
        # logNormal Likelihood
        lp_lbd_zeta = compute_log_pLbdZeta(_lambda, zeta,
                                       scatter_lambda, scatter_sze, rho,
                                       ln_lbd_pred, ln_zeta_pred)

        if debug:
            print("This is ln_lbd_pred:", ln_lbd_pred)
            print("This is ln_zeta_pred:", ln_zeta_pred)        
            print("This is lp_lbd_zeta:", lp_lbd_zeta)
            print("This is np.nansum(lp_lbd_zeta) or final result:", np.nansum(lp_lbd_zeta))
            
        return np.nansum(lp_lbd_zeta)
    
    ##### Prior Functions #####
    def set_prior_sze(theta_values):
        lp = 0.
        rhomin = 0.
        rhomax = 1.
        
        for i, prior_name in enumerate(['A_sze', 'B_sze', 'C_sze', 'scatter_sze']):
            mean, error = SZ_Priors[prior_name]
            param = theta_values[i]
            result = set_gaussian_prior(param, mean, error)
            lp += np.where(np.abs(result)>9., -np.inf, result)
            # outside a range of six sigmas (six standard deviations)
        # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
        lp = 0. if (theta_values[-1] > 0) else -np.inf
        return lp
    
    def set_prior_lambda(theta_values):
        lp = 0.
        rhomin = 0.
        rhomax = 1.
        
        for i, prior_name in enumerate(['A_lambda', 'B_lambda', 'C_lambda', 'scatter_lambda']):
            mean, error = Lambda_Priors[prior_name]
            param = theta_values[i]
            result = set_gaussian_prior(param, mean, error)
            lp += np.where(np.abs(result)>9., -np.inf, result)
            # outside a range of six sigmas (six standard deviations)
       
        # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
        lp = 0. if (theta_values[-1] > 0) else -np.inf
        return lp
    
    ########## ONLY VERY_SIMPLE_MODEL RESETS LP to 0 ###########
    def logprior(theta):
        lp = 0
    
        A_lambda, B_lambda, C_lambda, scatter_lambda = theta[4:8]
        A_sze, B_sze, C_sze, scatter_sze = theta[:4]
        rho = theta[-1]
    
        lp_lambda = set_prior_lambda([A_lambda, B_lambda, C_lambda, scatter_lambda])
        lp_sze = set_prior_sze([A_sze, B_sze, C_sze, scatter_sze])
    
        lp = 0. if ((rho > -1.) and (rho < 1)) else -np.inf
        return lp + lp_lambda + lp_sze
    
    ##### Posterior Functions #####
    def logposterior(theta):
        lp = logprior(theta)
        if not np.isfinite(lp):
            return -np.inf, -np.inf
        ll = log_likelihood_additional(theta)# + lp
        if not np.isfinite(ll):
            return lp, -np.inf
        return lp + ll, lp
    
    
elif simple_model:
    ### Likelihood Functions ###
    def log_likelihood(theta, indices, eps=1e-9):
        probs = []
        for ix in indices:
            probs.append(log_likelihood_additional(theta, ix))
        
        # missing a normalization factor
        p = np.array(probs)
        log_p = np.log(p)
        log_p = np.where(np.isnan(log_p), -np.inf, log_p)
        return np.sum(log_p)
    
    
    def log_likelihood_additional(theta, ix):
        # unfolding theta
        A_lambda, B_lambda, C_lambda, scatter_lambda = theta[4:8]
        A_sze, B_sze, C_sze, scatter_sze = theta[:4]
        rho = theta[-1]
    
        # calling the indepedent variables
        # there are defined as global variables
        mass_i = mass[ix]
        redshift_i = redshift[ix]
        p_chisi = prob_chisi_vec[ix]
        p_lbd_hat = prob_lbd_hat_vec[ix]
        
        # take the indices with non zero values for P_chisi, P_lbd_hat
        # this approx. makes the code to run faster by a factor 10
        llo, lup = list(lbd_indices_vec[ix])

        #llo, lup = 0, len(lbdvec)
        clo, cup = list(zeta_indices_vec[ix])

        # calling predictions;
        ln_lbd_pred = ln_lbd_given_M([A_lambda, B_lambda, C_lambda, scatter_lambda], mass_i, redshift_i)
        ln_zeta_pred= ln_zeta_given_M([A_sze, B_sze, C_sze, scatter_sze], mass_i, redshift_i)
        
        # the logNormal Distribution
        lp_lbd_zeta = compute_log_pLbdZeta(ll[clo:cup,llo:lup], zz[clo:cup,llo:lup],
                                      scatter_lambda, scatter_sze, rho,
                                      ln_lbd_pred, ln_zeta_pred)
    
        p_lbd_zeta = np.exp(lp_lbd_zeta)
    
        # integrate over zeta
        p_chisi = np.tile(p_chisi[clo:cup], (int(lup-llo), 1)).T
        p_lbd = np.trapz(p_lbd_zeta*p_chisi, x=zetavec[clo:cup], axis=0)
    
        # integrate over lambda
        norm = np.trapz(p_lbd, x=lbdvec[llo:lup])
        p = np.trapz(p_lbd*p_lbd_hat[llo:lup], x=lbdvec[llo:lup], axis=0)
        
        if debug:
            print("This is mass_i:", mass_i)
            print("This is redshift_i:", redshift_i)
            print("This is p_chisi:", p_chisi)
            print("This is p_lbd_hat:", p_lbd_hat)
            print("This is llo, lup:", llo, lup)  
            print("This is clo, cup:", clo, cup)
            print("This is ln_lbd_pred:", ln_lbd_pred)
            print("This is ln_zeta_pred:", ln_zeta_pred)
            print("This is lp_lbd_zeta:", lp_lbd_zeta)
            print("This is p_lbd_zeta:", p_lbd_zeta)
            print("This is p_chisi after integration over zeta:", p_chisi)
            print("This is p_lbd after integration over zeta:", p_lbd)
            print("This is the normal after integration over lambda:", norm)
            print("This is the probability after intergration over lambda:", p)
            print("This is the final result:", p/norm)

        return p/norm
    
    ##### Prior Functions #####
    def set_prior_sze(theta_values):
        lp = 0.
        rhomin = 0.
        rhomax = 1.
    
        for i, prior_name in enumerate(['A_sze', 'B_sze', 'C_sze', 'scatter_sze']):
            mean, error = SZ_Priors[prior_name]
            param = theta_values[i]
            result = set_gaussian_prior(param, mean, error)
            lp += np.where(np.abs(result)>9., -np.inf, result)
            # outside a range of six sigmas (six standard deviations)
    
        lp += set_uniform_prior(theta_values[-1], scatter_lower_bound, scatter_upper_bound)
        lp += 0. if (theta_values[-1] > 0) else -np.inf
        return lp
    
    def set_prior_lambda(theta_values):
        lp = 0.
        rhomin = -1.
        rhomax = 1.
        
        for i, prior_name in enumerate(['A_lambda', 'B_lambda', 'C_lambda', 'scatter_lambda']):
            mean, error = Lambda_Priors[prior_name]
            param = theta_values[i]
            result = set_gaussian_prior(param, mean, error)
            lp += np.where(np.abs(result)>9., -np.inf, result)
            # outside a range of six sigmas (six standard deviations)
       
        lp += set_uniform_prior(theta_values[-1], scatter_lower_bound, scatter_upper_bound)
         # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
        lp += 0. if (theta_values[-1] > 0) else -np.inf
        return lp
    

    def logprior(theta):
        A_lambda, B_lambda, C_lambda, scatter_lambda = theta[4:8]
        A_sze, B_sze, C_sze, scatter_sze = theta[:4]
        rho = theta[-1]
        
        lp_lambda = set_prior_lambda([A_lambda, B_lambda, C_lambda, scatter_lambda])
        lp_sze = set_prior_sze([A_sze, B_sze, C_sze, scatter_sze])
        
        lp = set_uniform_prior(rho, -0.78, 1)
        return lp + lp_lambda + lp_sze
    
    #### Posterior Functions ####
    def logposterior(theta, indices):
        lp = logprior(theta)
        if not np.isfinite(lp):
            return -np.inf, -np.inf
        ll = log_likelihood(theta, indices)# + lp
        if not np.isfinite(ll):
            return lp, -np.inf
        return lp + ll, lp
    
elif spt_model:
    ### Likelihood Functions ###
    def log_likelihood(theta, indices, eps=1e-9):
        probs = []
        for ix in indices:
            probs.append(log_likelihood_additional(theta, ix))
        p = np.array(probs)#/np.sum(probs)
        log_p = np.log(p)
        log_p = np.where(np.isnan(log_p), -np.inf, log_p)
        return np.sum(log_p)
    
    def log_likelihood_additional(theta, ix):
        # unfolding theta
        A_lambda, B_lambda, C_lambda, scatter_lambda = theta[4:8]
        A_sze, B_sze, C_sze, scatter_sze = theta[:4]
        rho = theta[-1]
    
        # forgot the mass
        redshift_i = redshift[ix]
        p_chisi = prob_chisi_vec[ix]
        p_lbd_hat = prob_lbd_hat_vec[ix]
        #p_lbd_hat_conv = prob_lbd_hat_conv[ix]
        llo, lup = list(lbd_indices_vec[ix])
        #llo, lup = 0, len(lbdvec)#list(lbd_indices_vec[ix])
        clo, cup = list(zeta_indices_vec[ix])
    
        # calling predictions;
        ln_lbd_pred = ln_lbd_given_M([A_lambda, B_lambda, C_lambda, scatter_lambda], mvec, redshift_i)
        ln_zeta_pred= ln_zeta_given_M([A_sze, B_sze, C_sze, scatter_sze], mvec, redshift_i)
        halo_mass_func = halo_mass_function2(redshift_i)
    
        ln_lbd_pred = ln_lbd_pred[:,np.newaxis,np.newaxis]
        ln_zeta_pred= ln_zeta_pred[:,np.newaxis,np.newaxis]
        hmf = halo_mass_func[:,np.newaxis,np.newaxis]
    
        # the logNormal Distribution
        lp_lbd_zeta = compute_log_pLbdZeta(ll[:,clo:cup,llo:lup], zz[:,clo:cup,llo:lup],
                                      scatter_lambda, scatter_sze, rho,
                                      ln_lbd_pred, ln_zeta_pred)
        dN_lbd_zeta = np.exp(lp_lbd_zeta)
    
        # integrate over M
        #norm = float(norm_function(redshift_i))
        p_lbd_zeta = np.trapz(dN_lbd_zeta*hmf, x=mvec, axis=0)#/norm

        # integrate over zeta
        p_chisi = np.tile(p_chisi[clo:cup], (int(lup-llo), 1)).T
        p_lbd = np.trapz(p_lbd_zeta*p_chisi, x=zetavec[clo:cup], axis=0)
    
        # integrate over lambda
        norm = np.trapz(p_lbd, x=lbdvec[llo:lup])
        p = np.trapz(p_lbd*p_lbd_hat[llo:lup], x=lbdvec[llo:lup], axis=0)
        return p/norm
    
    def _halo_mass_function(M, z):
        return mass_function.massFunction(M, z, mdef = '500c', model = 'bocquet16')
    
    halo_mass_function = np.vectorize(_halo_mass_function)
    
    #### Prior Functions ####
    def set_prior_sze(theta_values):
        lp = 0.
        rhomin = 0.
        rhomax = 1.
        
        for i, prior_name in enumerate(['A_sze', 'B_sze', 'C_sze', 'scatter_sze']):
            mean, error = SZ_Priors[prior_name]
            param = theta_values[i]
            result = set_gaussian_prior(param, mean, error)
            lp += np.where(np.abs(result)>9., -np.inf, result)
            # outside a range of six sigmas (six standard deviations)
    
        lp += set_uniform_prior(theta_values[-1], scatter_lower_bound, scatter_upper_bound)
        lp += 0. if (theta_values[-1] > 0) else -np.inf
        return lp
    
    def set_prior_lambda(theta_values):
        lp = 0.
        rhomin = -1.
        rhomax = 1.
        
        for i, prior_name in enumerate(['A_lambda', 'B_lambda', 'C_lambda', 'scatter_lambda']):
            mean, error = Lambda_Priors[prior_name]
            param = theta_values[i]
            result = set_gaussian_prior(param, mean, error)
            lp += np.where(np.abs(result)>9., -np.inf, result)
            # outside a range of six sigmas (six standard deviations)
       
        lp += set_uniform_prior(theta_values[-1], scatter_lower_bound, scatter_upper_bound)
        lp += 0. if (theta_values[-1] > 0) else -np.inf
        # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
        return lp
    
    def logprior(theta):
        A_lambda, B_lambda, C_lambda, scatter_lambda = theta[4:8]
        A_sze, B_sze, C_sze, scatter_sze = theta[:4]
        rho = theta[-1]
    
        lp_lambda = set_prior_lambda([A_lambda, B_lambda, C_lambda, scatter_lambda])
        lp_sze = set_prior_sze([A_sze, B_sze, C_sze, scatter_sze])
    
        lp = set_uniform_prior(rho, -0.78, 1)
        return lp + lp_lambda + lp_sze
    
    #### Posterior Functions ####
    def logposterior(theta, indices):
        lp = logprior(theta)
        if not np.isfinite(lp):
            return -np.inf, -np.inf
    
        ll = log_likelihood(theta, indices)# + lp
        if not np.isfinite(ll):
            return lp, -np.inf
        return lp + ll, lp
    

###############################################################################
############################## Shared Functions ###############################
###############################################################################

# FOR SIMPLE MODEL & SPT MODEL ONLY: 
# For now both the sz and lambda scatters have the same lower and upper bounds
scatter_lower_bound = 0.02
scatter_upper_bound = 0.50

def set_uniform_prior(param, lowerBound, upperBound):
    return np.where((param > lowerBound) & (param < upperBound), param, np.inf)

def set_gaussian_prior(param, mu, sigma):
    return -0.5*((param - mu)/(sigma))**2

def ln_zeta_given_M(theta_sze,M,z):
    A_sze, B_sze, C_sze, scatter_sze = theta_sze
    return (np.log(A_sze) + (B_sze)*np.log(M/M0) + (C_sze)*(np.log(E(z)/Ez0)))

def ln_lbd_given_M(theta_lambda,M,z):
    A_lambda, B_lambda, C_lambda, scatter_lambda = theta_lambda
    return (np.log(A_lambda) + (B_lambda)*np.log(M/M0) + (C_lambda)*(np.log(E(z)/Ez0)))

def logNormal_variance(mu,std):
    return (np.exp(std**2)-1)*np.exp(2*mu+std**2)

def E(z):
    # The Hubble constant at the value of z
    Hz = cosmo.H(z).value
    # The Hubble constant at z=0
    H0 = cosmo.H(0).value
    return (Hz/H0)

def gaussian(x,mu,std):
    return np.exp(-(x-mu)**2/std**2/2.)/np.sqrt(2*np.pi*std**2)

def prob_chisi(zeta, xi):
    res = np.exp(-(xi-np.sqrt(zeta**2+2))**2/2.)/np.sqrt(2*np.pi)
    return res

def prob_lbd_hat(x, mean , std):
    res = gaussian(x, mean , std)
    return res

def prob_mass(zeta, mass, z, params):
    params = A_sze, B_sze, C_sze, scatter_sze
    ln_zeta_pred = ln_zeta_given_M([A_sze, B_sze, C_sze, scatter_sze], mass, z)
    ln_prob = -0.5*((np.log(zeta)-ln_zeta_pred)/scatter_sze)**2
    return ln_prob
    
def slice_array(y,alpha=1e-2):
    cy = np.cumsum(y/np.sum(y),axis=0)
    ilo,iup = np.interp([alpha,1-alpha],cy,np.arange(len(y))).astype(int)+(0,1)
    return ilo, iup

def compute_log_pLbdZeta(Lambda, Zeta, scatter_lambda, scatter_sze, rho,
                         ln_lbd_pred, ln_zeta_pred, eps = 1e-9):
    # converting std to normal distribution
    s_zeta = scatter_sze
    s_lambda = scatter_lambda
    s_lambda_inv = np.where(s_lambda<=eps, np.inf, 1/s_lambda)
    s_zeta_inv = np.where(s_zeta<=eps, np.inf, 1/s_zeta)
    
    # avoid error messages
    rho2 = (1-rho**2)
    rho_inv = np.where(rho2<=eps, np.inf, 1/rho2)
    cov2 = (s_lambda)**(2)*(s_zeta)**(2)*rho2
    
    additional_cov = (-0.5)*np.log(np.pi*cov2)

    lbd_std = (np.log(Lambda) - ln_lbd_pred)*s_lambda_inv
    zeta_std = (np.log(Zeta) - ln_zeta_pred)*s_zeta_inv

    np.seterr(invalid='ignore')

    # lbd_likelihood
    lp_lbd  = (-rho_inv*lbd_std**2)/2.

    # zeta likelihood
    lp_zeta = (-rho_inv*zeta_std**2)/2.

    # corr likelihod
    lp_corr = rho*rho_inv*lbd_std*zeta_std
    
    # total likelihood
    lp_total_m = lp_lbd + lp_zeta + lp_corr + additional_cov
    
    if debug:
        print("This is s_zeta:", s_zeta)
        print("This is s_lambda:", s_lambda)
        print("This is s_lambda_inv:", s_lambda_inv)
        print("This is s_zeta_inv:", s_zeta_inv)
        print("This is rho2:", rho2)
        print("This is rho_inv:", rho_inv)
        print("This is cov2:", cov2)
        print("This is additional_cov:", additional_cov)
        print("This is lbd_std:", lbd_std)        
        print("This is lbd_std:", lbd_std)
        print("This is lp_lbd:", lp_lbd)    
        print("This is lp_zeta:", lp_zeta)    
        print("This is lp_corr:", lp_corr)
        print("This is lp_total_m:", lp_total_m)

    return lp_total_m

###############################################################################
############ Model-Specific Grid Settings & Tests #############################
###############################################################################


### Grid Setting & Output Information
if very_simple_model:
    nCores = 32
    Nzeta = 75 # Previously 75
    Nlbd = 150
    Nmass = 100 # Previously 100
    Nz = 100
    alpha = 0.0001 #(ORIGINALLY) 0.0001
    
    runname = "combined_oct_29"
    if is_real_data:
        filename = "very_simple_model_real_data_%s.h5"%runname
    else:
        filename = "very_simple_model_fake_data_%s.h5"%runname

if simple_model:
    nCores = 32
    Nzeta = 100
    Nlbd = 200
    Nmass = 150
    Nz = 100
    alpha = 0.001
    
    runname = "combined_oct_29"
    if is_real_data:
        filename = "simple_model_real_data_%s.h5"%runname
    else:
        filename = "simple_model_fake_data_%s.h5"%runname

if spt_model:
    nCores = 32
    Nzeta = 50 # Previously 100
    Nlbd = 100 # Previously 200
    Nmass = 100 # Previously 100
    Nz = 100
    alpha = 0.001
    
    runname = "combined_oct_29"
    if is_real_data:
        filename = "spt_model_real_data_%s.h5"%runname
    else:
        filename = "spt_model_fake_data_%s.h5"%runname

print('filename:',filename)
infile = 'fake_data_Jul4.csv'

## Fake Data
if not is_real_data:
    df = pd.read_csv(infile)
    mask = (df['lambda']>2.)&(df['zeta']>0.)
    ix = np.where(mask)[0]
    Np = ix.size

    # Set Variables
    yerr = 0.05*(df['lambda'].to_numpy())
    redshift = (np.array(df['z']))[ix]
    zeta = (np.array(df['zeta']))[ix]
    sz_signal = (np.array(df['chisi']))[ix]
    _lambda = (np.array(df['lambda_wo_noise']))[ix]
    _lambda_error = (np.array(yerr))[ix]
    mass = np.array(df['M'])[ix]
    
    if simple_model or spt_model:
        indices = np.arange(Np,dtype=int)

## Real Data
if is_real_data:
    fname = '../data_set/sptecs_catalog_oct919.fits'
    data = Table(getdata(fname))
    ix = np.where(data['LAMBDA_CHISQ']>30)[0]
    Np = ix.size
    
    sz_signal = np.array(data['XI'])[ix]
    zeta = np.sqrt(sz_signal**2-3)
    _lambda = np.array(data['LAMBDA_CHISQ'])[ix]
    _lambda_error = np.array(data['LAMBDA_CHISQ_E'])[ix]
    redshift = np.array(data['REDSHIFT'])[ix]
    mass = np.array(data['M500']*1e14)[ix]
    
    if simple_model or spt_model:
        indices = np.arange(Np,dtype=int)

# global variables
M0 = 3e14
Ez0 = E(0)

if very_simple_model or simple_model:
    zvec = np.linspace(np.min(redshift), np.max(redshift), Nz)
    mvec = np.logspace(13.8, 15.2, Nmass)
    lbdvec = np.linspace(0.8*np.min(_lambda), 1.2*np.max(_lambda), Nlbd)
    zetavec = np.linspace(1.5, 1.2*np.max(zeta), Nzeta)
    prob_lbd_hat_vec = np.array([prob_lbd_hat(lbdvec, _lambda_i, _lambda_error_i)
                             for _lambda_i, _lambda_error_i in zip(_lambda, _lambda_error)])
    prob_chisi_vec = np.array([prob_chisi(zetavec, sz_signal_i) for sz_signal_i in sz_signal])
    lbd_indices_vec = np.array([slice_array(pi, alpha=alpha) for pi in prob_lbd_hat_vec])
    zeta_indices_vec = np.array([slice_array(pi, alpha=alpha) for pi in prob_chisi_vec])
    zz, ll = np.meshgrid(zetavec, lbdvec, indexing='ij')
    if very_simple_model:
        ## ONLY SIMPLE MODEL HAS A STEP CONDITION
        step = np.where(lbdvec>5.,1.,0.)

elif spt_model:
    zvec = np.linspace(np.min(redshift), np.max(redshift), Nz)
    mvec = np.logspace(13.8, 15.2, Nmass)
    lbdvec = np.linspace(0.8*np.min(_lambda), 1.2*np.max(_lambda), Nlbd)
    zetavec = np.linspace(1.5, 1.2*np.max(zeta), Nzeta)
    prob_lbd_hat_vec = np.array([prob_lbd_hat(lbdvec, _lambda_i, _lambda_error_i)
                             for _lambda_i, _lambda_error_i in zip(_lambda, _lambda_error)])
    prob_chisi_vec = np.array([prob_chisi(zetavec, sz_signal_i) for sz_signal_i in sz_signal])
    lbd_indices_vec = np.array([slice_array(pi, alpha=alpha) for pi in prob_lbd_hat_vec])
    zeta_indices_vec = np.array([slice_array(pi, alpha=alpha) for pi in prob_chisi_vec])
    mm, zz, ll = np.meshgrid(mvec, zetavec, lbdvec, indexing='ij')
    zz2, mm2 = np.meshgrid(zvec, mvec)
    halo_mass_function2 = interpolate.interp1d(zvec, halo_mass_function(mm2, zz2), kind='cubic')

    
if debug:
    theta_true = [5.24, 1.534, 0.465, 0.161, 76.9, 1.02, 0.29, 0.16, 0.8]
    indices = np.arange(len(mass))
    print('Test LogPosterior')
    print('Vector size')
    print('lambda vec',lbdvec.size)
    print('zeta vec',zetavec.size)
    print('mass vec',mvec.size) 
    print("This is prob_lbd_hat_vec shape :", prob_lbd_hat_vec.shape)
    print("This is lbd_indices_vec:", lbd_indices_vec[:3])
    print("This is zeta_indices_vec shape:", zeta_indices_vec.shape)
    guess = (np.array(theta_true)[:, np.newaxis]*(1.+0.01*np.random.normal(size=(ndims,walkers)))).T
    sel = np.arange(len(redshift))#[:100]
    sel = np.random.randint(len(redshift), size=100, dtype=int)
    argslist = [sel]
    if very_simple_model:
        print(logposterior(theta_true))
        logposterior(sel)
    elif simple_model or spt_model:
        print(logposterior(theta_true, indices))
        logposterior(theta_true, sel)


if quick_fit:
    if very_simple_model:
        start = time.time()
        np.random.seed(42)
        nll = lambda *args: -log_likelihood_additional(*args)
        initial = theta_true + 0.05 * np.random.randn(9)
        soln = minimize(nll, initial)
        end = time.time()
        vsp_time = end - start
        print("Very Simple Model took {0:.1f} seconds".format(vsp_time))
        
        albd, blbd, clbd, slbd, rho = soln.x[4:]
        
        print("Maximum likelihood estimates:")
        print("Albd = {0:.3f}".format(albd))
        print("Blbd = {0:.3f}".format(blbd))
        print("Clbd = {0:.3f}".format(clbd))
        print("Scatter_lbd = {0:.3f}".format(slbd))
        print("rho: {0:.3f}".format(rho))
        
        theta_true = [5.24, 1.534, 0.465, 0.161, 76.9, 1.02, 0.29, 0.16, 0.8]
        indices = np.arange(len(mass))
        # a quick of 10% from the truth
        initial = theta_true + 0.2 * np.random.randn(9)
        log_likelihood_additional(initial)
        ## plotting this results
        np.random.seed(42)
        lps = [-1.*log_likelihood_additional(theta_true)]
        dist = [0.]
        # increase the quick from the truth
        for i in [1.,5.,10.,15.,20.,25.,30.,60]:
            initial = theta_true + (i/100.) * np.random.randn(9)
            nDist = np.linalg.norm(np.array(theta_true)-np.array(initial))
            lps.append(-1.*log_likelihood_additional(initial))
            dist.append(nDist)
    
        print("lps:", lps)
    
        plt.scatter(dist,np.array(lps)-np.min(lps)+1.)
        plt.ylabel('LogLikelihood - Min')
        plt.xlabel(r'|$\theta-\theta_{random}$|')
        plt.yscale('log')

    elif simple_model:
        start = time.time()
        np.random.seed(42)
        nll = lambda *args: -logposterior(*args)[0]
        initial = theta_true + 0.05 * np.random.randn(9)
        soln = minimize(nll, initial, args=indices)
        end = time.time()
        sp_time = end - start
        print("Simple Model took {0:.1f} seconds".format(sp_time))
        
        albd, blbd, clbd, slbd, rho = soln.x[4:]
        
        print("Maximum likelihood estimates:")
        print("Albd = {0:.3f}".format(albd))
        print("Blbd = {0:.3f}".format(blbd))
        print("Clbd = {0:.3f}".format(clbd))
        print("Scatter_lbd = {0:.3f}".format(slbd))
        print("rho: {0:.3f}".format(rho))
        
        initial = theta_true + 0.1 * np.random.randn(9)
        logposterior(initial,indices)
            
        ## plotting this results
        t0 = time.time()
        np.random.seed(42)
        lps = [-1.*logposterior(theta_true,indices)[0]]
        dist = [0.]
        # increase the quick from the truth
        for i in [1.,2.5,5.,7.5,10.,15.,20.,25.,30.,60]:
            initial = theta_true + (i/100.) * np.random.randn(9)
            #print(1-np.array(initial)/np.array(theta_true))
            nDist = np.linalg.norm(np.array(theta_true)-np.array(initial))
            lps.append(-1.*logposterior(initial,indices)[0])
            dist.append(nDist)
        
        print("lps:", lps)
    
        plt.scatter(dist,np.array(lps)-np.min(lps)+1.)
        plt.ylabel('LogLikelihood - Min')
        plt.xlabel(r'|$\theta-\theta_{random}$|')
        plt.yscale('log')
    
    elif spt_model:
        start = time.time()
        np.random.seed(42)
        nll = lambda *args: -logposterior(*args)[0]
        initial = theta_true + 0.05 * np.random.randn(9)
        soln = minimize(nll, initial, args=indices)
        end = time.time()
        sp_time = end - start
        print("SPT took {0:.1f} seconds".format(sp_time))

        albd, blbd, clbd, slbd, rho = soln.x[4:]

        print("Maximum likelihood estimates:")
        print("Albd = {0:.3f}".format(albd))
        print("Blbd = {0:.3f}".format(blbd))
        print("Clbd = {0:.3f}".format(clbd))
        print("Scatter_lbd = {0:.3f}".format(slbd))
        print("rho: {0:.3f}".format(rho))
    
        initial = theta_true + 0.1 * np.random.randn(9)
        logposterior(initial,indices)
    
        ## plotting this results
        t0 = time.time()
        np.random.seed(42)
        lps = [-1.*logposterior(theta_true,indices)[0]]
        dist = [0.]
        # increase the quick from the truth
        for i in [1.,2.5,5.,7.5,10.,15.,20.,25.,30.,60]:
            initial = theta_true + (i/100.) * np.random.randn(9)
            #print(1-np.array(initial)/np.array(theta_true))
            nDist = np.linalg.norm(np.array(theta_true)-np.array(initial))
            lps.append(-1.*logposterior(initial,indices)[0])
            dist.append(nDist)
            
        print("lps:", lps)
    
        # plt.scatter(dist,np.array(lps)-np.min(lps)+1.)
        # plt.ylabel('LogLikelihood - Min')
        # plt.xlabel(r'|$\theta-\theta_{random}$|')
        # plt.yscale('log')
        
    

## Put the new changes here
if run_mcmc:
    if very_simple_model:
        print('Starting MCMC')
        pool = Pool(processes=nCores)              # start 64 worker processes
        sampler = emcee.EnsembleSampler(walkers, ndims, logposterior, pool=pool)
        start = time.time()
        sampler.run_mcmc(guess, Nsamples+Nburnin)
        end = time.time()
        sp_mcmc_time = end - start
        print("Very Simple Model took {0:.1f} seconds".format(sp_mcmc_time))
        #print("It is {} slower than the very simple model".format(sp_mcmc_time/vsp_mcmc_time))
        
        flat_samples = sampler.flatchain
        fig, axes = plt.subplots(ndims, figsize=(10, 7), sharex=True)
        samples = flat_samples
        for i in range(ndims):
            ax = axes[i]
            ax.plot(samples[:, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            #ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        fig.savefig('mcmc_chain_very_simple_model_real_data_1.2.png',dpi=75)
        plt.clf()
        
        fig = corner.corner(flat_samples, truths=theta_true, show_titles = True);
        fig.savefig('mcmc_corner_very_simple_model_real_data_model_1.2.png',dpi=75)
        plt.clf()
    elif simple_model or spt_model:
        print('Starting MCMC')
        pool = Pool(processes=nCores)              # start 64 worker processes
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(walkers, ndims)
        sampler = emcee.EnsembleSampler(walkers, ndims, logposterior, args=[indices], backend=backend, pool=pool)
        start = time.time()
        sampler.run_mcmc(guess, Nsamples+Nburnin)
        end = time.time()
        sp_mcmc_time = end - start
        print("SPT Model took {0:.1f} seconds".format(sp_mcmc_time))
        
        flat_samples = sampler.flatchain
        ### Changed line to be like SPT Model - could fix the previous printing issue
        ### np.save(filename[:-2], flat_samples)
        np.save(filename, flat_samples)
        # Second place where the data is saved (precaution)
        
        fig, axes = plt.subplots(ndims, figsize=(10, 7), sharex=True)
        samples = flat_samples
        for i in range(ndims):
            ax = axes[i]
            ax.plot(samples[:, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            #ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        if simple_model:
            fig.savefig('mcmc_chain_simple_model.png',dpi=75)
            plt.clf()
            fig = corner.corner(flat_samples, truths=theta_true, show_titles = True);
            fig.savefig('mcmc_corner_simple_model.png',dpi=75)
            plt.clf()
        else:
            fig.savefig('mcmc_chain_spt_model.png',dpi=75)
            plt.clf()
            fig = corner.corner(flat_samples, truths=theta_true, show_titles = True);
            fig.savefig('mcmc_corner_spt_model.png',dpi=75)
            plt.clf()