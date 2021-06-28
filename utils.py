"""contains utility methods"""

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18_arXiv_v2 as Planck18
from scipy.interpolate import interp1d
from astropy import units as u


def get_z_interp(z_max):
    """Generates an interpolation to convert between lookback
    time and redshift

    Parameters
    ----------
    z_max : `float`
        maximum redshift for interpolator

    Returns
    -------
    z_interp : `scipy.interpolate interpolation`
        one dimension interpolator for redshift
    """
    z = np.expm1(np.linspace(0, np.log1p(z_max), 1000))
    tlb = Planck18.lookback_time(z).to(u.Myr).value
    z_interp = interp1d(tlb, z)
    return z_interp


def get_met_bins(mets):
    """
    Returns the bin edges for metallicity bins centered on
    supplied metallicities assuming a log_10 spaced metallicity grid

    Parameters
    ----------
    mets : `numpy.array`
        centers of metallicity bins

    Returns
    -------
    met_bins : `numpy.array`
        metallicity bin edges
    """
    bw = np.mean(np.log10(mets[1:]) - np.log10(mets[:-1])) / 2
    met_bins = 10 ** (np.arange(np.log10(min(mets)) - bw, np.log10(max(mets)) + 3 * bw, bw))

    return met_bins[::2]


def read_met_data(path, kstar_1, kstar_2, metallicity, SFstart=13700.0, SFduration=0.0):
    """
    Reads in all COSMIC data for specified metallicity grid

    Parameters
    ----------
    path : `string`
        path to COSMIC data where the path structure
        should be '{path}/dat_kstar1...'
    
    kstar_1 : `string`
        kstar for the primary following COSMIC notation
        
    kstar_2 : `string`
        kstar for the secondary following COSMIC notation

    mets : `numpy.array`
        metallicity grid for COSMIC data
        
    SFstart : `float`
        ZAMS lookback time for population
    
    SFduration : `float`
        Duration of star formation

    Returns
    -------
    BBH : `numpy.array`
        Data containing compact object binaries
        
    N_stars : `numpy.array`
        Total number of stars formed including singles to produce
        the data for each metallicity bin

    M_stars : `numpy.array`
        Total amount of stars formed in Msun to produce
        the data for each metallicity bin
    """
    metallicity = np.round(metallicity, 8)
    f = '{}/dat_kstar1_{}_kstar2_{}_SFstart_{}_SFduration_{}_metallicity_{}.h5'.format(path, 
                                                                                        kstar_1,
                                                                                        kstar_2,
                                                                                        SFstart,
                                                                                        SFduration,
                                                                                        metallicity)
    N_stars = np.max(pd.read_hdf(f, key='n_stars'))[0]
    M_stars = np.max(pd.read_hdf(f, key='mass_stars'))[0]

    bpp = pd.read_hdf(f, key='bpp')
    BBH = bpp.loc[(bpp.kstar_1 == 14) & (bpp.kstar_2 == 14) & (bpp.evol_type == 3)]

    return np.array(BBH, dtype=object), N_stars, M_stars


def get_cosmic_data(path, kstar_1, kstar_2, mets, SFstart=13700.0, SFduration=0.0):
    """
    Reads in all COSMIC data for specified metallicity grid

    Parameters
    ----------
    path : `string`
        path to COSMIC data where the path structure
        should be '{path}/dat_kstar1...'
    
    kstar_1 : `string`
        kstar for the primary following COSMIC notation
        
    kstar_2 : `string`
        kstar for the secondary following COSMIC notation

    mets : `numpy.array`
        metallicity grid for COSMIC data
        
    SFstart : `float`
        ZAMS lookback time for population
    
    SFduration : `float`
        Duration of star formation

    Returns
    -------
    Ms : `numpy.array`
        Total amount of stars formed in Msun to produce
        the data for each metallicity bin

    Ns : `numpy.array`
        Total number of stars formed including singles to produce
        the data for each metallicity bin

    ns : `numpy.array`
        The number of compact object binaries per metallicity bin

    dat : `numpy.array`
        Data containing compact object binaries
    """

    Ns = []
    Ms = []
    ns = []
    dat = []
    for m in mets:
        d, N, M = read_met_data(path, kstar_1, kstar_2, m, SFstart=13700.0, SFduration=0.0)
        Ms.append(M)
        Ns.append(N)
        dat.append(d)
        ns.append(len(d))
    Ms = np.array(Ms)
    Ns = np.array(Ns)
    ns = np.array(ns)
    dat = np.array(dat)

    return Ms, Ns, ns, dat
