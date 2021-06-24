"""Contains cosmic star formation information"""

import numpy as np
import astropy.units as u


def md_14(z):
    """The Madau & Dickinson (2014) star formation rate
    per comoving volume as a function of redshift

    Parameters
    ----------
    z : `float or numpy.array`
        redshift

    Returns
    -------
    sfr : `float or numpy.array`
        star formation rate per comoving volume at redshift z
        with astropy units
    """
    sfr = (0.015 * (1 + z)**2.7 / (1 + ((1 + z) / (1 + 1.9))**5.6) * u.Msun * u.Mpc**(-3) * u.yr**(-1))

    return sfr


def mf_17(z):
    """The Madau & Fragos (2017) star formation rate
    per comoving volume as a function of redshift

    Parameters
    ----------
    z : `float or numpy.array`
        redshift

    Returns
    -------
    sfr : `float or numpy.array`
        star formation rate per comoving volume at redshift z
        with astropy units
    """

    sfr = 0.01 * (1 + z)**2.6 / (1 + ((1 + z) / 3.2)**6.2) * u.Msun/(u.Mpc**3 * u.yr)

    return sfr


def mean_metal_log_z(z, Zsun=0.017):
    """
    Mass-weighted average log(metallicity) as a function of redshift
    From Madau & Fragos (2017)

    Parameters
    ----------
    z : `float or numpy.array`
        redshift

    Zsun : `float or numpy.array`
        metallicity of the sun
        NOTE: Madau & Fragos assume Zsun = 0.017

    Returns
    -------
    log_Z : `float or numpy.array`
        log(mean metallicity)
    """

    log_Z_Zsun = 0.153 - 0.074 * z ** 1.34
    log_Z = np.log(10 ** log_Z_Zsun * Zsun)

    return log_Z


def log_p_Z_z(Z, z, sigma_logZ):
    """Computes the metallicity and redshift log probability
    distribution function assuming a log normal metallicity
    distribution with sigma at each redshift

    Parameters
    ----------
    Z : `numpy.array`
        metallicities

    z : `numpy.array`
        redshifts

    sigma_logZ : `numpy.array`
        standard deviation of metallicity

    Returns
    -------
    log_pi : `numpy.array`
        log probability of the metallicity/redshift distribution at Z,z
    """
    mu = mean_metal_log_z(z)
    sigma = np.ones_like(z) * sigma_logZ

    return -np.log(Z) - np.log(sigma) - 0.5 * np.square((np.log(Z) - mu) / sigma)

