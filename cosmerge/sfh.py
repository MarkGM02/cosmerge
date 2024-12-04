"""Contains cosmic star formation information"""

import numpy as np
import astropy.units as u
from scipy.stats import norm as NormDist


def md_14(z):
    """The Madau & Dickinson (2014) star formation rate
    per comoving volume as a function of redshift

    Parameters
    ----------
    z : float or numpy.array
        redshift

    Returns
    -------
    sfr : float or numpy.array
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
    z : float or numpy.array
        redshift

    Returns
    -------
    sfr : float or numpy.array
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
    z : float or numpy.array
        redshift

    Zsun : float or numpy.array
        metallicity of the sun
        NOTE: Madau & Fragos assume Zsun = 0.017

    Returns
    -------
    log_Z : float or numpy.array
        log(mean metallicity)
    """

    log_Z_Zsun = 0.153 - 0.074 * z ** 1.34
    log_Z = np.log(10 ** log_Z_Zsun * Zsun)

    return log_Z


def log_p_Z_z(Z, z, sigma_log10Z):
    """Computes the metallicity and redshift log probability
    distribution function assuming a log normal metallicity
    distribution with sigma at each redshift

    Parameters
    ----------
    Z : numpy.array
        metallicities

    z : numpy.array
        redshifts

    sigma_log10Z : numpy.array
        standard deviation of metallicity in dex (convert to log)

    Returns
    -------
    log_pi : numpy.array
        log probability of the metallicity/redshift distribution at Z,z
    """
    mu = mean_metal_log_z(z)
    sigma = np.ones_like(z) * sigma_log10Z * np.log(10)

    return -np.log(Z) - np.log(sigma) - 0.5 * np.square((np.log(Z) - mu) / sigma)


# Below are all taken from van Son 2023 (Locations of Features...)
# Trying my best to make them flexible but I'm also prioritizing
# getting my project finished :D 
def van_son_tng(z):
    """Star formation rate given in van Son (2023), which was based on the TNG simulation. In units of mass per comoving volume as a function of redshift.
    
    Parameters
    ----------
    z : float or numpy.array
        redshift
    
    Returns
    -------
    sfr : float or numpy.array
        star formation rate per comoving volume at redshift z
        with astropy units"""
    
    sfr = 0.017 * (1 + z)**1.487 / (1 + ((1 + z) / 4.442)**5.886) * u.Msun/(u.Mpc**3 * u.yr)
    
    return sfr


def mu_z(z):
    """Redshift dependence of mean metallicity assuming no skew.
    
    Parameters
    ----------
    z: float or numpy.array
        redshift
    
    Returns
    -------
    mu : float or numpy.array
        mean metallicity at specified redshift for 0 skew distribution"""
    
    return 0.025 * 10**(-0.049 * z)


def omega_z(z):
    """Redshift dependence of scale of metallicity distribution.
    
    Parameters
    ----------
    z : float or numpy.array
        redshift
    
    Returns
    -------
    omega : float or numpy.array
        scale of metallicity distribution at specified metallicity"""
    
    return 1.129 * 10**(0.048 * z)


def mean_Z_z(z):
    """Redshift dependence of mean metallicity for skewed distribution.
    
    Parameters
    ----------
    z : float or numpy.array
        redshift
    
    Returns
    -------
    xi : float or numpy.array
        updated mean of metallicity distribution assuming a skewed
        log-normal distribution
    """
    omega = omega_z(z)
    mu = mu_z(z)
    beta = -1.79/(np.sqrt(1 + -1.79**2))
    return -omega**2 / 2 * np.log(mu / (2 * NormDist.cdf(beta * omega)))


def log_p_Z_z_skewed(Z, z):
    """The metallicity and redshift log probability distribution function. 
    Default values of constants correspond to the star formation rate given 
    in van Son (2023), which was based on the TNG simulation. 
    
    Parameters
    ----------
    Z : float or numpy.array
        metallicity
    z : float or numpy.array
        redshift
    
    Returns
    -------
    log_pi : numpy.array
        log probability (ln(dP/dlnZ)) of metallicity/redshift distribution at
        specified metallicity and redshift values"""
    
    omega = omega_z(z)
    xi = mean_Z_z(z)
    dPdlnZ = 2 / omega * NormDist.pdf((np.log(Z) - xi) / omega) * NormDist.cdf(-1.79 * (np.log(Z) - xi) / omega)
    
    return np.log(dPdlnZ)