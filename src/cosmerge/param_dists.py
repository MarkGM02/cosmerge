"""methods for making distribution functions for rates"""

from scipy.stats import gaussian_kde
import numpy as np
from scipy.special import logit
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18_arXiv_v2 as Planck18
import astropy.units as u


def get_dN_dtlb_dV(t_lb):
    """
    Creates a kde of the merger lookback times
    Note that since t_lb has values -inf < t_lb < inf
    the kde is properly normalized to be evaluated at
    t_lb > 0 or z(t_lb > 0)

    Parameters
    ----------
    t_lb : `numpy.array or pandas.Series`
        collection of merger lookback times with limits
        -inf < t_lb < inf

    Returns
    -------
    dN_dt : `scipy.stats.gaussian_kde`
        a kde which evaluates the pdf: dN/(dt_lb dV_com)
    """
    p_t_lb = gaussian_kde(t_lb)

    def dN_dt(t):
        return p_t_lb(t)

    return dN_dt


def get_dN_dtlb_dlnm_dV(t_lb, m):
    """
    Creates a kde of the merger lookback times and masses
    Note that since t_lb has values -inf < t_lb < inf
    the kde is properly normalized to be evaluated at
    t_lb > 0 or z(t_lb > 0) and to enforce proper mass
    boundaries, we return dln(m)

    Parameters
    ----------
    t_lb : `numpy.array or pandas.Series`
        collection of merger lookback times with limits
        -inf < t_lb < inf

    m : `numpy.array or pandas.Series`
        collection of merger masses with limits
        0 < m < inf

    Returns
    -------
    dN_d_t_lnm : `scipy.stats.gaussian_kde`
        a kde which evaluates : dN/(dt_lb dlnm dV_com)
    """

    t, lnm = np.broadcast_arrays(t_lb, np.log(m))
    p_tlb_lnm = gaussian_kde(np.vstack([t, lnm]))

    def dN_d_t_lnm(t_eval, m_eval):
        # set up the kde to evaluate properly with easy to use inputs
        lnm_eval = np.log(m_eval)
        t_eval, lnm_eval = np.broadcast_arrays(t_eval, lnm_eval)

        pts = np.vstack((t_eval, lnm_eval))

        return p_tlb_lnm(pts)

    return dN_d_t_lnm


def get_dN_dtlb_dlnm_dq_dV(t_lb, m, q):
    """
    Creates a kde of the merger lookback times, masses,
    and mass ratios
    Note that since t_lb has values -inf < t_lb < inf
    the kde is properly normalized to be evaluated at
    t_lb > 0 or z(t_lb > 0), to enforce proper mass
    boundaries we return dln(m) and dlogit(q)

    Parameters
    ----------
    t_lb : `numpy.array or pandas.Series`
        collection of merger lookback times with limits
        -inf < t_lb < inf

    m : `numpy.array or pandas.Series`
        collection of merger masses with limits
        0 < m < inf

    q : `numpy.array or pandas.Series`
        collection of mass ratios with limits
        0 < q <= 1

    Returns
    -------
    dN_d_t_lnm_logitq : `scipy.stats.gaussian_kde`
        a kde which evaluates the : dN/(dt_lb dlnm dlogitq dV_com)
    """

    t, lnm, logitq = np.broadcast_arrays(t_lb, np.log(m), logit(q))
    p_tlb_lnm_logitq = gaussian_kde(np.vstack([t, lnm, logitq]))

    def dN_d_t_lnm_logitq(t_eval, m_eval, q_eval):
        # set up the kde to evaluate properly with easy to use inputs
        lnm_eval = np.log(m_eval)
        logitq_eval = logit(q_eval)
        t_eval, lnm_eval, logitq_eval = np.broadcast_arrays(t_eval, lnm_eval, logitq_eval)

        pts = np.vstack((t_eval, lnm_eval, logitq_eval))

        return p_tlb_lnm_logitq(pts)

    return dN_d_t_lnm_logitq


def get_dN_dtlb_dlnm1_dlnm2_dV(t_lb, m1, m2):
    """
    Creates a kde of the merger lookback times, masses,
    and mass ratios
    Note that since t_lb has values -inf < t_lb < inf
    the kde is properly normalized to be evaluated at
    t_lb > 0 or z(t_lb > 0), to enforce proper mass
    boundaries we return dln(m) and dlogit(q)

    Parameters
    ----------
    t_lb : `numpy.array or pandas.Series`
        collection of merger lookback times with limits
        -inf < t_lb < inf

    m1 : `numpy.array or pandas.Series`
        collection of merger masses with limits
        0 < m < inf

    m2 : `numpy.array or pandas.Series`
        collection of merger masses with limits
        0 < m < inf

    Returns
    -------
    dN_d_t_lnm1_lnm2 : `scipy.stats.gaussian_kde`
        a kde which evaluates : dN/(dt_lb dlnm1 dlnm2 dV_com)
    """

    t, lnm1, lnm2 = np.broadcast_arrays(t_lb, np.log(m1), np.log(m2))
    p_tlb_lnm1_lnm2 = gaussian_kde(np.vstack([t, lnm1, lnm2]))

    def dN_d_t_lnm1_lnm2(t_eval, m1_eval, m2_eval):
        # set up the kde to evaluate properly with easy to use inputs
        lnm1_eval = np.log(m1_eval)
        lnm2_eval = np.log(m2_eval)
        t_eval, lnm1_eval, lnm2_eval = np.broadcast_arrays(t_eval, lnm1_eval, lnm2_eval)

        pts = np.vstack((t_eval, lnm1_eval, lnm2_eval))

        return p_tlb_lnm1_lnm2(pts)

    return dN_d_t_lnm1_lnm2


def get_dN_dtlb_dlnm_dZ_dV(t_lb, m, Z):
    """
    Creates a kde of the merger lookback times, masses,
    and mass ratios
    Note that since t_lb has values -inf < t_lb < inf
    the kde is properly normalized to be evaluated at
    t_lb > 0 or z(t_lb > 0), to enforce proper mass
    boundaries we return dln(m) and dln(Z)

    Parameters
    ----------
    t_lb : `numpy.array or pandas.Series`
        collection of merger lookback times with limits
        -inf < t_lb < inf

    m : `numpy.array or pandas.Series`
        collection of merger masses with limits
        0 < m < inf

    Z : `numpy.array or pandas.Series`
        collection of metallicities with limits
        0 < Z < inf

    Returns
    -------
    dN_d_t_lnm_lnZ : `scipy.stats.gaussian_kde`
        a kde which evaluates : dN/(dt_lb dlnm dlnZ dV_com)
    """

    t, lnm, lnZ = np.broadcast_arrays(t_lb, np.log(m), np.log(Z))
    p_tlb_lnm_lnZ = gaussian_kde(np.vstack([t, lnm, lnZ]))

    def dN_d_t_lnm_lnZ(t_eval, m_eval, Z_eval):
        # set up the kde to evaluate properly with easy to use inputs
        lnm_eval = np.log(m_eval)
        lnZ_eval = np.log(Z_eval)
        t_eval, lnm_eval, lnZ_eval = np.broadcast_arrays(t_eval, lnm_eval, lnZ_eval)

        pts = np.vstack((t_eval, lnm_eval, lnZ_eval))

        return p_tlb_lnm_lnZ(pts)

    return dN_d_t_lnm_lnZ


def get_pz(t_lb, z_max=15):
    """
    Creates a pdf which predicts the probability of merger redshifts
    based on the rate at the detector per redshift: dN_dz_dtd

    Parameters
    ----------
    t_lb : `numpy.array or pandas.Series`
        collection of merger lookback times with limits
        -inf < t_lb < inf

    z_max : `float`
        maximum redshift for evaluating the pdf

    Returns
    -------
    p_z : `scipy.stats.gaussian_kde`
        Probability of merger redshifts evaluated
        at the detector frame
    """
    # set up the redshifts/lookback times to evaluate pdf at
    zs = np.expm1(np.linspace(np.log(1), np.log(1 + z_max), 1024))
    ts = Planck18.lookback_time(zs).to(u.Myr).value

    # Get the comoving merger rate kde
    dN_dts_dV = get_dN_dtlb_dV(t_lb)

    # Set up the Jacobian to go from source frame lookback time
    # to redshift measured at the detector
    dV_dz = (4 * np.pi * u.sr * Planck18.differential_comoving_volume(zs).to(u.Gpc ** 3 * u.sr ** (-1))).value
    dts_dtd = 1 / (1 + zs)

    # Evaluate the rate kde and multiply by the Jacobian
    dN_dt_dz = dN_dts_dV(ts) * dV_dz * dts_dtd
    norm = np.trapz(dN_dt_dz, zs)

    # Create new kde of normalized dN_dt_dz
    p_z = interp1d(zs, dN_dt_dz/norm)

    return p_z
