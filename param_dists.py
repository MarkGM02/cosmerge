"""methods for making distribution functions"""

from scipy.stats import gaussian_kde
import numpy as np


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
    p_t : `scipy.stats.gaussian_kde`
        a kde which evaluates the pdf: dN/(dt_lb dV_com)
    """
    p_t_lb = gaussian_kde(t_lb, bw_method='scott')

    def p_t(t):
        return p_t_lb(t)

    return p_t


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
    p_t_lnm : `scipy.stats.gaussian_kde`
        a kde which evaluates the pdf: dN/(dt_lb dlnm dV_com)
    """

    t, lnm = np.broadcast_arrays(t_lb, np.log(m))
    p_tlb_lnm = gaussian_kde(np.vstack([t, lnm]), bw_method='scott')

    def p_t_lnm(t_eval, m_eval):
        lnm_eval = np.log(m_eval)
        t_eval, lnm_eval = np.broadcast_arrays(t_eval, lnm_eval)

        pts = np.vstack((t_eval, lnm_eval))

        return p_tlb_lnm(pts)

    return p_t_lnm
