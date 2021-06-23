"""methods for making distribution functions"""

from scipy.stats import gaussian_kde


def get_dN_dtlb(t_lb):
    """"""
    p_t_lb = gaussian_kde(t_lb, bw_method='scott')

    def pt(t):
        return p_t_lb(t)

    return pt
