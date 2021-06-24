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


def read_met_data(path, met_read):
    if met_read > 0.00009:
        f = '{}/dat_kstar1_13_14_kstar2_13_14_SFstart_13700.0_SFduration_0.0_metallicity_{}.h5'.format(met_read,
                                                                                                       met_read)
    else:
        f = '0.000085/dat_kstar1_13_14_kstar2_13_14_SFstart_13700.0_SFduration_0.0_metallicity_8.5e-05.h5'
    N_stars = np.max(pd.read_hdf(path + '/' + f, key='n_stars'))[0]
    M_stars = np.max(pd.read_hdf(path + '/' + f, key='mass_stars'))[0]

    # filter out the HG donor common envelope binaries
    bpp = pd.read_hdf(path + '/' + f, key='bpp')
    bpp_ce = bpp.loc[(bpp.evol_type == 7)]
    bpp_ce_1 = bpp_ce.loc[(bpp_ce.RRLO_1 > 1)]
    bpp_ce_2 = bpp_ce.loc[(bpp_ce.RRLO_2 > 1)]
    bpp_ce_1_pess = bpp_ce_1.loc[~bpp_ce_1.kstar_1.isin([2, 8])]
    bpp_ce_2_pess = bpp_ce_2.loc[~bpp_ce_2.kstar_2.isin([2, 8])]
    bpp_bin_num = bpp_ce_1_pess.bin_num.unique()
    bpp_bin_num = np.append(bpp_bin_num, bpp_ce_2_pess.bin_num.unique())
    bpp_bin_num = np.unique(bpp_bin_num)
    BBH = bpp.loc[(bpp.kstar_1 == 14) & (bpp.kstar_2 == 14) & (bpp.evol_type == 3) & (bpp.bin_num.isin(bpp_bin_num))]

    return np.array(BBH), N_stars, M_stars


def get_cosmic_data(path, mets):
    """
    Reads in all COSMIC data for specified metallicity grid

    Parameters
    ----------
    path : `string`
        path to COSMIC data where the path structure
        should be '{metallicity bin}/dat...'

    mets : `numpy.array`
        metallicity grid for COSMIC data

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
        d, N, M = read_met_data(path=path, met_read=m)
        Ms.append(M)
        Ns.append(N)
        dat.append(d)
        ns.append(len(d))
    Ms = np.array(Ms)
    Ns = np.array(Ns)
    ns = np.array(ns)
    dat = np.array(dat)

    return Ms, Ns, ns, dat
