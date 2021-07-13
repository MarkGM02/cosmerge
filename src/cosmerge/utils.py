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


def parse_kstar(kstar):
    """
    Parses the kstar string labels into kstar values
    to select merger types

    Parameters
    ----------
    kstar : `string`
        specifies the merger types for the primary and secondary stars
        where kstar = '13_14' contains both NSs and BHs
        and kstar = '13' just contains NSs

    Returns
    -------
    kstar_list : `list`
        list which can be used to select mergers of interest from COSMIC data
    """

    if len(kstar) == 2:
        kstar_list = [int(kstar)]
    else:
        kstar_hi = int(kstar[:2])
        kstar_lo = int(kstar[3:])
        kstar_list = range(kstar_hi, kstar_lo + 1)

    return kstar_list


def read_met_data(path, kstar_1, kstar_2, met_grid, SFstart=13700.0, SFduration=0.0,
                  pessimistic_cut=False, kstar_1_select=None, kstar_2_select=None):
    """
    Reads in all COSMIC data for specified metallicity grid

    Parameters
    ----------
    path : `string`
        path to COSMIC data where the path structure
        should be '{path}/dat_kstar1...'
    
    kstar_1 : `string`
        kstar for the primary following COSMIC dat file naming notation
        
    kstar_2 : `string`
        kstar for the secondary following COSMIC dat file naming notation

    met_grid : `numpy.array`
        metallicity grid for COSMIC data
        
    SFstart : `float`
        ZAMS lookback time for COSMIC population
    
    SFduration : `float`
        Duration of star formation for COSMIC population

    pessimistic_cut : `bool`
        Boolean to decide whether to apply the pessimistic
        cut to the merger data based on whether there where
        common envelope events with a Hertzsprung Gap donor

        Note: this is unnecessary if you specified
        cemergeflag = 1 in the Params file

    kstar_1_select : `list`
        If specified, will select kstars that are a subset of the
        kstar_1 data

    kstar_2_select : `list`
        If specified, will select kstars that are a subset of the
        kstar_2 data

    Returns
    -------
    mergers : `numpy.array`
        Data containing compact object binaries
        
    N_stars : `numpy.array`
        Total number of stars formed including singles to produce
        the data for each metallicity bin

    M_stars : `numpy.array`
        Total amount of stars formed in Msun to produce
        the data for each metallicity bin
    """
    met_grid = np.round(met_grid, 8)

    # read in the data
    f = '{}/dat_kstar1_{}_kstar2_{}_SFstart_{}_SFduration_{}_metallicity_{}.h5'.format(path,
                                                                                       kstar_1,
                                                                                       kstar_2,
                                                                                       SFstart,
                                                                                       SFduration,
                                                                                       met_grid)
    N_stars = np.max(pd.read_hdf(f, key='n_stars'))[0]
    M_stars = np.max(pd.read_hdf(f, key='mass_stars'))[0]

    bpp = pd.read_hdf(f, key='bpp')

    # filter out HG donors if requested
    if pessimistic_cut:
        bpp_pess_cut_1 = bpp.loc[((bpp.evol_type == 7) &
                                  (bpp.kstar_1.isin([0, 1, 2, 7, 8, 10, 11, 12])) &
                                  (bpp.RRLO_1 > 1))].bin_num
        bpp_pess_cut_2 = bpp.loc[((bpp.evol_type == 7) &
                                  (bpp.kstar_2.isin([0, 1, 2, 7, 8, 10, 11, 12])) &
                                  (bpp.RRLO_2 > 1))].bin_num

        bpp = bpp.loc[~bpp.bin_num.isin(bpp_pess_cut_1)]
        bpp = bpp.loc[~bpp.bin_num.isin(bpp_pess_cut_2)]

    if kstar_1_select is not None:
        kstar_1 = kstar_1_select
    else:
        # parse the kstars since they are supplied in a string format
        kstar_1 = parse_kstar(kstar=kstar_1)

    if kstar_2_select is not None:
        kstar_2 = kstar_2_select
    else:
        # parse the kstars since they are supplied in a string format
        kstar_2 = parse_kstar(kstar=kstar_2)

    # select mergers based on supplied kstars
    mergers = bpp.loc[(bpp.kstar_1.isin(kstar_1)) & (bpp.kstar_2.isin(kstar_2)) &
                      (bpp.evol_type == 3)]

    return np.array(mergers, dtype=object), N_stars, M_stars


def get_cosmic_data(path, kstar_1, kstar_2, mets,
                    SFstart=13700.0, SFduration=0.0, pessimistic_cut=False,
                    kstar_1_select=None, kstar_2_select=None):
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

    pessimistic_cut : `bool`
        Boolean to decide whether to apply the pessimistic
        cut to the merger data based on whether there where
        common envelope events with a Hertzsprung Gap donor

        Note: this is unnecessary if you specified
        cemergeflag = 1 in the Params file

    kstar_1_select : `list`
        If specified, will select kstars that are a subset of the
        kstar_1 data

    kstar_2_select : `list`
        If specified, will select kstars that are a subset of the
        kstar_2 data

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
        d, N, M = read_met_data(path, kstar_1, kstar_2, m, SFstart=SFstart, SFduration=SFduration,
                                pessimistic_cut=pessimistic_cut, kstar_1_select=kstar_1_select,
                                kstar_2_select=kstar_2_select)
        Ms.append(M)
        Ns.append(N)
        dat.append(d)
        ns.append(len(d))
    Ms = np.array(Ms)
    Ns = np.array(Ns)
    ns = np.array(ns)
    dat = np.array(dat)

    return Ms, Ns, ns, dat
