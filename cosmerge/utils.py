"""contains utility methods"""

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18
from scipy.interpolate import interp1d
from astropy import units as u
import tqdm

"""predefined masks for use in catalog generation"""
bbh_merger_mask = ('bpp', [lambda bpp: bpp.kstar_1 == 14,
                           lambda bpp: bpp.kstar_2 == 14,
                           lambda bpp: bpp.evol_type == 3]
)

bns_merger_mask = ('bpp', [lambda bpp: bpp.kstar_1 == 13,
                           lambda bpp: bpp.kstar_2 == 13,
                           lambda bpp: bpp.evol_type == 3]
)

sn_mask = ('bpp', [lambda bpp: bpp.evol_type.isin([15,16])])
mass_loss_mask = ('bcm', [lambda bcm: (bcm.deltam1 < -1e-4) | (bcm.deltam2 < -1e-4)])

def get_z_interp(z_max):
    """Generates an interpolation to convert between lookback
    time and redshift

    Parameters
    ----------
    z_max : float
        maximum redshift for interpolator

    Returns
    -------
    z_interp : scipy.interpolate interpolation
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
    mets : numpy.array
        centers of metallicity bins

    Returns
    -------
    met_bins : numpy.array
        metallicity bin edges
    """
    bw = np.mean(np.log10(mets[1:]) - np.log10(mets[:-1])) / 2
    met_bins = 10 ** (np.arange(np.log10(min(mets)) - bw, np.log10(max(mets)) + 3 * bw, bw))

    return met_bins[::2]

def read_met_data(path, kstar_1, kstar_2, met_grid, event_masks, SFstart=13700.0,
                  SFduration=0.0, pessimistic_cut=False, CE_cool_filter=False,
                  CE_cut=False, SMT_cut=False):
    """
    Reads in all COSMIC data for specified metallicity grid

    Parameters
    ----------
    path : string
        path to COSMIC data where the path structure
        should be '{path}/dat_kstar1...'
    
    kstar_1 : string
        kstar for the primary following COSMIC dat file naming notation
        
    kstar_2 : string
        kstar for the secondary following COSMIC dat file naming notation

    met_grid : numpy.array
        metallicity grid for COSMIC data

    event_masks : tuple
        tuple containing the dataframe name to apply the 
        masks to, and a list of functions to apply to that
        dataframe to select the desired events

    SFstart : float
        ZAMS lookback time for COSMIC population
    
    SFduration : float
        Duration of star formation for COSMIC population

    pessimistic_cut : bool, optional
        Boolean to decide whether to apply the pessimistic
        cut to the event data based on whether there where
        common envelope events with a Hertzsprung Gap donor

        Note: this is unnecessary if you specified
        cemergeflag = 1 in the Params file

    CE_cool_filter : bool, optional
        Boolean to decide whether to filter >40 Msun ZAMS
        based on the Klencki+2021 results (arXiv: 2006.11286)

    CE_cut : bool, optional
        Boolean to decide whether to throw out CE binaries

    SMT_cut : bool, optional
        Boolean to decide whether to throw out 
        stable mass transfer binaries

    Returns
    -------
    events : numpy.array
        Data containing user chosen events formatted as a bpp.
        Each bin_num corresponds to one row, where the other
        columns are lists of the data for that bin_num.
        
    N_stars : numpy.array
        Total number of stars formed including singles to produce
        the data for each metallicity bin

    M_stars : numpy.array
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
    N_stars = np.max(pd.read_hdf(f, key='n_stars'))
    M_stars = np.max(pd.read_hdf(f, key='mass_stars'))

    bpp = pd.read_hdf(f, key='bpp')
    bcm = pd.read_hdf(f, key='bcm')

    if len(bpp.bin_num.unique()) > 1e5:
        bin_num_keep = np.random.choice(bpp.bin_num.unique(), 100000, replace=False)
        downsamp_fac = 1e5/len(bpp.bin_num.unique())
        bpp = bpp.loc[bpp.bin_num.isin(bin_num_keep)]
        bcm = bcm.loc[bcm.bin_num.isin(bin_num_keep)]
        N_stars = N_stars*downsamp_fac
        M_stars = M_stars*downsamp_fac

    if CE_cut and SMT_cut:
        raise Error("You can't cut everything! You should leave at least one of CE_cut or SMT_cut False")

    if CE_cut:
        bpp_CE_bin_num = bpp.loc[bpp.evol_type == 7].bin_num.unique()
        bpp = bpp.loc[~bpp.bin_num.isin(bpp_CE_bin_num)]
        bcm = bcm.loc[~bcm.bin_num.isin(bpp_CE_bin_num)]
    if SMT_cut:
        bpp_CE_bin_num = bpp.loc[bpp.evol_type == 7].bin_num.unique()
        bpp = bpp.loc[bpp.bin_num.isin(bpp_CE_bin_num)]
        bcm = bcm.loc[bcm.bin_num.isin(bpp_CE_bin_num)]
    
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

        bcm = bcm.loc[~bcm.bin_num.isin(bpp_pess_cut_1)]
        bcm = bcm.loc[~bcm.bin_num.isin(bpp_pess_cut_2)]
    
    if CE_cool_filter:
        bpp_ce_1 = bpp.loc[((bpp.evol_type == 7) & (bpp.RRLO_1 > 1))].bin_num
        bpp_ce_1_zams = bpp.loc[bpp.bin_num.isin(bpp_ce_1)].groupby('bin_num').first().reset_index()
        
        bpp_cut_1 = bpp_ce_1_zams.loc[bpp_ce_1_zams.mass_1 > 40].bin_num
        
        bpp_ce_2 = bpp.loc[((bpp.evol_type == 7) & (bpp.RRLO_2 > 1))].bin_num
        bpp_ce_2_zams = bpp.loc[bpp.bin_num.isin(bpp_ce_2)].groupby('bin_num').first().reset_index()
        
        bpp_cut_2 = bpp_ce_2_zams.loc[bpp_ce_2_zams.mass_2 > 40].bin_num
        
        bpp = bpp.loc[~bpp.bin_num.isin(bpp_cut_1)]
        bpp = bpp.loc[~bpp.bin_num.isin(bpp_cut_2)]

        bcm = bcm.loc[~bcm.bin_num.isin(bpp_cut_1)]
        bcm = bcm.loc[~bcm.bin_num.isin(bpp_cut_2)]        

    # select events based on provided masks. All masks correspond to one dataframe
    df_name, mask_funcs = event_masks
    events = bpp if df_name == 'bpp' else bcm
    
    for mask_func in mask_funcs:
        events = events.loc[mask_func(events)]
    
    #collapse to one row per bin_num, keep exact bpp format
    if df_name == 'bcm': events = events.reindex(columns=bpp.columns)
    events = events.groupby('bin_num', as_index=False).agg(list) 

    return np.array(events, dtype=object), N_stars, M_stars


def get_cosmic_data(path, kstar_1, kstar_2, mets, event_masks,
                    SFstart=13700.0, SFduration=0.0, pessimistic_cut=False,
                    CE_cool_filter=False, CE_cut=False, SMT_cut=False):
    """
    Reads in all COSMIC data for specified metallicity grid

    Parameters
    ----------
    path : string
        path to COSMIC data where the path structure
        should be '{path}/dat_kstar1...'
    
    kstar_1 : string
        kstar for the primary following COSMIC notation
        
    kstar_2 : string
        kstar for the secondary following COSMIC notation

    mets : numpy.array
        metallicity grid for COSMIC data
    
    event_masks : tuple
        tuple containing the dataframe name to apply the 
        masks to, and a list of functions to apply to that
        dataframe to select the desired events
        
    SFstart : float
        ZAMS lookback time for population
    
    SFduration : float
        Duration of star formation

    pessimistic_cut : bool, optional
        Boolean to decide whether to apply the pessimistic
        cut to the event data based on whether there where
        common envelope events with a Hertzsprung Gap donor

        Note: this is unnecessary if you specified
        cemergeflag = 1 in the Params file
        
    CE_cool_filter : bool, optional
        Boolean to decide whether to allow >40 Msun ZAMS
        based on the Klencki+2021 results (arXiv: 2006.11286)

    CE_cut : bool, optional
        Boolean to decide whether to throw out CE binaries

    SMT_cut : bool, optional
        Boolean to decide whether to throw out 
        stable mass transfer binaries

    Returns
    -------
    Ms : numpy.array
        Total amount of stars formed in Msun to produce
        the data for each metallicity bin

    Ns : numpy.array
        Total number of stars formed including singles to produce
        the data for each metallicity bin

    ns : numpy.array
        The number of event binaries per metallicity bin

    dat : numpy.array
        Data containing event binaries
    """

    Ns = []
    Ms = []
    ns = []
    dat = []
    for m in tqdm.tqdm(mets):
        d, N, M = read_met_data(path, kstar_1, kstar_2, m, event_masks, SFstart=SFstart,
                                SFduration=SFduration, pessimistic_cut=pessimistic_cut,
                                CE_cool_filter=CE_cool_filter,
                                CE_cut=CE_cut, SMT_cut=SMT_cut)
        Ms.append(M)
        Ns.append(N)
        dat.append(d)
        ns.append(len(d))
    Ms = np.array(Ms)
    Ns = np.array(Ns)
    ns = np.array(ns)
    dat = np.array(dat, dtype=object)

    return Ms, Ns, ns, dat
