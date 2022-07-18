"""methods to build cosmic merger populations"""

from astropy.cosmology import Planck18_arXiv_v2 as Planck18
import astropy.units as u
import pandas as pd
import numpy as np
import tqdm
import scipy.integrate as integrate

from cosmerge import utils
from cosmerge import sfh


def md_zs(sfr_model, z_max):
    """A generator that returns redshifts of formation drawn from
    the user-specified SFR model

    Parameters
    ----------
    sfr_model : str
        function that returns the star formation rate model in units
        of Msun per comoving volume per time
        choose from: sfh.md_14 or sfh.md_17 or supply your own!

    z_max : float
        maximum redshift for star formation

    Returns
    -------
    redshifts : numpy.array
        redshifts sampled from the supplied star formation rate model
    """
    zs = np.expm1(np.linspace(np.log(1), np.log(1 + z_max), 1024))
    pzs = sfr_model(zs) * Planck18.lookback_time_integrand(zs) * Planck18.hubble_time.to(u.yr).value
    czs = integrate.cumtrapz(pzs, zs, initial=0)  # Cumulative distribution which integrates to 1

    while True:
        redshifts = np.interp(np.random.uniform(low=0, high=czs[-1]), czs, zs)
        yield redshifts


def draw_metallicities_and_redshifts(mets, ns, Ns, sfr_model, sigma_log10Z, z_max):
    """Generator for draws of formation metallicities and redshifts from a
    log-normal metallicity distribution based on Madau & Fragos (2017)
    from the sfh module and a user specified star formation rate model
    and connects to populations synthesized on a regular
    grid of metallicity spaced uniform in log10(Z)
    
    Parameters
    ----------
    mets : numpy.array
        The center of each metallicity bin in the COSMIC data grid
    
    ns : numpy.array
        The number of mergers within each metallicity bin.
        
    Ns : numpy.array
        The number of stars sampled to produce the mergers within each metallicity bin

    sfr_model : str
        Function which returns the star formation rate model

    sigma_log10Z : float
        Function giving the standard deviation of the metallicity distribution in dex
        Default : sigma(log10(Z)) = 0.5.

    z_max : float
        maximum redshift for star formation


    Returns
    -------
    Yields a series of `i, j, z, Z` for each drawn metallicity and redshift
    
    `i` : metallicity bin index
    `j` : the index of the system within that bin
    `z` : the redshift of formation
    `Z` : the randomly-assigned metallicity within the bin `i`.    
    """

    # set up metallicity bin centers and edges
    n_bin = len(mets)
    met_bins = utils.get_met_bins(mets)
    dZs = met_bins[1:] - met_bins[:-1]

    # select an initial metallicity bin from the metallicity bins randomly 
    # with equal weights for each metallicity
    i = np.random.randint(n_bin)

    # select an initial BBH merger index from the BBH dataset for metallicity index: i
    j = np.random.randint(ns[i])

    # sample an initial redshift from our SFR generator
    z = next(md_zs(sfr_model, z_max))

    # Assign an initial metallicity from the metallicity bins based on the metallicity index: i
    Z = np.random.uniform(low=met_bins[i], high=met_bins[i + 1])

    # repeat the selection for a new redshift (zp), metallicity index (ip), 
    # BBH merger index (jp), and metallicity (Zp)
    for zp in md_zs(sfr_model, z_max):
        ip = np.random.randint(n_bin)
        jp = np.random.randint(ns[ip])
        Zp = np.random.uniform(low=met_bins[ip], high=met_bins[ip + 1])

        # log acceptance probability is the difference between the current and next data points
        # Should we be doing Ns, or Ms here? The rates normalize to the M instead of the N?
        log_Pacc = sfh.log_p_Z_z(Zp, zp, sigma_log10Z) - sfh.log_p_Z_z(Z, z, sigma_log10Z) + \
                   np.log(Ns[i]) + np.log(ns[ip]) + np.log(dZs[ip]) - \
                   (np.log(Ns[ip]) + np.log(ns[i]) + np.log(dZs[i]))

        if np.log(np.random.rand()) < log_Pacc:
            i = ip
            j = jp
            z = zp
            Z = Zp
        else:
            pass
        yield i, j, z, Z


def generate_universe(n_sample, n_downsample, mets, M_sim, N_sim,
                      n_merger, mergers, sfh_model, sigma_log10Z=0.5,
                      z_max=15):
    """Generates a universe of star formation by sampling metallicities and
    redshifts according to the user specified star formation rate model,
    a mean metallicity evolution from Madau & Fragos (2017)
    and a log normal metallicity distribution with sigma_log10Z out to redshift z_max
    then connects these formation redshifts and metallicities to COSMIC
    data for merging compact objects to create a merger catalog

    Parameters
    ----------
    n_sample : integer
        number of formation samples to draw

    n_downsample : integer
        downsample factor

    mets : numpy.array
        The center of each metallicity bin in the COSMIC data grid

    M_sim : numpy.array
        Total amount of stars formed in Msun to produce
        the data for each metallicity bin

    N_sim : numpy.array
        Total number of stars formed to produce
        the data for each metallicity bin

    n_merger : numpy.array
        The number of compact object binaries per metallicity bin

    mergers : numpy.array
        A ragged edge numpy array that contains all mergers for each metallicity bin

    sfh_model : function
        Function which returns the star formation history model

    sigma_log10Z : float
        Function giving the standard deviation of the metallicity distribution in dex

    z_max : float
        maximum redshift for star formation

    Returns
    -------
    dat : pandas.DataFrame
        merger catalog containing formation metallicities, redshifts,
        and lookback times as well as merger lookback times, masses,
        and COSMIC bin_num indexes

    ibins : numpy.array
        Metallicity bin indices for each of the mergers in the catalog
    """

    # ibins: metallicity indices
    # j_s: bbh merger indices
    # z_s: formation redshifts
    # Z_s: metallicities
    ibins, j_s, z_s, Z_s = zip(*[x for (x, i) in
                                 zip(draw_metallicities_and_redshifts(mets, n_merger, N_sim,
                                                                      sfh_model, sigma_log10Z, z_max),
                                     tqdm.tqdm(range(n_sample))) if
                                 i % n_downsample == 0])
    # we want all of these indices to be in arrays to do array manipulation later
    ibins = np.array(ibins)
    j_s = np.array(j_s)
    z_s = np.array(z_s)
    Z_s = np.array(Z_s)

    # Now that we have a bunch of formation metallicities and redshifts
    # let's connect them to the COSMIC merger data to build a merger catalog
    t_delay_ind = 0
    m1_ind = 1
    m2_ind = 2
    bin_num_ind = 43

    dat = []
    # loop through our metallicity grid for easy COSMIC data access
    for ii in tqdm.tqdm(range(len(mets))):
        # select all the formation redshift and metallicities in that bin
        met_mask = ibins == ii
        if len(met_mask[met_mask]) > 0:
            # get all the formation lookback times from the formation redshifts
            # note that the units are in Myr since the COSMIC delay times are also in Myr
            t_form = Planck18.lookback_time(z_s[met_mask]).to(u.Myr).value

            # get all the merger lookback times by subtracting the delay time
            # for the selected BBH merger at metallicity ii, and rows j_s[met_mas]
            t_merge = t_form - mergers[ii][j_s[met_mask], t_delay_ind]

            # Note we don't remove anything that will merge in the future
            # This is because we want to keep the merger time pdf well behaved
            # for future rate kernel density estimates

            # connect the formation redshifts and metallicities to the COSMIC data
            if len(dat) == 0:
                dat = np.vstack([t_form, t_merge, z_s[met_mask],
                                 Z_s[met_mask], np.ones(len(t_form)) * mets[ii],
                                 mergers[ii][j_s[met_mask], m1_ind],
                                 mergers[ii][j_s[met_mask], m2_ind],
                                 mergers[ii][j_s[met_mask], bin_num_ind]])
            else:
                dat = np.append(dat, np.vstack(
                    [t_form, t_merge, z_s[met_mask],
                     Z_s[met_mask], np.ones(len(t_form)) * mets[ii],
                     mergers[ii][j_s[met_mask], m1_ind],
                     mergers[ii][j_s[met_mask], m2_ind],
                     mergers[ii][j_s[met_mask], bin_num_ind]]),
                                axis=1)

    dat = pd.DataFrame(dat.T,
                       columns=['t_form', 't_merge', 'z_form',
                                'met', 'met_cosmic', 'm1', 'm2',
                                'bin_num'],
                       dtype=float)

    # return merger catalog, merger fraction, and formation statistics
    return dat, ibins
