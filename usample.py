from astropy.cosmology import Planck18_arXiv_v2 as Planck18
from astropy.cosmology import z_at_value

import astropy.units as u
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.stats import gaussian_kde
import seaborn as sns
import tqdm

import utils
import numpy as np
from scipy.interpolate import interp1d
import sfh

## Set up interpolator to go between redshift and lookback time
def get_z_interp(t_merge_list):
    z_merge_list = []
    for t in t_merge_list:
        z_merge_list.append(z_at_value(Planck18.lookback_time, t * u.Myr))
    
    z_interp = interp1d(t_merge_list, z_merge_list)
    
    return z_interp

def get_cosmic_data(path, mets):
    Ns = []
    Ms = []
    ns = []
    bbhs = []
    for m in tqdm.tqdm(mets):
        bbh, N, M = utils.get_cosmic_data(path=path, met_read=m)
        Ms.append(M)
        Ns.append(N)
        bbhs.append(bbh)    
        ns.append(len(bbh))
    Ms = np.array(Ms)
    Ns = np.array(Ns)
    ns = np.array(ns)
    bbhs = np.array(bbhs)
    
    return Ms, Ns, ns, bbhs


def md_zs(sfr):
    """A generator that returns redshifts of formation drawn from the MF17 SFR."""
    zmax = 20
    zs = np.expm1(np.linspace(np.log(1), np.log(1+zmax), 1024))
    #pzs = sfr(zs) * Planck18.differential_comoving_volume(zs).to(u.Gpc**3/u.sr).value * 4 * np.pi
    pzs = sfr(zs) * Planck18.lookback_time_integrand(zs) * Planck18.hubble_time.to(u.yr).value
    czs = cumtrapz(pzs, zs, initial=0) # Cumulative distribution
        
    while True:
        yield np.interp(np.random.uniform(low=0, high=czs[-1]), czs, zs)
        
def get_met_bins(mets):
    bw = np.mean(np.log10(mets[1:]) - np.log10(mets[:-1]))/2
    met_bins = 10**(np.arange(np.log10(min(mets)) - bw, np.log10(max(mets)) + 3*bw, bw))
    
    return met_bins[::2]

## We just assume that sigma_logZ is 0.5
def sigma_logZ(z):
    return 0.5*np.ones_like(z)


def mergers_uniform_proposal(mets, Ms, ns, Ns, mu_logZ, sigma_logZ, sfr, bbh_dat):
    """Generator for draws from a log-normal metallicity distribution and 
    Madau & Fragos 2017 SFR given populations synthesized on a regular 
    grid of metallicity.
    
    Parameters
    ----------
    mets : `numpy.array`
        The center of each metallicity bin.
    
    dZs : `numpy.array`
        The width of each metallicity bin.
    
    Ms : `numpy.array`
        The total mass of ZAMS/IC draws in each metallicity bin. 
        (this includes the effects of the IMF/initial orbital parameter distributions/binary fraction)
    
    ns : `numpy.array`
        The number of BBH mergers within each metallicity bin.
        
    Ns : `numpy.array`
        The number of stars sampled to produce the BBH mergers within each metallicity bin
    
    mu_logZ : `function`
        Function giving the mean (natural) log metallicity at redshift `z`: `mu_logZ(z)`.
    
    sigma_logZ : `function`
        Function giving the standard deviation of the (natural) log metallicity at redshift `z`: `sigma_logZ(z)`.
    
    sfr : `function`
        Function which returns the star formation rate model
    
    bbh_dat : `numpy.array`
        Contains all the BBH data with index: [metallicity, BBH system, BBH parameters]
    
    Returns
    -------
    Yields a series of `(i, j, z, Z)` where 
    
    `i` : metallicity bin index
    `j` : the index of the system within that bin
    `z` : the redshift of formation
    `Z` : the randomly-assigned metallicity within the bin `i`.    
    """
    
    def log_pi(Z, z):
        """Computes the metallicity and redshift distribution assuming a log normal metallicity
        distribution at each redshift
        
        Parameters
        ----------
        Z : `numpy.array`
            metallicities
            
        z : `numpy.array`
            redshifts
            
        Returns
        -------
        log_pi : `numpy.array`
            log probability of the metallicity/redshift distribution at Z,z
        """
        mu = mu_logZ(z)
        sigma = sigma_logZ(z)
        
        return -np.log(Z) - np.log(sigma) - 0.5*np.square((np.log(Z) - mu)/sigma)
    
    # set up metallicity bin centers and edges
    Nbin = len(mets)
    met_bins = get_met_bins(mets)
    dZs = met_bins[1:] - met_bins[:-1]
    
    # select an initial metallicity bin from the metallicity bins randomly 
    # with equal weights for each metallicity
    i = np.random.randint(Nbin)
    
    # select an initial BBH merger index from the BBH dataset for metallicity index: i
    j = np.random.randint(ns[i])
    
    # sample an initial redshift from our SFR generator
    z = next(md_zs(sfr))
    # Assign an initial metallicity from the metallicity bins based on the metallicity index: i
    Z = np.random.uniform(low=met_bins[i], high=met_bins[i+1])
    
    
    # repeat the selection for a new redshift (zp), metallicity index (ip), 
    # BBH merger index (jp), and metallicity (Zp)
    for zp in md_zs(sfr):
        ip = np.random.randint(Nbin)
        jp = np.random.randint(ns[ip])
        Zp = np.random.uniform(low=met_bins[ip], high=met_bins[ip+1])
        
        # log acceptance probability is the difference between the current and next data points
        # Should we be doing Ns, or Ms here? The rates normalize to the M instead of the N?
        log_Pacc = log_pi(Zp, zp) - log_pi(Z, z) + np.log(Ns[i]) + np.log(ns[ip]) + np.log(dZs[ip]) - (np.log(Ns[ip]) + np.log(ns[i]) + np.log(dZs[i]))
        
        if np.log(np.random.rand()) < log_Pacc:
            i = ip
            j = jp
            z = zp
            Z = Zp
        else:
            pass
        yield (i,j,z,Z)

        
def sample_universe(n_sample, n_downsample, mets, path, sfr):
    Ms, Ns, ns, bbhs = get_cosmic_data(path, mets)
    
    z = np.expm1(np.linspace(0, np.log1p(18), 1000))
    tlb = Planck18.lookback_time(z).to(u.Myr).value
    z_interp = interp1d(tlb, z)
    
    
    
    # ibins: metallicity indices
    # j_s: bbh merger indices
    # z_s: formation redshifts
    # Z_s: metallicities
    ibins, j_s, z_s, Z_s = zip(*[x for (x, i) in zip(mergers_uniform_proposal(mets, Ms, ns, Ns, sfh.mean_metal_z, sigma_logZ, sfr, bbhs), tqdm.tqdm(range(n_sample))) if i%n_downsample == 0])
    
    t_form_ind = 0
    m1_ind = 1
    m2_ind = 2
    bin_num_ind = 43
    
    n_merge = 0
    dat = []    
    for ii in tqdm.tqdm(range(len(mets))):
    
        met_mask = np.array(ibins) == ii
        if len(met_mask) > 0:
            breakpoint()
            # get all the formation lookback times from the formation redshifts
            # note that the units are in Myr since the COSMIC delay times are also in Myr
            t_form = Planck18.lookback_time(np.array(z_s)[met_mask]).to(u.Myr).value
            
            # get all the merger lookback times by subtracting the delay time
            # for the selected BBH merger at metallicity ii, and rows j_s[ind]
            t_merge = t_form - bbhs[ii][np.array(j_s)[met_mask], 0]
            #t_merge = t_form
            
            # select out anything that merges after the present
            merging_mask = t_merge >= 0
            
            # count up the number of mergers 
            n_mergers = np.count_nonzero(merging_mask)
            n_merge += n_mergers
            # convert the merger lookback times to merger redshifts
            z_merge = z_interp(t_merge[merging_mask])
            
            # record the data 
            if n_mergers > 0:
                if len(dat) == 0:
                    dat = np.vstack([t_form[merging_mask], t_merge[merging_mask], np.array(z_s)[met_mask][merging_mask],\
                                     z_merge, np.array(Z_s)[met_mask][merging_mask], np.ones(n_mergers) * mets[ii],\
                                     bbhs[ii][np.array(j_s)[met_mask][merging_mask], m1_ind],\
                                     bbhs[ii][np.array(j_s)[met_mask][merging_mask], m2_ind],\
                                     bbhs[ii][np.array(j_s)[met_mask][merging_mask], bin_num_ind]])
                else:
                    dat = np.append(dat, np.vstack([t_form[merging_mask], t_merge[merging_mask], np.array(z_s)[met_mask][merging_mask],\
                                                    z_merge, np.array(Z_s)[met_mask][merging_mask], np.ones(n_mergers) * mets[ii],\
                                                    bbhs[ii][np.array(j_s)[met_mask][merging_mask], m1_ind],\
                                                    bbhs[ii][np.array(j_s)[met_mask][merging_mask], m2_ind],\
                                                    bbhs[ii][np.array(j_s)[met_mask][merging_mask], bin_num_ind]]),
                                    axis=1)
    # calculate the fraction of systems that merge from the systems that form
    f_merge = n_merge/len(ibins)
                    
    dat = pd.DataFrame(dat.T, columns=['t_form', 't_merge', 'z_form', 'z_merge', 'met', 'met_cosmic', 'm1', 'm2', 'bin_num'])
    
    # return merger catalog, merger fraction, and formation statistics
    return dat, f_merge, Ms, ns, ibins
  
def sample_universe_future(n_sample, n_downsample, mets, path, sfr):
    Ms, Ns, ns, bbhs = get_cosmic_data(path, mets)
    
    # ibins: metallicity indices
    # j_s: bbh merger indices
    # z_s: formation redshifts
    # Z_s: metallicities
    ibins, j_s, z_s, Z_s = zip(*[x for (x, i) in zip(mergers_uniform_proposal(mets, Ms, ns, Ns, sfh.mean_metal_z, sigma_logZ, sfr, bbhs), tqdm.tqdm(range(n_sample))) if i%n_downsample == 0])
    
    t_form_ind = 0
    m1_ind = 1
    m2_ind = 2
    bin_num_ind = 43
    
    n_merge = 0
    dat = []    
    for ii in tqdm.tqdm(range(len(mets))):
    
        met_mask = np.array(ibins) == ii
        if np.count_nonzero(met_mask) > 0:
            # get all the formation lookback times from the formation redshifts
            # note that the units are in Myr since the COSMIC delay times are also in Myr
            t_form = Planck18.lookback_time(np.array(z_s)[met_mask]).to(u.Myr).value
            
            # get all the merger lookback times by subtracting the delay time
            # for the selected BBH merger at metallicity ii, and rows j_s[ind]
            t_merge = t_form - bbhs[ii][np.array(j_s)[met_mask], 0]
            
            # count up the number of mergers 
            n_mergers = len(t_merge)
            n_merge += n_mergers
            
            # record the data 
            if n_mergers > 0:
                if len(dat) == 0:
                    dat = np.vstack([t_form, t_merge, np.array(z_s)[met_mask],\
                                     np.array(Z_s)[met_mask], np.ones(n_mergers) * mets[ii],\
                                     bbhs[ii][np.array(j_s)[met_mask], m1_ind],\
                                     bbhs[ii][np.array(j_s)[met_mask], m2_ind],\
                                     bbhs[ii][np.array(j_s)[met_mask], bin_num_ind]])
                else:
                    dat = np.append(dat, np.vstack([t_form, t_merge, np.array(z_s)[met_mask],\
                                                    np.array(Z_s)[met_mask], np.ones(n_mergers) * mets[ii],\
                                                    bbhs[ii][np.array(j_s)[met_mask], m1_ind],\
                                                    bbhs[ii][np.array(j_s)[met_mask], m2_ind],\
                                                    bbhs[ii][np.array(j_s)[met_mask], bin_num_ind]]),
                                    axis=1)
    # calculate the fraction of systems that merge from the systems that form
    f_merge = n_merge/len(ibins)
                    
    dat = pd.DataFrame(dat.T, columns=['t_form', 't_merge', 'z_form', 'met', 'met_cosmic', 'm1', 'm2', 'bin_num'])
    
    # return merger catalog, merger fraction, and formation statistics
    return dat, f_merge, Ms, ns, ibins

    
def sample_universe_no_cut(n_sample, n_downsample, mets, path, sfr):
    Ms, Ns, ns, bbhs = get_cosmic_data(path, mets)
    
    z = np.expm1(np.linspace(np.log(1+0.001), np.log(1+18), 5000))
    tlb = Planck18.lookback_time(z).to(u.Myr).value
    z_interp = interp1d(tlb, z)
    
    # ibins: metallicity indices
    # j_s: bbh merger indices
    # z_s: formation redshifts
    # Z_s: metallicities
    ibins, j_s, z_s, Z_s = zip(*[x for (x, i) in zip(mergers_uniform_proposal(mets, Ms, ns, Ns, sfh.mean_metal_z, sigma_logZ, sfr, bbhs), tqdm.tqdm(range(n_sample))) if i%n_downsample == 0])
    
    t_form_ind = 0
    m1_ind = 1
    m2_ind = 2
    bin_num_ind = 43
    
    n_merge = 0
    dat = []    
    for ii in tqdm.tqdm(range(len(mets))):
    
        met_mask = np.array(ibins) == ii
        if len(met_mask) > 0:
            # get all the formation lookback times from the formation redshifts
            # note that the units are in Myr since the COSMIC delay times are also in Myr
            t_form = Planck18.lookback_time(np.array(z_s)[met_mask]).to(u.Myr).value
            
            # get all the merger lookback times by subtracting the delay time
            # for the selected BBH merger at metallicity ii, and rows j_s[ind]
            t_merge = t_form - bbhs[ii][np.array(j_s)[met_mask], 0]
            #t_merge = t_form
            
            # select out anything that merges after the present
            merging_mask = t_merge >= 0
            
            # count up the number of mergers 
            n_mergers = np.count_nonzero(merging_mask)
            n_merge += n_mergers
            # convert the merger lookback times to merger redshifts
            #z_merge = z_interp(t_merge[merging_mask])
            
            # record the data 
            if n_mergers > 0:
                if len(dat) == 0:
                    dat = np.vstack([t_form, t_merge, np.array(z_s)[met_mask],\
                                     np.array(Z_s)[met_mask], np.ones(len(t_form)) * mets[ii],\
                                     bbhs[ii][np.array(j_s)[met_mask], m1_ind],\
                                     bbhs[ii][np.array(j_s)[met_mask], m2_ind],\
                                     bbhs[ii][np.array(j_s)[met_mask], bin_num_ind]])
                else:
                    dat = np.append(dat, np.vstack([t_form, t_merge, np.array(z_s)[met_mask],\
                                                    np.array(Z_s)[met_mask], np.ones(len(t_form)) * mets[ii],\
                                                    bbhs[ii][np.array(j_s)[met_mask], m1_ind],\
                                                    bbhs[ii][np.array(j_s)[met_mask], m2_ind],\
                                                    bbhs[ii][np.array(j_s)[met_mask], bin_num_ind]]),
                                    axis=1)
    # calculate the fraction of systems that merge from the systems that form
    f_merge = n_merge/len(ibins)
                    
    dat = pd.DataFrame(dat.T, columns=['t_form', 't_merge', 'z_form', 'met', 'met_cosmic', 'm1', 'm2', 'bin_num'])
    
    # return merger catalog, merger fraction, and formation statistics
    return dat, f_merge, Ms, ns, ibins
    
    
    