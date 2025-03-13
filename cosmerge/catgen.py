"""class for generating catalogs"""

from astropy.cosmology import Planck18
from astropy import units as u
import numpy as np
from cosmerge import usample, utils


class Catalog():
    """Class for building a generic catalog of merging compact objects
    from COSMIC data in metallicity grid

    Attributes
    ----------
    dat_path : string
        specifies directory where COSMIC data is stored
        NOTE: we expect all dat_kstar1... files in the
        metallicity grid to be stored in the same path

    sfh_model : method in sfh module
        function that returns the star formation rate model in units
        of Msun per comoving volume per time
        choose from: sfh.md_14, sfh.md_17, sfh.van_son_tng, or supply your own!

    met_grid : numpy.array
        metallicity grid for COSMIC data

    kstar_1 : string
        kstar for the primary following COSMIC dat file naming notation

    kstar_2 : string
        kstar for the secondary following COSMIC dat file naming notation

    SFstart : float
        ZAMS lookback time for COSMIC population

    SFduration : float
        Duration of star formation for COSMIC population

    skew : bool
        Whether to use the skewed log-normal metallicity distribution.

    pessimistic_cut : bool, optional
        kwarg -- decides whether to apply the pessimistic
        cut to the merger data based on whether there where
        common envelope events with a Hertzsprung Gap donor

        Note: this is unnecessary if you specified
        cemergeflag = 1 in the Params file

    CE_cut : bool, optional
        kwarg -- decides whether to throw out
        CE binaries

    SMT_cut : bool, optional
        kwarg -- decides whether to throw out 
        stable mass transfer binaries

    CE_cool_filter : bool, optional
        kwarg -- decides whether to filter out stars with ZAMS mass
        > 40 Msun

    kstar_1_select : list, optional
        kwarg -- If specified, will select kstars that are a subset of the
        kstar_1 data

    kstar_2_select : list, optional
        kwarg -- If specified, will select kstars that are a subset of the
        kstar_2 data

    """

    def __init__(self, dat_path, sfh_model, met_grid, kstar_1, kstar_2, skew, **kwargs):
        self.dat_path = dat_path
        self.sfh_model = sfh_model
        self.met_grid = met_grid
        self.kstar_1 = kstar_1
        self.kstar_2 = kstar_2
        self.skew = skew

        kwarg_list = ['kstar_1_select', 'kstar_2_select', 'pessimistic_cut', 'CE_cut', 'SMT_cut', 'CE_cool_filter']
        for k in kwarg_list:
            if 'kstar' in k:
                setattr(self, k, None)
            elif ('cut' in k) or ('filter' in k):
                setattr(self, k, False)
            elif k == 'SFstart':
                setattr(self, k, 13700.0)
            elif k == 'SFduration':
                setattr(self, k, 0.0)

        for key, value in kwargs.items():
            setattr(self, key, value)

        Ms, Ns, ns, merger_dat = utils.get_cosmic_data(path=self.dat_path,
                                                       kstar_1=self.kstar_1,
                                                       kstar_2=self.kstar_2,
                                                       mets=self.met_grid,
                                                       SFstart=self.SFstart,
                                                       SFduration=self.SFduration,
                                                       pessimistic_cut=self.pessimistic_cut,
                                                       CE_cool_filter=self.CE_cool_filter,
                                                       CE_cut = self.CE_cut,
                                                       SMT_cut = self.SMT_cut,
                                                       kstar_1_select=self.kstar_1_select,
                                                       kstar_2_select=self.kstar_2_select)
        self.M_sim = Ms
        self.N_sim = Ns
        self.n_merger = ns
        self.merger_dat = merger_dat

    def build_cat(self, n_sample, n_downsample, sigma_log10Z=0.5, z_max=20):
        mergers, ibins = usample.generate_universe(n_sample=n_sample,
                                                   n_downsample=n_downsample,
                                                   mets=self.met_grid,
                                                   M_sim=self.M_sim, 
                                                   N_sim=self.N_sim, 
                                                   n_merger=self.n_merger,
                                                   mergers=self.merger_dat,
                                                   sfh_model=self.sfh_model,
                                                   skew=self.skew,
                                                   sigma_log10Z=sigma_log10Z,
                                                   z_max=z_max)

        z = np.expm1(np.linspace(0, np.log1p(z_max), 1000))
        M_merger = np.mean(self.M_sim[ibins] / self.n_merger[ibins])

        # divide by 1e6 because COSMIC time is in Myr
        M_star_U = np.trapz(self.sfh_model(z).to(u.Msun * u.yr**(-1) * u.Gpc**(-3)).value,
                            Planck18.lookback_time(z).to(u.yr).value)/1e6

        norm_fac = M_star_U / M_merger

        return mergers, norm_fac
