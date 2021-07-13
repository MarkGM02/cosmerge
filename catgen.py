"""class for generating catalogs"""

from astropy.cosmology import Planck18_arXiv_v2 as Planck18
from astropy import units as u
import numpy as np
import usample, utils


class Catalog():
    """Class for building a generic catalog of merging compact objects
    from COSMIC data in metallicity grid

    Attributes
    ----------
    dat_path : `string`
        specifies directory where COSMIC data is stored
        NOTE: we expect all dat_kstar1... files in the
        metallicity grid to be stored in the same path

    sfh_model : `method in sfh module`
        function that returns the star formation rate model in units
        of Msun per comoving volume per time
        choose from: sfh.md_14 or sfh.md_17 or supply your own!

    met_grid : `numpy.array`
        metallicity grid for COSMIC data

    kstar_1 : `string`
        kstar for the primary following COSMIC dat file naming notation

    kstar_2 : `string`
        kstar for the secondary following COSMIC dat file naming notation

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
        kwarg -- If specified, will select kstars that are a subset of the
        kstar_1 data

    kstar_2_select : `list`
        kwarg -- If specified, will select kstars that are a subset of the
        kstar_2 data

    """

    def __init__(self, dat_path, sfh_model, met_grid, kstar_1, kstar_2, SFstart, SFduration, pessimistic_cut, **kwargs):
        self.dat_path = dat_path
        self.sfh_model = sfh_model
        self.met_grid = met_grid
        self.kstar_1 = kstar_1
        self.kstar_2 = kstar_2
        self.SFstart = SFstart
        self.SFduration = SFduration
        self.pessimistic_cut = pessimistic_cut
        if 'kstar_1_select' not in kwargs:
            self.kstar_1_select = None
        if 'kstar_2_select' not in kwargs:
            self.kstar_2_select = None
        for key, value in kwargs.items():
            setattr(self, key, value)

        Ms, Ns, ns, merger_dat = utils.get_cosmic_data(path=self.dat_path,
                                                       kstar_1=self.kstar_1,
                                                       kstar_2=self.kstar_2,
                                                       mets=self.met_grid,
                                                       SFstart=13700.0,
                                                       SFduration=0.0,
                                                       pessimistic_cut=self.pessimistic_cut,
                                                       kstar_1_select=self.kstar_1_select,
                                                       kstar_2_select=self.kstar_2_select)
        self.M_sim = Ms
        self.N_sim = Ns
        self.n_merger = ns
        self.merger_dat = merger_dat

    def build_cat(self, n_sample, n_downsample, sigma_logZ=0.5, z_max=20):
        mergers, ibins = usample.generate_universe(n_sample=n_sample,
                                                   n_downsample=n_downsample,
                                                   mets=self.met_grid,
                                                   M_sim=self.M_sim, 
                                                   N_sim=self.N_sim, 
                                                   n_BBH=self.n_merger,
                                                   mergers=self.merger_dat,
                                                   sfh_model=self.sfh_model,
                                                   sigma_logZ=sigma_logZ,
                                                   z_max=z_max)

        z = np.expm1(np.linspace(0, np.log1p(z_max), 1000))
        M_merger = np.mean(self.M_sim[ibins] / self.n_merger[ibins])

        # divide by 1e6 because COSMIC time is in Myr
        M_star_U = np.trapz(self.sfr_model(z).to(u.Msun * u.yr**(-1) * u.Gpc**(-3)).value,
                            Planck18.lookback_time(z).to(u.yr).value)/1e6

        norm_fac = M_star_U / M_merger

        return mergers, norm_fac
