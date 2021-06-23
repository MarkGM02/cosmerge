"""class for generating catalogs"""

from astropy.cosmology import Planck18_arXiv_v2 as Planck18
from astropy import units as u
import numpy as np
import usample


class Catalog():
    """Class for building a generic catalog of merging compact objects
    from COSMIC data in metallicity grid

    Attributes
    ----------
    merger_type : `str`
        Defines the type of merging compact objects with options
        including: 'BNS', 'NSBH', 'BBH'


    """

    def __init__(self, dat_path, merger_type, sfr_model, met_grid):
        self.dat_path = dat_path
        self.merger_type = merger_type
        self.sfr_model = sfr_model
        self.met_grid = met_grid

    def build_cat(self, n_sample, n_downsample, sigma_logZ=0.5, z_max=20):
        mergers, Ms, ns, ibins = usample.generate_universe(n_sample=n_sample,
                                                           n_downsample=n_downsample,
                                                           mets=self.met_grid,
                                                           path=self.dat_path,
                                                           sfr_model=self.sfr_model,
                                                           sigma_logZ=sigma_logZ,
                                                           z_max=z_max)

        z = np.expm1(np.linspace(0, np.log1p(z_max), 1000))
        M_merger = np.mean(Ms[ibins] / ns[ibins])

        # divide by 1e6 because COSMIC time is in Myr
        M_star_U = np.trapz(self.sfr_model(z).to(u.Msun * u.yr**(-1) * u.Gpc**(-3)).value,
                            Planck18.lookback_time(z).to(u.yr).value)/1e6

        norm_fac = M_star_U / M_merger

        return mergers, norm_fac


