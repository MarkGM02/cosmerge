"""contains utility methods"""

import numpy as np
import pandas as pd


def get_cosmic_data(path, met_read):
    if met_read > 0.00009:
        f = '{}/dat_kstar1_13_14_kstar2_13_14_SFstart_13700.0_SFduration_0.0_metallicity_{}.h5'.format(met_read,met_read)
    else:
        f = '0.000085/dat_kstar1_13_14_kstar2_13_14_SFstart_13700.0_SFduration_0.0_metallicity_8.5e-05.h5'
    N_stars = np.max(pd.read_hdf(path+'/'+f, key='n_stars'))[0]
    M_stars = np.max(pd.read_hdf(path+'/'+f, key='mass_stars'))[0]
    bpp = pd.read_hdf(path+'/'+f, key='bpp')
    bpp_ce = bpp.loc[(bpp.evol_type == 7)]
    bpp_ce_1 = bpp_ce.loc[(bpp_ce.RRLO_1 > 1)]
    bpp_ce_2 = bpp_ce.loc[(bpp_ce.RRLO_2 > 1)]
    bpp_ce_1_pess = bpp_ce_1.loc[~bpp_ce_1.kstar_1.isin([2,8])]
    bpp_ce_2_pess = bpp_ce_2.loc[~bpp_ce_2.kstar_2.isin([2,8])]
    bpp_bin_num = bpp_ce_1_pess.bin_num.unique()
    bpp_bin_num = np.append(bpp_bin_num, bpp_ce_2_pess.bin_num.unique())
    bpp_bin_num = np.unique(bpp_bin_num)
    BBH = bpp.loc[(bpp.kstar_1 == 14) & (bpp.kstar_2 == 14) & (bpp.evol_type == 3) & (bpp.bin_num.isin(bpp_bin_num))]
    bpp = []
    return np.array(BBH), N_stars, M_stars

