"""Rest-frame UV (M1450) quasar luminosity functions.

The UV counterpart of the bolometric QLF in :mod:`qlf_obs_data`, kept
separate because the measurements are published per magnitude rather than per
dex of luminosity. Conversions between M1450 and log L_bol, and between
``dPhi/dM`` and ``dPhi/dlogL``, are applied on load so the resulting
:class:`Qlf_data` objects share the bolometric convention used elsewhere.

Those conversions carry a bolometric-correction systematic that the bolometric
QLF does not, so these datasets are used for comparison rather than as
likelihood inputs.
"""

import numpy as np
import pandas as pd
import os

from qhtools.utils.magnitude_conversion import get_log_Lbol_from_M1450, get_M1450_from_log_Lbol,\
                                                    get_logphi_dlogL_from_logphi_dm, get_logphi_dm_from_logphi_dlogL

from qhtools.utils import my_utils


class Qlf_data:
    """One rest-frame UV luminosity function, converted to the bolometric axis.

    Mirrors :class:`~..qlf_obs_data.Qlf_data` so the same plotting helpers
    work, but the points originate as ``dPhi/dM`` at M1450 and are converted on
    load. That conversion carries a bolometric-correction systematic the
    directly-bolometric QLFs do not, so these are comparison data rather than
    likelihood inputs.
    """

    def __init__(self, x_data, y_data, errs, label, systematic_err=None, log_L_axis = None, log_qlf_fit = None,
                 covariance=None):
        self.x = x_data

        self.data = y_data

        self.errs = errs

        self.err_down = errs[0]
        self.err_up = errs[1]
        self.err = (self.err_down + self.err_up) / 2.

        self.label = label

        if systematic_err is not None:
            self.sys_err = systematic_err
            self.err = np.sqrt(self.err ** 2 + self.sys_err ** 2)
        else:
            self.sys_err = None

        self.log_L_axis = log_L_axis
        self.log_qlf_fit = log_qlf_fit

        self.covariance = covariance

    def check_nans(self):
        var = np.asarray(self.data)

        return np.isnan(np.sum(var))


# luminosity bins
log_L_axis = np.linspace(7, 16, 200)







# Kulkarni+19 z=2 UV data

# using fit for magn phi in Kulkarni
alpha = -3.53092475
beta = -1.60534524
log_phi = -6.3503986
log_M = -25.75633118
M_min = -35.
M_max = -15.
# sys_err = 0.1
sys_err = 0.05

params = [log_phi, log_M, alpha, beta]


x_data_M1450 = np.flip(np.asarray([-27.,  -26.4, -25.8, -25.2, -24.6, -24.,  -23.4, -22.8]))
x_data = my_utils.to_solar(get_log_Lbol_from_M1450(x_data_M1450))

y_data_dm = np.flip(np.asarray([-7.50505261, -7.11859175, -6.66916511, -6.34802073, -6.11617929, -5.92514892,
                         -5.85597776, -5.52286128]))
y_data = get_logphi_dlogL_from_logphi_dm(y_data_dm)


uperr = np.flip(np.asarray([0.04134302, 0.01933686, 0.01156437, 0.00844055, 0.00700755, 0.00659533,
                            0.00752911, 0.01403184]))
downerr = np.flip(np.asarray([0.04151245, 0.01935509, 0.01156835, 0.0084421,  0.00700844, 0.00659608,
                            0.00753022, 0.0140389 ]))

err_data_dm = (uperr + downerr)/2.
err_data = err_data_dm
# err_data = get_logphi_dlogL_from_logphi_dm(err_data_dm )


data_z2_UV_Kulkarni = Qlf_data(x_data=x_data, y_data=y_data, errs=[err_data, err_data], label="Kulkarni+19 QLF z=2 UV",
                      systematic_err=None, log_L_axis=None, log_qlf_fit=None)


data_z2_UV_Kulkarni_sys_err = Qlf_data(x_data=x_data, y_data=y_data, errs=[err_data, err_data], label="Kulkarni+19 QLF z=2 UV + sys err",
                      systematic_err=sys_err, log_L_axis=None, log_qlf_fit=None)





# Kulkarni+19 z=4 UV data

# using fit for magn phi in Kulkarni
alpha = -4.7616435
beta = -2.14357471
log_phi = -8.16796726
log_M = -27.36738842
M_min = -35.
M_max = -15.
sys_err = 0.2

params = [log_phi, log_M, alpha, beta]

x_data_M1450 = np.asarray([-22.5, -23.5, -24.5, -25.5,
                              -26.4, -27., -27.6, -28.2, -28.8])
x_data_M1450_z4 = x_data_M1450
x_data = my_utils.to_solar(get_log_Lbol_from_M1450(x_data_M1450))

y_data_dm = np.asarray([-6.3617414, -6.28928198, -6.57807238, -6.60310764,
                    -7.75746929, -8.12246818,  -8.7061302, -9.49744685, -10.06191718])
y_data_dm_z4 = y_data_dm
y_data = get_logphi_dlogL_from_logphi_dm(y_data_dm)

# threshold_lum = log_L_axis4 > 13.517710847782716
# num_den = np.trapezoid(np.power(10, log_qlf_fit4[threshold_lum]), log_L_axis4[threshold_lum])
# num_den_error = 0.04 * num_den
# print("num den L: ", num_den, "num den error: ", num_den_error)
# 
# threshold_m = M1450_axis4 < -26.82
# num_den = np.trapezoid(np.power(10, log_qlf_dm_fit4[threshold_m]), M1450_axis4[threshold_m])
# num_den_error = 0.04 * num_den
# print("num den M: ", num_den, "num den error: ", num_den_error)

uperr = np.asarray([0.3652876, 0.17410131, 0.22440214, 0.22440214,
                    0.01812821, 0.02773402, 0.05559563, 0.14659368,  0.29506734])

downerr = np.asarray([0.450883, 0.1844543, 0.24560444, 0.24560444,
                    0.01814328, 0.02778674, 0.05599574, 0.1529649,  0.34125893])

err_data_dm = (uperr + downerr)/2
err_data_dm_z4 = err_data_dm
err_data = err_data_dm
# err_data = get_logphi_dlogL_from_logphi_dm(err_data_dm )

data_z4_UV_Kulkarni = Qlf_data(x_data=x_data, y_data=y_data, errs=[err_data, err_data], label="Kulkarni+19 QLF z=4 UV",
                      systematic_err=None, log_L_axis=None, log_qlf_fit=None)


data_z4_UV_Kulkarni_sys_err = Qlf_data(x_data=x_data, y_data=y_data, errs=[err_data, err_data], label="Kulkarni+19 QLF z=4 UV + sys err",
                      systematic_err=sys_err, log_L_axis=None, log_qlf_fit=None)







# Niida+20 z=5 UV data (arXiv:2010.00481, ApJ 904, 89, "The Faint End of the
# Quasar Luminosity Function at z ~ 5 from the Subaru Hyper Suprime-Cam Survey").
# Two binned 1/Vmax samples covering -28.76 < M_1450 < -22.32 together:
#   * HSC-SSP Wide (81.8 deg^2, Lyman-break selected to i = 24.1) — the faint end,
#     ASYMMETRIC linear errors;
#   * SDSS DR7 spectroscopic quasars — the bright end, SYMMETRIC linear errors.
# Phi is LINEAR, in 1e-8 Mpc^-3 mag^-1 for BOTH tables (verified against the
# published tables and against their own DPL fit: at M_1450 = -26.38 the fit
# gives 0.85e-8 vs the tabulated 0.772e-8).
#
# Transcribed and checked against the paper.

# Their double power-law fit (max-likelihood, beta fixed). NB the paper's alpha
# is the FAINT-end slope; this module's convention is alpha = bright, beta = faint.
alpha = -2.90
beta = -1.22
log_phi = np.log10(1.01e-7)
log_M = -25.05
M_min = -35.
M_max = -15.
sys_err = 0.1

params = [log_phi, log_M, alpha, beta]

# A 1-sigma lower bound that runs to Phi <= 0 (the SDSS M = -28.13 bin, 9.2 +- 9.2)
# has no finite log error; cap it rather than emit -inf, which breaks errorbar().
_DOWNERR_CAP = 2.0


def _niida_arrays(M1450, phi_1e8, up_1e8, down_1e8):
    """M_1450 + linear Phi/1e-8 per mag -> (log L_bol [Lsun], log Phi/dex, errs)."""
    x_data = my_utils.to_solar(get_log_Lbol_from_M1450(M1450))

    y_data_dm = np.log10(phi_1e8) - 8.
    y_data = get_logphi_dlogL_from_logphi_dm(y_data_dm)

    with np.errstate(divide="ignore"):
        limup_dm = np.log10(phi_1e8 + up_1e8) - 8.
        limdown_dm = np.log10(np.clip(phi_1e8 - down_1e8, 0., None)) - 8.

    uperr = limup_dm - y_data_dm
    downerr = np.where(np.isfinite(limdown_dm), y_data_dm - limdown_dm, _DOWNERR_CAP)

    # The M_1450 tables below are already ordered faint -> bright, so x_data is
    # increasing in log L_bol like every other set in this module (no flip needed).
    return x_data, y_data, [downerr, uperr]


# --- HSC-SSP Wide (faint end) ---
x_data_M1450 = np.asarray([-22.57, -23.07, -23.57, -24.07, -24.57,
                           -25.07, -25.57, -26.07, -26.57, -27.07])
phi_1e8 = np.asarray([68.9, 23.1, 12.5, 10.7, 5.39, 7.78, 2.98, 1.80, 0.60, 1.20])
up_1e8 = np.asarray([6.9, 4.4, 3.4, 3.2, 2.46, 2.81, 2.01, 1.75, 1.39, 1.58])
down_1e8 = np.asarray([6.9, 3.7, 2.7, 2.5, 1.76, 2.13, 1.29, 0.98, 0.50, 0.78])

# The faintest bin sits ~4x ABOVE Niida+20's own best-fit DPL (their fit gives
# 16e-8 there vs the tabulated 69e-8) — it carries the largest incompleteness /
# contamination correction of the sample. Kept in the default object (it is
# published data); `_faintcut` drops it for figures that prefer not to show it.
mask_faintcut = x_data_M1450 < -22.8

x_data, y_data, errs = _niida_arrays(x_data_M1450, phi_1e8, up_1e8, down_1e8)

data_z5_UV_Niida_hsc = Qlf_data(x_data=x_data, y_data=y_data, errs=errs,
                                label="Niida+20 QLF z=5 UV (HSC)",
                      systematic_err=None, log_L_axis=None, log_qlf_fit=None)

data_z5_UV_Niida_hsc_sys_err = Qlf_data(x_data=x_data, y_data=y_data, errs=errs,
                                        label="Niida+20 QLF z=5 UV (HSC) + sys err",
                      systematic_err=sys_err, log_L_axis=None, log_qlf_fit=None)

_x_fc, _y_fc, _errs_fc = _niida_arrays(x_data_M1450[mask_faintcut], phi_1e8[mask_faintcut],
                                       up_1e8[mask_faintcut], down_1e8[mask_faintcut])

data_z5_UV_Niida_hsc_faintcut = Qlf_data(x_data=_x_fc, y_data=_y_fc, errs=_errs_fc,
                                         label="Niida+20 QLF z=5 UV (HSC, faint bin cut)",
                      systematic_err=None, log_L_axis=None, log_qlf_fit=None)


# --- SDSS DR7 (bright end); symmetric linear errors ---
x_data_M1450 = np.asarray([-26.13, -26.38, -26.63, -26.88, -27.13, -27.38,
                           -27.63, -27.88, -28.13, -28.38, -28.63])
phi_1e8 = np.asarray([0.639, 0.772, 0.628, 0.499, 0.277, 0.212,
                      0.083, 0.055, 0.0092, 0.018, 0.018])
err_1e8 = np.asarray([0.171, 0.087, 0.076, 0.068, 0.051, 0.044,
                      0.028, 0.023, 0.0092, 0.013, 0.013])

x_data, y_data, errs = _niida_arrays(x_data_M1450, phi_1e8, err_1e8, err_1e8)

data_z5_UV_Niida_sdss = Qlf_data(x_data=x_data, y_data=y_data, errs=errs,
                                 label="Niida+20 QLF z=5 UV (SDSS)",
                      systematic_err=None, log_L_axis=None, log_qlf_fit=None)

data_z5_UV_Niida_sdss_sys_err = Qlf_data(x_data=x_data, y_data=y_data, errs=errs,
                                         label="Niida+20 QLF z=5 UV (SDSS) + sys err",
                      systematic_err=sys_err, log_L_axis=None, log_qlf_fit=None)




# Schindler+23 z=6 UV data


# using fit for magn phi in Schindler
alpha = -3.84
beta = -1.70
log_phi = -8.75
log_M = -26.38
M_min = -35.
M_max = -15.
sys_err = 0.1

params = [log_phi, log_M, alpha, beta]

x_data_M1450 = np.flip(np.asarray([-22.75, -23.25, -23.75, -24.25, -24.75,
                                   -25.38, -25.99, -26.5, -27., -27.5]))

mask_bright = x_data_M1450 < -25.

x_data = my_utils.to_solar(get_log_Lbol_from_M1450(x_data_M1450))

y_data_dm_gpc = np.flip(np.asarray([22.391389122219632, 10.936991964398082, 8.742852680230358,
                                6.110286701868284, 4.957923688812892, 3.4650401067758763,
                                1.6924861426537576, 0.8517417617362988, 0.39781014182518903,
                                0.15075857012804558]))
limdown_dm_gpc = np.flip(np.asarray([14.742038344097162, 7.418914338109374, 6.019752089691551, 4.08339196329787,
                            3.1681838279256445, 2.8538391900705573, 1.4577911179928136, 0.712052729566329,
                            0.32278556179729945, 0.09075292175672141]))
limup_dm_gpc = np.flip(np.asarray([33.505920670754975, 15.884459706129727, 12.509637717136844, 9.143281838874717,
                                7.758706135491627, 4.207140676790739, 2.054962446679654, 1.0188346994013966,
                                0.5127274250089656, 0.23947184325526777]))


y_data_dm = np.log10(y_data_dm_gpc) - 9.
y_data = get_logphi_dlogL_from_logphi_dm(y_data_dm)

limup_dm = np.log10(limup_dm_gpc) - 9.
limdown_dm = np.log10(limdown_dm_gpc) - 9.

uperr = limup_dm - y_data_dm
downerr = y_data_dm - limdown_dm

err_data_dm = (uperr + downerr)/2.
err_data = err_data_dm

data_z6_UV_Schindler = Qlf_data(x_data=x_data, y_data=y_data, errs=[err_data, err_data],
                                label="Schindler+23 QLF z=6 UV",
                      systematic_err=None, log_L_axis=None, log_qlf_fit=None)


data_z6_UV_Schindler_sys_err = Qlf_data(x_data=x_data, y_data=y_data, errs=[err_data, err_data],
                                        label="Schindler+23 QLF z=6 UV + sys err",
                      systematic_err=sys_err, log_L_axis=None, log_qlf_fit=None)


data_z6_UV_Schindler_bright = Qlf_data(x_data=x_data[mask_bright], y_data=y_data[mask_bright],
                                       errs=[err_data[mask_bright], err_data[mask_bright]],
                                label="Schindler+23 QLF z=6 UV bright",
                      systematic_err=None, log_L_axis=None, log_qlf_fit=None)




# Wang+19 z=7 UV data


# using fit for magn phi in Wang
alpha = -2.54
beta = -1.90
log_phi = 0.5-9.
log_M = -25.2
M_min = -35.
M_max = -15.
sys_err = 0.1


params = [log_phi, log_M, alpha, beta]


x_data_M1450 = np.flip(np.asarray([-25.84, -26.45, -27.19]))

x_data = my_utils.to_solar(get_log_Lbol_from_M1450(x_data_M1450))

y_data_dm_gpc = np.flip(np.asarray([0.819, 0.35652, 0.149]))
limdown_dm_gpc = np.flip(np.asarray([0.51089, 0.28049, 0.10325]))
limup_dm_gpc = np.flip(np.asarray([1.146, 0.432, 0.1973]))


y_data_dm = np.log10(y_data_dm_gpc) - 9.
y_data = get_logphi_dlogL_from_logphi_dm(y_data_dm)

limup_dm = np.log10(limup_dm_gpc) - 9.
limdown_dm = np.log10(limdown_dm_gpc) - 9.

uperr = limup_dm - y_data_dm
downerr = y_data_dm - limdown_dm

err_data_dm = (uperr + downerr)/2.
err_data = err_data_dm

data_z7_UV_Wang = Qlf_data(x_data=x_data, y_data=y_data, errs=[err_data, err_data],
                                label="Wang+19 QLF z=7 UV",
                      systematic_err=None, log_L_axis=None, log_qlf_fit=None)


data_z7_UV_Wang_sys_err = Qlf_data(x_data=x_data, y_data=y_data, errs=[err_data, err_data],
                                        label="Wang+19 QLF z=7 UV + sys err",
                      systematic_err=sys_err, log_L_axis=None, log_qlf_fit=None)




# Matsuoka+23 z=7 UV data (arXiv:2305.11225, "Quasar Luminosity Function at z=7")
# Binned LF, Table 2, 6.55 < z < 7.15 (35 quasars: 22 SHELLQs + 13 brighter lit).
# Phi in Gpc^-3 mag^-1, symmetric linear errors. Same UV-QLF form as Wang above.

sys_err = 0.1

x_data_M1450 = np.flip(np.asarray([-23.25, -23.75, -24.25, -24.75, -25.25,
                                   -25.75, -26.25, -26.75, -27.50]))

x_data = my_utils.to_solar(get_log_Lbol_from_M1450(x_data_M1450))

y_data_dm_gpc = np.flip(np.asarray([2.5, 3.0, 3.5, 3.2, 1.58, 0.75, 0.63, 0.18, 0.082]))
err_dm_gpc    = np.flip(np.asarray([1.8, 1.5, 1.4, 1.3, 0.91, 0.53, 0.26, 0.10, 0.047]))
limup_dm_gpc   = y_data_dm_gpc + err_dm_gpc
limdown_dm_gpc = y_data_dm_gpc - err_dm_gpc

y_data_dm = np.log10(y_data_dm_gpc) - 9.
y_data = get_logphi_dlogL_from_logphi_dm(y_data_dm)

limup_dm = np.log10(limup_dm_gpc) - 9.
limdown_dm = np.log10(limdown_dm_gpc) - 9.

uperr = limup_dm - y_data_dm
downerr = y_data_dm - limdown_dm

data_z7_UV_Matsuoka = Qlf_data(x_data=x_data, y_data=y_data, errs=[downerr, uperr],
                               label="Matsuoka+23 QLF z=7 UV",
                      systematic_err=None, log_L_axis=None, log_qlf_fit=None)


data_z7_UV_Matsuoka_sys_err = Qlf_data(x_data=x_data, y_data=y_data, errs=[downerr, uperr],
                                       label="Matsuoka+23 QLF z=7 UV + sys err",
                      systematic_err=sys_err, log_L_axis=None, log_qlf_fit=None)





if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax_main = plt.subplots()

    # colors = [ "gray", "C2"]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    datas = [data_z4_UV_Kulkarni]
    for data, color in zip(datas, colors):
        print("x data", data.x)
        print("y data", data.data)
        print("y err", data.err)
        if data.log_L_axis is not None:
            ax_main.plot(my_utils.to_ergs(data.log_L_axis), data.log_qlf_fit, label=data.label + " fit", linestyle="--", color=color)

        ax_main.errorbar(my_utils.to_ergs(data.x), data.data, data.err, label=data.label, linestyle="", marker="o", color=color)

    ax_main.set_xlabel("log L")
    ax_main.set_ylabel("log Phi [Mpc^-3 dex^-1]")

    ax_main.set_xlim(44, 49)
    ax_main.set_ylim(-12, -4)
    ax_main.legend(loc="lower left")


    plt.show()