"""Shen+2020 bolometric quasar luminosity function fits.

Tabulated double-power-law parameters -- ``phi_star``, ``L_star``, ``gamma_1``,
``gamma_2`` -- from Shen et al. (2020), keyed by redshift string, in two
flavours: the ``_free`` fits (each redshift fitted independently) and the
``_polished`` fits (smoothed across redshift). The polished set is the single
source of truth imported by :mod:`qlf_obs_data`.

Also provides the model evaluation and plotting helpers used to overlay these
fits on the measured QLF points, in both the "A" and "B" parameterisations of
that paper.
"""

import numpy as np
import unyt




redshift_obs = np.array(['0.2','0.8','1.2','1.6','2.0','3.0','4.0','5.0'])

c = 3e8*unyt.m/unyt.s
L_sun = 3.828*1e33 #erg s^-1
factor = 3.2*10**4
efficiency = 0.1


dic_gamma_1_free = {'0.2': 0.812, '0.4': 0.561, '0.8': 0.599, '1.2': 0.504, '1.6': 0.484, '2.0': 0.411, '3.0': 0.424, '4.0': 0.403, '5.0': 0.260, '6.0': 1.196}
dic_gamma_2_free = {'0.2': 1.753, '0.4': 2.108, '0.8': 2.199, '1.2': 2.423, '1.6': 2.546, '2.0': 2.487, '3.0': 1.787, '4.0': 1.988, '5.0': 1.916, '6.0': 2.349}
dic_phi_star_free = {'0.2': -4.405, '0.4': -4.151, '0.8': -4.412, '1.2': -4.530, '1.6': -4.668, '2.0': -4.679, '3.0': -4.698, '4.0': -5.244, '5.0': -5.258, '6.0': -8.019}
dic_L_star_free = {'0.2': 11.407, '0.4': 11.650, '0.8': 12.223, '1.2': 12.662, '1.6': 12.919, '2.0': 13.011, '3.0': 12.708, '4.0': 12.730, '5.0': 12.319, '6.0': 13.709} 

dic_gamma_1_error_free = {'0.2': 0.046, '0.4': 0.041, '0.8': 0.031, '1.2': 0.030, '1.6': 0.034, '2.0': 0.029, '3.0': 0.070, '4.0': 0.162, '5.0': 0.425, '6.0': 0.246}
dic_gamma_2_error_free = {'0.2': 0.087, '0.4': 0.075, '0.8': 0.070, '1.2': 0.060, '1.6': 0.082, '2.0': 0.063, '3.0': 0.058, '4.0': 0.099, '5.0': 0.123, '6.0': 0.692}
dic_phi_star_error_free = {'0.2': 0.278, '0.4': 0.111, '0.8': 0.080, '1.2': 0.052, '1.6': 0.058, '2.0': 0.046, '3.0': 0.107, '4.0': 0.174, '5.0': 0.357, '6.0': 1.099}
dic_L_star_error_free = {'0.2': 0.223, '0.4': 0.080, '0.8': 0.059, '1.2': 0.036, '1.6': 0.040, '2.0': 0.032, '3.0': 0.086, '4.0': 0.134, '5.0': 0.261, '6.0': 0.639} 

dic_gamma_1_polished = {'0.2': 0.787, '0.4': 0.561, '0.8': 0.599, '1.2': 0.504, '1.6': 0.484, '2.0': 0.411, '3.0': 0.424, '4.0': 0.213, '5.0': 0.245, '6.0': 1.509}
dic_gamma_2_polished = {'0.2': 1.713, '0.4': 2.108, '0.8': 2.199, '1.2': 2.423, '1.6': 2.546, '2.0': 2.487, '3.0': 1.878, '4.0': 1.885, '5.0': 1.912, '6.0': 1.509}
dic_phi_star_polished = {'0.2': -4.240, '0.4': -4.151, '0.8': -4.412, '1.2': -4.530, '1.6': -4.668, '2.0': -4.679, '3.0': -4.698, '4.0': -5.034, '5.0': -5.243, '6.0': -5.452}
dic_L_star_polished = {'0.2': 11.275, '0.4': 11.650, '0.8': 12.223, '1.2': 12.662, '1.6': 12.919, '2.0': 13.011, '3.0': 12.708, '4.0': 12.562, '5.0': 12.308, '6.0': 11.978} 

dic_gamma_1_error_polished = {'0.2': 0.024, '0.4': 0.041, '0.8': 0.031, '1.2': 0.030, '1.6': 0.034, '2.0': 0.029, '3.0': 0.070, '4.0': 0.092, '5.0': 0.211, '6.0': 0.058}
dic_gamma_2_error_polished = {'0.2': 0.046, '0.4': 0.075, '0.8': 0.070, '1.2': 0.060, '1.6': 0.082, '2.0': 0.063, '3.0': 0.058, '4.0': 0.052, '5.0': 0.086, '6.0': 0.058}
dic_phi_star_error_polished = {'0.2': 0.000, '0.4': 0.111, '0.8': 0.080, '1.2': 0.052, '1.6': 0.058, '2.0': 0.046, '3.0': 0.107, '4.0': 0, '5.0': 0, '6.0': 0}
dic_L_star_error_polished = {'0.2': 0.223, '0.4': 0.080, '0.8': 0.059, '1.2': 0.036, '1.6': 0.040, '2.0': 0.032, '3.0': 0.086, '4.0': 0.027, '5.0': 0.062, '6.0': 0.055} 

a0_A = 0.8569
a0_err_up_A = 0.0247
a0_err_down_A = 0.0253

a1_A = -0.2614
a1_err_up_A = 0.0162
a1_err_down_A = 0.0164

a2_A = 0.0200
a2_err_up_A = 0.0011
a2_err_down_A = 0.0011

b0_A = 2.5375
b0_err_up_A = 0.0177
b0_err_down_A = 0.0187

b1_A = -1.0425
b1_err_up_A = 0.0164
b1_err_down_A = 0.0182

b2_A = 1.1201
b2_err_up_A = 0.0199
b2_err_down_A = 0.0207

c0_A = 13.0088
c0_err_up_A = 0.0090
c0_err_down_A = 0.0091

c1_A = -0.5759
c1_err_up_A = 0.0018
c1_err_down_A = 0.0020

c2_A = 0.4554
c2_err_up_A = 0.0028
c2_err_down_A = 0.0027

d0_A = -3.5326
d0_err_up_A = 0.0235
d0_err_down_A = 0.0209

d1_A = -0.3936
d1_err_up_A = 0.0070
d1_err_down_A = 0.0073


a0_B = 0.3653
a0_err_up_B = 0.0115
a0_err_down_B = 0.0114

a1_B = -0.6006
a1_err_up_B = 0.0422
a1_err_down_B = 0.0417

b0_B = 2.4709
b0_err_up_B = 0.0163
b0_err_down_B = 0.0169

b1_B = -0.9963
b1_err_up_B = 0.0167
b1_err_down_B = 0.0161

b2_B = 1.0716
b2_err_up_B = 0.0180
b2_err_down_B = 0.0181

c0_B = 12.9656
c0_err_up_B = 0.0092
c0_err_down_B = 0.0089

c1_B = -0.5758
c1_err_up_B = 0.0020
c1_err_down_B = 0.0019

c2_B = 0.4698
c2_err_up_B = 0.0025
c2_err_down_B = 0.0026

d0_B = -3.6276
d0_err_up_B = 0.0209
d0_err_down_B = 0.0203

d1_B = -0.3444
d1_err_up_B = 0.0063
d1_err_down_B = 0.0061

z_ref = 2

num_z = 20
arr_z = np.linspace(0, 8, num_z, endpoint=True)

def parameter_fit_B(z,a0=a0_B,a1=a1_B,b0=b0_B,b1=b1_B,b2=b2_B,c0=c0_B,c1=c1_B,c2=c2_B,d0=d0_B,d1=d1_B):
    """Shen+2020 model-B redshift evolution of one DPL parameter.

    Model B parameterises each parameter as a function of (1+z) with a pivot,
    giving a smooth evolution rather than independent per-redshift fits.
    """
    gamma1_z = a0*((1+z)/(1+z_ref))**a1
    gamma2_z = 2*b0/(((1+z)/(1+z_ref))**b1+((1+z)/(1+z_ref))**b2)
    logLstar_z = 2*c0/(((1+z)/(1+z_ref))**c1+((1+z)/(1+z_ref))**c2)
    logphi_z = d0+d1*(1+z)

    return gamma1_z, gamma2_z, logLstar_z, logphi_z

def parameter_fit_A(z,a0=a0_A,a1=a1_A,a2=a2_A,b0=b0_A,b1=b1_A,b2=b2_A,c0=c0_A,c1=c1_A,c2=c2_A,d0=d0_A,d1=d1_A):
    """Shen+2020 model-A redshift evolution of one DPL parameter.

    The alternative parameterisation to :func:`parameter_fit_B`; the two differ
    in how they extrapolate beyond the fitted redshift range.
    """
    gamma1_z = a0+a1*(1+z)+a2*(2*(1+z)**2-1)
    gamma2_z = 2*b0/(((1+z)/(1+z_ref))**b1+((1+z)/(1+z_ref))**b2)
    logLstar_z = 2*c0/(((1+z)/(1+z_ref))**c1+((1+z)/(1+z_ref))**c2)
    logphi_z = d0+d1*(1+z)

    return gamma1_z, gamma2_z, logLstar_z, logphi_z


def luminosity_model(L, gamma_1, gamma_2, log_L_star, log_phi_star):
    """Double power law: phi(L) for given slopes, break and normalisation.

    ``gamma_1`` is the faint-end slope, ``gamma_2`` the bright-end slope, with
    the break at ``log_L_star`` and amplitude ``log_phi_star``.
    """
    phi_star = 10**(log_phi_star)
    Lstar_star = 10**(log_L_star)*L_sun

    ratio = 10**L/Lstar_star

    phi = phi_star/((ratio)**gamma_1+(ratio)**gamma_2)
    integral_number = np.trapezoid(phi,L)
    return np.log10(integral_number)


def plot_observation_B(z,Lmin,Lmax,N_sample=1000, num_model=30, keys='free'):
    """Overlay model-B QLF realisations on the observed points at one redshift."""
    arr = np.linspace(Lmin, Lmax, num_model, endpoint=True)
        
    a0_sample = np.random.normal(a0_B, (a0_err_up_B+a0_err_down_B)/2, size=N_sample)
    a1_sample = np.random.normal(a1_B, (a1_err_up_B+a1_err_down_B)/2, size=N_sample)
#    a2_sample = np.random.normal(a2, (a2_err_up+a2_err_down)/2, size=N_sample)

    b0_sample = np.random.normal(b0_B, (b0_err_up_B+b0_err_down_B)/2, size=N_sample)
    b1_sample = np.random.normal(b1_B, (b1_err_up_B+b1_err_down_B)/2, size=N_sample)
    b2_sample = np.random.normal(b2_B, (b2_err_up_B+b2_err_down_B)/2, size=N_sample)

    c0_sample = np.random.normal(c0_B, (c0_err_up_B+c0_err_down_B)/2, size=N_sample)
    c1_sample = np.random.normal(c1_B, (c1_err_up_B+c1_err_down_B)/2, size=N_sample)
    c2_sample = np.random.normal(c2_B, (c2_err_up_B+c2_err_down_B)/2, size=N_sample)

    d0_sample = np.random.normal(d0_B, (d0_err_up_B+d0_err_down_B)/2, size=N_sample)
    d1_sample = np.random.normal(d1_B, (d1_err_up_B+d1_err_down_B)/2, size=N_sample)

    Result = []

    for i in range(N_sample):
        gamma1_z, gamma2_z, logLstar_z, logphi_z = parameter_fit_B(z,a0=a0_sample[i],a1=a1_sample[i],b0=b0_sample[i],b1=b1_sample[i],b2=b2_sample[i],c0=c0_sample[i],c1=c1_sample[i],c2=c2_sample[i],d0=d0_sample[i],d1=d1_sample[i])
        result = luminosity_model(arr, gamma1_z, gamma2_z, logLstar_z, logphi_z)
        Result.append(result)

    Result = np.array(Result)
    ebar = np.std(Result)

    Result_mean = np.mean(Result)

    return Result_mean, ebar

def plot_observation_A(z,Lmin,Lmax,N_sample=1000, num_model=30, keys='free'):
    """Overlay model-A QLF realisations on the observed points at one redshift."""
    arr = np.linspace(Lmin, Lmax, num_model, endpoint=True)
        
    a0_sample = np.random.normal(a0_A, (a0_err_up_A+a0_err_down_A)/2, size=N_sample)
    a1_sample = np.random.normal(a1_A, (a1_err_up_A+a1_err_down_A)/2, size=N_sample)
    a2_sample = np.random.normal(a2_A, (a2_err_up_A+a2_err_down_A)/2, size=N_sample)

    b0_sample = np.random.normal(b0_A, (b0_err_up_A+b0_err_down_A)/2, size=N_sample)
    b1_sample = np.random.normal(b1_A, (b1_err_up_A+b1_err_down_A)/2, size=N_sample)
    b2_sample = np.random.normal(b2_A, (b2_err_up_A+b2_err_down_A)/2, size=N_sample)

    c0_sample = np.random.normal(c0_A, (c0_err_up_A+c0_err_down_A)/2, size=N_sample)
    c1_sample = np.random.normal(c1_A, (c1_err_up_A+c1_err_down_A)/2, size=N_sample)
    c2_sample = np.random.normal(c2_A, (c2_err_up_A+c2_err_down_A)/2, size=N_sample)

    d0_sample = np.random.normal(d0_A, (d0_err_up_A+d0_err_down_A)/2, size=N_sample)
    d1_sample = np.random.normal(d1_A, (d1_err_up_A+d1_err_down_A)/2, size=N_sample)

    Result = []

    for i in range(N_sample):
        gamma1_z, gamma2_z, logLstar_z, logphi_z = parameter_fit_A(z,a0=a0_sample[i],a1=a1_sample[i],a2=a2_sample[i],b0=b0_sample[i],b1=b1_sample[i],b2=b2_sample[i],c0=c0_sample[i],c1=c1_sample[i],c2=c2_sample[i],d0=d0_sample[i],d1=d1_sample[i])
        result = luminosity_model(arr, gamma1_z, gamma2_z, logLstar_z, logphi_z)
        Result.append(result)

    Result = np.array(Result)
    ebar = np.std(Result)

    Result_mean = np.mean(Result)

    return Result_mean, ebar



def plot_phi_obs_B(Lmin,Lmax):
    """Plot the model-B phi_star evolution against the per-redshift fits."""
    Phi_obs = []
    Error_obs = []
    Z = []
    for i in range(num_z):
        z = float(arr_z[i])
        phi_obs, error_obs = plot_observation_B(z=z,Lmin=Lmin,Lmax=Lmax)
        Phi_obs.append(phi_obs)
        Error_obs.append(error_obs)
    #    Z.append(float(redshift_obs[i]))
        Z.append(z)

    return np.asarray(Phi_obs), np.asarray(Error_obs), np.asarray(Z)

def plot_phi_obs_A(Lmin,Lmax):
    """Plot the model-A phi_star evolution against the per-redshift fits."""
    Phi_obs = []
    Error_obs = []
    Z = []
    for i in range(num_z):
        z = float(arr_z[i])
        phi_obs, error_obs = plot_observation_A(z=z,Lmin=Lmin,Lmax=Lmax)
        Phi_obs.append(phi_obs)
        Error_obs.append(error_obs)
    #    Z.append(float(redshift_obs[i]))
        Z.append(z)

    return np.asarray(Phi_obs), np.asarray(Error_obs), np.asarray(Z)



