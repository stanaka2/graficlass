import sys
import argparse
import numpy as np
import pathlib
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import quad
import classy
from classy import Class

import re
from packaging import version
classy_version = re.sub("[^0-9^.]", "", classy.__version__)

grafic_format = True
use_tk_dtot = False

kmin = -5
kmax = 2
klist = np.logspace(kmin, kmax, 2001)
logklist = np.log(klist)

# Numerical oscillations occur at Tk in the CDM when log_kmax exceeds 4 in the Class computation.
# When a larger kmax is required, it is obtained by interpolation.
class_kmax = min(3.0, kmax)
P_k_max_h = 10**class_kmax

if kmax > 3.0:
    print("# Note. For k>1000, the transfer function is calculated approximately using extrapolation.")

if version.parse(classy_version) >= version.parse('3.0.0'):
    output_setting = 'dTk vTk'
else:
    output_setting = ['dTk', 'vTk']

# defalutsetting set to massless neutrino
defaultsetting = {
    'h': 0.67556,
    'T_cmb': 2.7255,
    'omega_cdm': 0.12038,
    'omega_b': 0.022032,
    'N_ur': 3.044,  # N_eff
    'N_ncdm': 0,
    'deg_ncdm': 0,
    'm_ncdm': 0.0,
    'P_k_max_h/Mpc': P_k_max_h,
    'Omega_fld': 0,
    'Omega_k': 0,
    'k_pivot': 0.05,
    'A_s': 2.215e-9,
    'n_s': 0.9619,
    'z_max_pk': 300,
    'gauge': 'synchronous',
    # 'gauge': 'Newtonian',
    'output': output_setting,
    'background_verbose': 1

    # Note N_ur : in default.ini in https://github.com/wullm/class_public/blob/master/default.ini
    # If you have respectively 0,1,2, or 3 MASSIVE neutrinos and the default T_ncdm of 0.71611,
    # designed to give M_tot/omega_nu of 93.14 eV, and if you want N_eff equal to 3.044,
    # then you should pass for N_ur 3.044,2.0308,1.0176, or 0.00441
}


def set_precisionsetting_dict(no_high_precision=False):
    ncdm_precisionsettings = {
        'tol_ncdm': 1.e-8,  # defalut 1e-3
        'tol_ncdm_bg': 1.e-10  # defalut 1e-5
    }

    ncdm_high_precisionsettings = {
        'tol_ncdm': 1.e-8,  # defalut 1e-3
        'tol_ncdm_bg': 1.e-10,  # defalut 1e-5
        'ncdm_fluid_approximation': 3,
        'l_max_ncdm': 100,  # defalut 17
        'tol_ncdm_synchronous': 1.e-10,  # default 1e-3
        'tol_ncdm_newtonian': 1.e-10,  # default 1e-5

        'delta_l_max': 1000.,
        # 'k_min_tau0': 0.002,
        # 'k_max_tau0_over_l_max': 3.,
        # 'k_step_sub': 0.015,
        # 'k_step_super': 0.0001,
        # 'k_step_super_reduction': 0.1,
    }

    if version.parse(classy_version) >= version.parse('3.0.0'):
        precisionsettings = {  # These are sufficient for this purpose
            'tol_perturbations_integration': 1.e-8,  # default 1e-5
            'perturbations_sampling_stepsize': 1e-4,  # default 0.1
            'background_Nloga': 4620,  # default 3000
        }
        ncdm_high_precisionsettings.update({
            # 'ncdm_quadrature_strategy': 3, # the trapezoidal rule on [0,q_max] where q_max is the next input.
            # 'ncdm_maximum_q': 15,  # default 15
            # 'ncdm_N_momentum_bins': 150
        })
    else:
        precisionsettings = {  # These are sufficient for this purpose
            'tol_perturb_integration': 1.e-8,
            'perturb_sampling_stepsize': 1e-4,
        }
        ncdm_high_precisionsettings.update({
            # 'Quadrature strategy': 3,
            # 'Maximum q': 15,
            # 'Number of momentum bins': 150
        })

    if no_high_precision:
        precisionsettings.update(ncdm_precisionsettings)
    else:
        precisionsettings.update(ncdm_high_precisionsettings)

    return precisionsettings


def get_pk(cosmo, trans):
    As = float(cosmo.pars['A_s'])
    ns = float(cosmo.pars['n_s'])
    k0 = float(cosmo.pars['k_pivot']) / (float(cosmo.pars['h']))
    return 2 * np.pi**2 / klist**3 * As * (klist / k0)**(ns - 1) * trans**2


def get_pk_dimensionless(cosmo, trans):
    As = float(cosmo.pars['A_s'])
    ns = float(cosmo.pars['n_s'])
    k0 = float(cosmo.pars['k_pivot']) / (float(cosmo.pars['h']))
    return As * (klist / k0)**(ns - 1) * trans**2


def get_pk_ij(cosmo, transi, transj):
    As = float(cosmo.pars['A_s'])
    ns = float(cosmo.pars['n_s'])
    k0 = float(cosmo.pars['k_pivot']) / (float(cosmo.pars['h']))
    return 2 * np.pi**2 / klist**3 * As * (klist / k0)**(ns - 1) * (transi * transj)


def get_pk_ij_dimensionless(cosmo, transi, transj):
    As = float(cosmo.pars['A_s'])
    ns = float(cosmo.pars['n_s'])
    k0 = float(cosmo.pars['k_pivot']) / (float(cosmo.pars['h']))
    return As * (klist / k0)**(ns - 1) * (transi * transj)


def calc_pk(cosmo, tk_tbl):
    tk_spl = ius(logklist, tk_tbl)
    pk_spl = ius(logklist, get_pk(cosmo, tk_spl(logklist)))
    pk = pk_spl(logklist)
    return pk


def calc_pk_ij(cosmo, tki_tbl, tkj_tbl):
    tki_spl = ius(logklist, tki_tbl)
    tkj_spl = ius(logklist, tkj_tbl)
    pk_spl = ius(logklist, get_pk_ij(cosmo, tki_spl(logklist), tkj_spl(logklist)))
    pk = pk_spl(logklist)
    return pk


def get_THwindow(kR):
    return 3 * (np.sin(kR) - kR * np.cos(kR)) / kR**3


def get_sigma8(cosmo, trans_totmat0):
    R = 8
    DeltakW = ius(logklist, get_pk_dimensionless(cosmo, trans_totmat0) * get_THwindow(klist * R)**2)
    return np.sqrt(quad(lambda x: DeltakW(x), logklist[0], logklist[-1], limit=200)[0])


def calc_ddplus(a, cosmo):
    if a == 0:
        return 0

    Om = cosmo.Omega0_m()
    Ov = cosmo.Omega_Lambda()
    Ok = cosmo.Omega0_k()

    eta = np.sqrt(Om / a + Ov * a * a + Ok)
    ddplus = (2.5 / (eta * eta * eta))
    return ddplus


def calc_dplus(a, cosmo):
    if a == 0:
        return 0

    Om = cosmo.Omega0_m()
    Ov = cosmo.Omega_Lambda()
    Ok = cosmo.Omega0_k()

    eta = np.sqrt(Om / a + Ov * a * a + Ok)
    dplus = eta / a * quad(calc_ddplus, 0.0, a, limit=200, args=(cosmo))[0]
    return dplus


def is_num(s):
    try:
        float(s)
    except ValueError:
        return s
    else:
        return float(s)


def load_input_paramfile(inputfile):
    input_params = {}
    with open(inputfile) as f:
        for line in f.readlines():
            if line == "\n":
                continue
            if line.startswith('#'):
                continue
            line = line.split('#')[0]
            line = line.split()

            key_name = line[0]

            if len(line) > 2:
                vals = line[1:]
                vals = [i.replace(',', '') for i in vals]
                vals = [is_num(i) for i in vals]
                input_params[key_name] = vals
            else:
                vals = is_num(line[1])
                input_params[key_name] = vals

    input_params["H0"] = 100 * input_params["h0"]
    input_params["Nu_massive_num"] = int(input_params["Nu_massive_num"])

    if input_params["Nu_massive_num"] > 0:
        if type(input_params["Nu_mass"]) is not list:
            input_params["Nu_mass"] = [input_params["Nu_mass"]]

        if type(input_params["Nu_mass_degeneracies"]) is not list:
            input_params["Nu_mass_degeneracies"] = [input_params["Nu_mass_degeneracies"]]

        for inu_d in range(len(input_params["Nu_mass_degeneracies"])):
            input_params["Nu_mass_degeneracies"][inu_d] = int(input_params["Nu_mass_degeneracies"][inu_d])

    else:
        input_params["Nu_massive_num"] = 1  # Reset to zero later.
        input_params["Nu_mass_degeneracies"] = [0]
        input_params["Nu_mass"] = [0.0]

    print(input_params)
    return input_params


def set_neutrino_parameters(input_params):
    num_nu_massive = sum(input_params["Nu_mass_degeneracies"])

    if num_nu_massive == 0:
        num_nu_massless = 3.044
    elif num_nu_massive == 1:
        num_nu_massless = 2.0308
    elif num_nu_massive == 2:
        num_nu_massless = 1.0176
    elif num_nu_massive == 3:
        num_nu_massless = 0.00441

    print("input massive and massless", num_nu_massive, num_nu_massless)

    Nu_mass_fractions = [0] * len(input_params["Nu_mass"])

    nu_mass_tot = 0.0
    for inu in range(len(input_params["Nu_mass"])):
        nu_mass_tot += input_params["Nu_mass"][inu] * input_params["Nu_mass_degeneracies"][inu]

    if nu_mass_tot > 0.0:
        for inu in range(len(input_params["Nu_mass"])):
            ifrac = input_params["Nu_mass"][inu] * input_params["Nu_mass_degeneracies"][inu]
            Nu_mass_fractions[inu] = ifrac / nu_mass_tot
    else:
        Nu_mass_fractions = [0.0]

    input_params["Nu_mass_tot"] = nu_mass_tot
    input_params["Nu_mass_fractions"] = Nu_mass_fractions
    input_params["Nu_massless_num"] = num_nu_massless

    return input_params


def set_class_parameters(input_params):
    params = defaultsetting

    params['h'] = input_params['h0']
    # params['H0'] = 100.0*params['h']
    h2 = (input_params['h0'])**2.0

    params['omega_b'] = input_params['Omega_b0'] * h2
    params['N_ur'] = input_params['Nu_massless_num']

    params['N_ncdm'] = input_params['Nu_massive_num']
    params['deg_ncdm'] = ",".join(map(str, input_params['Nu_mass_degeneracies']))
    params['m_ncdm'] = ",".join(map(str, input_params["Nu_mass"]))

    params['k_pivot'] = input_params['k_pivot']
    params['A_s'] = input_params['A_s']
    # params['ln10^{10}A_s'] = np.log(1e10 * params['A_s'])
    params['n_s'] = input_params['ns']

    Omm0 = input_params["Omega_m0"]
    Omb0 = input_params["Omega_b0"]
    Omnu0 = input_params["Nu_mass_tot"] / (93.14 * h2)
    Omc0 = Omm0 - Omb0 - Omnu0
    params['omega_cdm'] = Omc0 * h2

    return params


def print_cosmo_parameters(cosmo):
    h0 = cosmo.h()
    h02 = cosmo.h() * cosmo.h()

    print("")
    print("h0 = {0:.5f}".format(h0))
    print("Omega_m0, Omega_cdm0, Omega_b0, Omega_Lambda0, Omega_nu0, Omega_rad0, Omega_k = \n {0:.5f}, {1:.5f}, {2:.5f}, {3:.5f}, {4:.5e}, {5:.5e}, {6:.5f}".format(
        cosmo.Omega0_m(), cosmo.Omega0_cdm(), cosmo.Omega_b(), cosmo.Omega_Lambda(), cosmo.Omega_nu, cosmo.Omega_r(), cosmo.Omega0_k()))
    print("omega_m0, omega_cdm0, omega_b0, omega_Lambda0, omega_nu0, omega_rad0, omega_k = \n {0:.5f}, {1:.5f}, {2:.5f}, {3:.5f}, {4:.5e}, {5:.5e}, {6:.5f}".format(
        cosmo.Omega0_m() * h02, cosmo.Omega0_cdm() * h02, cosmo.Omega_b() * h02, cosmo.Omega_Lambda() * h02, cosmo.Omega_nu * h02, cosmo.Omega_r() * h02, cosmo.Omega0_k() * h02))

    print("Omega_cb0 = {0:.5f} , omega_cb0 = {1:.5f}".format(
        cosmo.Omega0_cdm() + cosmo.Omega_b(), (cosmo.Omega0_cdm() + cosmo.Omega_b()) * h02))

    print("ns = {0:.5f}".format(cosmo.n_s()))
    print("As = {0:14.10e}".format(cosmo.pars['A_s']))
    print("k_pivot = {0:.5f} [Mpc^-1]".format(cosmo.pars['k_pivot']))
    print("sig8 = {0:14.10e}".format(cosmo.pars['sig8']))

    print("")

    print("Massless_Nu_num = {0:.4f} ".format(cosmo.pars["N_ur"]))
    print("Massive_Nu_num = {0:d} ".format(sum(map(int, cosmo.pars['deg_ncdm'].split(",")))))
    print("Massive_Nu_num (degeneracies) = {0:d} ".format(cosmo.pars["N_ncdm"]))

    tot_nu_mass = np.array(list(map(float, cosmo.pars['m_ncdm'].split(",")))) * np.array(list(map(int, cosmo.pars['deg_ncdm'].split(","))))

    sum_tot_nu_mass = sum(tot_nu_mass)

    print("sum(Nu_mass) = {0:.5f} ".format(sum_tot_nu_mass), "[eV]")

    if sum_tot_nu_mass <= 1.0e-10:
        sum_tot_nu_mass = 1.0e-10

    print("Nu_mass_i = ", cosmo.pars['m_ncdm'], "[eV]")
    print("Nu_degeneracies_i = ", cosmo.pars['deg_ncdm'])

    print("Omega_nu0 = ", cosmo.Omega_nu)
    print("Omega_nu0_i*deg = ", cosmo.Omega_nu *
          tot_nu_mass / (sum_tot_nu_mass))

    print("Omega_nu0/Omega_m = ", cosmo.Omega_nu / cosmo.Omega0_m())
    print("Omega_nu0_i*deg/Omega_m = ",
          cosmo.Omega_nu * tot_nu_mass / (sum_tot_nu_mass) / cosmo.Omega0_m())


def get_transfer_funcs(cosmo, zred):
    """
    output_format = CLASS (default) : T(k) : curveture R = 1
    output_format = CAMB : -T(k)/K^2 : curveture R = -1/k^2
    """

    ## calc sigma8 at z=0 ##
    trans0 = cosmo.get_transfer(z=0, output_format='class')
    kh = trans0['k (h/Mpc)']

    if use_tk_dtot:
        tk_tot0 = trans0['d_tot']
    else:
        ascale = 1.0 / (1.0 + zred)
        cbackg = cosmo.get_background()
        cbackg_ascale = 1. / (1 + cbackg['z'])
        rho_cdm = ius(cbackg_ascale, cbackg['(.)rho_cdm'])(ascale)
        rho_b = ius(cbackg_ascale, cbackg['(.)rho_b'])(ascale)
        rho_nu = 0.0
        trans_nu = 0.0
        if 'd_ncdm[0]' in trans0.keys():
            rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
            trans_nu += trans0['d_ncdm[0]'] * rho_ncdm0
            rho_nu += rho_ncdm0
        if 'd_ncdm[1]' in trans0.keys():
            rho_ncdm1 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[1]'])(ascale)
            trans_nu += trans0['d_ncdm[1]'] * rho_ncdm1
            rho_nu += rho_ncdm1
        if 'd_ncdm[2]' in trans0.keys():
            rho_ncdm2 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[2]'])(ascale)
            trans_nu += trans0['d_ncdm[2]'] * rho_ncdm2
            rho_nu += rho_ncdm2
        tk_tot0 = (trans0['d_cdm'] * rho_cdm + trans0['d_b'] * rho_b + trans_nu) / (rho_cdm + rho_b + rho_nu)

    tk_tot0 *= -1.0
    tot_spl = ius(np.log(kh), tk_tot0)
    sig8_0 = get_sigma8(cosmo, tot_spl(logklist))
    cosmo.pars['sig8'] = sig8_0
    print("sigma8(z=0) ", sig8_0)

    ## calc transfer function ##
    tk_funcs = []
    transz = cosmo.get_transfer(z=zred, output_format='class')
    kh = transz['k (h/Mpc)']

    ascale = 1.0 / (1.0 + zred)
    cbackg = cosmo.get_background()
    cbackg_ascale = 1. / (1 + cbackg['z'])
    rho_cdm = ius(cbackg_ascale, cbackg['(.)rho_cdm'])(ascale)
    rho_b = ius(cbackg_ascale, cbackg['(.)rho_b'])(ascale)

    if use_tk_dtot:
        tk_totz = transz['d_tot']
    else:
        rho_nu = 0.0
        trans_nu = 0.0
        if 'd_ncdm[0]' in transz.keys():
            rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
            trans_nu += transz['d_ncdm[0]'] * rho_ncdm0
            rho_nu += rho_ncdm0
        if 'd_ncdm[1]' in transz.keys():
            rho_ncdm1 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[1]'])(ascale)
            trans_nu += transz['d_ncdm[1]'] * rho_ncdm1
            rho_nu += rho_ncdm1
        if 'd_ncdm[2]' in transz.keys():
            rho_ncdm2 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[2]'])(ascale)
            trans_nu += transz['d_ncdm[2]'] * rho_ncdm2
            rho_nu += rho_ncdm2
        tk_totz = (transz['d_cdm'] * rho_cdm + transz['d_b'] * rho_b + trans_nu) / (rho_cdm + rho_b + rho_nu)

    tk_totz *= -1.0
    tot_spl = ius(np.log(kh), tk_totz)
    sig8_grafic = get_sigma8(cosmo, tot_spl(logklist))
    # dfact = cosmo.scale_independent_growth_factor(0) / cosmo.scale_independent_growth_factor(zred)
    ascale = 1.0 / (1.0 + zred)
    print(calc_dplus(1, cosmo), calc_dplus(ascale, cosmo))
    dfact = calc_dplus(1, cosmo) / calc_dplus(ascale, cosmo)
    print("sigma8(z=0) for grafic input (sig8(z)*D+(z=0)/D+(z))", sig8_grafic, sig8_grafic * dfact)
    print("growth rate", dfact)
    cosmo.pars['sig8'] = sig8_grafic * dfact

    # Total matter
    if use_tk_dtot:
        tk_tot = transz['d_tot']
    else:
        rho_nu = 0.0
        trans_nu = 0.0
        if 'd_ncdm[0]' in transz.keys():
            rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
            trans_nu += transz['d_ncdm[0]'] * rho_ncdm0
            rho_nu += rho_ncdm0
        if 'd_ncdm[1]' in transz.keys():
            rho_ncdm1 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[1]'])(ascale)
            trans_nu += transz['d_ncdm[1]'] * rho_ncdm1
            rho_nu += rho_ncdm1
        if 'd_ncdm[2]' in transz.keys():
            rho_ncdm2 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[2]'])(ascale)
            trans_nu += transz['d_ncdm[2]'] * rho_ncdm2
            rho_nu += rho_ncdm2
        tk_tot = (transz['d_cdm'] * rho_cdm + transz['d_b'] * rho_b + trans_nu) / (rho_cdm + rho_b + rho_nu)

    tk_tot *= -1.0
    tot_spl = ius(np.log(kh), tk_tot)
    tk_tot = tot_spl(logklist)
    tk_tot /= (klist * cosmo.h())**2
    tk_tot0 = tk_tot[0]
    tk_tot /= tk_tot0
    tk_tot = np.abs(tk_tot)

    # CDM+baryon
    tk_cb = (transz['d_cdm'] * rho_cdm + transz['d_b'] * rho_b) / (rho_cdm + rho_b)
    tk_cb *= -1.0
    cb_spl = ius(np.log(kh), tk_cb)
    tk_cb = cb_spl(logklist)
    tk_cb /= (klist * cosmo.h())**2
    tk_cb /= tk_tot0
    tk_cb = np.abs(tk_cb)

    tk_cdm = -transz['d_cdm']
    cdm_spl = ius(np.log(kh), tk_cdm)
    tk_cdm = cdm_spl(logklist)
    tk_cdm /= (klist * cosmo.h())**2
    tk_cdm /= tk_tot0
    tk_cdm = np.abs(tk_cdm)

    tk_bar = -transz['d_b']
    bar_spl = ius(np.log(kh), tk_bar)
    tk_bar = bar_spl(logklist)
    tk_bar /= (klist * cosmo.h())**2
    tk_bar /= tk_tot0
    tk_bar = np.abs(tk_bar)

    tk_funcs.append(klist)
    tk_funcs.append(tk_tot)
    tk_funcs.append(tk_cb)
    tk_funcs.append(tk_cdm)
    tk_funcs.append(tk_bar)

    if 'd_ncdm[0]' in transz.keys():
        tk_nu0 = -transz['d_ncdm[0]']
        nu0_spl = ius(np.log(kh), tk_nu0)
        tk_nu0 = nu0_spl(logklist)
        tk_nu0 /= (klist * cosmo.h())**2
        tk_nu0 /= tk_tot0
        tk_nu0 = np.abs(tk_nu0)
        tk_funcs.append(tk_nu0)

    if 'd_ncdm[1]' in transz.keys():
        tk_nu1 = -transz['d_ncdm[1]']
        nu1_spl = ius(np.log(kh), tk_nu1)
        tk_nu1 = nu1_spl(logklist)
        tk_nu1 /= (klist * cosmo.h())**2
        tk_nu1 /= tk_tot0
        tk_nu1 = np.abs(tk_nu1)
        tk_funcs.append(tk_nu1)

    if 'd_ncdm[2]' in transz.keys():
        tk_nu2 = -transz['d_ncdm[2]']
        nu2_spl = ius(np.log(kh), tk_nu2)
        tk_nu2 = nu2_spl(logklist)
        tk_nu2 /= (klist * cosmo.h())**2
        tk_nu2 /= tk_tot0
        tk_nu2 = np.abs(tk_nu2)
        tk_funcs.append(tk_nu2)

    return cosmo, tk_funcs, transz


def get_powers(cosmo, tk_z, zred):
    pkz = []
    kh = tk_z['k (h/Mpc)']

    ascale = 1.0 / (1.0 + zred)
    cbackg = cosmo.get_background()
    cbackg_ascale = 1. / (1 + cbackg['z'])
    rho_cdm = ius(cbackg_ascale, cbackg['(.)rho_cdm'])(ascale)
    rho_b = ius(cbackg_ascale, cbackg['(.)rho_b'])(ascale)

    # Total matter
    if use_tk_dtot:
        tkz = tk_z['d_tot']
    else:
        rho_nu = 0.0
        trans_nu = 0.0
        if 'd_ncdm[0]' in tk_z.keys():
            rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
            trans_nu += tk_z['d_ncdm[0]'] * rho_ncdm0
            rho_nu += rho_ncdm0
        if 'd_ncdm[1]' in tk_z.keys():
            rho_ncdm1 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[1]'])(ascale)
            trans_nu += tk_z['d_ncdm[1]'] * rho_ncdm1
            rho_nu += rho_ncdm1
        if 'd_ncdm[2]' in tk_z.keys():
            rho_ncdm2 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[2]'])(ascale)
            trans_nu += tk_z['d_ncdm[2]'] * rho_ncdm2
            rho_nu += rho_ncdm2
        tkz = (tk_z['d_cdm'] * rho_cdm + tk_z['d_b'] * rho_b + trans_nu) / (rho_cdm + rho_b + rho_nu)

    tkz_spl = ius(np.log(kh), tkz)
    pkz_tot = calc_pk(cosmo, tkz_spl(logklist))

    # CDM+baryon
    tkz = (tk_z['d_cdm'] * rho_cdm + tk_z['d_b'] * rho_b) / (rho_cdm + rho_b)
    tkz_spl = ius(np.log(kh), tkz)
    pkz_cb = calc_pk(cosmo, tkz_spl(logklist))

    tkz = tk_z['d_cdm']
    tkz_spl = ius(np.log(kh), tkz)
    pkz_cdm = calc_pk(cosmo, tkz_spl(logklist))

    tkz = tk_z['d_b']
    tkz_spl = ius(np.log(kh), tkz)
    pkz_bar = calc_pk(cosmo, tkz_spl(logklist))

    pkz.append(klist)
    pkz.append(pkz_tot)
    pkz.append(pkz_cb)
    pkz.append(pkz_cdm)
    pkz.append(pkz_bar)

    if 'd_ncdm[0]' in tk_z.keys():
        tkz = tk_z['d_ncdm[0]']
        tkz_spl = ius(np.log(kh), tkz)
        pkz_nu0 = calc_pk(cosmo, tkz_spl(logklist))
        pkz.append(pkz_nu0)

    if 'd_ncdm[1]' in tk_z.keys():
        tkz = tk_z['d_ncdm[1]']
        tkz_spl = ius(np.log(kh), tkz)
        pkz_nu1 = calc_pk(cosmo, tkz_spl(logklist))
        pkz.append(pkz_nu1)

    if 'd_ncdm[2]' in tk_z.keys():
        tkz = tk_z['d_ncdm[2]']
        tkz_spl = ius(np.log(kh), tkz)
        pkz_nu2 = calc_pk(cosmo, tkz_spl(logklist))
        pkz.append(pkz_nu2)

    # for nu_tot check
    if 0:
        rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
        rho_ncdm1 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[1]'])(ascale)
        tkz = (tk_z['d_ncdm[0]'] * rho_ncdm0 + tk_z['d_ncdm[1]'] * rho_ncdm1) / (rho_ncdm0 + rho_ncdm1)
        tkz_spl = ius(np.log(kh), tkz)
        pkz_nu_tot = calc_pk(cosmo, tkz_spl(logklist))
        pkz.append(pkz_nu_tot)

        # P_tot = f1^2 P1 + 2 f1f2 P12 + f2^2 P2
        rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
        rho_ncdm1 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[1]'])(ascale)
        tkzi = tk_z['d_ncdm[0]']
        tkzj = tk_z['d_ncdm[1]']
        tkzi_spl = ius(np.log(kh), tkzi)
        tkzj_spl = ius(np.log(kh), tkzj)
        pkz_nu_ij = calc_pk_ij(cosmo, tkzi_spl(logklist), tkzj_spl(logklist))
        pkz.append(pkz_nu_ij)

    return pkz


def get_powers_vpk(cosmo, tk_z, zred):
    # T_cdm cannot be obtained with synchronous gauge. On the other hand,
    # the small-k of d_ncdm is wrong in Newtonian gauge. Therefore, we use synchronous gauge.

    vpkz = []
    kh = tk_z['k (h/Mpc)']

    ascale = 1.0 / (1.0 + zred)
    cbackg = cosmo.get_background()
    cbackg_ascale = 1. / (1 + cbackg['z'])
    rho_cdm = ius(cbackg_ascale, cbackg['(.)rho_cdm'])(ascale)
    rho_b = ius(cbackg_ascale, cbackg['(.)rho_b'])(ascale)

    # Total matter
    # if use_tk_dtot:
    tkz = tk_z['t_tot']
    """
    else:
        rho_nu = 0.0
        trans_nu = 0.0
        if 't_ncdm[0]' in tk_z.keys():
            rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
            trans_nu += tk_z['t_ncdm[0]'] * rho_ncdm0
            rho_nu += rho_ncdm0
        if 't_ncdm[1]' in tk_z.keys():
            rho_ncdm1 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[1]'])(ascale)
            trans_nu += tk_z['t_ncdm[1]'] * rho_ncdm1
            rho_nu += rho_ncdm1
        if 't_ncdm[2]' in tk_z.keys():
            rho_ncdm2 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[2]'])(ascale)
            trans_nu += tk_z['t_ncdm[2]'] * rho_ncdm2
            rho_nu += rho_ncdm2
        tkz = (tk_z['t_cdm'] * rho_cdm + tk_z['t_b'] * rho_b + trans_nu) / (rho_cdm + rho_b + rho_nu)
    """
    tkz_spl = ius(np.log(kh), tkz)
    vpkz_tot = calc_pk(cosmo, tkz_spl(logklist))

    """
    # CDM+baryon
    tkz = (tk_z['t_cdm'] * rho_cdm + tk_z['t_b'] * rho_b) / (rho_cdm + rho_b)
    tkz_spl = ius(np.log(kh), tkz)
    vpkz_cb = calc_pk(cosmo, tkz_spl(logklist))

    tkz = tk_z['t_cdm']
    tkz_spl = ius(np.log(kh), tkz)
    vpkz_cdm = calc_pk(cosmo, tkz_spl(logklist))
    """
    tkz = tk_z['t_b']
    tkz_spl = ius(np.log(kh), tkz)
    vpkz_bar = calc_pk(cosmo, tkz_spl(logklist))

    vpkz.append(klist)
    vpkz.append(vpkz_tot)
    # vpkz.append(vpkz_cb)
    # vpkz.append(vpkz_cdm)
    vpkz.append(vpkz_bar)

    if 't_ncdm[0]' in tk_z.keys():
        tkz = tk_z['t_ncdm[0]']
        tkz_spl = ius(np.log(kh), tkz)
        vpkz_nu0 = calc_pk(cosmo, tkz_spl(logklist))
        vpkz.append(vpkz_nu0)

    if 't_ncdm[1]' in tk_z.keys():
        tkz = tk_z['t_ncdm[1]']
        tkz_spl = ius(np.log(kh), tkz)
        vpkz_nu1 = calc_pk(cosmo, tkz_spl(logklist))
        vpkz.append(vpkz_nu1)

    if 't_ncdm[2]' in tk_z.keys():
        tkz = tk_z['t_ncdm[2]']
        tkz_spl = ius(np.log(kh), tkz)
        vpkz_nu2 = calc_pk(cosmo, tkz_spl(logklist))
        vpkz.append(vpkz_nu2)

    return vpkz


def output_grafic_file(cosmo, params, args):
    """ sample : # = "\n"
    transfer_0.1eV_Z10.dat
    power_0.1eV_Z10.dat
    2
    0.308, 0.692, 0.0484, 67.8, 5.45517e-05
    0.96
    -0.80362321922246249                #-0.823
    1.e-4, 50.0
    -2000.0
    10.0
    3             #Nu_massive_num
    1 1 1         #Nu_mass_degeneracies
    0.1 0.2 0.3   #Nu_mass
    0.1667, 0.3333, 0.5000 ##Nu_frac
    1
    0
    #
    #
    #
    #
    1
    123456789
    white_noise.dat
    """

    """
    1          #Nu_massive_num
    3          #Nu_mass_degeneracies
    0.133333   #Nu_mass

    2          #Nu_massive_num
    1 1        #Nu_mass_degeneracies
    0.1 0.3    #Nu_mass
    """

    if len(params["Nu_mass_fractions"]) == 1:
        frac_str = "{0:.5f}".format(*params["Nu_mass_fractions"])
    elif len(params["Nu_mass_fractions"]) == 2:
        frac_str = "{0:.5f},{1:.5f}".format(*params["Nu_mass_fractions"])
    elif len(params["Nu_mass_fractions"]) == 3:
        frac_str = "{0:.5f},{1:.5f},{2:.5f}".format(
            *params["Nu_mass_fractions"])

    if cosmo.pars["m_ncdm"] == "0.0" and cosmo.pars["deg_ncdm"] == "0":
        print("Massless Neutrino")
        cosmo.pars["N_ncdm"] = 0

    input_str = "{0:s}\n".format(params["tkfile"]) + \
        "{0:s}\n".format(params["pkfile"]) + \
        "2\n" + \
        "{0:.4e}, {1:.4e}, {2:.4e}, {3:.4e}, {4:.4e}\n".format(
        cosmo.Omega0_m(), cosmo.Omega_Lambda(), cosmo.Omega_b(), 100 * cosmo.h(), cosmo.Omega_r()) + \
        "{0:.4f}\n".format(cosmo.n_s()) + \
        "{0:.8f}\n".format(-cosmo.pars['sig8']) + \
        "{0:f}, {1:f}\n".format(10**kmin, 10**kmax) + \
        "{0:.1f}\n".format(-params["boxsize"]) + \
        "{0:.1f}\n".format(params["zini"]) + \
        "{0:d}\n".format(cosmo.pars["N_ncdm"]) + \
        "{0:s}\n".format(cosmo.pars["deg_ncdm"]) + \
        "{0:s}\n".format(cosmo.pars["m_ncdm"]) + \
        "{0:s}\n".format(frac_str) + \
        "{0:d}\n{1:d}\n".format(1, 0) + \
        "\n\n\n\n" + \
        "{0:d}\n".format(1) + \
        "{0:d}\n".format(int(params["seed"])) + \
        "{0:s}\n".format("white_noise.dat")

    with open(args.output_grafic, mode='w') as f:
        f.write(input_str)

    print("output grafic file : ", args.output_grafic)


def set_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_parameter_file', help='input parameter file',
                        type=str, default="./input_class_param_sample.ini")
    parser.add_argument('-o', '--output_grafic', help='output grafic param filename', type=str,
                        default="input_param.grafic")
    parser.add_argument('-z', '--zinit', help='input zini. The default value is to use the value in the input file.', type=float,
                        default=None)
    parser.add_argument('--no_high_precision', help='No high precision for massive neutrinos.',
                        action="store_true")
    args = parser.parse_args(argv[1:])
    return args


if __name__ == '__main__':
    args = set_args(sys.argv)
    print(vars(args))

    input_params = load_input_paramfile(args.input_parameter_file)
    input_params = set_neutrino_parameters(input_params)

    if args.zinit is not None:
        input_params['zini'] = args.zinit

    cparams = set_class_parameters(input_params)
    pparams = set_precisionsetting_dict(args.no_high_precision)

    cosmo_obj = Class()
    cosmo_obj.set(cparams)
    cosmo_obj.set(pparams)
    cosmo_obj.compute()

    zred = input_params['zini']
    cosmo, tk_grafic, tk_class_z = get_transfer_funcs(cosmo_obj, zred)

    print_cosmo_parameters(cosmo)
    pkz = get_powers(cosmo, tk_class_z, zred)
    vpkz = get_powers_vpk(cosmo, tk_class_z, zred)

    output_grafic_file(cosmo, input_params, args)

    if grafic_format == True:
        klist *= cosmo.h()
        theaders = "k (1/Mpc) Tot cb CDM baryon nu[0] nu[1] nu[2]"
    else:
        theaders = "k (h/Mpc) Tot cb CDM baryon nu[0] nu[1] nu[2]"

    np.savetxt(input_params["tkfile"], np.transpose(tk_grafic), fmt="%.10e", header=theaders)

    if grafic_format == True:
        # reverse k unit
        klist /= cosmo.h()

    pheaders = "k (h/Mpc) Tot cb CDM baryon nu[0] nu[1] nu[2]"
    filename = "class_powers_{0:.2f}eV_z{1:.1f}.dat".format(input_params["Nu_mass_tot"], zred)
    np.savetxt(filename, np.transpose(pkz), fmt="%.10e", header=pheaders)

    pheaders = "k (h/Mpc) Tot baryon nu[0] nu[1] nu[2]"
    filename = "class_vel_powers_{0:.2f}eV_z{1:.1f}.dat".format(input_params["Nu_mass_tot"], zred)
    np.savetxt(filename, np.transpose(vpkz), fmt="%.10e", header=pheaders)
