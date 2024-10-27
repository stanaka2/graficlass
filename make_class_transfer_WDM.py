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

import make_class_transfer as mct

grafic_format = True
use_tk_dtot = False

kmin = -5
kmax = 6
# kmax = 3
klist = np.logspace(kmin, kmax, 2001)
logklist = np.log(klist)

# Numerical oscillations occur at Tk in the CDM when log_kmax exceeds 4 in the Class computation.
# When a larger kmax is required, it is obtained by interpolation.
class_kmax = min(3.0, kmax)
P_k_max_h = 10**class_kmax

if kmax > 3.0:
    print("# Note. For k>1000, the transfer function is calculated approximately using extrapolation.")


# <https://github.com/lesgourg/class_public/issues/418>
evolver = 1

# <https://github.com/lesgourg/class_public/issues/222>
# k_per_decade_for_pk = P_k_max_h
k_per_decade_for_pk = 100


def set_precisionsetting_dict_WDM(high_precision=False):
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
            # Smaller values will cause a small step error with a setting of P_k_max_h/Mpc > 1000.
            'tol_perturbations_integration': 1.e-6,  # default 1e-5
            'perturbations_sampling_stepsize': 1e-2,  # default 0.1
            'background_Nloga': 4620,  # default 3000
        }
        ncdm_high_precisionsettings.update({
            # 'ncdm_quadrature_strategy': 3, # the trapezoidal rule on [0,q_max] where q_max is the next input.
            # 'ncdm_maximum_q': 15,  # default 15
            # 'ncdm_N_momentum_bins': 150
        })
    else:
        precisionsettings = {  # These are sufficient for this purpose
            # Smaller values will cause a small step error with a setting of P_k_max_h/Mpc > 1000.
            'tol_perturb_integration': 1.e-6,
            'perturb_sampling_stepsize': 1e-2,
        }
        ncdm_high_precisionsettings.update({
            # 'Quadrature strategy': 3,
            # 'Maximum q': 15,
            # 'Number of momentum bins': 150
        })

    if high_precision:
        precisionsettings.update(ncdm_high_precisionsettings)
    else:
        precisionsettings.update(ncdm_precisionsettings)

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


def Viel05_fit_model(k, Mwdm, Omega_dm, h, mu=1.12):
    # <https://arxiv.org/abs/astro-ph/0511630>
    # Eq.6, 7
    a = 0.049
    b = -1.11
    c = 0.11
    d = 1.22
    alpha = a * ((Mwdm / 1000.0)**b) * ((Omega_dm / 0.25)**c) * ((h / 0.7)**d)
    tk = (1.0 + (alpha * k)**(2.0 * mu))**(-5.0 / mu)
    return tk


def Abazajian06_fit_model(k, Mwdm, Omega_dm, h, nu=2.25, mu=3.08):
    # <https://arxiv.org/abs/astro-ph/0511630>
    # Eq.11, 12
    a = 0.189
    b = -0.858
    c = -0.136
    d = 0.692
    alpha = a * ((Mwdm / 1000.0)**b) * ((Omega_dm / 0.26)**c) * ((h / 0.7)**d)
    tk = (1.0 + (alpha * k)**(nu))**(-mu)
    return tk


def Murgia17_fit_model(k, alpha, beta=3.5, gamma=-10):
    # <https://arxiv.org/abs/1704.07838>
    # Eq.2.4, 2.5
    tk = (1.0 + (alpha * k)**(beta))**(gamma)
    return tk


def getNearestIndex(l, num):
    idx = np.abs(np.asarray(l) - num).argmin()
    return idx


def khalf(alpha, beta=3.5, gamma=-10):
    # T^2 (=Pcdm/Pwdm) = 0.5 or T (=Tcdm/Twdm) = sqrt(0.5) of wave number
    kh = (0.5**(1.0 / (2 * gamma)) - 1)**(1.0 / beta) / alpha
    return kh


def alpha_from_khalf(kh, beta=3.5, gamma=-10):
    alpha = (0.5**(1.0 / (2 * gamma)) - 1)**(1.0 / beta) / kh
    return alpha


def get_transfer_funcs_WDM(cosmo, zred, wdm_flag):
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
        rho_wdm = 0.0
        trans_wdm = 0.0
        if 'd_ncdm[0]' in trans0.keys():
            rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
            trans_wdm += trans0['d_ncdm[0]'] * rho_ncdm0
            rho_wdm += rho_ncdm0
        tk_tot0 = (trans0['d_cdm'] * rho_cdm + trans0['d_b'] * rho_b + trans_wdm) / (rho_cdm + rho_b + rho_wdm)

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
        rho_wdm = 0.0
        trans_wdm = 0.0
        if 'd_ncdm[0]' in transz.keys():
            rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
            trans_wdm += transz['d_ncdm[0]'] * rho_ncdm0
            rho_wdm += rho_ncdm0
        tk_totz = (transz['d_cdm'] * rho_cdm + transz['d_b'] * rho_b + trans_wdm) / (rho_cdm + rho_b + rho_wdm)

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

    # Total matter (CDM+baryon or WDM+baryon)
    if use_tk_dtot:
        tk_tot = transz['d_tot']
    else:
        rho_wdm = 0.0
        trans_wdm = 0.0
        if 'd_ncdm[0]' in transz.keys():
            rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
            trans_wdm += transz['d_ncdm[0]'] * rho_ncdm0
            rho_wdm += rho_ncdm0
        tk_tot = (transz['d_cdm'] * rho_cdm + transz['d_b'] * rho_b + trans_wdm) / (rho_cdm + rho_b + rho_wdm)

    tk_tot *= -1.0
    tot_spl = ius(np.log(kh), tk_tot, ext=3)
    tk_tot = tot_spl(logklist)
    tk_tot /= (klist * cosmo.h())**2
    tk_tot0 = tk_tot[0]
    tk_tot /= tk_tot0
    tk_tot = np.abs(tk_tot)

    if wdm_flag:
        tk_dm = -transz['d_ncdm[0]']
    else:
        tk_dm = -transz['d_cdm']
    cdm_spl = ius(np.log(kh), tk_dm, ext=3)
    tk_dm = cdm_spl(logklist)
    tk_dm /= (klist * cosmo.h())**2
    tk_dm /= tk_tot0
    tk_dm = np.abs(tk_dm)

    tk_bar = -transz['d_b']
    bar_spl = ius(np.log(kh), tk_bar, ext=3)
    tk_bar = bar_spl(logklist)
    tk_bar /= (klist * cosmo.h())**2
    tk_bar /= tk_tot0
    tk_bar = np.abs(tk_bar)

    tk_funcs.append(klist)
    tk_funcs.append(tk_tot)
    tk_funcs.append(tk_tot)
    tk_funcs.append(tk_dm)
    tk_funcs.append(tk_bar)

    return cosmo, tk_funcs, transz


def get_powers_WDM(cosmo, tk_z, zred, wdm_flag):
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
        rho_wdm = 0.0
        trans_wdm = 0.0
        if 'd_ncdm[0]' in tk_z.keys():
            rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
            trans_wdm += tk_z['d_ncdm[0]'] * rho_ncdm0
            rho_wdm += rho_ncdm0
        tkz = (tk_z['d_cdm'] * rho_cdm + tk_z['d_b'] * rho_b + trans_wdm) / (rho_cdm + rho_b + rho_wdm)

    tkz_spl = ius(np.log(kh), tkz, ext=3)
    pkz_tot = calc_pk(cosmo, tkz_spl(logklist))

    if wdm_flag:
        tkz = tk_z['d_ncdm[0]']
    else:
        tkz = tk_z['d_cdm']
    tkz_spl = ius(np.log(kh), tkz, ext=3)
    pkz_dm = calc_pk(cosmo, tkz_spl(logklist))

    tkz = tk_z['d_b']
    tkz_spl = ius(np.log(kh), tkz, ext=3)
    pkz_bar = calc_pk(cosmo, tkz_spl(logklist))

    pkz.append(klist)
    pkz.append(pkz_tot)
    pkz.append(pkz_tot)
    pkz.append(pkz_dm)
    pkz.append(pkz_bar)

    return pkz


def load_input_paramfile_WDM(inputfile):
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
                vals = [mct.is_num(i) for i in vals]
                input_params[key_name] = vals
            else:
                vals = mct.is_num(line[1])
                input_params[key_name] = vals

    input_params["WDM_flag"] = int(input_params["WDM_flag"])
    if input_params["WDM_flag"] == 0:
        input_params["WDM_model"] = "Class"

    input_params["H0"] = 100 * input_params["h0"]

    print(input_params)
    return input_params


def set_class_parameters_WDM(input_params, wdm_flag):
    params = mct.defaultsetting
    params['P_k_max_h/Mpc'] = P_k_max_h
    params['k_per_decade_for_pk'] = k_per_decade_for_pk
    params['evolver'] = evolver

    params['h'] = input_params['h0']
    # params['H0'] = 100.0*params['h']
    h2 = (input_params['h0'])**2.0

    params['omega_b'] = input_params['Omega_b0'] * h2
    params['k_pivot'] = input_params['k_pivot']
    params['A_s'] = input_params['A_s']
    # params['ln10^{10}A_s'] = np.log(1e10 * params['A_s'])
    params['n_s'] = input_params['ns']

    Omm0 = input_params["Omega_m0"]
    Omb0 = input_params["Omega_b0"]
    Omnu0 = 0.0
    Omdm0 = Omm0 - Omb0 - Omnu0

    if wdm_flag:
        params['N_ncdm'] = 1
        params['deg_ncdm'] = 1
        params['m_ncdm'] = input_params["WDM_mass"]
        params['omega_ncdm'] = Omdm0 * h2
        params['omega_cdm'] = 0.0
        params['T_ncdm'] = 0.2062 * (1.0 / (input_params["WDM_mass"] / 1000.0))**(1.0 / 3.0),  # Mastache 2023, eq.(17)
    else:
        params['N_ncdm'] = 1
        params['deg_ncdm'] = 0
        params['m_ncdm'] = 0.0
        params['omega_ncdm'] = 0.0
        params['omega_cdm'] = Omdm0 * h2
    return params


def print_cosmo_parameters_WDM(cosmo):
    h0 = cosmo.h()
    h02 = cosmo.h() * cosmo.h()

    print("")
    print("h0 = {0:.5f}".format(h0))
    print("Omega_m0, Omega_cdm0, Omega_wdm0, Omega_b0, Omega_Lambda0, Omega_rad0, Omega_k = \n {0:.5f}, {1:.5f}, {2:.5f}, {3:.5f}, {4:.5e}, {5:.5e}, {6:.5f}".format(
        cosmo.Omega0_m(), cosmo.Omega0_cdm(), cosmo.Om_ncdm(0), cosmo.Omega_b(),
        cosmo.Omega_Lambda(), cosmo.Omega_r(), cosmo.Omega0_k()))
    print("omega_m0, omega_cdm0, omega_wdm0, omega_b0, omega_Lambda0, omega_rad0, omega_k = \n {0:.5f}, {1:.5f}, {2:.5f}, {3:.5f}, {4:.5e}, {5:.5e}, {6:.5f}".format(
        cosmo.Omega0_m() * h02, cosmo.Omega0_cdm() * h02, cosmo.Om_ncdm(0) * h02, cosmo.Omega_b() * h02,
        cosmo.Omega_Lambda() * h02, cosmo.Omega_r() * h02, cosmo.Omega0_k() * h02))

    print("Omega_cb0 = {0:.5f} , omega_cb0 = {1:.5f}".format(
        cosmo.Omega0_cdm() + cosmo.Omega_b(), (cosmo.Omega0_cdm() + cosmo.Omega_b()) * h02))

    print("Omega_wb0 = {0:.5f} , omega_wb0 = {1:.5f}".format(
        cosmo.Om_ncdm(0) + cosmo.Omega_b(), (cosmo.Om_ncdm(0) + cosmo.Omega_b()) * h02))

    print("ns = {0:.5f}".format(cosmo.n_s()))
    print("As = {0:14.10e}".format(cosmo.pars['A_s']))
    print("k_pivot = {0:.5f} [Mpc^-1]".format(cosmo.pars['k_pivot']))
    print("sig8 = {0:14.10e}".format(cosmo.pars['sig8']))

    print("")

    WDM_mass = cosmo.pars['m_ncdm']
    print("WDM_mass = {0:.5f} ".format(WDM_mass), "[eV]")


def output_grafic_file_WDM(cosmo, params, args):
    """ sample : # = "\n"
    transfer_0.1eV_Z10.dat
    power_0.1eV_Z10.dat
    2
    0.308, 0.692, 0.0484, 67.8, 5.45517e-05
    0.96
    -0.80362321922246249                #-0.823
    1.e-4, 800.0
    -2000.0
    10.0
    1             #WDM_num
    1             #WDM_degeneracies
    100000        #WDM_mass
    1.0           #WDM_frac
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

    if params["WDM_flag"] == 0:
        print("CDM model")
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
        "{0:d}\n".format(cosmo.pars["deg_ncdm"]) + \
        "{0:f}\n".format(cosmo.pars["m_ncdm"]) + \
        "{0:.2f}\n".format(1.0) + \
        "{0:d}\n{1:d}\n".format(1, 0) + \
        "\n\n\n\n" + \
        "{0:d}\n".format(1) + \
        "{0:d}\n".format(int(params["seed"])) + \
        "{0:s}\n".format("white_noise.dat")

    with open(args.output_grafic, mode='w') as f:
        f.write(input_str)

    print("output grafic file : ", args.output_grafic)


def class_tk_outputs(input_params, args):
    wdm_flag = input_params["WDM_flag"]
    cparams = set_class_parameters_WDM(input_params, wdm_flag)
    pparams = set_precisionsetting_dict_WDM(args.high_precision)

    cosmo_obj = Class()
    cosmo_obj.set(cparams)
    cosmo_obj.set(pparams)
    cosmo_obj.compute()

    cosmo, tk_grafic, tk_class_z = get_transfer_funcs_WDM(cosmo_obj, zred, wdm_flag)
    pkz = get_powers_WDM(cosmo, tk_class_z, zred, wdm_flag)
    output_grafic_file_WDM(cosmo, input_params, args)

    print_cosmo_parameters_WDM(cosmo)
    return tk_grafic, pkz, cosmo


def Viel05_tk_outputs(input_params, args):
    # this mode calculate only CDM
    wdm_flag = 0
    cparams = set_class_parameters_WDM(input_params, wdm_flag)
    pparams = set_precisionsetting_dict_WDM(args.high_precision)

    cosmo_obj = Class()
    cosmo_obj.set(cparams)
    cosmo_obj.set(pparams)
    cosmo_obj.compute()

    cosmo, tk_grafic, tk_class_z = get_transfer_funcs_WDM(cosmo_obj, zred, wdm_flag)

    Omega_dm = cosmo.Omega0_cdm()
    Mwdm = input_params["WDM_mass"]
    h = cosmo.h()

    klist = tk_grafic[0]
    cdm_tk = tk_grafic[1]
    tk_grafic[1] = Viel05_fit_model(klist, Mwdm, Omega_dm, h) * cdm_tk
    tk_grafic[2] = Viel05_fit_model(klist, Mwdm, Omega_dm, h) * cdm_tk
    tk_grafic[3] = Viel05_fit_model(klist, Mwdm, Omega_dm, h) * cdm_tk

    # override tk data
    kh = tk_class_z['k (h/Mpc)']
    tk_spl = ius(np.log(kh), tk_class_z['d_cdm'], ext=3)
    tk_class_z['d_cdm'] = tk_spl(logklist)
    tk_spl = ius(np.log(kh), tk_class_z['d_b'], ext=3)
    tk_class_z['d_b'] = tk_spl(logklist)
    tk_class_z['k (h/Mpc)'] = klist

    tk_class_z['d_cdm'] = Viel05_fit_model(tk_class_z['k (h/Mpc)'], Mwdm, Omega_dm, h) * tk_class_z['d_cdm']
    tk_class_z['d_b'] = Viel05_fit_model(tk_class_z['k (h/Mpc)'], Mwdm, Omega_dm, h) * tk_class_z['d_b']

    pkz = get_powers_WDM(cosmo, tk_class_z, zred, wdm_flag)

    # here, input_params["WDM_flag"] == 1
    output_grafic_file_WDM(cosmo, input_params, args)

    print_cosmo_parameters_WDM(cosmo)
    return tk_grafic, pkz, cosmo


def Abazajian06_tk_outputs(input_params, args):
    # this mode calculate only CDM
    wdm_flag = 0
    cparams = set_class_parameters_WDM(input_params, wdm_flag)
    pparams = set_precisionsetting_dict_WDM(args.high_precision)

    cosmo_obj = Class()
    cosmo_obj.set(cparams)
    cosmo_obj.set(pparams)
    cosmo_obj.compute()

    cosmo, tk_grafic, tk_class_z = get_transfer_funcs_WDM(cosmo_obj, zred, wdm_flag)

    Omega_dm = cosmo.Omega0_cdm()
    Mwdm = input_params["WDM_mass"]
    h = cosmo.h()

    klist = tk_grafic[0]
    cdm_tk = tk_grafic[1]
    tk_grafic[1] = Abazajian06_fit_model(klist, Mwdm, Omega_dm, h) * cdm_tk
    tk_grafic[2] = Abazajian06_fit_model(klist, Mwdm, Omega_dm, h) * cdm_tk
    tk_grafic[3] = Abazajian06_fit_model(klist, Mwdm, Omega_dm, h) * cdm_tk

    # override tk data
    kh = tk_class_z['k (h/Mpc)']
    tk_spl = ius(np.log(kh), tk_class_z['d_cdm'], ext=3)
    tk_class_z['d_cdm'] = tk_spl(logklist)
    tk_spl = ius(np.log(kh), tk_class_z['d_b'], ext=3)
    tk_class_z['d_b'] = tk_spl(logklist)
    tk_class_z['k (h/Mpc)'] = klist

    tk_class_z['d_cdm'] = Abazajian06_fit_model(tk_class_z['k (h/Mpc)'], Mwdm, Omega_dm, h) * tk_class_z['d_cdm']
    tk_class_z['d_b'] = Abazajian06_fit_model(tk_class_z['k (h/Mpc)'], Mwdm, Omega_dm, h) * tk_class_z['d_b']

    pkz = get_powers_WDM(cosmo, tk_class_z, zred, wdm_flag)

    # here, input_params["WDM_flag"] == 1
    output_grafic_file_WDM(cosmo, input_params, args)

    print_cosmo_parameters_WDM(cosmo)
    return tk_grafic, pkz, cosmo


def Murgia17_tk_outputs(input_params, args):
    # this mode calculate CDM first
    wdm_flag = 0
    cparams = set_class_parameters_WDM(input_params, wdm_flag)
    pparams = set_precisionsetting_dict_WDM(args.high_precision)
    cosmo_obj = Class()
    cosmo_obj.set(cparams)
    cosmo_obj.set(pparams)
    cosmo_obj.compute()
    cdm_cosmo, cdm_tk_grafic, cdm_tk_class_z = get_transfer_funcs_WDM(cosmo_obj, zred, wdm_flag)

    # this mode calculate WDM second
    wdm_flag = 1
    cparams = set_class_parameters_WDM(input_params, wdm_flag)
    pparams = set_precisionsetting_dict_WDM(args.high_precision)
    cosmo_obj = Class()
    cosmo_obj.set(cparams)
    cosmo_obj.set(pparams)
    cosmo_obj.compute()
    wdm_cosmo, wdm_tk_grafic, wdm_tk_class_z = get_transfer_funcs_WDM(cosmo_obj, zred, wdm_flag)

    # calculate fitting parameter
    beta = input_params.setdefault("beta", 2.4)
    gamma = input_params.setdefault("gamma", -10.0)

    klist = cdm_tk_grafic[0]
    cdm_tk = cdm_tk_grafic[1]
    wdm_tk = wdm_tk_grafic[1]

    kid = getNearestIndex(wdm_tk / cdm_tk, 0.5**0.5)
    alpha = alpha_from_khalf(klist[kid], beta, gamma)
    wdm_tk_grafic[1] = Murgia17_fit_model(klist, alpha, beta, gamma) * cdm_tk_grafic[1]
    wdm_tk_grafic[2] = Murgia17_fit_model(klist, alpha, beta, gamma) * cdm_tk_grafic[2]
    wdm_tk_grafic[3] = Murgia17_fit_model(klist, alpha, beta, gamma) * cdm_tk_grafic[3]
    wdm_tk_grafic[4] = Murgia17_fit_model(klist, alpha, beta, gamma) * cdm_tk_grafic[4]

    # override tk data
    kh = cdm_tk_class_z['k (h/Mpc)']
    tk_spl = ius(np.log(kh), cdm_tk_class_z['d_cdm'], ext=3)
    cdm_tk_class_z['d_cdm'] = tk_spl(logklist)
    tk_spl = ius(np.log(kh), cdm_tk_class_z['d_b'], ext=3)
    cdm_tk_class_z['d_b'] = tk_spl(logklist)
    cdm_tk_class_z['k (h/Mpc)'] = klist

    cdm_tk_class_z['d_cdm'] = Murgia17_fit_model(cdm_tk_class_z['k (h/Mpc)'], alpha, beta, gamma) * cdm_tk_class_z['d_cdm']
    cdm_tk_class_z['d_b'] = Murgia17_fit_model(cdm_tk_class_z['k (h/Mpc)'], alpha, beta, gamma) * cdm_tk_class_z['d_b']

    # cdm mode
    pkz = get_powers_WDM(cdm_cosmo, cdm_tk_class_z, zred, 0)

    # here, input_params["WDM_flag"] == 1
    output_grafic_file_WDM(wdm_cosmo, input_params, args)
    print_cosmo_parameters_WDM(wdm_cosmo)

    return wdm_tk_grafic, pkz, wdm_cosmo


def set_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_parameter_file', help='input parameter file',
                        type=str, default="./input_class_param_sample_WDM.ini")
    parser.add_argument('-o', '--output_grafic', help='output grafic param filename', type=str,
                        default="input_param_WDM.grafic")
    parser.add_argument('-z', '--zinit', help='input zini. The default value is to use the value in the input file.', type=float,
                        default=None)
    parser.add_argument('--high_precision', help='high precision setting. vely slow (several hours)',
                        action="store_true")
    args = parser.parse_args(argv[1:])
    return args


if __name__ == '__main__':
    args = set_args(sys.argv)
    print(vars(args))

    input_params = load_input_paramfile_WDM(args.input_parameter_file)
    wdm_flag = input_params["WDM_flag"]
    wdm_model = input_params["WDM_model"]

    if args.zinit is not None:
        input_params['zini'] = args.zinit

    zred = input_params['zini']

    if wdm_model == "Class":
        tk_grafic, pkz, cosmo = class_tk_outputs(input_params, args)
    if wdm_model == "Viel05":
        tk_grafic, pkz, cosmo = Viel05_tk_outputs(input_params, args)
    if wdm_model == "Abazajian06":
        tk_grafic, pkz, cosmo = Abazajian06_tk_outputs(input_params, args)
    if wdm_model == "Murgia17":
        tk_grafic, pkz, cosmo = Murgia17_tk_outputs(input_params, args)

    if wdm_flag:
        if grafic_format == True:
            klist *= cosmo.h()
            theaders = "k (1/Mpc) Tot Tot DM baryon dummy"
        else:
            theaders = "k (h/Mpc) Tot Tot DM baryon dummy"

        tk_grafic = np.vstack([tk_grafic, np.zeros(len(klist))])

    else:
        if grafic_format == True:
            klist *= cosmo.h()
            theaders = "k (1/Mpc) Tot Tot DM baryon"
        else:
            theaders = "k (h/Mpc) Tot Tot DM baryon"

    np.savetxt(input_params["tkfile"], np.transpose(tk_grafic), fmt="%.10e", header=theaders)

    if grafic_format == True:
        # reverse k unit
        klist /= cosmo.h()

    pheaders = "k (h/Mpc) Tot Tot DM baryon"

    if wdm_flag:
        filename = "class_powers_wdm_{0:.1f}keV_z{1:.1f}.dat".format(float(input_params["WDM_mass"]) / 1000.0, zred)
    else:
        filename = "class_powers_cdm_z{0:.1f}.dat".format(zred)

    np.savetxt(filename, np.transpose(pkz), fmt="%.10e", header=pheaders)

    print("The grafic code uses the cb component in the third column for input.")
    print("The WDM model uses the same code, so we put the total component in column 3. (This duplicates column 2.)")
