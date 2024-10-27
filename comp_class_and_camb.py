import sys
import argparse
import numpy as np
import pathlib
import pprint
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import quad

import classy
from classy import Class
import camb

import make_class_transfer as mct

npoint = 400
kmin = 2e-5
kmax = 100
logkmin = np.log10(kmin)
logkmax = np.log10(kmax)
klist = np.logspace(logkmin, logkmax, npoint)
logklist = np.log(klist)

# for background table
atab_bg = np.logspace(-np.log10(201), 0, 1001)

delta_z = 1e-5
zlist = np.array([200, 100, 50, 40, 30, 20, 10,
                  9, 8, 7, 6, 5, 4, 3, 2,
                  1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
                  0.4, 0.3, 0.2, 0.1, 0.0])


# high-z side
zlist_pdz = zlist + delta_z
# low-z side
zlist_mdz = zlist - delta_z

zlist_wdz = np.sort(np.concatenate([zlist, zlist_pdz, zlist_mdz[zlist_mdz > 0]]))[::-1]

zlist_idx = np.where(np.isin(zlist_wdz, zlist))[0]
assert np.all(zlist_wdz[zlist_idx] == zlist)
assert np.all(zlist_wdz[1::3] == zlist)


zlist_data_dir = "class_camb_zlist"
zlist_str = [(format(i + 1, "d") + ":z=" + format(s, ".1f")) for i, s in enumerate(zlist)]
zlist_str = " ".join(zlist_str)


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


# pk against klist (not logklist)
def calc_class_pk(_kh, _tk, cosmo):
    tk_spl = ius(np.log(_kh), _tk)
    pk = calc_pk(cosmo, tk_spl(logklist))
    return klist, pk


def calc_class_pk_ij(_kh, _tki, _tkj, cosmo):
    tki_spl = ius(np.log(_kh), _tki)
    tkj_spl = ius(np.log(_kh), _tkj)
    pk = calc_pk_ij(cosmo, tki_spl(logklist), tkj_spl(logklist))
    return klist, pk


# pk against klist (not logklist)
def calc_camb_pk(_kh, _tk, _pars):
    class _cosmo:
        pars = {}
        pars = {"h": _pars.h,
                "A_s": _pars.InitPower.As,
                "k_pivot": _pars.InitPower.pivot_scalar,
                "n_s": _pars.InitPower.ns}

    camb2class = -1.0 * (_kh * _cosmo.pars["h"])**2  # -1/k^2 (1/Mpc^2 not (h/Mpc)^2)
    tk_spl = ius(np.log(_kh), _tk * camb2class)
    pk = calc_pk(_cosmo, tk_spl(logklist))
    return klist, pk


def calc_camb_pk_ij(_kh, _tki, _tkj, _pars):
    class _cosmo:
        pars = {}
        pars = {"h": _pars.h,
                "A_s": _pars.InitPower.As,
                "k_pivot": _pars.InitPower.pivot_scalar,
                "n_s": _pars.InitPower.ns}

    camb2class = -1.0 * (_kh * _cosmo.pars["h"])**2  # -1/k^2 (1/Mpc^2 not (h/Mpc)^2)
    tki_spl = ius(np.log(_kh), _tki * camb2class)
    tkj_spl = ius(np.log(_kh), _tkj * camb2class)
    pk = calc_pk_ij(_cosmo, tki_spl(logklist), tkj_spl(logklist))
    return klist, pk


# pk against klist (not logklist)
def interp_tk(_kh, _tk):
    tmp_spl = ius(np.log(_kh), _tk)
    tk = tmp_spl(logklist)
    # tk = np.abs(tk)
    return klist, tk


def output_pktk_file(output_dir, target):
    kh = target[0]
    func = target[1]
    label = target[2]
    headers = "k (h/Mpc) {0:s}".format(label) + zlist_str
    filename = output_dir + label + ".dat"
    if len(func): np.savetxt(filename, np.transpose(np.vstack((kh, func))),
                             fmt="%.8e", header=headers)


def calc_and_save_class_dtk_vtk(input_params, output_suffix):
    """
    output_format = CLASS (default) : T(k) : curveture R = 1
    output_format = CAMB : -T(k)/K^2 : curveture R = -1/k^2 (not -1/(kh)^2)
    """

    cparams = mct.set_class_parameters(input_params)
    cparams['gauge'] = 'synchronous'
    # cparams['gauge'] = 'newtonian'
    pparams = mct.set_precisionsetting_dict(args.no_high_precision)
    cosmo_sync = Class()
    cosmo_sync.set(cparams)
    cosmo_sync.set(pparams)
    cosmo_sync.compute()

    print("done CLASS " + cparams['gauge'] + " setup")

    cparams['gauge'] = 'newtonian'
    cosmo_newt = Class()
    cosmo_newt.set(cparams)
    cosmo_newt.set(pparams)
    cosmo_newt.compute()

    print("done CLASS " + cparams['gauge'] + " setup")

    d_tot_tk, d_tot_pk = [], []
    d_cb_tk, d_cb_pk = [], []
    d_cdm_tk, d_cdm_pk = [], []
    d_bar_tk, d_bar_pk = [], []

    d_nu0_tk, d_nu0_pk = [], []
    d_nu1_tk, d_nu1_pk = [], []
    d_nu2_tk, d_nu2_pk = [], []
    d_nu_tot_tk, d_nu_tot_pk = [], []

    v_tot_tk, v_tot_pk = [], []
    v_cdm_tk, v_cdm_pk = [], []
    v_bar_tk, v_bar_pk = [], []

    v_nu_tot_tk, v_nu_tot_pk = [], []
    v_nu0_tk, v_nu0_pk = [], []
    v_nu1_tk, v_nu1_pk = [], []
    v_nu2_tk, v_nu2_pk = [], []

    relv_nutc_tk, relv_nutc_pk = [], []
    relv_nu0c_tk, relv_nu0c_pk = [], []
    relv_nu1c_tk, relv_nu1c_pk = [], []
    relv_nu2c_tk, relv_nu2c_pk = [], []
    relv_bc_tk, relv_bc_pk = [], []

    relv_nutc_Del2_pk = []
    relv_nu0c_Del2_pk = []
    relv_nu1c_Del2_pk = []
    relv_nu2c_Del2_pk = []
    relv_bc_Del2_pk = []
    relv_bc_Del2_pk_from_tk = []

    cbackg = cosmo_sync.get_background()
    cbackg_ascale = 1. / (1 + cbackg['z'])
    rho_cdm_spl = ius(cbackg_ascale, cbackg['(.)rho_cdm'])
    rho_b_spl = ius(cbackg_ascale, cbackg['(.)rho_b'])
    calH_spl = ius(cbackg_ascale, cbackg['H [1/Mpc]'] * cbackg_ascale)  # H*a
    Da_spl = ius(cbackg_ascale, cbackg['gr.fac. D'])
    fz_spl = ius(cbackg_ascale, cbackg['gr.fac. f'])
    h0 = cosmo_sync.h()

    for zred in zlist:
        transz = cosmo_sync.get_transfer(z=zred, output_format='class')
        kh = transz['k (h/Mpc)']
        ascale = 1.0 / (1.0 + zred)
        rho_cdm = rho_cdm_spl(ascale)
        rho_b = rho_b_spl(ascale)

        rho_nu = 0.0
        trans_nu = 0.0
        if 'd_ncdm[0]' in transz.keys():
            rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
            trans_nu += transz['d_ncdm[0]'] * rho_ncdm0
            rho_nu += rho_ncdm0
            d_nu0_pk.append(calc_class_pk(kh, transz['d_ncdm[0]'], cosmo_sync)[1])
            d_nu0_tk.append(interp_tk(kh, transz['d_ncdm[0]'])[1])

        if 'd_ncdm[1]' in transz.keys():
            rho_ncdm1 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[1]'])(ascale)
            trans_nu += transz['d_ncdm[1]'] * rho_ncdm1
            rho_nu += rho_ncdm1
            d_nu1_pk.append(calc_class_pk(kh, transz['d_ncdm[1]'], cosmo_sync)[1])
            d_nu1_tk.append(interp_tk(kh, transz['d_ncdm[1]'])[1])

        if 'd_ncdm[2]' in transz.keys():
            rho_ncdm2 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[2]'])(ascale)
            trans_nu += transz['d_ncdm[2]'] * rho_ncdm2
            rho_nu += rho_ncdm2
            d_nu2_pk.append(calc_class_pk(kh, transz['d_ncdm[2]'], cosmo_sync)[1])
            d_nu2_tk.append(interp_tk(kh, transz['d_ncdm[2]'])[1])

        # matter total
        tmp_tk = (transz['d_cdm'] * rho_cdm + transz['d_b'] * rho_b + trans_nu) / (rho_cdm + rho_b + rho_nu)
        d_tot_pk.append(calc_class_pk(kh, tmp_tk, cosmo_sync)[1])
        d_tot_tk.append(interp_tk(kh, tmp_tk)[1])

        # neutrino total
        if np.all(trans_nu != 0.0):
            tmp_tk = trans_nu / rho_nu
            d_nu_tot_pk.append(calc_class_pk(kh, tmp_tk, cosmo_sync)[1])
            d_nu_tot_tk.append(interp_tk(kh, tmp_tk)[1])

        # CDM+baryon
        tmp_tk = (transz['d_cdm'] * rho_cdm + transz['d_b'] * rho_b) / (rho_cdm + rho_b)
        d_cb_pk.append(calc_class_pk(kh, tmp_tk, cosmo_sync)[1])
        d_cb_tk.append(interp_tk(kh, tmp_tk)[1])

        d_cdm_pk.append(calc_class_pk(kh, transz['d_cdm'], cosmo_sync)[1])
        d_cdm_tk.append(interp_tk(kh, transz['d_cdm'])[1])

        d_bar_pk.append(calc_class_pk(kh, transz['d_b'], cosmo_sync)[1])
        d_bar_tk.append(interp_tk(kh, transz['d_b'])[1])

    cbackg = cosmo_newt.get_background()
    cbackg_ascale = 1. / (1 + cbackg['z'])
    rho_cdm_spl = ius(cbackg_ascale, cbackg['(.)rho_cdm'])
    rho_b_spl = ius(cbackg_ascale, cbackg['(.)rho_b'])
    calH_spl = ius(cbackg_ascale, cbackg['H [1/Mpc]'] * cbackg_ascale)  # H*a
    Da_spl = ius(cbackg_ascale, cbackg['gr.fac. D'])
    fz_spl = ius(cbackg_ascale, cbackg['gr.fac. f'])
    h0 = cosmo_newt.h()

    pkk = cosmo_sync.get_primordial()["k [1/Mpc]"]
    ppk = cosmo_sync.get_primordial()["P_scalar(k)"]
    ppk = ius(pkk, ppk)(klist)

    for zred in zlist:
        ascale = 1.0 / (1.0 + zred)
        transz = cosmo_newt.get_transfer(z=zred, output_format='class')
        kh = transz['k (h/Mpc)']

        calH = calH_spl(ascale)  # calH = H*a
        rho_cdm = rho_cdm_spl(ascale)
        rho_b = rho_b_spl(ascale)

        vpk_factor = 1.0 / calH

        v_cdm_pk.append(calc_class_pk(kh, transz['t_cdm'] * vpk_factor, cosmo_newt)[1])
        v_cdm_tk.append(interp_tk(kh, transz['t_cdm'])[1])

        v_bar_pk.append(calc_class_pk(kh, transz['t_b'] * vpk_factor, cosmo_newt)[1])
        v_bar_tk.append(interp_tk(kh, transz['t_b'])[1])

        rho_nu = 0.0
        trans_nu = 0.0
        if 't_ncdm[0]' in transz.keys():
            rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
            trans_nu += transz['t_ncdm[0]'] * rho_ncdm0
            rho_nu += rho_ncdm0
            v_nu0_pk.append(calc_class_pk(kh, transz['t_ncdm[0]'] * vpk_factor, cosmo_newt)[1])
            v_nu0_tk.append(interp_tk(kh, transz['t_ncdm[0]'])[1])
        if 't_ncdm[1]' in transz.keys():
            rho_ncdm1 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[1]'])(ascale)
            trans_nu += transz['t_ncdm[1]'] * rho_ncdm1
            rho_nu += rho_ncdm1
            v_nu1_pk.append(calc_class_pk(kh, transz['t_ncdm[1]'] * vpk_factor, cosmo_newt)[1])
            v_nu1_tk.append(interp_tk(kh, transz['t_ncdm[1]'])[1])
        if 't_ncdm[2]' in transz.keys():
            rho_ncdm2 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[2]'])(ascale)
            trans_nu += transz['t_ncdm[2]'] * rho_ncdm2
            rho_nu += rho_ncdm2
            v_nu2_pk.append(calc_class_pk(kh, transz['t_ncdm[2]'] * vpk_factor, cosmo_newt)[1])
            v_nu2_tk.append(interp_tk(kh, transz['t_ncdm[2]'])[1])

        tmp_tk = (transz['t_cdm'] * rho_cdm + transz['t_b'] * rho_b + trans_nu) / (rho_cdm + rho_b + rho_nu)
        v_tot_pk.append(calc_class_pk(kh, tmp_tk * vpk_factor, cosmo_newt)[1])
        v_tot_tk.append(interp_tk(kh, tmp_tk)[1])

        # neutrino total
        if np.all(trans_nu != 0.0):
            tmp_tk = trans_nu / rho_nu
            v_nu_tot_pk.append(calc_class_pk(kh, tmp_tk * vpk_factor, cosmo_newt)[1])
            v_nu_tot_tk.append(interp_tk(kh, tmp_tk)[1])

    for iz in range(len(zlist)):
        zred = zlist[iz]
        transz = cosmo_newt.get_transfer(z=zred, output_format='class')
        kh = transz['k (h/Mpc)']

        ascale = 1.0 / (1.0 + zred)
        rvtk_factor = -1.0 / (kh * h0)
        to_kms2 = 299792**2

        rel_vel_bc = (transz['t_b'] - transz['t_cdm']) * rvtk_factor
        relv_bc_tk.append(interp_tk(kh, rel_vel_bc)[1])

        tmp_k, tmp_rvpk = calc_class_pk(kh, rel_vel_bc, cosmo_newt)
        relv_bc_pk.append(tmp_rvpk)
        relv_bc_Del2_pk.append(tmp_k**3 / (2.0 * np.pi**2) * tmp_rvpk * to_kms2)

        _rvtk_factor = -1.0 / (klist * h0)
        tmp_tk = (_rvtk_factor * (v_bar_tk[iz] - v_cdm_tk[iz]))**2
        relv_bc_Del2_pk_from_tk.append(ppk * tmp_tk * to_kms2)

        rho_nu = 0.0
        trans_nu = 0.0
        if 't_ncdm[0]' in transz.keys():
            rho_ncdm0 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[0]'])(ascale)
            trans_nu += transz['t_ncdm[0]'] * rho_ncdm0
            rho_nu += rho_ncdm0

            rel_vel_nuc = (transz['t_ncdm[0]'] - transz['t_cdm']) * rvtk_factor
            relv_nu0c_tk.append(interp_tk(kh, rel_vel_nuc)[1])
            tmp_k, tmp_rvpk = calc_class_pk(kh, rel_vel_nuc, cosmo_newt)
            relv_nu0c_pk.append(tmp_rvpk)
            relv_nu0c_Del2_pk.append(tmp_k**3 / (2.0 * np.pi**2) * tmp_rvpk * to_kms2)

        if 't_ncdm[1]' in transz.keys():
            rho_ncdm1 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[1]'])(ascale)
            trans_nu += transz['t_ncdm[1]'] * rho_ncdm1
            rho_nu += rho_ncdm1

            rel_vel_nuc = (transz['t_ncdm[1]'] - transz['t_cdm']) * rvtk_factor
            relv_nu1c_tk.append(interp_tk(kh, rel_vel_nuc)[1])
            tmp_k, tmp_rvpk = calc_class_pk(kh, rel_vel_nuc, cosmo_newt)
            relv_nu1c_pk.append(tmp_rvpk)
            relv_nu1c_Del2_pk.append(tmp_k**3 / (2.0 * np.pi**2) * tmp_rvpk * to_kms2)

        if 't_ncdm[2]' in transz.keys():
            rho_ncdm2 = ius(cbackg_ascale, cbackg['(.)rho_ncdm[2]'])(ascale)
            trans_nu += transz['t_ncdm[2]'] * rho_ncdm2
            rho_nu += rho_ncdm2

            rel_vel_nuc = (transz['t_ncdm[2]'] - transz['t_cdm']) * rvtk_factor
            relv_nu2c_tk.append(interp_tk(kh, rel_vel_nuc)[1])
            tmp_k, tmp_rvpk = calc_class_pk(kh, rel_vel_nuc, cosmo_newt)
            relv_nu2c_pk.append(tmp_rvpk)
            relv_nu2c_Del2_pk.append(tmp_k**3 / (2.0 * np.pi**2) * tmp_rvpk * to_kms2)

        # neutrino total
        if np.all(trans_nu != 0.0):
            tmp_tk = trans_nu / rho_nu
            rel_vel_nuc = (tmp_tk - transz['t_cdm']) * rvtk_factor
            relv_nutc_tk.append(interp_tk(kh, rel_vel_nuc)[1])
            tmp_k, tmp_rvpk = calc_class_pk(kh, rel_vel_nuc, cosmo_newt)
            relv_nutc_pk.append(tmp_rvpk)
            relv_nutc_Del2_pk.append(tmp_k**3 / (2.0 * np.pi**2) * tmp_rvpk * to_kms2)

    print("done CLASS setup")

    class_dir = zlist_data_dir + "/class" + output_suffix + "/"
    pathlib.Path(class_dir).mkdir(parents=True, exist_ok=True)

    with open(class_dir + "/class_params.dat", "w") as f:
        pprint.pprint(cosmo_sync.pars, stream=f)

    filename = class_dir + "/background_param_table.dat"
    headers = "0:z  1:a  2:H [1/Mpc]  3:calH(H*a)  4:D(a)  5:f(z)=dln(D)/dln(a)"
    atab = atab_bg
    ztab = 1.0 / atab - 1.0
    calHtab = calH_spl(atab)
    Htab = calHtab / atab
    Datab = Da_spl(atab)
    fztab = fz_spl(atab)
    np.savetxt(filename, np.transpose(np.vstack((ztab, atab, Htab, calHtab, Datab, fztab))), fmt="%.8e", header=headers)

    filename = class_dir + "/background_param_zlist.dat"
    headers = "0:z  1:a  2:H [1/Mpc]  3:calH(H*a)  4:D(a)  5:f(z)=dln(D)/dln(a)"
    ztab = zlist
    atab = 1.0 / (1.0 + ztab)
    calHtab = calH_spl(atab)
    Htab = calHtab / atab
    Datab = Da_spl(atab)
    fztab = fz_spl(atab)
    np.savetxt(filename, np.transpose(np.vstack((ztab, atab, Htab, calHtab, Datab, fztab))), fmt="%.8e", header=headers)

    filename = class_dir + "/primordial_pk.dat"
    headers = "k [1/Mpc] primordial_pk [Mpc^3]"
    pkk = cosmo_sync.get_primordial()["k [1/Mpc]"]
    ppk = cosmo_sync.get_primordial()["P_scalar(k)"]
    ppk = ius(pkk, ppk)(klist * h0)
    np.savetxt(filename, np.transpose((klist * h0, ppk)), fmt="%.8e", header=headers)

    tk_targets = [(klist, d_tot_tk, "dens_tot_tk"), (klist, d_cb_tk, "dens_cb_tk"), (klist, d_cdm_tk, "dens_cdm_tk"),
                  (klist, d_bar_tk, "dens_baryon_tk"), (klist, d_nu_tot_tk, "dens_nu_tot_tk"), (klist, d_nu0_tk, "dens_nu0_tk"),
                  (klist, d_nu1_tk, "dens_nu1_tk"), (klist, d_nu2_tk, "dens_nu2_tk"),
                  (klist, v_tot_tk, "velc_tot_tk"), (klist, v_cdm_tk, "velc_cdm_tk"), (klist, v_bar_tk, "velc_baryon_tk"),
                  (klist, v_nu_tot_tk, "velc_nu_tot_tk"), (klist, v_nu0_tk, "velc_nu0_tk"), (klist, v_nu1_tk, "velc_nu1_tk"), (klist, v_nu2_tk, "velc_nu2_tk"),
                  (klist, relv_nu0c_tk, "rel_velc_nu0c"), (klist, relv_nu1c_tk, "rel_velc_nu1c"),
                  (klist, relv_nu2c_tk, "rel_velc_nu2c"), (klist, relv_nutc_tk, "rel_velc_nutc"),
                  (klist, relv_bc_tk, "rel_velc_bc")]

    pk_targets = [(klist, d_tot_pk, "dens_tot_pk"), (klist, d_cb_pk, "dens_cb_pk"), (klist, d_cdm_pk, "dens_cdm_pk"),
                  (klist, d_bar_pk, "dens_baryon_pk"), (klist, d_nu_tot_pk, "dens_nu_tot_pk"), (klist, d_nu0_pk, "dens_nu0_pk"),
                  (klist, d_nu1_pk, "dens_nu1_pk"), (klist, d_nu2_pk, "dens_nu2_pk"),
                  (klist, v_tot_pk, "velc_tot_pk"), (klist, v_cdm_pk, "velc_cdm_pk"), (klist, v_bar_pk, "velc_baryon_pk"),
                  (klist, v_nu_tot_pk, "velc_nu_tot_pk"), (klist, v_nu0_pk, "velc_nu0_pk"), (klist, v_nu1_pk, "velc_nu1_pk"), (klist, v_nu2_pk, "velc_nu2_pk"),
                  (klist, relv_nu0c_pk, "rel_velc_nu0c"), (klist, relv_nu1c_pk, "rel_velc_nu1c"),
                  (klist, relv_nu2c_pk, "rel_velc_nu2c"), (klist, relv_nutc_pk, "rel_velc_nutc"),
                  (klist, relv_bc_pk, "rel_velc_bc"), (klist, relv_bc_Del2_pk,
                                                       "rel_velc_bc_Del2_pk"), (klist, relv_bc_Del2_pk_from_tk, "rel_velc_bc_Del2_pk_from_tk"),
                  (klist, relv_nu0c_Del2_pk, "rel_velc_nu0c_Del2_pk"), (klist, relv_nu1c_Del2_pk, "rel_velc_nu1c_Del2_pk"),
                  (klist, relv_nu2c_Del2_pk, "rel_velc_nu2c_Del2_pk"), (klist, relv_nutc_Del2_pk, "rel_velc_nutc_Del2_pk")]

    for target in tk_targets:
        output_pktk_file(class_dir, target)

    for target in pk_targets:
        output_pktk_file(class_dir, target)

    return cosmo_sync, cosmo_newt


def calc_and_save_camb_dtk_vtk(input_params, output_suffix):
    h0 = input_params["h0"]
    Omm0 = input_params["Omega_m0"]
    Omb0 = input_params["Omega_b0"]
    Omnu0 = input_params["Nu_mass_tot"] / (93.14 * (h0**2))
    Omc0 = Omm0 - Omb0 - Omnu0

    print(Omm0, Omb0, Omc0, Omnu0, input_params["Nu_mass_tot"])

    pars = camb.CAMBparams()

    pars.set_cosmology(H0=100 * h0, ombh2=Omb0 * h0 ** 2,
                       omch2=Omc0 * h0 ** 2, mnu=input_params["Nu_mass_tot"])

    pars.num_nu_massive = sum(input_params["Nu_mass_degeneracies"])
    pars.num_nu_massless = 3.044 - pars.num_nu_massive
    pars.nu_mass_numbers = input_params["Nu_mass_degeneracies"]
    pars.nu_mass_eigenstates = len(input_params["Nu_mass_degeneracies"])
    pars.nu_mass_degeneracies = input_params["Nu_mass_degeneracies"]
    pars.nu_mass_fractions = input_params["Nu_mass_fractions"]

    if not args.no_high_precision:
        pars.Transfer.accurate_massive_neutrinos = True
        pars.Transfer.high_precision = True
        pars.Accuracy.AccuracyBoost = 4.0  # 1-4 default 1.0
        pars.Accuracy.lAccuracyBoost = 5.0  # 1-5 default 1.0
        pars.Accuracy.neutrino_q_boost = 5.0  # 1-5 default 1.0
        pars.MassiveNuMethod = 'Nu_best'

    nl_model = camb.model.NonLinear_none

    pars.InitPower.set_params(As=input_params["A_s"],
                              ns=input_params["ns"], pivot_scalar=input_params["k_pivot"])
    pars.set_matter_power(redshifts=zlist_wdz, kmax=100)
    pars.NonLinear = nl_model
    results = camb.get_results(pars)

    d_cdm_tk, d_cdm_pk = [], []
    d_bar_tk, d_bar_pk = [], []
    d_nu_tk, d_nu_pk = [], []
    d_tot_tk, d_tot_pk = [], []
    d_cb_tk, d_cb_pk = [], []
    v_cdm_tk, v_cdm_pk = [], []  # -vNc k / calH , vNc: Newtonian-gauge baryon velocity
    v_bar_tk, v_bar_pk = [], []  # -vNb k / calH , calH: time-dependent hubble parameter
    relv_bc_tk, relv_bc_pk = [], []  # vb-vc, relative baryon-CDM velocity

    v_cdm_tk_calc, v_cdm_pk_calc = [], []
    v_bar_tk_calc, v_bar_pk_calc = [], []
    v_nu_tk_calc, v_nu_pk_calc = [], []

    relv_nuc_tk_calc, relv_nuc_pk_calc = [], []

    relv_nuc_Del2_pk = []
    relv_bc_Del2_pk = []

    tfunc = results.get_matter_transfer_data().transfer_data

    for iz in zlist_idx:
        kh = tfunc[0][:, iz]

        tmp_tk = tfunc[1][:, iz]
        d_cdm_pk.append(calc_camb_pk(kh, tmp_tk, pars)[1])
        d_cdm_tk.append(interp_tk(kh, tmp_tk)[1])

        tmp_tk = tfunc[2][:, iz]
        d_bar_pk.append(calc_camb_pk(kh, tmp_tk, pars)[1])
        d_bar_tk.append(interp_tk(kh, tmp_tk)[1])

        tmp_tk = tfunc[5][:, iz]
        d_nu_pk.append(calc_camb_pk(kh, tmp_tk, pars)[1])
        d_nu_tk.append(interp_tk(kh, tmp_tk)[1])

        tmp_tk = tfunc[6][:, iz]
        d_tot_pk.append(calc_camb_pk(kh, tmp_tk, pars)[1])
        d_tot_tk.append(interp_tk(kh, tmp_tk)[1])

        tmp_tk = tfunc[7][:, iz]
        d_cb_pk.append(calc_camb_pk(kh, tmp_tk, pars)[1])
        d_cb_tk.append(interp_tk(kh, tmp_tk)[1])

        tmp_tk = tfunc[10][:, iz]
        v_cdm_pk.append(calc_camb_pk(kh, tmp_tk, pars)[1])
        v_cdm_tk.append(interp_tk(kh, tmp_tk)[1])

        tmp_tk = tfunc[11][:, iz]
        v_bar_pk.append(calc_camb_pk(kh, tmp_tk, pars)[1])
        v_bar_tk.append(interp_tk(kh, tmp_tk)[1])

        tmp_tk = tfunc[12][:, iz]
        relv_bc_pk.append(calc_camb_pk(kh, tmp_tk, pars)[1])
        relv_bc_tk.append(interp_tk(kh, tmp_tk)[1])

    for iz in zlist_idx:
        zred = zlist_wdz[iz]
        nu_dtk_m1 = tfunc[5][:, iz - 1]
        cdm_dtk_m1 = tfunc[1][:, iz - 1]
        bar_dtk_m1 = tfunc[2][:, iz - 1]

        if iz + 1 >= len(zlist_wdz):
            nu_dtk_p1 = tfunc[5][:, iz]
            cdm_dtk_p1 = tfunc[1][:, iz]
            bar_dtk_p1 = tfunc[2][:, iz]
            nu_vtk = (nu_dtk_p1 - nu_dtk_m1) / (delta_z)
            cdm_vtk = (cdm_dtk_p1 - cdm_dtk_m1) / (delta_z)
            bar_vtk = (bar_dtk_p1 - bar_dtk_m1) / (delta_z)
        else:
            nu_dtk_p1 = tfunc[5][:, iz + 1]
            cdm_dtk_p1 = tfunc[1][:, iz + 1]
            bar_dtk_p1 = tfunc[2][:, iz + 1]
            nu_vtk = (nu_dtk_p1 - nu_dtk_m1) / (2.0 * delta_z)
            cdm_vtk = (cdm_dtk_p1 - cdm_dtk_m1) / (2.0 * delta_z)
            bar_vtk = (bar_dtk_p1 - bar_dtk_m1) / (2.0 * delta_z)

        v_nu_pk_calc.append(calc_camb_pk(kh, nu_vtk, pars)[1])
        v_cdm_pk_calc.append(calc_camb_pk(kh, cdm_vtk, pars)[1])
        v_bar_pk_calc.append(calc_camb_pk(kh, bar_vtk, pars)[1])
        v_nu_tk_calc.append(interp_tk(kh, nu_vtk)[1])
        v_cdm_tk_calc.append(interp_tk(kh, cdm_vtk)[1])
        v_bar_tk_calc.append(interp_tk(kh, bar_vtk)[1])

    for iz in range(len(zlist)):
        zred = zlist[iz]
        kh = klist
        to_kms2 = 299792**2
        anow = 1.0 / (1.0 + zred)
        calH = results.h_of_z(zred) * anow  # camb_H*a = calH
        rvtk_factor = -calH / (kh * h0)

        tmp_k, tmp_rvpk = calc_camb_pk(kh, relv_bc_tk[iz], pars)
        relv_bc_Del2_pk.append(tmp_k**3 / (2.0 * np.pi**2) * tmp_rvpk * to_kms2)

        rel_vel_nuc = (v_nu_tk_calc[iz] - v_cdm_tk[iz]) * rvtk_factor
        tmp_k, tmp_rvpk = calc_camb_pk(kh, rel_vel_nuc, pars)
        relv_nuc_tk_calc.append(rel_vel_nuc)
        relv_nuc_pk_calc.append(tmp_rvpk)
        relv_nuc_Del2_pk.append(tmp_k**3 / (2.0 * np.pi**2) * tmp_rvpk * to_kms2)

    print("done CAMB setup")

    camb_dir = zlist_data_dir + "/camb" + output_suffix + "/"
    pathlib.Path(camb_dir).mkdir(parents=True, exist_ok=True)

    with open(camb_dir + "/camb_params.dat", "w") as f:
        pprint.pprint(results.get_background_outputs, stream=f)

    filename = camb_dir + "/background_param_table.dat"
    headers = "0:z  1:a  2:H [1/Mpc]  3:calH(H*a)  4:D(a)  5:f(z)=dln(D)/dln(a)"
    atab = atab_bg
    ztab = 1.0 / atab - 1.0
    Htab = results.h_of_z(ztab)
    calHtab = Htab * atab
    np.savetxt(filename, np.transpose(np.vstack((ztab, atab, Htab, calHtab))), fmt="%.8e", header=headers)

    filename = camb_dir + "/background_param_zlist.dat"
    headers = "0:z  1:a  2:H [1/Mpc]  3:calH(H*a)  4:D(a)  5:f(z)=dln(D)/dln(a)"
    ztab = zlist
    atab = 1.0 / (1.0 + ztab)
    Htab = results.h_of_z(ztab)
    calHtab = Htab * atab
    np.savetxt(filename, np.transpose(np.vstack((ztab, atab, Htab, calHtab))), fmt="%.8e", header=headers)

    filename = camb_dir + "primordial_pk.dat"
    headers = "k [1/Mpc] primordial_pk [Mpc^3]"
    primordial_pk = pars.scalar_power(klist * h0)
    np.savetxt(filename, np.transpose((klist * h0, primordial_pk)), fmt="%.8e", header=headers)

    tk_targets = [(klist, d_tot_tk, "dens_tot_tk"), (klist, d_cb_tk, "dens_cb_tk"), (klist, d_cdm_tk, "dens_cdm_tk"),
                  (klist, d_bar_tk, "dens_baryon_tk"), (klist, d_nu_tk, "dens_nu_tot_tk"),
                  (klist, v_cdm_tk, "velc_cdm_tk"),
                  (klist, v_bar_tk, "velc_baryon_tk"), (klist, v_nu_tk_calc, "velc_nu_tot_tk"), (klist, relv_nuc_tk_calc, "rel_velc_nutc"),
                  (klist, relv_bc_tk, "rel_velc_bc")]

    pk_targets = [(klist, d_tot_pk, "dens_tot_pk"), (klist, d_cb_pk, "dens_cb_pk"), (klist, d_cdm_pk, "dens_cdm_pk"),
                  (klist, d_bar_pk, "dens_baryon_pk"), (klist, d_nu_pk, "dens_nu_tot_pk"), (klist, v_cdm_pk, "velc_cdm_pk"),
                  (klist, v_bar_pk, "velc_baryon_pk"), (klist, v_nu_pk_calc, "velc_nu_tot_pk"), (klist, relv_nuc_pk_calc, "rel_velc_nutc"),
                  (klist, relv_bc_pk, "rel_velc_bc"), (klist, relv_bc_Del2_pk, "rel_velc_bc_Del2_pk"), (klist, relv_nuc_Del2_pk, "rel_velc_nutc_Del2_pk")]

    for target in tk_targets:
        output_pktk_file(camb_dir, target)

    for target in pk_targets:
        output_pktk_file(camb_dir, target)

    return results


def set_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_parameter_file', help='input parameter file',
                        type=str, default="./input_class_param_sample.ini")
    parser.add_argument('-no_hp', '--no_high_precision', help='No high precision for massive neutrinos.',
                        action="store_true")
    parser.add_argument('-suffix', '--output_suffix', help="Suffix of output directory.",
                        type=str, default="")
    args = parser.parse_args(argv[1:])
    return args


if __name__ == '__main__':
    args = set_args(sys.argv)
    print(vars(args))

    input_params = mct.load_input_paramfile(args.input_parameter_file)
    input_params = mct.set_neutrino_parameters(input_params)

    camb_obj = calc_and_save_camb_dtk_vtk(input_params, args.output_suffix)
    class_sync_obj, class_newt_obj = calc_and_save_class_dtk_vtk(input_params, args.output_suffix)

    pathlib.Path(zlist_data_dir).mkdir(parents=True, exist_ok=True)
    """
    These output format is below.

                 CLASS       CAMB
    dens_tk      1          -1/k^2
    dens_pk      1           1
    velc_tk      v*(-k)      v*(-k/calH)*(-1/k^2)
    velc_pk      1          -1/k^2

    rel_velc_ij  vi-vj       vi-vj
    rel_velc_nu0c_Del2_pk  1    1   [(km/s)^2]


    k unit is 1/Mpc.
    kh unit is h/Mpc.
    H unit is 1/Mpc.
    calH is H*a.
    """
