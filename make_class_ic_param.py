import sys
import argparse
import numpy as np

# model="Planck2015"
model = "Planck2018"


def set_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nu_mass', help='neutrino total mass',
                        type=float, default=0.1)
    parser.add_argument('--mass_type', help='neutrino mass type',
                        type=str, default="equal", choices=["massless", "equal", "normal", "inverted"])
    parser.add_argument('-o', '--output_file', help='output filename', type=str,
                        default="input_class.ini")
    parser.add_argument('-z', '--zinit', help='input zini. The default value is to use the value in the input file.', type=float,
                        default=None)
    args = parser.parse_args(argv[1:])
    return args


args = set_args(sys.argv)
print(vars(args))


def get_EQ_mass(mtot):
    m1 = mtot / 3.0
    m2 = mtot / 3.0
    m3 = mtot / 3.0

    if np.isclose(m1 + m2 + m3, mtot):
        return m1, m2, m3
    else:
        print(f"Impossible to define mass for mtot={mtot}")
        return -1, -1, -1


def get_NH_mass(mtot, m21_2=7.42e-5, m31_2=2.510e-3):
    """
    mtot = m1 + m2 + m3
    m2 = np.sqrt(m21_2 + m1**2)
    m3 = np.sqrt(m31_2 + m1**2)
    0 = m1 + np.sqrt(m21_2 + m1**2) + np.sqrt(m31_2 + m1**2) - mtot
    """

    def nh_f(m1):
        m2 = np.sqrt(m21_2 + m1**2)
        m3 = np.sqrt(m31_2 + m1**2)
        return m1 + m2 + m3 - mtot

    m1_low = 0
    m1_high = mtot
    eps = 1e-12

    while m1_high - m1_low > eps:
        m1_mid = 0.5 * (m1_low + m1_high)
        if nh_f(m1_mid) > 0:
            m1_high = m1_mid
        else:
            m1_low = m1_mid

    m1 = 0.5 * (m1_low + m1_high)
    m2 = np.sqrt(m21_2 + m1**2)
    m3 = np.sqrt(m31_2 + m1**2)

    if np.isclose(m1 + m2 + m3, mtot):
        return m1, m2, m3
    else:
        print(f"Impossible to define mass for mtot={mtot}")
        return -1, -1, -1


def get_IH_mass(mtot, m21_2=7.42e-5, m32_2=-2.490e-3):
    """
    mtot = m1 + m2 + m3
    m2 = np.sqrt(m21_2 + m1**2)
    m3 = np.sqrt(m32_2 + m2**2)
    0 = m1 + np.sqrt(m21_2 + m1**2) + np.sqrt(m32_2 + m21_2 + m1**2) - mtot
    """

    def ih_f(m1):
        m2 = np.sqrt(m21_2 + m1**2)
        m3 = np.sqrt(m32_2 + m2**2)
        return m1 + m2 + m3 - mtot

    m1_low = np.sqrt(-m21_2 - m32_2)
    m1_high = mtot
    eps = 1e-12

    if m1_high < m1_low:
        print(f"Impossible to define mass for mtot={mtot}")
        return -1, -1, -1

    while m1_high - m1_low > eps:
        m1_mid = 0.5 * (m1_low + m1_high)
        if ih_f(m1_mid) > 0:
            m1_high = m1_mid
        else:
            m1_low = m1_mid

    m1 = 0.5 * (m1_low + m1_high)
    m2 = np.sqrt(m21_2 + m1**2)
    m3 = np.sqrt(m32_2 + m2**2)

    if np.isclose(m1 + m2 + m3, mtot):
        return m1, m2, m3
    else:
        print(f"Impossible to define mass for mtot={mtot}")
        return -1, -1, -1
