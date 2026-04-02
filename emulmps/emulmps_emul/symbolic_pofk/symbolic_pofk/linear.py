import numpy as np
import warnings
import scipy.integrate


## Vic edits
def safe_log_w_param(w0, wa=0, epsilon=1e-10):
    """
    Safely compute log of w-parameter expressions.
    
    For expressions like log(-w0) or log(-w0 - wa):
    - If the argument is positive, return log normally
    - If negative or zero, use log(abs(x) + epsilon)
    
    Args:
        w0: dark energy equation of state parameter (constant)
        wa: dark energy equation of state parameter (evolving)
        epsilon: small value to prevent log(0)
    
    Returns:
        Logarithm of the expression, handling edge cases
    """
    arg = -w0 - wa if wa != 0 else -w0
    
    # Use abs to handle both positive and negative cases
    # Add epsilon to prevent log(0)
    return np.log(np.abs(arg) + epsilon)
## Vic edits

def get_eisensteinhu_nw(k, As, Om, Ob, h, ns, mnu, w0, wa):
    """
    Compute the no-wiggles Eisenstein & Hu approximation
    to the linear P(k) at redshift zero.

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum

    Returns:
        :pk (np.ndarray): Approxmate linear power spectrum at corresponding k values [(Mpc/h)^3]
    """
    ombom0 = Ob / Om
    om0h2 = Om * h**2
    ombh2 = Ob * h**2
    theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin

    # Compute scale factor s, alphaGamma, and effective shape Gamma
    s = 44.5 * np.log(9.83 / om0h2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)
    alphaGamma = 1.0 - 0.328 * \
        np.log(431.0 * om0h2) * ombom0 + 0.38 * \
        np.log(22.3 * om0h2) * ombom0**2
    Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) /
                      (1.0 + (0.43 * k * h * s)**4))

    # Compute q, C0, L0, and tk_eh
    q = k * theta2p7**2 / Gamma
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
    tk_eh = L0 / (L0 + C0 * q**2)

    kpivot = 0.05

    pk = (
        2 * np.pi ** 2 / k ** 3
        * (As * 1e-9) * (k * h / kpivot) ** (ns - 1)
        * (2 * k ** 2 * 2998**2 / 5 / Om) ** 2
        * tk_eh ** 2
    )
    return pk

def get_approximate_D(k, As, Om, Ob, h, ns, mnu, w0, wa, a):
    """
    Approximation to the growth factor using the results of
    Bond et al. 1980, Lahav et al. 1992, Carrol et al. 1992 
    and Eisenstein & Hu 1997 (D_cbnu).

    There are two differences between our method and theirs. 
    First, in Eisenstein & Hu 1997 D is chosen to be (1 + zeq) a at 
    early times, whereas we instead choose D -> a at early times. 
    Second, the formulae reported there assume that w=-1, whereas we
    change the Omega_Lambda terms to include a w0-wa parameterisation.

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS
        :a (float): Scale factor to consider

    Returns:
        :D (np.ndarray): Approximate linear growth factor at corresponding k values
    """

    # avoid singularities
    mnu = mnu + 1e-10

    # Get fitting formula without free-streaming
    z = 1 / a - 1
    theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin
    zeq = 2.5e4 * Om * h ** 2 / theta2p7 ** 4

    Omega = Om * a ** (-3)
    OL = (1 - Om) * a ** (-3 * (1 + w0 + wa)) * np.exp(- 3 * wa * (1 - a))
    g = np.sqrt(Omega + OL)
    Omega /= g ** 2
    OL /= g ** 2

    D1 = (
        (1 + zeq) / (1 + z) * 5 * Omega / 2 /
        (Omega ** (4/7) - OL + (1 + Omega/2) * (1 + OL/70))
    )

    # Split Omega_m into CDM, Baryons and Neutrinos
    Onu = mnu / 93.14 / h ** 2
    Oc = Om - Ob - Onu
    fc = Oc / Om
    fb = Ob / Om
    fnu = Onu / Om
    fcb = fc + fb

    # Add Bond et al. 1980 suppression
    pcb = 1/4 * (5 - np.sqrt(1 + 24 * fcb))
    Nnu = (3 if mnu != 0.0 else 0)
    q = k * h * theta2p7 ** 2 / (Om * h ** 2)
    yfs = 17.2 * fnu * (1 + 0.488 / fnu ** (7/6)) * (Nnu * q / fnu) ** 2
    Dcbnu = (fcb ** (0.7/pcb) + (D1 / (1 + yfs)) **
             0.7) ** (pcb / 0.7) * D1 ** (1 - pcb)

    # Remove 1+zeq normalisation given in Eisenstein & Hu 1997
    D = Dcbnu / (1 + zeq)

    return D

def log10_S(k, As, Om, Ob, h, ns, mnu, w0, wa):
    """
    Corrections to the present-day linear power spectrum

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS
        :a (float): Scale factor to consider

    Returns:
        :result (np.ndarray): Corrections to the present-day linear power spectrum
    """
    e = np.array([0.2841, 0.1679, 0.0534, 0.0024, 0.1183, 0.3971,
                  0.0985, 0.0009, 0.1258, 0.2476, 0.1841, 0.0316,
                  0.1385, 0.2825, 0.8098, 0.019, 0.1376, 0.3733])

    ## Vic changes
    epsilon = 1e-10  # Small value to prevent log(0)
    ## Vic changes end

    part1 = -e[0] * h
    part2 = -e[1] * w0
    part3 = -e[2] * mnu / np.sqrt(e[3] + k**2)

    part4 = -(e[4] * h) / (e[5] * h + mnu)

    part5 = e[6] * mnu / (h * np.sqrt(e[7] + (Om * e[8] + k)**2))

    numerator_inner = (e[9] * Ob - e[10] * w0 - e[11] * wa +
                       (e[12] * w0 + e[13]) / (e[14] * wa + w0))
    
    ## Vic changes: Use safe log with abs
    log_term = np.log(np.abs(-e[17] * w0) + epsilon)
    denominator_inner = np.sqrt(e[15] + (Om + e[16] * log_term)**2)
    ## Vic changes end

    part6 = numerator_inner / denominator_inner

    # Sum all parts to get the final result
    result = part1 + part2 + part3 + part4 + part5 + part6

    return result/10

def growth_correction_R(As, Om, Ob, h, ns, mnu, w0, wa, a):
    """
    Correction to the growth factor 

    Args:
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Om
        :Ob (float): The z=0 baryonic density parameter, Ob
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS
        :a (float): The scale factor to evaluate P(k) at

    Returns:
        :result (float): correction to the growth factor
    """
    d = np.array([0.8545, 0.394, 0.7294, 0.5347, 0.4662, 4.6669,
                  0.4136, 1.4769, 0.5959, 0.4553, 0.0799, 5.8311,
                  5.8014, 6.7085, 0.3445, 1.2498, 0.3756, 0.2136])

    ## Vic changes
    epsilon = 1e-10  # Small value to prevent log(0)
    ## Vic changes end

    part1 = d[0]

    ## Vic changes: Use safe log with abs
    log_term1 = np.log(np.abs(-d[5] * w0 - d[6] * wa) + epsilon)
    denominator_inner1 = a * d[1] + d[2] + (Om * d[3] - a * d[4]) * log_term1
    ## Vic changes end
    part2 = -1 / denominator_inner1

    ## Vic changes: Use safe log with abs
    log_term2 = np.log(np.abs(-d[9] * w0 - d[10] * wa) + epsilon)
    numerator_inner2 = Om * d[7] - a * d[8] + log_term2
    ## Vic changes end

    denominator_inner2 = -a * d[11] + d[12] + d[13] * \
        (Om * d[14] + a * d[15] - 1) * (d[16] * w0 + d[17] * wa + 1)
    part3 = -numerator_inner2 / denominator_inner2

    result = 1 + (1 - a) * (part1 + part2 + part3)

    return result

def logF_fiducial(k, Om, Ob, h, ns):
    """
    Compute the emulated logarithm of the ratio between the true linear
    power spectrum and the Eisenstein & Hu 1998 fit. Here we use the fiducial exprssion
    given in Bartlett et al. 2023.
    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
    Returns:
        :logF (np.ndarray): The logarithm of the ratio between the linear P(k) and the
            Eisenstein & Hu 1998 zero-baryon fit
    """
    b = [0.05448654, 0.00379, 0.0396711937097927, 0.127733431568858, 1.35,
         4.053543862744234, 0.0008084539054750851, 1.8852431049189666,
         0.11418372931475675, 3.798, 14.909, 5.56, 15.8274343004709, 0.0230755621512691,
         0.86531976, 0.8425442636372944, 4.553956000000005, 5.116999999999995,
         70.0234239999998, 0.01107, 5.35, 6.421, 134.309, 5.324, 21.532,
         4.741999999999985, 16.68722499999999, 3.078, 16.987, 0.05881491,
         0.0006864690561825617, 195.498, 0.0038454457516892, 0.276696018851544,
         7.385, 12.3960625361899, 0.0134114370723638]

    line1 = b[0] * h - b[1]

    line2 = (
        ((Ob * b[2]) / np.sqrt(h ** 2 + b[3])) ** (b[4] * Om) *
        (
            (b[5] * k - Ob) / np.sqrt(b[6] + (Ob - b[7] * k) ** 2)
            * b[8] * (b[9] * k) ** (-b[10] * k) * np.cos(Om * b[11]
                                                         - (b[12] * k) / np.sqrt(b[13] + Ob ** 2))
            - b[14] * ((b[15] * k) / np.sqrt(1 + b[16] * k ** 2) - Om)
            * np.cos(b[17] * h / np.sqrt(1 + b[18] * k ** 2))
        )
    )

    line3 = (
        b[19] * (b[20] * Om + b[21] * h - np.log(b[22] * k)
                 + (b[23] * k) ** (- b[24] * k)) * np.cos(b[25] / np.sqrt(1 + b[26] * k ** 2))
    )

    line4 = (
        (b[27] * k) ** (-b[28] * k) * (b[29] * k - (b[30] * np.log(b[31] * k))
                                       / np.sqrt(b[32] + (Om - b[33] * h) ** 2))
        * np.cos(Om * b[34] - (b[35] * k) / np.sqrt(Ob ** 2 + b[36]))
    )
    logF = line1 + line2 + line3 + line4
    return logF

def plin_emulated(k, Om, Ob, h, ns, As=1e-9, w0=-1, wa=0.0, a=1):
    """
    Compute the emulated linear matter power spectrum using the fits of
    Eisenstein & Hu 1998 and Bartlett et al. 2023.

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :a (float, default=1): The scale factor to evaluate P(k) at
        :emulator (str, default='fiducial'): Which emulator to use from Bartlett et al.
            2023. 'fiducial' uses the fiducial one, and 'max_precision' uses the
            most precise one.
        :extrapolate (bool, default=False): If using the Bartlett et al. 2023 fit, then
            if true we extrapolate outside range tested in paper. Otherwise, uses E&H with
            baryons for k < kmin and k > kmax. Due to problems with the Colossus
            Eisenstein & Hu fit on large scales (typically a 2 percent offset), it is
            strongly recommended to use extrapolate=False.
        :kmin (float, default=9.e-3): Minimum k value to use Bartlett et al. formula
            if extrapolate=False
        :kmax (float, default=9): Maximum k value to use Bartlett et al. formula
            if extrapolate=False

    Returns:
        :pk_lin (np.ndarray): The emulated linear P(k) [(Mpc/h)^3]
    """
    eh = get_eisensteinhu_nw(k, As, Om, Ob, h, ns, mnu=0.06, w0=w0, wa=wa)
    D1 = get_approximate_D(k=k, As=As, Om=Om, Ob=Ob, h=h, 
                           ns=ns, mnu=0.06, w0=w0, wa=wa, a=a)
    R1 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, 
                             ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
    F1 = np.exp(logF_fiducial(k, Om, Ob, h, ns))
    S1 = np.power(10, log10_S(k, As, Om, Ob, h, ns, 0.06, w0, wa))    
    return eh*D1*D1*F1*R1*S1

def As_to_sigma8(As, Om, Ob, h, ns, mnu, w0, wa):
    """
    Compute the emulated conversion As -> sigma8, using the most accurate expression

    Args:
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Om
        :Ob (float): The z=0 baryonic density parameter, Ob
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS

    Returns:
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
    """
    b = np.array([0.0246, 2.1062, 2.9355, 0.7626, 0.2962, 0.5096, 
                  4.4025, 3.6495, 0.4144, 0.8615, 0.6188, 0.1751, 
                  0.824, 0.5466, 0.5519, 0.3689, 0.3261, 0.2002, 
                  0.8892, 0.4462, 1.215, 3.4829, 2.5852, 0.0242, 
                  0.0051, 0.1614, 1.2991, 4.1426, 3.3055, 0.5716, 
                  6.0094, 1.9569, 2.1477, 1.1902, 0.128, 0.6931, 
                  0.2661])

    ## Vic changes
    epsilon = 1e-10  # Small value to prevent log(0)
    ## Vic changes end

    term1_inner = (Om * b[1] + 
                   (b[2] * mnu - b[3] * ns + np.log(b[4] * h - b[5] * mnu)) *
                   (b[6] * h + b[7] * mnu - b[8] * ns + 1))
    term1 = b[0] * term1_inner
    
    term2 = b[9] * h - mnu

    ## Vic changes: Use safe log with abs for all w0, wa log terms
    log_term1 = np.log(np.abs(-b[18] * w0 - b[19] * wa) + epsilon)
    log_term2 = np.log(np.abs(-b[21] * w0 - b[22] * wa) + epsilon)
    log_term3 = np.log(np.abs(-b[23] * w0 - b[24] * wa) + epsilon)
    
    term3_inner1 = (b[12] * w0 - b[13] * wa - np.log(Om * b[14])) * \
                   (Om * b[15] + b[16] * w0 + b[17] * wa + log_term1)
    term3_inner2 = np.log(Om * b[20] + log_term2)
    term3 = b[10] * w0 - b[11] * mnu - term3_inner1 - term3_inner2 + log_term3
    ## Vic changes end
    
    term4_inner1 = Ob * b[30] - b[31] * h - np.log(Om * b[32])
    term4_inner2 = Om * b[33] - b[34] * h - b[35] * mnu - b[36] * ns
    term4 = b[25] * mnu - np.sqrt(Ob) * b[26] - Ob * b[27] + Om * b[28] - b[29] * h + 1 + term4_inner1 * term4_inner2

    result = term1 * term2 * term3 * term4
    
    return result*np.sqrt(As)