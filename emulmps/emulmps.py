"""
Fast Matter Power Spectrum Emulator for Cobaya

This module provides a neural network-based emulator for cosmological matter
power spectra within the Cobaya framework. It uses a local emulmps module
that handles all model loading and prediction internally.

The emulator predicts matter power spectra P(k,z) from cosmological parameters.
By default, both linear and nonlinear P(k) are computed (using halofit+).

Supports LCDM, wCDM, and w0waCDM cosmologies.

Author: Victoria Lloyd & V. Miranda
Date: 2025
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from cobaya.theory import Theory
from typing import Mapping, Tuple, Optional
from cobaya.typing import empty_dict, InfoDict
from cobaya.log import LoggedError, get_logger
from pathlib import Path
import logging

# --- Path Management ---
def _get_project_root() -> Path:
    """Returns the root directory Path object, relative to this file."""
    return Path(__file__).resolve().parent

# --- Dependency Guards ---
ROOT = _get_project_root()
# Optional imports for nonlinear corrections
try:
    # Issues installing symbolic_pofk as a package, including a local copy in the emulator
    import sys; sys.path.insert(0, f"{ROOT}/emulmps_emul/symbolic_pofk")
    from symbolic_pofk.linear import As_to_sigma8, plin_emulated
    from symbolic_pofk.syrenhalofit import run_halofit, run_halofit_vec
    # Create module-level aliases for compatibility
    class symbolic_linear:
        As_to_sigma8 = As_to_sigma8
        plin_emulated = plin_emulated
    class syrenhalofit:
        run_halofit = run_halofit
        run_halofit_vec = run_halofit_vec
    SYMBOLIC_POFK_AVAILABLE = True
except ImportError as e:
    logging.error(f"ERROR: A required dependency could not be imported. Please ensure all dependencies are installed.")
    logging.error(f"Missing component: {e.name if hasattr(e, 'name') else 'symbolic_pofk'}")
    logging.error(f"If running this package locally, ensure the symbolic_pofk library is accessible.")
    SYMBOLIC_POFK_AVAILABLE = False
    symbolic_linear = None
    syrenhalofit = None


# Import the local emulmps module
try:
    from .emulmps_emul import emulmps_w0wa as emulmps_emul
    from emulmps_emul import get_pks
except ImportError:
    try:
        # Alternative import path if running from different location
        from .emulmps_emul.emulmps_w0wa import get_pks
    except ImportError as e:
        raise ImportError(
            "Could not import emulmps module. Ensure the emulmps directory "
            "is present in the same directory as this theory file. "
            f"Error: {e}"
        )

########## Taken from cobaya/theories/cosmo/boltzmannbase.py ############
class PowerSpectrumInterpolator(RectBivariateSpline):
    r"""
    2D spline interpolation object (scipy.interpolate.RectBivariateSpline)
    to evaluate matter power spectrum as function of z and k.

    *This class is adapted from CAMB's own P(k) interpolator, by Antony Lewis;
    it's mostly interface-compatible with the original.*

    :param z: values of z for which the power spectrum was evaluated.
    :param k: values of k for which the power spectrum was evaluated.
    :param P_or_logP: Values of the power spectrum (or log-values, if logP=True).
    :param logP: if True (default: False), log of power spectrum are given and used
        for the underlying interpolator.
    :param logsign: if logP is True, P_or_logP is log(logsign*Pk)
    :param extrap_kmax: if set, use power law extrapolation beyond kmax up to
        extrap_kmax; useful for tails of integrals.
    """

    def __init__(self, z, k, P_or_logP, extrap_kmin=None, extrap_kmax=None, logP=False,
                 logsign=1):
        self.islog = logP
        #  Check order
        z, k = (np.atleast_1d(x) for x in [z, k])
        if len(z) < 4:
            raise ValueError('Require at least four redshifts for Pk interpolation.'
                             'Consider using Pk_grid if you just need a small number'
                             'of specific redshifts (doing 1D splines in k yourself).')
        z, k, P_or_logP = np.array(z), np.array(k), np.array(P_or_logP)
        i_z = np.argsort(z)
        i_k = np.argsort(k)
        self.logsign = logsign
        self.z, self.k, P_or_logP = z[i_z], k[i_k], P_or_logP[i_z, :][:, i_k]
        self.zmin, self.zmax = self.z[0], self.z[-1]
        self.extrap_kmin, self.extrap_kmax = extrap_kmin, extrap_kmax
        logk = np.log(self.k)
        # Start from extrap_kmin using a (log,log)-linear extrapolation
        if extrap_kmin and extrap_kmin < self.input_kmin:
            if not logP:
                raise ValueError('extrap_kmin must use logP')
            logk = np.hstack(
                [np.log(extrap_kmin),
                 np.log(self.input_kmin) * 0.1 + np.log(extrap_kmin) * 0.9, logk])
            logPnew = np.empty((P_or_logP.shape[0], P_or_logP.shape[1] + 2))
            logPnew[:, 2:] = P_or_logP
            diff = (logPnew[:, 3] - logPnew[:, 2]) / (logk[3] - logk[2])
            delta = diff * (logk[2] - logk[0])
            logPnew[:, 0] = logPnew[:, 2] - delta
            logPnew[:, 1] = logPnew[:, 2] - delta * 0.9
            P_or_logP = logPnew
        # Continue until extrap_kmax using a (log,log)-linear extrapolation
        if extrap_kmax and extrap_kmax > self.input_kmax:
            if not logP:
                raise ValueError('extrap_kmax must use logP')
            logk = np.hstack(
                [logk, np.log(self.input_kmax) * 0.1 + np.log(extrap_kmax) * 0.9,
                 np.log(extrap_kmax)])
            logPnew = np.empty((P_or_logP.shape[0], P_or_logP.shape[1] + 2))
            logPnew[:, :-2] = P_or_logP
            diff = (logPnew[:, -3] - logPnew[:, -4]) / (logk[-3] - logk[-4])
            delta = diff * (logk[-1] - logk[-3])
            logPnew[:, -1] = logPnew[:, -3] + delta
            logPnew[:, -2] = logPnew[:, -3] + delta * 0.9
            P_or_logP = logPnew
        super().__init__(self.z, logk, P_or_logP)

    @property
    def input_kmin(self):
        """Minimum k for the interpolation (not incl. extrapolation range)."""
        return self.k[0]

    @property
    def input_kmax(self):
        """Maximum k for the interpolation (not incl. extrapolation range)."""
        return self.k[-1]

    @property
    def kmin(self):
        """Minimum k of the interpolator (incl. extrapolation range)."""
        if self.extrap_kmin is None:
            return self.input_kmin
        return self.extrap_kmin

    @property
    def kmax(self):
        """Maximum k of the interpolator (incl. extrapolation range)."""
        if self.extrap_kmax is None:
            return self.input_kmax
        return self.extrap_kmax

    def check_ranges(self, z, k):
        """Checks that we are not trying to extrapolate beyond the interpolator limits."""
        z = np.atleast_1d(z).flatten()
        min_z, max_z = min(z), max(z)
        if min_z < self.zmin and not np.allclose(min_z, self.zmin):
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to z={min(z)} "
                              f"(minimum z computed is {self.zmin}).")
        if max_z > self.zmax and not np.allclose(max_z, self.zmax):
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to z={max(z)} "
                              f"(maximum z computed is {self.zmax}).")
        k = np.atleast_1d(k).flatten()
        min_k, max_k = min(k), max(k)
        if min_k < self.kmin and not np.allclose(min_k, self.kmin):
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to k={min(k)} 1/Mpc "
                              f"(minimum k possible is {self.kmin} 1/Mpc).")
        if max_k > self.kmax and not np.allclose(max_k, self.kmax):
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to k={max(k)} 1/Mpc "
                              f"(maximum k possible is {self.kmax} 1/Mpc).")

    def P(self, z, k, grid=None):
        """
        Get the power spectrum at (z,k).
        """
        self.check_ranges(z, k)
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self.logsign * np.exp(self(z, np.log(k), grid=grid, warn=False))
        else:
            return self(z, np.log(k), grid=grid, warn=False)

    def logP(self, z, k, grid=None):
        """
        Get the log power spectrum at (z,k). (or minus log power spectrum if
        islog and logsign=-1)
        """
        self.check_ranges(z, k)
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self(z, np.log(k), grid=grid, warn=False)
        else:
            return np.log(self(z, np.log(k), grid=grid, warn=False))

    def __call__(self, *args, warn=True, **kwargs):
        if warn:
            get_logger(self.__class__.__name__).warning(
                "Do not call the instance directly. Use instead methods P(z, k) or "
                "logP(z, k) to get the (log)power spectrum. (If you know what you are "
                "doing, pass warn=False)")
        return super().__call__(*args, **kwargs)


class emulmps(Theory):
    """
    Fast Matter Power Spectrum Emulator Theory Code.
    
    This class provides a fast neural network emulator that replaces expensive
    Boltzmann calculations. It predicts matter power spectra P(k,z) from 
    cosmological parameters. By default, both linear and nonlinear P(k) are 
    computed using halofit+ corrections.
    
    The emulator internally expects parameters in the format:
        [As_1e9, ns, H0, omegab, omegam, w0, wa]
    
    And outputs:
        - k: array of k-modes in h/Mpc, shape (2400,), range [10^-5, 10^2]
        - z: array of redshifts, shape (122,), range [0, 50]
        - Pk: power spectrum in (Mpc/h)^3, shape (122, 2400)
    
    Supports LCDM, wCDM (constant w), and w0waCDM cosmologies.

    Parameter aliasing:
        - 'w' is automatically treated as 'w0' with wa=0 (wCDM)
        - If neither w0 nor w are provided, defaults to w0=-1, wa=0 (LCDM)
    
    Nonlinear corrections:
        Set 'nonlinear_method' in extra_args to choose method:
        - 'syrenhalofit': Uses SYREN-halofit with ML corrections
        - 'halofit+': Uses halofit+ without ML corrections (DEFAULT)
        - None: Linear P(k) only
        
        When nonlinear_method is configured (including default), BOTH linear and 
        nonlinear P(k) are computed. Use the 'nonlinear' flag in get_Pk_grid() and 
        get_Pk_interpolator() to select which to retrieve:
            - get_Pk_grid(nonlinear=False) -> returns linear P(k)
            - get_Pk_grid(nonlinear=True) -> returns nonlinear P(k)
        
        The nonlinear boost B(k,z) = P_nl/P_lin is computed using symbolic_pofk,
        then applied to the emulated linear spectrum: P_nl,emul = P_lin,emul x B
    
    Attributes:
        renames: Mapping for parameter name translations
        extra_args: Configuration dictionary
            - param_order: List of parameter names (see initialize())
            - nonlinear_method: Nonlinear correction method (default: 'halofit+')
            - use_syren: If True, bypass emulator and use symbolic approx only (default: False)
        
        # Runtime attributes
        req: Dictionary of required parameters
        param_order: List of parameter names in the order expected by emulator
        nonlinear_method: Nonlinear correction method
        use_syren: Flag controlling emulator mode
    """
    
    # Class-level attributes required by Cobaya
    renames: Mapping[str, str]
    extra_args: InfoDict
    path: str

    def initialize(self):
        """
        Initialize the emulator theory code.
        
        This method is called by Cobaya during setup. Since the emulmps module
        handles all loading internally, initialization is minimal - we only need
        to specify which parameters to use and in what order.
        
        The method expects extra_args to contain:
            - 'param_order': List of parameter names in order
              Default: ["As_1e9", "ns", "H0", "omegab", "omegam", "w0", "wa"]
            - 'nonlinear_method': Nonlinear method (default: 'halofit+')
            - 'use_syren': If True, bypass emulator corrections and use only 
              symbolic approximation (default: False)
        
        Cosmology options:
            - LCDM: ["As_1e9", "ns", "H0", "omegab", "omegam"]
            - wCDM: ["As_1e9", "ns", "H0", "omegab", "omegam", "w"]
            - w0waCDM: ["As_1e9", "ns", "H0", "omegab", "omegam", "w0", "wa"]
        """
        super().initialize()

        # Set defaults
        self.renames = empty_dict
        self.extra_args = getattr(self, 'extra_args', {})
        
        # Get use_syren flag (default: True, meaning use only symbolic approximation)
        # When False, uses emulator corrections
        self.use_syren = self.extra_args.get('use_syren', True)
        
        # Get nonlinear correction method
        # Default is 'halofit+' (from VM)
        self.nonlinear_method = self.extra_args.get('nonlinear_method', None)
        
        # Validate nonlinear method
        if self.nonlinear_method is not None:
            if not SYMBOLIC_POFK_AVAILABLE:
                raise LoggedError(
                    self.log,
                    "nonlinear_method specified but symbolic_pofk is not installed. "
                    "Install with: pip install symbolic_pofk"
                )
            valid_methods = ['syrenhalofit', 'halofit+']
            if self.nonlinear_method not in valid_methods:
                raise LoggedError(
                    self.log,
                    f"Invalid nonlinear_method '{self.nonlinear_method}'. "
                    f"Valid options: {valid_methods}"
                )
        else:
            self.nonlinear_method = 'halofit+' # VM set a default
        
        # Get parameter ordering from config, or use default w0waCDM ordering
        self.param_order = self.extra_args.get(
            'param_order',
            ["As_1e9", "ns", "H0", "omegab", "omegam", "w0", "wa"]
        )
        
        # Validate parameter ordering
        valid_params = {"As_1e9", "ns", "H0", "omegab", "omegam", "w0", "wa", "w"}
        for param in self.param_order:
            if param not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{param}' in param_order. "
                    f"Valid parameters: {valid_params}"
                )
        
        # Check that we have the required minimum parameters
        required_base = {"As_1e9", "ns", "H0", "omegab", "omegam"}
        provided = set(self.param_order)
        if not required_base.issubset(provided):
            missing = required_base - provided
            raise ValueError(
                f"Missing required parameters: {missing}. "
                f"At minimum, param_order must include {required_base}"
            )
        
        # Check for conflicting dark energy parameters
        if "w" in self.param_order and "w0" in self.param_order:
            raise ValueError(
                "Cannot specify both 'w' and 'w0' in param_order. "
                "Use 'w' for wCDM (constant w) or 'w0'+'wa' for w0waCDM."
            )
        
        # Build requirements dictionary
        self.req = {param: None for param in self.param_order}
        
        # Log initialization with cosmology info
        # Note: 'w' is treated as 'w0' internally, so check for wa presence
        if ("w0" in self.param_order or "w" in self.param_order) and "wa" in self.param_order:
            cosmo_type = "w0waCDM"
        elif "w" in self.param_order:
            cosmo_type = "wCDM (constant w)"
        elif "w0" in self.param_order:
            cosmo_type = "wCDM (w0 only, wa=0)"
        else:
            cosmo_type = "LCDM"
        
        self.log.info(
            f"emulmps emulator initialized with {cosmo_type} cosmology"
        )
        self.log.info(f"Parameter order: {self.param_order}")
        
        # Log emulator mode
        if self.use_syren:
            self.log.info("Emulator mode: SYMBOLIC ONLY (use_syren=True, bypassing emulated corrections)")
        else:
            self.log.info("Emulator mode: FULL EMULATOR (using emulated corrections)")

    def get_requirements(self):
        """
        Return the parameters required by this theory code.
        
        Returns:
            dict: Dictionary with parameter names as keys
        """
        return self.req

    def calculate(self, state, want_derived=True, **params):
        """
        Calculate the emulated power spectrum and store in state.
        
        This is the main calculation method called by Cobaya during sampling.
        It extracts the required parameters, converts to emulator format,
        calls the emulmps emulator, and stores the result.
        
        The emulator returns P(k,z) in units of h/Mpc for k and (Mpc/h)^3 for Pk.
        We convert to standard Cobaya units: 1/Mpc for k and Mpc^3 for Pk.
        
        By default, both linear and nonlinear P(k) are computed and stored.
        
        Parameter handling:
            - 'w' is treated as 'w0' with wa=0
            - If neither w0/wa nor w are provided, defaults to w0=-1, wa=0 (LCDM)
        
        Args:
            state: Dictionary where results should be stored
            want_derived: Whether to compute derived parameters
            **params: Dictionary of all parameter values for this sample point
        
        Returns:
            bool: True if calculation succeeded, False otherwise
        """
        try:
            # Extract parameter values and convert to emulator format
            # The emulator expects: [As_1e9, ns, H0, omegab, omegam, w0, wa]
            emul_params = []
            
            for p in self.param_order:
                if p == "w":
                    # wCDM: constant w -> treat as w0
                    emul_params.append(params[p])
                else:
                    # All other parameters (As_1e9, ns, H0, w0, wa)
                    emul_params.append(params[p])
            
            # Now handle dark energy parameters for emulator
            # Emulator always expects 7 params: [As_1e9, ns, H0, omegab, omegam, w0, wa]
            
            has_w0 = "w0" in self.param_order
            has_wa = "wa" in self.param_order
            has_w = "w" in self.param_order
            
            if has_w:
                # User provided 'w' -> append wa=0 for wCDM
                if not has_wa:
                    emul_params.append(0.0)
                self.log.debug(f"wCDM mode: using w={emul_params[-2]:.4f}, wa=0.0")
                
            elif has_w0 and not has_wa:
                # User provided only w0 -> append wa=0
                emul_params.append(0.0)
                self.log.debug(f"wCDM mode: using w0={emul_params[-2]:.4f}, wa=0.0")
                
            elif has_w0 and has_wa:
                # User provided both w0 and wa -> already in emul_params
                self.log.debug(
                    f"w0waCDM mode: using w0={emul_params[-2]:.4f}, "
                    f"wa={emul_params[-1]:.4f}"
                )
                
            else:
                # Neither provided -> LCDM defaults
                emul_params.append(-1.0)  # w0 = -1
                emul_params.append(0.0)   # wa = 0
                self.log.debug("LCDM mode: using w0=-1.0, wa=0.0")
            
            # Call the emulmps emulator
            # Returns: k_modes (h/Mpc), z_modes, Pk_linear ((Mpc/h)^3)
            # Pass use_syren flag to control whether to apply emulator corrections
            k_mpc, z_array, Pk_lin_mpc = get_pks(emul_params, use_syren=self.use_syren)
            
            # Extract h from H0
            h = params['H0'] / 100.0

            # ===================================================================
            # NONLINEAR CORRECTIONS (computed by default)
            # ===================================================================
            # Compute nonlinear P(k) using boost method
            Pk_nl_mpc = None
            if self.nonlinear_method is not None:
                Pk_nl_mpc = self._apply_nonlinear_boost(
                    Pk_lin_mpc*h**3, k_mpc/h, z_array, params
                )

           # Store NONLINEAR P(k) if computed
            if Pk_nl_mpc is not None:
                state[("Pk_grid", True, "delta_tot", "delta_tot")] = (
                    k_mpc, z_array, Pk_nl_mpc
                )

            # Store LINEAR P(k) in state dictionary with key matching Cobaya convention
            # Key format: ("Pk_grid", nonlinear, var_pair_sorted)
            state[("Pk_grid", False, "delta_tot", "delta_tot")] = (
                k_mpc, z_array, Pk_lin_mpc
            )
            # Also store in simple format for backward compatibility (linear)
            state["Pk_grid"] = {
                'k': k_mpc,          # Shape: (nk,), in 1/Mpc
                'z': z_array,        # Shape: (nz,)
                'Pk': Pk_lin_mpc     # Shape: (nz, nk), in Mpc^3 (LINEAR)
            }
            
            # Optionally compute and store derived parameters
            if want_derived:
                derived = {}
                
                # Compute sigma8 at z=0 if requested (using linear P(k))
                if 'sigma8' in self.output_params:
                    derived['sigma8'] = self._compute_sigma8(
                        Pk_lin_hmpc, k_hmpc, z_array, z=0.0
                    )
                state["derived"] = derived
            
            # Return True to indicate successful calculation
            return True
            
        except Exception as e:
            if self.stop_at_error:
                self.log.error(
                    f"emulmps emulator evaluation failed: {e}\n"
                    f"Parameters: {params}"
                )
                raise
            else:
                self.log.debug(
                    f"emulmps emulator evaluation failed: {e}\n"
                    f"Returning False (likelihood=0)."
                )
                return False

    def get_Pk_grid(self, var_pair=("delta_tot", "delta_tot"), nonlinear=False):
        r"""
        Get matter power spectrum grid, e.g. suitable for splining.
        
        Returns P(k,z) in standard Cobaya units: k in 1/Mpc, Pk in Mpc^3.
        The arrays z and k are in ascending order.
        
        Args:
            var_pair: which power spectrum (only delta_tot supported)
            nonlinear: if True, return nonlinear P(k); if False, return linear P(k)
        
        Returns:
            Tuple of (k, z, Pk) where k and z are 1-d arrays,
            and Pk[i,j] is P(z[i], k[j]) in units of Mpc^3
        """
        # Validate inputs
        if var_pair != ("delta_tot", "delta_tot"):
            raise LoggedError(
                self.log,
                f"emulmps only supports delta_tot power spectra, not {var_pair}"
            )
        
        # Check if nonlinear requested but nonlinear_method explicitly disabled
        if nonlinear and self.nonlinear_method is None:
            raise LoggedError(
                self.log,
                "Nonlinear P(k) requested but nonlinear_method=None. "
                "Set nonlinear_method='syrenhalofit' or 'halofit+' in extra_args."
            )
        
        # Try to get from state with standard key format
        key = ("Pk_grid", nonlinear) + tuple(sorted(var_pair))
        if key in self.current_state:
            return self.current_state[key]
        
        # Fallback: get from simple format (always linear)
        if not nonlinear and "Pk_grid" in self.current_state:
            pk_dict = self.current_state["Pk_grid"]
            return pk_dict['k'], pk_dict['z'], pk_dict['Pk']
        
        raise LoggedError(
            self.log,
            f"Matter power spectrum (nonlinear={nonlinear}) not computed. "
            f"This should not happen."
        )

    def get_Pk_interpolator(
        self,
        var_pair=("delta_tot", "delta_tot"),
        nonlinear=False,
        extrap_kmin=None,
        extrap_kmax=None
    ) -> PowerSpectrumInterpolator:
        r"""
        Get a P(z,k) bicubic interpolation object (PowerSpectrumInterpolator).
        
        The interpolator works in standard Cobaya units: k in 1/Mpc, Pk in Mpc^3.
        
        Args:
            var_pair: variable pair for power spectrum (only delta_tot supported)
            nonlinear: if True, return nonlinear interpolator; if False, return linear
            extrap_kmin: use log-linear extrapolation from extrap_kmin up to min k
            extrap_kmax: use log-linear extrapolation beyond max k up to extrap_kmax
        
        Returns:
            PowerSpectrumInterpolator instance with methods:
                - P(z, k): get power spectrum at (z, k)
                - logP(z, k): get log power spectrum at (z, k)
        """
        # Validate inputs
        if var_pair != ("delta_tot", "delta_tot"):
            raise LoggedError(
                self.log,
                f"emulmps only supports delta_tot power spectra, not {var_pair}"
            )
        
        # Check if nonlinear requested but nonlinear_method explicitly disabled
        if nonlinear and self.nonlinear_method is None:
            raise LoggedError(
                self.log,
                "Nonlinear P(k) requested but nonlinear_method=None. "
                "Set nonlinear_method='syrenhalofit' or 'halofit+' in extra_args."
            )
        
        # Create unique key for caching
        key = (
            ("Pk_interpolator", nonlinear, extrap_kmin, extrap_kmax) +
            tuple(sorted(var_pair))
        )
        
        # Return cached interpolator if available
        if key in self.current_state:
            return self.current_state[key]
        
        # Get the power spectrum grid (linear or nonlinear based on flag)
        k, z, pk = self.get_Pk_grid(var_pair=var_pair, nonlinear=nonlinear)
        
        # Check if we should use log interpolation
        log_p = True
        sign = 1
        
        # Handle negative values (shouldn't happen for matter PS, but be safe)
        if np.any(pk < 0):
            if np.all(pk < 0):
                sign = -1
            else:
                log_p = False
                self.log.warning(
                    "Power spectrum has both positive and negative values. "
                    "Using linear interpolation instead of log."
                )
        
        # Check if extrapolation is requested
        extrapolating = (
            (extrap_kmax and extrap_kmax > k[-1]) or
            (extrap_kmin and extrap_kmin < k[0])
        )
        
        # Prepare data for interpolator
        if log_p:
            pk_for_interp = np.log(sign * pk)
        elif extrapolating:
            raise LoggedError(
                self.log,
                f'Cannot do log extrapolation with zero-crossing Pk for {var_pair}'
            )
        else:
            pk_for_interp = pk
        
        # Create the interpolator
        result = PowerSpectrumInterpolator(
            z, k, pk_for_interp,
            logP=log_p,
            logsign=sign,
            extrap_kmin=extrap_kmin,
            extrap_kmax=extrap_kmax
        )
        
        # Cache and return
        self.current_state[key] = result
        return result

    def _apply_nonlinear_boost(self, Pk_lin_hmpc, k_hmpc, z_array, params, emulator='EH'):
        """
        Apply nonlinear corrections using symbolic_pofk boost.
        
        This method computes the boost B(k,z) = P_nl(k,z) / P_lin(k,z) using
        symbolic_pofk, then applies it to the emulated linear spectrum:
            P_nl,emul = P_lin,emul x B_symbolic
        
        Args:
            Pk_lin_hmpc: Linear power spectrum in (Mpc/h)^3, shape (n_z, n_k)
            k_hmpc: k-modes in h/Mpc, shape (n_k,)
            z_array: Redshift array, shape (n_z,)
            params: Dictionary of cosmological parameters
            
        Returns:
            np.ndarray: Nonlinear power spectrum in (Mpc/h)^3, same shape as input
        """
        # Extract parameters needed for symbolic_pofk
        As_1e9 = params['As_1e9']
        ns = params['ns']
        H0 = params['H0']
        h = H0 / 100.0
        Ob = params['omegab']
        Om = params['omegam']
        w0 = params['w']
        wa = params['wa']
        a_array = 1.0 / (1.0 + z_array)

        # Compute sigma8 from As using symbolic_pofk's conversion
        sigma8 = symbolic_linear.As_to_sigma8(As_1e9, Om, Ob, h, ns, 0.06, w0, wa)
                        
        boost = run_halofit_vec(
            k_hmpc, sigma8, Om, Ob, h, ns, a_array,
            return_boost=True,
            Plin_in=Pk_lin_hmpc,   # (nz, nk)
        )
        Pk_nl_hmpc = Pk_lin_hmpc * boost / h**3
        return Pk_nl_hmpc

    def _compute_sigma8(self, Pk_2d, k_array, z_array, z=0.0):
        """
        Compute sigma8 from power spectrum.
        
        sigma8^2 = (1/2π^2) int[ P(k) k^2 W^2(kR) dk], with R = 8 Mpc/h
        
        where W(x) = 3(sin(x) - x*cos(x))/x^3 is the spherical top-hat window.
        
        Note: This uses the emulator's native units (h/Mpc and (Mpc/h)^3).
        
        Args:
            Pk_2d: Power spectrum array, shape (n_z, n_k) in (Mpc/h)^3
            k_array: k-modes in h/Mpc
            z_array: Redshift array
            z: Redshift at which to compute sigma8 (default 0.0)
        
        Returns:
            float: sigma8 value
        """
        # Find nearest redshift
        z_idx = np.argmin(np.abs(z_array - z))
        
        # Check if we need interpolation
        if np.abs(z_array[z_idx] - z) > 0.01:
            # Interpolate if requested z is not close to a grid point
            from scipy.interpolate import interp1d
            Pk_interp = interp1d(z_array, Pk_2d, axis=0, kind='cubic')
            Pk_z = Pk_interp(z)
        else:
            # Use nearest grid point
            Pk_z = Pk_2d[z_idx, :]
        
        # Spherical top-hat window function in Fourier space
        R = 8.0  # Mpc/h
        x = k_array * R
        
        # Handle x=0 (W(0) = 1)
        W = np.zeros_like(x)
        mask = x > 1e-6
        W[mask] = 3.0 * (np.sin(x[mask]) - x[mask] * np.cos(x[mask])) / x[mask]**3
        W[~mask] = 1.0
        
        # Integrand: k^2 P(k) W^2(kR) / (2π^2)
        integrand = k_array**2 * Pk_z * W**2 / (2.0 * np.pi**2)
        
        # Integrate in log(k) space for better accuracy
        log_k = np.log(k_array)
        integrand_logk = integrand * k_array  # Jacobian: d(log k) = dk/k
        
        # Trapezoidal rule
        sigma8_squared = np.trapz(integrand_logk, log_k)
        sigma8 = np.sqrt(sigma8_squared)
        return sigma8

    def get_can_support_params(self):
        """
        Return list of parameters that this theory can provide.
        
        This tells Cobaya which derived parameters or observables this theory
        code can compute. Other components (likelihoods) can then request these.
        
        Returns:
            list: List of available products
        """
        return ['Pk_grid', 'Pk_interpolator', 'sigma8']

    def get_sigma8(self):
        """
        Retrieve sigma8 from derived parameters.
        
        Returns:
            float: sigma8 value at z=0
        """
        return self.current_state.get('derived', {}).get('sigma8')

    def get_param(self, param_name):
        """
        Get a derived parameter value.
        
        Args:
            param_name: Name of the derived parameter
        
        Returns:
            float: Parameter value
        """
        return self.current_state.get('derived', {}).get(param_name)
    
    def get_version(self):
        """Return emulator version."""
        return '1.0.0'
    
    def get_speed(self):
        """
        Return relative speed estimate.
        
        The emulmps emulator is very fast, but this number is not yet calibrated.
        """
        return 20.0