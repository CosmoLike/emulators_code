"""
Fast Matter Power Spectrum Emulator for Cobaya

This module provides a neural network-based emulator for cosmological matter
power spectra within the Cobaya framework. It uses a local emulmps module
that handles all model loading and prediction internally.

The emulator predicts matter power spectra P(k,z) from cosmological parameters.
By default, both linear and nonlinear P(k) are computed (using halofit+).

Currently supports LCDM, wCDM, and w0waCDM cosmologies.

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
    import sys; sys.path.insert(0, f"{ROOT}/emulmps_emul/symbolic_pofk")
    from symbolic_pofk.linear import As_to_sigma8, plin_emulated
    from symbolic_pofk.syrenhalofit import run_halofit, run_halofit_vec
    class symbolic_linear:
        As_to_sigma8 = As_to_sigma8
        plin_emulated = plin_emulated
    class syrenhalofit:
        run_halofit = run_halofit
        run_halofit_vec = run_halofit_vec
    SYMBOLIC_POFK_AVAILABLE = True
except ImportError as e:
    logging.error(f"ERROR: A required dependency could not be imported.")
    logging.error(f"Missing component: {e.name if hasattr(e, 'name') else 'symbolic_pofk'}")
    SYMBOLIC_POFK_AVAILABLE = False
    symbolic_linear = None
    syrenhalofit = None


# Import the local emulmps module
try:
    from .emulmps_emul import emulmps_w0wa as emulmps_emul
    from emulmps_emul import get_pks
except ImportError:
    try:
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
        return self.k[0]

    @property
    def input_kmax(self):
        return self.k[-1]

    @property
    def kmin(self):
        if self.extrap_kmin is None:
            return self.input_kmin
        return self.extrap_kmin

    @property
    def kmax(self):
        if self.extrap_kmax is None:
            return self.input_kmax
        return self.extrap_kmax

    def check_ranges(self, z, k):
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
        self.check_ranges(z, k)
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self.logsign * np.exp(self(z, np.log(k), grid=grid, warn=False))
        else:
            return self(z, np.log(k), grid=grid, warn=False)

    def logP(self, z, k, grid=None):
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

    Nonlinear P(k) is computed using one of three strategies, tried in order:

    1. **Emulated boost** (preferred): if nl_model_file and nl_metadata_file are
       supplied in extra_args, the boost network predicts B(k,z) = P_nl/P_lin
       directly.  P_nl,emul = B_emul * P_lin,emul.

    2. **Symbolic fallback**: if no boost model is available but
       nonlinear_method is set ('syrenhalofit' or 'halofit+'), the boost is
       computed analytically via symbolic_pofk.

    3. **Linear only**: if nonlinear_method=None and no boost model is loaded,
       only linear P(k) is stored.

    Extra args
    ----------
    model_file : str
        Path to the linear Keras model.
    metadata_file : str
        Path to the linear metadata bundle.
    model_type : str
        'mlp' or 'npce' (default: 'mlp').
    nl_model_file : str, optional
        Path to the nonlinear boost Keras model.
    nl_metadata_file : str, optional
        Path to the nonlinear boost metadata bundle.
    nl_model_type : str
        'mlp' or 'npce' for the boost model (default: 'mlp').
    nonlinear_method : str or None
        Symbolic fallback method when no boost model is loaded.
        'syrenhalofit', 'halofit+', or None (default: 'halofit+').
    use_syren : bool
        If True, bypass the linear emulator and use only the symbolic
        approximation (default: False).
    param_order : list of str
        Order of cosmological parameters (default: w0waCDM ordering).
    """

    renames: Mapping[str, str]
    extra_args: InfoDict
    path: str

    def initialize(self):
        super().initialize()

        self.renames    = empty_dict
        self.extra_args = getattr(self, 'extra_args', {})

        self.use_syren         = self.extra_args.get('use_syren', True)
        self.nonlinear_method  = self.extra_args.get('nonlinear_method', 'halofit+')

        # Validate symbolic fallback method
        if self.nonlinear_method is not None:
            if not SYMBOLIC_POFK_AVAILABLE:
                raise LoggedError(
                    self.log,
                    "nonlinear_method specified but symbolic_pofk is not installed."
                )
            valid_methods = ['syrenhalofit', 'halofit+']
            if self.nonlinear_method not in valid_methods:
                raise LoggedError(
                    self.log,
                    f"Invalid nonlinear_method '{self.nonlinear_method}'. "
                    f"Valid options: {valid_methods}"
                )

        self.param_order = self.extra_args.get(
            'param_order',
            ["As_1e9", "ns", "H0", "omegab", "omegam", "w0", "wa"]
        )

        valid_params = {"As_1e9", "ns", "H0", "omegab", "omegam", "w0", "wa", "w"}
        for param in self.param_order:
            if param not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{param}' in param_order. "
                    f"Valid parameters: {valid_params}"
                )

        required_base = {"As_1e9", "ns", "H0", "omegab", "omegam"}
        provided = set(self.param_order)
        if not required_base.issubset(provided):
            missing = required_base - provided
            raise ValueError(f"Missing required parameters: {missing}.")

        if "w" in self.param_order and "w0" in self.param_order:
            raise ValueError(
                "Cannot specify both 'w' and 'w0' in param_order."
            )

        self.req = {param: None for param in self.param_order}

        model_file       = self.extra_args.get('model_file', None)
        metadata_file    = self.extra_args.get('metadata_file', None)
        model_type       = self.extra_args.get('model_type', 'mlp')
        nl_model_file    = self.extra_args.get('nl_model_file', None)
        nl_metadata_file = self.extra_args.get('nl_metadata_file', None)
        nl_model_type    = self.extra_args.get('nl_model_type', 'mlp')

        from .emulmps_emul.emulmps_w0wa import PkEmulator
        self._emulator = PkEmulator(
            model_file=model_file,
            metadata_file=metadata_file,
            model_type=model_type,
            nl_model_file=nl_model_file,
            nl_metadata_file=nl_metadata_file,
            nl_model_type=nl_model_type,
        )

        if self._emulator.has_nl_model():
            self.log.info("Nonlinear boost: emulated (boost neural network loaded).")
        elif self.nonlinear_method is not None:
            self.log.info(
                f"Nonlinear boost: symbolic fallback ({self.nonlinear_method})."
            )
        else:
            self.log.info("Nonlinear boost: disabled (linear P(k) only).")

        self.log.info(
            f"PkEmulator loaded (model_file={model_file}, "
            f"metadata_file={metadata_file})"
        )

        if ("w0" in self.param_order or "w" in self.param_order) and "wa" in self.param_order:
            cosmo_type = "w0waCDM"
        elif "w" in self.param_order:
            cosmo_type = "wCDM (constant w)"
        elif "w0" in self.param_order:
            cosmo_type = "wCDM (w0 only, wa=0)"
        else:
            cosmo_type = "LCDM"

        self.log.info(f"emulmps emulator initialized with {cosmo_type} cosmology")
        self.log.info(f"Parameter order: {self.param_order}")

    def get_requirements(self):
        return self.req

    def calculate(self, state, want_derived=True, **params):
        """
        Calculate P(k,z) and store linear (and optionally nonlinear) grids in
        state.

        Nonlinear strategy (in priority order):
          1. Emulated boost (get_boost on the loaded NL model).
          2. Symbolic fallback (via symbolic_pofk).
          3. Linear only.
        """
        try:
            emul_params = []
            for p in self.param_order:
                emul_params.append(params[p])

            has_w0 = "w0" in self.param_order
            has_wa = "wa" in self.param_order
            has_w  = "w"  in self.param_order

            if has_w:
                if not has_wa:
                    emul_params.append(0.0)
            elif has_w0 and not has_wa:
                emul_params.append(0.0)
            elif not has_w0 and not has_w:
                emul_params.extend([-1.0, 0.0])

            # ------------------------------------------------------------------
            # Linear P(k)
            # ------------------------------------------------------------------
            k_mpc, z_array, Pk_lin_mpc = self._emulator.get_pks(
                emul_params, use_syren=self.use_syren
            )

            if not np.all(np.isfinite(Pk_lin_mpc)) or np.any(~(Pk_lin_mpc > 0)):
                self.log.debug(f"Non-finite or non-positive Pk_lin at params={params} — rejecting point.")
                return False

            # ------------------------------------------------------------------
            # Nonlinear P(k)
            # ------------------------------------------------------------------
            _, _, boost = self._emulator.get_boost(emul_params, pk_lin=Pk_lin_mpc, use_syren=self.use_syren)

            if not np.all(np.isfinite(boost)) or np.any(~(boost > 0)):
                self.log.debug(
                    f"Non-finite or non-positive boost at params={params} — "
                    "rejecting point."
                )
                return False

            Pk_nl_mpc  = (boost * Pk_lin_mpc).astype(np.float32)

            if not np.all(np.isfinite(Pk_nl_mpc)) or np.any(~(Pk_nl_mpc > 0)):
                self.log.debug(
                    f"Non-finite or non-positive Pk_nl at params={params} — "
                    "rejecting point."
                )
                return False

            # ------------------------------------------------------------------
            # Store results
            # ------------------------------------------------------------------
            if Pk_nl_mpc is not None:
                state[("Pk_grid", True, "delta_tot", "delta_tot")] = (
                    k_mpc, z_array, Pk_nl_mpc
                )

            state[("Pk_grid", False, "delta_tot", "delta_tot")] = (
                k_mpc, z_array, Pk_lin_mpc
            )
            state["Pk_grid"] = {
                'k': k_mpc,
                'z': z_array,
                'Pk': Pk_lin_mpc,
            }

            if want_derived:
                derived = {}
                if 'sigma8' in self.output_params:
                    derived['sigma8'] = self._compute_sigma8(
                        Pk_lin_mpc, k_mpc, z_array, z=0.0
                    )
                state["derived"] = derived

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
                    "Returning False (likelihood=0)."
                )
                return False

    def get_Pk_grid(self, var_pair=("delta_tot", "delta_tot"), nonlinear=False):
        r"""
        Get matter power spectrum grid.

        Returns P(k,z) in standard Cobaya units: k in 1/Mpc, Pk in Mpc^3.
        """
        if var_pair != ("delta_tot", "delta_tot"):
            raise LoggedError(
                self.log,
                f"emulmps only supports delta_tot power spectra, not {var_pair}"
            )

        nl_available = (
            self._emulator.has_nl_model() or self.nonlinear_method is not None
        )
        if nonlinear and not nl_available:
            raise LoggedError(
                self.log,
                "Nonlinear P(k) requested but no boost model or nonlinear_method "
                "is configured."
            )

        key = ("Pk_grid", nonlinear) + tuple(sorted(var_pair))
        if key in self.current_state:
            return self.current_state[key]

        if not nonlinear and "Pk_grid" in self.current_state:
            pk_dict = self.current_state["Pk_grid"]
            return pk_dict['k'], pk_dict['z'], pk_dict['Pk']

        raise LoggedError(
            self.log,
            f"Matter power spectrum (nonlinear={nonlinear}) not computed."
        )

    def get_Pk_interpolator(
        self,
        var_pair=("delta_tot", "delta_tot"),
        nonlinear=False,
        extrap_kmin=None,
        extrap_kmax=None,
    ) -> PowerSpectrumInterpolator:
        r"""Get a P(z,k) bicubic interpolation object."""
        if var_pair != ("delta_tot", "delta_tot"):
            raise LoggedError(
                self.log,
                f"emulmps only supports delta_tot power spectra, not {var_pair}"
            )

        nl_available = (
            self._emulator.has_nl_model() or self.nonlinear_method is not None
        )
        if nonlinear and not nl_available:
            raise LoggedError(
                self.log,
                "Nonlinear P(k) requested but no boost model or nonlinear_method "
                "is configured."
            )

        key = (
            ("Pk_interpolator", nonlinear, extrap_kmin, extrap_kmax) +
            tuple(sorted(var_pair))
        )

        if key in self.current_state:
            return self.current_state[key]

        k, z, pk = self.get_Pk_grid(var_pair=var_pair, nonlinear=nonlinear)

        log_p = True
        sign  = 1
        if np.any(pk < 0):
            if np.all(pk < 0):
                sign = -1
            else:
                log_p = False
                self.log.debug(
                    "Power spectrum has both positive and negative values; "
                    "using linear interpolation."
                )

        extrapolating = (
            (extrap_kmax and extrap_kmax > k[-1]) or
            (extrap_kmin and extrap_kmin < k[0])
        )

        if log_p:
            pk_for_interp = np.log(sign * pk)
        elif extrapolating:
            raise LoggedError(
                self.log,
                f'Cannot do log extrapolation with zero-crossing Pk for {var_pair}'
            )
        else:
            pk_for_interp = pk

        result = PowerSpectrumInterpolator(
            z, k, pk_for_interp,
            logP=log_p,
            logsign=sign,
            extrap_kmin=extrap_kmin,
            extrap_kmax=extrap_kmax,
        )

        self.current_state[key] = result
        return result

    def _compute_sigma8(self, Pk_2d, k_array, z_array, z=0.0):
        """Compute sigma8 from linear P(k) at redshift z."""
        z_idx = np.argmin(np.abs(z_array - z))
        if np.abs(z_array[z_idx] - z) > 0.01:
            from scipy.interpolate import interp1d
            Pk_interp = interp1d(z_array, Pk_2d, axis=0, kind='cubic')
            Pk_z = Pk_interp(z)
        else:
            Pk_z = Pk_2d[z_idx, :]

        R  = 8.0
        x  = k_array * R
        W  = np.zeros_like(x)
        mask = x > 1e-6
        W[mask]  = 3.0 * (np.sin(x[mask]) - x[mask] * np.cos(x[mask])) / x[mask]**3
        W[~mask] = 1.0

        integrand = k_array**2 * Pk_z * W**2 / (2.0 * np.pi**2)
        log_k     = np.log(k_array)
        return np.sqrt(np.trapz(integrand * k_array, log_k))

    def get_can_support_params(self):
        return ['Pk_grid', 'Pk_interpolator', 'sigma8']

    def get_sigma8(self):
        return self.current_state.get('derived', {}).get('sigma8')

    def get_param(self, param_name):
        return self.current_state.get('derived', {}).get(param_name)

    def get_version(self):
        return '1.0.0'

    def get_speed(self):
        return 20.0