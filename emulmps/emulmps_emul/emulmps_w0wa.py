# Author: Victoria Lloyd (2025) & V. Miranda

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import warnings
warnings.filterwarnings("ignore", message=".*TF-TRT.*")

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# Clean warning format: category name + message only, no file/line noise
warnings.formatwarning = lambda msg, cat, *a, **kw: f"{cat.__name__}: {msg}\n"

import numpy as np
import joblib
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys
from . import train_utils_pk_emulator as utils
from . train_utils_pk_emulator import CustomActivationLayer, TComponentScaler
import tensorflow as tf

class _AliasLoader:
    """Redirect any train_utils_pk_emulator_vN import to the canonical module."""
    def find_module(self, name, path=None):
        if name.startswith('train_utils_pk_emulator'):
            return self
    
    def load_module(self, name):
        if name not in sys.modules:
            sys.modules[name] = utils
        return sys.modules[name]
    
sys.meta_path.append(_AliasLoader())

# --- Custom warning class ---
class EmulatorWarning(UserWarning):
    """Warnings emitted by the emulmps PkEmulator."""
    pass


def _warn(msg: str) -> None:
    """Emit a clean emulator warning that bypasses cobaya's logger."""
    warnings.warn(f"[emulmps] {msg}", EmulatorWarning, stacklevel=3)


# --- Path Management ---
def _get_project_root() -> Path:
    return Path(__file__).resolve().parent


ROOT = _get_project_root()

try:
    from tensorflow import keras
    import sys; sys.path.insert(0, f"{ROOT}/symbolic_pofk")
    from symbolic_pofk.linear import plin_emulated, get_approximate_D, growth_correction_R, As_to_sigma8
    from symbolic_pofk.syrenhalofit import run_halofit_vec
    _DEPENDENCIES_LOADED = True
except ImportError as e:
    _warn(f"A required dependency could not be imported: {e.name}. "
          f"If running locally, ensure symbolic_pofk is accessible.")
    _DEPENDENCIES_LOADED = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError:
    pass


# --- CORE EMULATOR CLASS ---
class PkEmulator:
    """
    Cosmology Emulator for the Matter Power Spectrum P(k, z).
    Encapsulates models and handles I/O robustly.

    Input convention
    ----------------
    All public methods expect params = [10^9 A_s, ns, H0, Ob, Om, w0, wa].
    The scaler conversion (wa -> w0+wa) is handled internally before calling
    param_scaler.transform(), matching the training convention (legacy version).

    Boost model
    -----------
    An optional second neural network can be loaded to predict the nonlinear
    boost B(k,z) = P_nonlin(k,z) / P_lin(k,z).  The boost network was trained
    on log(P_nonlin / P_lin) and so exp(network output) is the boost directly.

    Use get_boost() to obtain the boost array, optionally multiplied by a
    supplied P_lin to yield P_nonlin.  If no boost model is loaded,
    get_boost() raises RuntimeError.
    """

    N_PCS     = 25
    N_K_MODES = 500

    K_MODES = np.logspace(-5.1, 2, N_K_MODES)
    Z_MODES = np.concatenate((
        np.linspace(0, 3,  33, endpoint=False),
        np.linspace(3, 10,  7, endpoint=False),
        np.linspace(10, 50, 12)
    ))
    N_ZS = len(Z_MODES)

    def __init__(self,
                 model_file: str = None,
                 metadata_file: str = None,
                 model_type: str = "mlp",
                 base_model_path: str = "models",
                 metadata_path: str = "metadata_w0wacdm",
                 num_batches: int = 10,
                 nl_model_file: str = None,
                 nl_metadata_file: str = None,
                 nl_model_type: str = "mlp"):
        """
        Parameters
        ----------
        model_file : str, optional
            Path to the linear Keras model (.h5 or .keras).
        metadata_file : str, optional
            Path to the linear metadata bundle (.joblib).
        model_type : str
            Architecture of the linear model: 'mlp' or 'npce'.
        base_model_path : str
            Directory containing model files (used when model_file is None).
        metadata_path : str
            Directory containing metadata (used when metadata_file is None).
        num_batches : int
            Number of training batches used to locate default metadata files.
        nl_model_file : str, optional
            Path to the nonlinear boost Keras model.  If None, get_boost() will
            raise RuntimeError unless a fallback is configured at the call site.
        nl_metadata_file : str, optional
            Path to the nonlinear boost metadata bundle.
        nl_model_type : str
            Architecture of the nonlinear boost model: 'mlp' or 'npce'.
        """
        if not _DEPENDENCIES_LOADED:
            raise RuntimeError("Cannot initialize PkEmulator due to missing dependencies.")

        self.MODEL_DIR    = ROOT / base_model_path
        self.METADATA_DIR = ROOT / metadata_path
        self.NUM_BATCHES  = num_batches

        resolved_model_path    = Path(model_file)    if model_file    is not None else self.MODEL_DIR / "emulator_w0wacdm.h5"
        resolved_metadata_path = Path(metadata_file) if metadata_file is not None else self.METADATA_DIR / "metadata.joblib"

        try:
            # -----------------------------------------------------------------
            # Load linear model + metadata
            # -----------------------------------------------------------------
            _bundle           = joblib.load(resolved_metadata_path)
            self.param_scaler = _bundle["param_scaler"]
            self.t_comp_pca   = _bundle["t_comp_pca"]
            self.model_type   = model_type

            self._params_buf = np.empty((1, 7), dtype=np.float32)
            self._a_array    = (1.0 / (self.Z_MODES + 1)).astype(np.float32)

            # Infer model type for dummy evaluation, if not directly supplied in metadata
            if model_file is not None:
                bundle_model_type = _bundle.get("model_type", None)
                if bundle_model_type is not None:
                    self.model_type = bundle_model_type
                else:
                    try:
                        n_inputs = self.model.input_shape[-1]
                        self.model_type = "mlp" if n_inputs == 7 else "npce"
                        _warn(f"'model_type' not found in bundle. "
                              f"Inferred '{self.model_type}' from model input shape.")
                    except Exception:
                        _warn("Could not infer model_type from model input shape. "
                              "Defaulting to 'mlp'.")

            self.t_comp_scaler = _bundle.get("t_comp_scaler", None)
            if self.t_comp_scaler is None:
                _warn("t_comp_scaler not in bundle — assuming raw t-components.")

            self._pce_indices = _bundle.get("npce_indices", None)
            if model_type == "npce" and self._pce_indices is None:
                raise ValueError(
                    "model_type='npce' but bundle['npce_indices'] is None. "
                    "Please (re)package your metadata."
                )

            self._metadata_bundle = _bundle

            self.model = keras.models.load_model(
                resolved_model_path,
                custom_objects={"CustomActivationLayer": CustomActivationLayer},
                compile=False,
            )

            @tf.function(jit_compile=False)
            def compiled_inference(x):
                return self.model(x, training=False)
            self._compiled_inference = compiled_inference

            self.PCAS:    Dict[float, PCA]            = {}
            self.SCALERS: Dict[float, StandardScaler] = {}
            self._pcas_loaded               = False
            self.INVERSE_TRANSFORM_MATRICES = None
            self.INVERSE_TRANSFORM_OFFSETS  = None
            self._load_pcas_and_scalers()

            # -----------------------------------------------------------------
            # Load nonlinear boost model + metadata (optional)
            # -----------------------------------------------------------------
            self._nl_model          = None
            self._nl_compiled       = None
            self._nl_param_scaler   = None
            self._nl_t_comp_pca     = None
            self._nl_t_comp_scaler  = None
            self._nl_pce_indices    = None
            self._nl_model_type     = None
            self._nl_n_pcs          = None  # learned from NL metadata
            self.NL_INVERSE_TRANSFORM_MATRICES = None
            self.NL_INVERSE_TRANSFORM_OFFSETS  = None

            if nl_model_file is not None and nl_metadata_file is not None:
                self._load_nl_model(nl_model_file, nl_metadata_file, nl_model_type)

            # -----------------------------------------------------------------
            # Warm up linear model
            # -----------------------------------------------------------------
            dummy_params = self.param_scaler.transform(
                np.array([[2.0, 0.96, 67.0, 0.05, 0.3, -1.0, 0.0]], dtype=np.float32))
            dummy_input = self._make_network_input(dummy_params)
            dummy_tf    = tf.constant(dummy_input, dtype=tf.float32)
            _ = self._compiled_inference(dummy_tf)
            _ = self._compiled_inference(dummy_tf)

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"[emulmps] Required model or metadata file not found: {e.filename}\n"
                "Ensure 'models' and 'metadata' directories are correctly placed."
            ) from e

    # -------------------------------------------------------------------------
    # Nonlinear model loading
    # -------------------------------------------------------------------------

    def _load_nl_model(
        self,
        nl_model_file: str,
        nl_metadata_file: str,
        nl_model_type: str,
    ) -> None:
        """
        Load the nonlinear boost model and its metadata bundle.

        The bundle is expected to have the same structure as the linear one
        (param_scaler, t_comp_pca, t_comp_scaler, pcas, scalers, optionally
        npce_indices).  The boost network was trained on log(P_nonlin/P_lin),
        so exp(network output) is the boost directly.

        This method populates:
            self._nl_model, self._nl_compiled
            self._nl_param_scaler, self._nl_t_comp_pca, self._nl_t_comp_scaler
            self._nl_pce_indices, self._nl_n_pcs
            self.NL_INVERSE_TRANSFORM_MATRICES, self.NL_INVERSE_TRANSFORM_OFFSETS
        """
        nl_model_path    = Path(nl_model_file)
        nl_metadata_path = Path(nl_metadata_file)

        nl_bundle = joblib.load(nl_metadata_path)

        if nl_model_file is not None:
            nl_bundle_model_type = nl_bundle.get("model_type", None)
            if nl_bundle_model_type is not None:
                self.nl_model_type = nl_bundle_model_type
            else:
                try:
                    n_inputs = self.model.input_shape[-1]
                    self.nl_model_type = "mlp" if n_inputs == 7 else "npce"
                    _warn(f"'model_type' not found in nonlinear bundle. "
                            f"Inferred '{self.nl_model_type}' from model input shape.")
                except Exception:
                    _warn("Could not infer model_type from nonlinear model input shape. "
                            "Defaulting to 'mlp'.")

        self._nl_param_scaler  = nl_bundle["param_scaler"]
        self._nl_t_comp_pca    = nl_bundle["t_comp_pca"]
        self._nl_t_comp_scaler = nl_bundle.get("t_comp_scaler", None)
        if self._nl_t_comp_scaler is None:
            _warn("nl_metadata: t_comp_scaler not found — assuming raw t-components.")

        self._nl_pce_indices = nl_bundle.get("npce_indices", None)
        if self.nl_model_type == "npce" and self._nl_pce_indices is None:
            raise ValueError(
                "nl_model_type='npce' but nl_bundle['npce_indices'] is None."
            )

        self._nl_model = keras.models.load_model(
            nl_model_path,
            custom_objects={"CustomActivationLayer": CustomActivationLayer},
            compile=False,
        )

        @tf.function(jit_compile=False)
        def nl_compiled_inference(x):
            return self._nl_model(x, training=False)
        self._nl_compiled = nl_compiled_inference

        # Pre-compute fused inverse-transform matrices for the NL PCA/scalers.
        # Each per-z PCA reconstructs log(boost) on the full k grid.
        nl_pcas    = nl_bundle["pcas"]
        nl_scalers = nl_bundle["scalers"]

        # Infer N_PCS from the first stored PCA object
        first_z_key      = next(iter(nl_pcas))
        self._nl_n_pcs   = nl_pcas[first_z_key].n_components_

        inverse_matrices, inverse_offsets = [], []
        for z in self.Z_MODES:
            z_key  = float(f"{z:.3f}")
            pca    = nl_pcas[z_key]
            scaler = nl_scalers[z_key]

            if hasattr(scaler, "scale_"):
                scale, mean = scaler.scale_, scaler.mean_
            elif hasattr(scaler, "std"):
                scale, mean = scaler.std, scaler.mean
            else:
                raise AttributeError(
                    f"NL scaler for z={z:.3f} has neither 'scale_' nor 'std'."
                )

            inverse_matrices.append(pca.components_ * scale[None, :])
            inverse_offsets.append(pca.mean_ * scale + mean)

        self.NL_INVERSE_TRANSFORM_MATRICES = np.stack(inverse_matrices, axis=0).astype(np.float32)
        self.NL_INVERSE_TRANSFORM_OFFSETS  = np.stack(inverse_offsets,  axis=0).astype(np.float32)

        # Warm up
        dummy_norm = self._nl_param_scaler.transform(
            np.array([[2.0, 0.96, 67.0, 0.05, 0.3, -1.0, -1.0]], dtype=np.float32)
        )
        dummy_input = self._make_nl_network_input(dummy_norm)
        dummy_tf    = tf.constant(dummy_input, dtype=tf.float32)
        _ = self._nl_compiled(dummy_tf)
        _ = self._nl_compiled(dummy_tf)

        _warn(f"Nonlinear boost model loaded from {nl_model_path}")

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    def _load_pcas_and_scalers(self):
        if self._pcas_loaded:
            return

        try:
            self.PCAS    = self._metadata_bundle["pcas"]
            self.SCALERS = self._metadata_bundle["scalers"]

            inverse_matrices, inverse_offsets = [], []
            for z in self.Z_MODES:
                z_key  = float(f"{z:.3f}")
                pca    = self.PCAS[z_key]
                scaler = self.SCALERS[z_key]

                if hasattr(scaler, 'scale_'):
                    scale, mean = scaler.scale_, scaler.mean_
                elif hasattr(scaler, 'std'):
                    scale, mean = scaler.std, scaler.mean
                else:
                    raise AttributeError(
                        f"Scaler for z={z:.3f} has neither 'scale_' nor 'std'."
                    )

                inverse_matrices.append(pca.components_ * scale[None, :])
                inverse_offsets.append(pca.mean_ * scale + mean)

            self.INVERSE_TRANSFORM_MATRICES = np.stack(inverse_matrices, axis=0).astype(np.float32)
            self.INVERSE_TRANSFORM_OFFSETS  = np.stack(inverse_offsets,  axis=0).astype(np.float32)
            self._pcas_loaded = True

        except (KeyError, FileNotFoundError) as e:
            raise RuntimeError(
                f"[emulmps] Could not load metadata from bundle: {e}"
            ) from e

    def _evaluate_pce_basis(self, X_norm: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Evaluate a PCE multi-index basis at X_norm."""
        X_clipped = np.clip(X_norm, -1.0, 1.0).astype(np.float64)
        N, D      = X_clipped.shape
        max_deg   = int(indices.max()) if indices.size > 0 else 0

        pow_table = np.empty((D, max_deg + 1, N), dtype=np.float64)
        pow_table[:, 0, :] = 1.0
        if max_deg >= 1:
            pow_table[:, 1, :] = X_clipped.T
        for p in range(2, max_deg + 1):
            pow_table[:, p, :] = pow_table[:, p - 1, :] * X_clipped.T

        d_idx   = np.arange(D)[np.newaxis, :]
        powered = pow_table[d_idx, indices, :]
        return powered.prod(axis=1).T.astype(np.float32)

    def _make_network_input(self, params_norm: np.ndarray) -> np.ndarray:
        """Build linear-model network input (MLP: pass-through; NPCE: PCE basis)."""
        if self.model_type == "mlp":
            return params_norm.astype(np.float32)
        return self._evaluate_pce_basis(params_norm, self._pce_indices)

    def _make_nl_network_input(self, params_norm: np.ndarray) -> np.ndarray:
        """Build NL-model network input."""
        if self._nl_model_type == "mlp":
            return params_norm.astype(np.float32)
        return self._evaluate_pce_basis(params_norm, self._nl_pce_indices)

    def _norm_params_for_scaler(
        self,
        params_wa: np.ndarray,
        scaler,
    ) -> np.ndarray:
        """
        Convert [As, ns, H0, Ob, Om, w0, wa] -> [As, ns, H0, Ob, Om, w0, w0+wa]
        then apply scaler.transform().

        Both the linear and nonlinear models were trained with w0+wa stored in
        column 6, so we do this conversion before every scaler call.
        """
        buf = params_wa.copy().reshape(1, -1)
        buf[0, 6] = buf[0, 5] + buf[0, 6]   # wa -> w0+wa
        return scaler.transform(buf)

    # -------------------------------------------------------------------------
    # Linear P(k) inference
    # -------------------------------------------------------------------------

    def _compute_mps_approximation(self, params: np.ndarray) -> np.ndarray:
        """Symbolic P_lin(k, z) approximation (syren). params has wa in col 6."""
        As, ns, H0_in, Ob, Om, w0, wa = params
        h   = float(H0_in) / 100.0
        k_h = self.K_MODES / h

        pk_fid = plin_emulated(k_h, Om, Ob, h, ns, As=As, w0=w0, wa=wa)

        kref = 1e-4
        D0 = get_approximate_D(k=kref, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
        Dz = get_approximate_D(k=kref, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=self._a_array)
        R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
        Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=self._a_array)

        growth = (Dz / D0) ** 2 * (Rz / R0)
        return ((pk_fid * growth[:, None]) / h**3).astype(np.float32)

    def _compute_boost_approximation(self, Pk_lin, params: np.ndarray) -> np.ndarray:
        """Symbolic nonlinear boost fallback via symbolic_pofk."""
        As, ns, H0_in, Ob, Om, w0, wa = params
        h   = float(H0_in) / 100.0
        k_h = self.K_MODES / h

        sigma8 = As_to_sigma8(As, Om, Ob, h, ns, 0.06, w0, wa)

        boost = run_halofit_vec(
            k_h, sigma8, Om, Ob, h, ns, self._a_array,
            return_boost=True,
            Plin_in=Pk_lin * h**3,
        )
        return boost

    def _predict_fracs_all_z(self, params_norm: np.ndarray) -> np.ndarray:
        """Linear model: normalised params -> log-frac for all z."""
        net_input    = self._make_network_input(params_norm)
        t_comps_norm = self._compiled_inference(
            tf.constant(net_input, dtype=tf.float32)).numpy()

        if self.t_comp_scaler is not None:
            t_comps_raw = self.t_comp_scaler.inverse_transform(t_comps_norm)
        else:
            t_comps_raw = t_comps_norm

        pcs_z = self.t_comp_pca.inverse_transform(t_comps_raw).reshape(self.N_ZS, self.N_PCS)

        return (np.einsum('zp,zpk->zk',
                          pcs_z.astype(np.float32),
                          self.INVERSE_TRANSFORM_MATRICES,
                          optimize=True)
                + self.INVERSE_TRANSFORM_OFFSETS)

    def get_pks(
        self,
        params: List[float],
        use_syren: bool = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return linear P(k, z) for a single cosmology.

        Parameters
        ----------
        params : list of 7 floats
            [10^9 A_s, ns, H0, Ob, Om, w0, wa]  — wa in column 6, not w0+wa.
        use_syren : bool, optional
            If True, skip the NN and return only the symbolic approximation.

        Returns
        -------
        k_modes : (N_K_MODES,)
        z_modes : (N_ZS,)
        pks     : (N_ZS, N_K_MODES)  — linear P(k,z) in Mpc³
        """
        buf = self._params_buf
        buf[0, :] = params   # wa stays in col 6 for the approximation

        pk_mps = self._compute_mps_approximation(buf[0])

        if use_syren is True:
            return self.K_MODES, self.Z_MODES, pk_mps

        # Convert wa -> w0+wa before the scaler (matches training convention)
        x_norm = self._norm_params_for_scaler(buf, self.param_scaler)

        pks = (np.exp(self._predict_fracs_all_z(x_norm)) * pk_mps).astype(np.float32)
        return self.K_MODES, self.Z_MODES, pks

    # -------------------------------------------------------------------------
    # Nonlinear boost inference
    # -------------------------------------------------------------------------

    def _predict_nl_fracs_all_z(self, params_norm: np.ndarray) -> np.ndarray:
        """
        NL boost model: normalised params -> log(P_nonlin/P_lin) for all z.

        The inference chain is identical to the linear model; only the network,
        t-component PCA/scaler, and per-z PCA/scaler objects differ.
        """
        net_input    = self._make_nl_network_input(params_norm)
        t_comps_norm = self._nl_compiled(
            tf.constant(net_input, dtype=tf.float32)).numpy()

        if self._nl_t_comp_scaler is not None:
            t_comps_raw = self._nl_t_comp_scaler.inverse_transform(t_comps_norm)
        else:
            t_comps_raw = t_comps_norm

        pcs_z = self._nl_t_comp_pca.inverse_transform(t_comps_raw).reshape(
            self.N_ZS, self._nl_n_pcs
        )

        return (np.einsum('zp,zpk->zk',
                          pcs_z.astype(np.float32),
                          self.NL_INVERSE_TRANSFORM_MATRICES,
                          optimize=True)
                + self.NL_INVERSE_TRANSFORM_OFFSETS)

    def get_boost(
        self,
        params: List[float],
        pk_lin: Optional[np.ndarray] = None,
        use_syren: bool = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the nonlinear boost B(k,z) = P_nonlin/P_lin predicted by the
        boost neural network.

        Parameters
        ----------
        params : list of 7 floats
            [10^9 A_s, ns, H0, Ob, Om, w0, wa]
        pk_lin : (N_ZS, N_K_MODES) array, optional
            If supplied, the returned ``pks`` array is B * pk_lin, i.e. the
            emulated nonlinear power spectrum.  If None, returns the raw boost.

        Returns
        -------
        k_modes : (N_K_MODES,)
        z_modes : (N_ZS,)
        result  : (N_ZS, N_K_MODES)
            Boost B(k,z) when pk_lin is None, or B(k,z)*pk_lin otherwise.

        Raises
        ------
        RuntimeError
            If no nonlinear boost model was loaded (nl_model_file=None at init).
        """
        if self._nl_model is None:
            raise RuntimeError(
                "No nonlinear boost model loaded. "
                "Pass nl_model_file and nl_metadata_file to PkEmulator.__init__()."
            )

        buf = self._params_buf
        buf[0, :] = params

        syren_boost = self._compute_boost_approximation(pk_lin, params)

        if use_syren is True:
            return self.K_MODES, self.Z_MODES, syren_boost

        # Convert wa -> w0+wa before the NL scaler (same training convention)
        x_norm = self._norm_params_for_scaler(buf, self._nl_param_scaler)

        boost = (np.exp(self._predict_nl_fracs_all_z(x_norm)) * syren_boost).astype(np.float32)

        k_t = 0.005  # [1/Mpc]
        n   = 2.0   # 1 = pure exponential, larger = sharper transition
        self._lin_to_nl_weight = (
            1.0 - np.exp(-(self.K_MODES / k_t)**n)
        ).astype(np.float32)

        self._k_lin_mask = self.K_MODES < k_t
        boost = 1.0 + (boost - 1.0) * self._lin_to_nl_weight

        return self.K_MODES, self.Z_MODES, boost

    def has_nl_model(self) -> bool:
        """Return True if a nonlinear boost model has been loaded."""
        return self._nl_model is not None

# --- Public Module-Level Interface ---

_pk_emulator_instance = None
if _DEPENDENCIES_LOADED:
    try:
        _pk_emulator_instance = PkEmulator()
    except Exception as e:
        _warn(f"PkEmulator module-level instance failed during initialization: {e}")


def get_pks(params: List[float], use_syren: bool = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Module-level convenience function to get linear P(k, z).

    Parameters
    ----------
    params : list of 7 floats [10^9 A_s, ns, H0, Ob, Om, w0, wa]
    use_syren : if True, return only the symbolic approximation.

    Returns
    -------
    (k_modes, z_modes, P(k,z))
    """
    _pk_emulator_instance = None
    if _DEPENDENCIES_LOADED:
        try:
            _pk_emulator_instance = PkEmulator()
        except Exception as e:
            _warn(f"Module-level singleton not loaded "
                  f"(expected if using explicit model_file/metadata_file): {e}")