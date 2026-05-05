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
from typing import Dict, Any, List, Tuple
from pathlib import Path
import sys
from . import train_utils_pk_emulator as utils
sys.modules['train_utils_pk_emulator'] = utils
from . train_utils_pk_emulator import CustomActivationLayer, TComponentScaler
import tensorflow as tf


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
    #from colossus.cosmology import cosmology as Cosmo
    import sys; sys.path.insert(0, f"{ROOT}/symbolic_pofk")
    from symbolic_pofk.linear import plin_emulated, get_approximate_D, growth_correction_R
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
                 num_batches: int = 10):

        if not _DEPENDENCIES_LOADED:
            raise RuntimeError("Cannot initialize PkEmulator due to missing dependencies.")

        self.MODEL_DIR    = ROOT / base_model_path
        self.METADATA_DIR = ROOT / metadata_path
        self.NUM_BATCHES  = num_batches

        resolved_model_path    = Path(model_file)    if model_file    is not None else self.MODEL_DIR / "emulator_w0wacdm.h5"
        resolved_metadata_path = Path(metadata_file) if metadata_file is not None else self.METADATA_DIR / "metadata.joblib"

        try:
            _bundle           = joblib.load(resolved_metadata_path)
            self.param_scaler = _bundle["param_scaler"]
            self.t_comp_pca   = _bundle["t_comp_pca"]
            self.model_type   = model_type

            self._params_buf = np.empty((1, 7), dtype=np.float32)
            self._a_array    = (1.0 / (self.Z_MODES + 1)).astype(np.float32)

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

            # Warm up
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

    def _evaluate_pce_basis(self, X_norm: np.ndarray) -> np.ndarray:
        indices   = self._pce_indices
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
        if self.model_type == "mlp":
            return params_norm.astype(np.float32)
        return self._evaluate_pce_basis(params_norm)

    def _compute_mps_approximation(self, params: np.ndarray) -> np.ndarray:
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

    def _predict_fracs_all_z(self, params: np.ndarray) -> np.ndarray:
        net_input    = self._make_network_input(params)
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

    def get_pks(self, params: List[float], use_syren: bool = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        buf = self._params_buf
        buf[0, 0] = params[0]
        buf[0, 1] = params[1]
        buf[0, 2] = params[2]
        buf[0, 3] = params[3]
        buf[0, 4] = params[4]
        buf[0, 5] = params[5]
        buf[0, 6] = params[6]  # wa for mps approximation

        pk_mps = self._compute_mps_approximation(buf[0])

        if use_syren is True:
            return self.K_MODES, self.Z_MODES, pk_mps

        # Convert wa -> w0+wa for the scaler
        buf[0, 6] = params[5] + params[6]
        x_norm = self.param_scaler.transform(buf)

        pks = (np.exp(self._predict_fracs_all_z(x_norm)) * pk_mps).astype(np.float32)
        return self.K_MODES, self.Z_MODES, pks


# --- Public Module-Level Interface ---

_pk_emulator_instance = None
if _DEPENDENCIES_LOADED:
    try:
        _pk_emulator_instance = PkEmulator()
    except Exception as e:
        _warn(f"PkEmulator module-level instance failed during initialization: {e}")


def get_pks(params: List[float], use_syren: bool = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Module-level function to get P(k, z).

    Args:
        params: [10^9 A_s, ns, H0, Ob, Om, w0, wa]
        use_syren: if True, return only the symbolic approximation.

    Returns:
        (k_modes, z_modes, P(k,z))
    """
    _pk_emulator_instance = None
    if _DEPENDENCIES_LOADED:
        try:
            _pk_emulator_instance = PkEmulator()
        except Exception as e:
            _warn(f"Module-level singleton not loaded "
                  f"(expected if using explicit model_file/metadata_file): {e}")