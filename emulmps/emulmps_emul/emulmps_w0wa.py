# Author: Victoria Lloyd (2025) & V. Miranda
import os
import numpy as np
import joblib
from typing import Dict, Any, List, Tuple
import logging
from pathlib import Path
import sys
from . import train_utils_pk_emulator as utils
sys.modules['train_utils_pk_emulator'] = utils
from . train_utils_pk_emulator import CustomActivationLayer
from keras.losses import MeanSquaredError
import time
import tensorflow as tf


# Set up logging for the module
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Path Management ---
def _get_project_root() -> Path:
    """Returns the root directory Path object, relative to this file."""
    return Path(__file__).resolve().parent

# --- Dependency Guards ---
ROOT = _get_project_root()
try:

    from tensorflow import keras
    from colossus.cosmology import cosmology as Cosmo
    # Issues installing symbolic_pofk as a package, including a local copy in the emulator
    import sys; sys.path.insert(0, f"{ROOT}/symbolic_pofk"); from symbolic_pofk.linear import plin_emulated, get_approximate_D, growth_correction_R
    
    _DEPENDENCIES_LOADED = True
except ImportError as e:
    logging.error(f"FATAL ERROR: A required dependency could not be imported. Please ensure all dependencies are installed.")
    logging.error(f"Missing component: {e.name}")
    logging.error(f"If running this package locally, ensure the symbolic_pofk library is accessible.")
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

    # --- Configuration Constants (Class Attributes) ---
    N_PCS = 25          # Number of k-space PCs per redshift
    N_K_MODES = 240     # Number of k modes
    
    # Fixed k and z grids
    K_MODES = np.logspace(-5.1, 2, N_K_MODES)
    Z_MODES = np.concatenate((
        np.linspace(0, 2, 30, endpoint=False),
        np.linspace(2, 10, 10, endpoint=False),
        np.linspace(10, 50, 12)
    ))
    N_ZS = len(Z_MODES)  # Number of redshift bins
    
    def __init__(self, base_model_path: str = "models", metadata_path: str = "metadata", num_batches: int = 10):
        """Initializes the emulator by loading all necessary models and metadata."""
        
        if not _DEPENDENCIES_LOADED:
            raise RuntimeError("Cannot initialize PkEmulator due to missing dependencies.")

        logging.info("[PkEmulator] Initializing and loading models...")
        
        # Paths are relative to the package root
        self.MODEL_DIR = ROOT / base_model_path
        self.METADATA_DIR = ROOT / metadata_path
        self.NUM_BATCHES = num_batches
        
        # Load the core components using Path objects
        try:
            self.param_scaler = joblib.load(self.METADATA_DIR / f"param_scaler_lowk_{num_batches}_batches")
            self.t_comp_pca = joblib.load(self.METADATA_DIR / "t_components_pca_lowk")
            self.model = keras.models.load_model(
                self.MODEL_DIR / f"emulator.h5",
                custom_objects={
                    "CustomActivationLayer": CustomActivationLayer,
                    "mse": MeanSquaredError()
                },
            )

            # Create compiled inference function with XLA
            @tf.function(jit_compile=False)
            def compiled_inference(x):
                return self.model(x, training=False)
            
            self._compiled_inference = compiled_inference

            # Initialize empty dictionaries for PCA and Scaler objects
            self.PCAS: Dict[float, PCA] = {}
            self.SCALERS: Dict[float, StandardScaler] = {}
            self._pcas_loaded = False

            # Pre-computed inverse transform matrices
            self.INVERSE_TRANSFORM_MATRICES = None
            self.INVERSE_TRANSFORM_OFFSETS = None

            # Load PCA/Scalers
            self._load_pcas_and_scalers()
            
            # Warm up both the neural network AND the compiled function
            logging.info("[PkEmulator] Warming up neural network...")
            dummy_params = self.param_scaler.transform(np.array([[2.0, 0.96, 67.0, 0.05, 0.3, -1.0, 0.0]], dtype=np.float32))
            dummy_tf = tf.constant(dummy_params, dtype=tf.float32)
            
            # Warm up compiled function (first call triggers XLA compilation)
            _ = self._compiled_inference(dummy_tf)
            # Second call to ensure compilation is complete
            _ = self._compiled_inference(dummy_tf)

            logging.info("[PkEmulator] Core models loaded successfully.")

        except FileNotFoundError as e:
            logging.error(f"CRITICAL: Required model or metadata file not found: {e.filename}")
            logging.warning("Please ensure the 'models' and 'metadata' directories are correctly placed relative to pk_emulator.py.")
            raise
    
    def _load_pcas_and_scalers(self):
        """
        Eager loading of PCA and Scaler objects with pre-computation of inverse transforms.
        
        Only loads these if they haven't been loaded yet. This is called
        automatically during __init__ to pre-compute inverse transformation matrices.
        
        Pre-computes the combined inverse PCA + inverse scaling
        transformation matrices for maximum performance.
        """
        if self._pcas_loaded:
            return  # Already loaded
        
        logging.info("[PkEmulator] Loading PCA and Scaler objects...")
        
        try:
            # Load individual PCA and Scaler objects
            for z in self.Z_MODES:
                z_key = float(f"{z:.3f}")
                self.PCAS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.pca")
                self.SCALERS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.frac_pks_scaler")
            
            logging.info("[PkEmulator] PCA and Scaler objects loaded successfully.")
            logging.info("[PkEmulator] Pre-computing inverse transformation matrices...")
            
            # Pre-compute combined inverse transformation matrices
            # This combines: (PCs @ PCA.components_ + PCA.mean_) * std + mean
            # Into a single matrix multiplication + offset addition
            
            inverse_matrices = []
            inverse_offsets = []
            
            for z in self.Z_MODES:
                z_key = float(f"{z:.3f}")
                pca = self.PCAS[z_key]
                scaler = self.SCALERS[z_key]
                
                # Get the scaling factor (handle both sklearn and custom Scaler)
                if hasattr(scaler, 'scale_'):
                    scale = scaler.scale_  # sklearn StandardScaler
                    mean = scaler.mean_
                elif hasattr(scaler, 'std'):
                    scale = scaler.std     # Custom Scaler
                    mean = scaler.mean
                else:
                    raise AttributeError(f"Scaler object has neither 'scale_' nor 'std' attribute")
                
                # Combined transformation matrix
                # Shape: (N_PCS, N_K_MODES)
                # Each row of PCA.components_ is scaled element-wise by scaler.std
                combined_matrix = pca.components_ * scale[None, :]
                
                # Combined offset vector
                # Shape: (N_K_MODES,)
                # This is: PCA.mean_ * scaler.std + scaler.mean
                combined_offset = pca.mean_ * scale + mean
                
                inverse_matrices.append(combined_matrix)
                inverse_offsets.append(combined_offset)
            
            # Stack into arrays for vectorized operations
            # Shape: (N_ZS, N_PCS, N_K_MODES)
            self.INVERSE_TRANSFORM_MATRICES = np.stack(inverse_matrices, axis=0).astype(np.float32)
            
            # Shape: (N_ZS, N_K_MODES)
            self.INVERSE_TRANSFORM_OFFSETS = np.stack(inverse_offsets, axis=0).astype(np.float32)
            
            self._pcas_loaded = True
            logging.info("[PkEmulator] Inverse transformation matrices pre-computed successfully.")
            
        except FileNotFoundError as e:
            logging.error(f"CRITICAL: Required PCA/Scaler file not found: {e.filename}")
            logging.warning("Please ensure the 'metadata' directory contains all PCA and scaler files.")
            raise
    
    def _compute_mps_approximation(self, params: np.ndarray) -> np.ndarray:
        """
        Computes the analytical P_lin(k, z) approximation.
        
        This function uses Colossus to calculate the growth factors.
        
        Args:
            params: 1D array of cosmological parameters [10^9 A_s, ns, H0, Ob, Om, w0, wa].
            
        Returns:
            np.ndarray: P_lin(k, z) array of shape (N_ZS, N_K_MODES).
        """
        # Destructure parameters (clearer than indexing)
        As, ns, H0_in, Ob, Om, w0, wa = params
        h = H0_in / 100.0 # Convert H0 to h
        # Compute fiducial P(k) at z=0
        # Convert k for plin_emulated: 1/Mpc -> h/Mpc
        k_for_plin = self.K_MODES / h
        pk_fid = plin_emulated(k_for_plin, Om, Ob, h, ns, As=As, w0=w0, wa=wa)
        a_array = 1.0/(self.Z_MODES + 1)
        D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, 
                               ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
        Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, 
                               ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
        R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, 
                                 ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
        Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, 
                                 ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
        growth_factors = (Dz/D0)*(Dz/D0)*(Rz/R0)
        # pk_fid[None, :] broadcasting works cleanly: (1, K) * (Z, 1) = (Z, K)
        result = pk_fid[None, :] * growth_factors[:, None]
        result = result / h**3
        return result.astype(np.float32)

    def _predict_fracs_all_z(self, params: np.ndarray) -> np.ndarray:
        """Optimized NN inference using compiled TensorFlow function.
        Performs the full NN prediction and PCA reconstructions,
        optimized using compiled TensorFlow function.
        
        This method requires PCA and Scaler objects, which are loaded
        at initialization to avoid unnecessary loading during evaluation.
        
        Args:
            params: Normalized 2D array of parameters (1, N_params).
            
        Returns:
            np.ndarray: Final predicted log fractional differences, shape (N_ZS, N_K_MODES).
        """
        # Convert to TF tensor and use compiled inference
        params_tf = tf.constant(params, dtype=tf.float32)

        # Predict T-components
        t_comps_pred = self._compiled_inference(params_tf).numpy()
        
        # Inverse T-components to flat k-PCs (shape 1, N_ZS * N_PCS)
        pcs_flat = self.t_comp_pca.inverse_transform(t_comps_pred).astype(np.float32)

        # Reshape to individual redshifts (shape N_ZS, N_PCS)
        # We drop the single batch dimension (axis 0) since we only have one cosmology
        pcs_pred_z_stack = pcs_flat.reshape(self.N_ZS, self.N_PCS)
        
        reconstructed_fracs = (
            np.einsum('zp,zpk->zk', pcs_pred_z_stack, self.INVERSE_TRANSFORM_MATRICES) +
            self.INVERSE_TRANSFORM_OFFSETS
        )
        
        return reconstructed_fracs.astype(np.float32)

    def get_pks(self, params: List[float], use_syren: bool = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns P(k, z) for all emulator redshifts for a given cosmology.
        
        Args:
            params: List or 1D array of 7 cosmological parameters [10^9 A_s, ns, H0, Ob, Om, w0, wa].
            use_syren: Optional flag to bypass emulator corrections.
                      - None (default): Apply emulator corrections (full emulator)
                      - True: Bypass emulator, return only symbolic approximation
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (k_modes, z_modes, P(k,z)).
        """
        # Normalize cosmological parameters
        # Ensure input is a 2D array (1, N_params) for the scaler/model
        x_norm = self.param_scaler.transform(np.array(params, dtype=np.float32).reshape(1, -1))
        
        # Compute MPS approximation (symbolic P(k) - always needed)
        pk_mps = self._compute_mps_approximation(np.array(params, dtype=np.float32))

        # Decide whether to apply emulator corrections
        if use_syren is True:
            # Bypass emulator - return only symbolic approximation
            pks = pk_mps
        else:
            # Default behavior: apply emulator corrections
            # Generate predicted fractional differences (shape N_ZS, N_K_MODES)
            frac = np.exp(self._predict_fracs_all_z(x_norm), dtype=np.float32)
            
            # Full emulated P(k, z)
            pks = (frac * pk_mps).astype(np.float32)

        # Return the k-modes, z-modes, and the P(k,z) array
        return self.K_MODES, self.Z_MODES, pks



# --- Public Module-Level Interface ---

# Instantiate the emulator once when the module is imported.
_pk_emulator_instance = None
if _DEPENDENCIES_LOADED:
    try:
        _pk_emulator_instance = PkEmulator()
    except Exception as e:
        logging.warning(f"PkEmulator instance failed during initialization: {e}")

def get_pks(params: List[float], use_syren: bool = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Module-level function to get P(k, z). This is the streamlined public interface.
    
    It calls the get_pks method of the globally-instantiated PkEmulator object.
    
    Args:
        params: List or 1D array of cosmo paras, [10^9 A_s, ns, H0, Ob, Om, w0, wa].
        use_syren: Optional flag to bypass emulator corrections.
                  - None (default): Apply emulator corrections (full emulator)
                  - True: Bypass emulator, return only symbolic approximation
            
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (k_modes, z_modes, P(k,z)).
    """
    if _pk_emulator_instance is None:
        raise RuntimeError("PkEmulator is not loaded. Check prior error messages regarding dependencies or files.")
    return _pk_emulator_instance.get_pks(params, use_syren=use_syren)