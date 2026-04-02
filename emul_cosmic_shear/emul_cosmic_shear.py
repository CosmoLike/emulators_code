import torch, os, sys
import torch.nn as nn
import numpy as np
from cobaya.theory import Theory
from cobaya.theories.emul_cosmic_shear.emulator import ResTRF, ResCNN
from typing import Mapping, Iterable
from cobaya.typing import empty_dict, InfoDict
import h5py as h5
sys.path.append(os.path.dirname(__file__))
try:
    import torch_xla.core.xla_model as xm
    _tpu_ok = bool(xm.get_xla_supported_devices("TPU"))
except Exception:
    xm, _tpu_ok = None, False

def get_device(dev: str):
    if dev == "tpu":
        if xm is None or not _tpu_ok:
            raise RuntimeError("TPU requested but torch_xla is not available.")
        return xm.xla_device()
    return torch.device(dev)

class emul_cosmic_shear(Theory):
    renames: Mapping[str, str] = empty_dict
    extra_args: InfoDict = { }
    _must_provide: dict
    path: str

    def initialize(self):
        super().initialize()
        RT = os.environ.get("ROOTDIR")
        imax = 1
        self.eval  = [True] * imax
        for name in ("M", "info", "ord", "extrapar"):
            setattr(self, name, [None] * imax)
        for name in ("X_mean", "X_std", "Y_mean", "dv_evals", "dv_evecs", "inv_dv_evecs"):
            setattr(self, name, [None] * imax)
        self.req   = [] 
        self.device = "cpu" if (d := self.extra_args.get("device")) is None else d.lower()
        self.device = (
            "cuda" if ((req := self.device) == "cuda" and torch.cuda.is_available()) 
            else "mps" if (req in ("cuda","mps") 
                        and hasattr(torch.backends, "mps") 
                        and torch.backends.mps.is_built() 
                        and torch.backends.mps.is_available()) 
            else "tpu" if (req in ("cuda","tpu") and _tpu_ok)
            else "cpu"
        )
        self.device = get_device(self.device)

        # BASIC CHECKS BEGINS ------------------------------------------------
        _required_lists = [
            ("extra", "Emulator Cosmic Shear: Missing emulator file (extra) option"),
            ("ord", "Emulator Cosmic Shear: Missing ord (parameter ordering) option"),
            ("file", "Emulator Cosmic Shear: Missing emulator file option"),
            ("extrapar", "Emulator Cosmic Shear: Missing extrapar option"),
        ]
        for key, msg in _required_lists:
            if (tmp := self.extra_args.get(key)) is None or (len(tmp)<imax):
                raise ValueError(msg)        
            if any(self.eval[i] and 
                   (tmp[i] is None or (isinstance(tmp[i], str) and tmp[i].strip().lower()=="none")) 
                   for i in range(imax)):
                raise ValueError(msg)
        for i in range(imax):
            if not self.eval[i]:
                continue
            self.extrapar[i] = self.extra_args["extrapar"][i].copy()
            if not isinstance(self.extrapar[i], dict):
                raise ValueError('Emulator Cosmic Shear: extrapar option not a dictionary')
        # BASIC CHECKS ENDS ------------------------------------------------
        
        for i in range(imax):
            self.info[i] = self.extra_args.get('extra')[i]
            # load extra stuff for
            if 'h5' in self.info[i]:
                with h5.File(self.info[i], 'r', locking=False) as f:
                    self.X_mean[i] = torch.Tensor(f['sample_mean'][:]).to(self.device)
                    self.X_std[i] = torch.Tensor(f['sample_std'][:]).to(self.device)
                    self.Y_mean[i] = torch.Tensor(f['dv_fid'][:]).to(self.device)
                    self.dv_evals[i] = torch.Tensor(f['dv_evals'][:]).to(self.device)
                    self.dv_evecs[i] = torch.Tensor(f['dv_evecs'][:]).to(self.device)
            # invert the rotation matrix so that we don't do it every time we evaluate
            
            # do the inverse on CPU (MPS can abort on linalg.inv for some shapes/dtypes)
            #self.inv_dv_evecs[i] = torch.linalg.inv(self.dv_evecs[i])
            if self.device.type == "mps":
                A = self.dv_evecs[i].detach().to("cpu")
                inv_cpu = torch.linalg.inv(A)
                self.inv_dv_evecs[i] = inv_cpu.to(self.device)
            else:
                self.inv_dv_evecs[i] = torch.linalg.inv(self.dv_evecs[i])

            self.ord[i] = self.extra_args.get('ord')[i]

            if self.extrapar[i]['MLA'] == 'TRF':
                self.M[i] = ResTRF(input_dim = len(self.ord[i]),
                                   output_dim  = self.extrapar[i]['OUTPUT_DIM'],
                                   int_dim_res = self.extrapar[i]['INT_DIM_RES'],
                                   int_dim_trf = self.extrapar[i]['INT_DIM_TRF'],
                                   N_channels  = self.extrapar[i]['NC_TRF'])

            elif self.extrapar[i]['MLA'] == 'CNN':
                self.M[i] = ResCNN(input_dim = len(self.ord[i]),
                                   output_dim = self.extrapar[i]['OUTPUT_DIM'],
                                   int_dim = self.extrapar[i]['INT_DIM_RES'],
                                   cnn_dim = self.extrapar[i]['CNN_DIM'],
                                   kernel_size = self.extrapar[i]['KERNEL_DIM'])
            else:
                print("MLA must be one of [TRF, CNN]")
                exit()

            self.M[i] = self.M[i].to(self.device)
            self.M[i].load_state_dict(torch.load(self.extra_args.get('file')[i],map_location=self.device))
            self.M[i] = self.M[i].eval()
            self.req.extend(self.ord[i])

        self.req = list(set(self.req))
        d = {}
        for i in self.req:
            d[i] = None
        self.req = d

    def get_requirements(self):
        return self.req

    def predict_data_vector(self, X, i):
        X = torch.tensor(X, dtype=self.X_mean[i].dtype, device=self.device)
        with torch.no_grad():
            X_norm = torch.nan_to_num((X - self.X_mean[i]) / self.X_std[i], nan=0.0)
            M_pred = self.M[i](X_norm)
        res = (M_pred * self.dv_evals[i]) @ self.inv_dv_evecs[i] + self.Y_mean[i]
        return res[0].detach().cpu().numpy()

    def calculate(self, state, want_derived=True, **params):
        i = 0
        X = [params[p] for p in self.ord[i]]
        state["cosmic_shear"] = self.predict_data_vector(X,i)
        return True

    def get_cosmic_shear(self):
        return self.current_state['cosmic_shear']




