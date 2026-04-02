import torch
import torch.nn as nn
import numpy as np
import sys, os
from cobaya.theory import Theory
from cobaya.typing import InfoDict
from cobaya.theories.emultheta.emulator import ResMLP2
import joblib
import scipy
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from typing import Mapping, Iterable
from cobaya.typing import empty_dict, InfoDict
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

class emultheta(Theory):
    renames: Mapping[str, str] = empty_dict
    extra_args: InfoDict = { }
    _must_provide: dict
    path: str
    
    def initialize(self):
        super().initialize()
        RT = os.environ.get("ROOTDIR")
        imax = 1
        for name in ("M", "info", "ord", "extrapar"):
            setattr(self, name, [None] * imax)
        self.req    = [] 
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
            ("extra", "Emulator Theta: Missing emulator file (extra) option"),
            ("ord", "Emulator Theta: Missing ord (parameter ordering) option"),
            ("file", "Emulator Theta: Missing emulator file option"),
            ("extrapar", "Emulator Theta: Missing extrapar option"),
        ]
        for key, msg in _required_lists:
            if (tmp := self.extra_args.get(key)) is None:
                raise ValueError(msg)        
            if any(x is None or (isinstance(x, str) and x.strip().lower() == "none") for x in tmp[:imax]):
                raise ValueError(msg)
        _mla_requirements = {
            "GP":  [],
            "ResMLP": ["INTDIM","NLAYER"],
        }
        self.extrapar = self.extra_args["extrapar"][0]
        if not isinstance(self.extrapar, dict):
            raise ValueError('Emulator Theta: extrapar option not a dictionary')
        self.MLA = self.extrapar.get('MLA')
        if self.MLA is None or (isinstance(self.MLA, str) and self.MLA.strip().lower() == "none"):
            raise ValueError(f'Emulator Theta: Missing extrapar MLA option')
        try:
            req_keys = _mla_requirements[self.MLA]
        except KeyError:
            raise KeyError(f"Emulator Theta: Unknown MLA option: {self.MLA}")
        miss = [k for k in req_keys if k not in self.extrapar]
        if miss:
            raise KeyError(f"Emulator Theta: Missing extrapar keys for {self.MLA}: {miss}")
        # BASIC CHECKS ENDS ------------------------------------------------
        
        file = os.path.join(RT, self.extra_args.get("extra")[0])
        self.info[0] = np.load(file,allow_pickle=True)
        self.ord[0] = self.extra_args.get('ord')[0]
        
        if self.MLA == "GP":
            file = os.path.join(RT, self.extra_args.get("file")[0])
            self.M[0] = joblib.load(file)
        elif self.MLA == "ResMLP":
            self.M[0] = ResMLP2(input_dim = len(self.ord[0]),
                                output_dim = 1,
                                int_dim = self.extrapar['INTDIM'],
                                N_layer = self.extrapar['NLAYER'])
            self.M[0] = self.M[0].to(self.device)
            self.M[0] = nn.DataParallel(self.M[0])
            file = os.path.join(RT, self.extra_args.get("file")[0])
            self.M[0].load_state_dict(torch.load(file, map_location=self.device))
            self.M[0] = self.M[0].module.to(self.device)
            self.M[0].eval()
        self.req.extend(self.ord[0])

        self.req = list(set(self.req))
        d = {'omegamh2' : None}
        for i in self.req:
            d[i] = None
        self.req = d

    def get_requirements(self):
        return self.req

    def predict(self, X, Y_mean, Y_std, model):
        with torch.no_grad():
            X = torch.Tensor(X).to(self.device)
            pred = model(X)
            M_pred = pred.to(self.device).float().cpu().numpy()
            y_pred = (M_pred *Y_std+Y_mean)
        return y_pred

    def calculate(self, state, want_derived=False, **params):
        par = params.copy()

        params = self.ord[0]
        X_mean = self.info[0].item()['X_mean']
        Y_mean = self.info[0].item()['Y_mean']
        X_std  = self.info[0].item()['X_std']
        Y_std  = self.info[0].item()['Y_std']
        p =  np.array([par[key] for key in params]) - X_mean
        if self.MLA == "GP":
            H0 = self.M[0].predict(p/X_std)[0]*Y_std[0] + Y_mean[0]
        elif self.MLA == "ResMLP":
            p0 = p/X_std
            H0 = self.predict(p0, Y_mean, Y_std, self.M[0])[0][0]
        
        h2       = (H0/100.0)**2
        omegamh2 = par["omegamh2"]
        omegabh2 = par["omegabh2"]
        par.update({"H0": H0})
        par.update({"omegam": omegamh2/h2})   
        par.update({"omegab": omegabh2/h2})   
        state.update({"H0": par["H0"]})
        state.update({"omegam": par["omegam"]})
        state.update({"omegab": par["omegab"]})
        state["derived"].update({"H0": par["H0"]})
        state["derived"].update({"omegam": par["omegam"]})
        state["derived"].update({"omegab": par["omegab"]})
        return True

    def get_H0(self):
        state = self.current_state.copy()
        return state["H0"]

    def get_omegam(self):
        state = self.current_state.copy()
        return state["omegam"]

    def get_omegab(self):
        state = self.current_state.copy()
        return state["omegab"]