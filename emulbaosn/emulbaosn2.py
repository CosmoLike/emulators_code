import torch, os
import torch.nn as nn
import numpy as np 
import os.path as path
from cobaya.theories.emulbaosn.emulator import ResBlock, ResMLP
from scipy import interpolate
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

class emulbaosn():    
    def __init__(self, extra_args):
        self.extra_args = extra_args
        RT = os.environ.get("ROOTDIR")
        self.imax = 2
        for name in ("M", "info", "ord", "z", "tmat", "extrapar", "offset"):
            setattr(self, name, [None] * self.imax)
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

        # BASIC CHECKS BEGINS --------------------------------------------------
        _required_lists = [
            ("extra", "Emulator BAOSN: Missing emulator file (extra) option"),
            ("ord", "Emulator BAOSN: Missing ord (parameter ordering) option"),
            ("file", "Emulator BAOSN: Missing emulator file option"),
            ("extrapar", "Emulator BAOSN: Missing extrapar option"),
        ]
        for key, msg in _required_lists:
            if (tmp := self.extra_args.get(key)) is None or (len(tmp)<self.imax):
                raise ValueError(msg)        
            # H(Z) -------------------------------------------------------------
            i = 1
            if tmp[i] is None or (isinstance(tmp[i], str) and tmp[i].strip().lower()=="none"):
                raise ValueError(msg)            
        _mla_requirements = {
            "ResMLP": ["INTDIM","NLAYER","TMAT",'ZLIN','offset'],
            "INT": ["ZMIN","ZMAX","NZ"],
        }
        for i in range(self.imax):
            self.extrapar[i] = self.extra_args["extrapar"][i].copy()
            if not isinstance(self.extrapar[i], dict):
                raise ValueError('Emulator BAOSN: extrapar option not a dictionary')
            mla = self.extrapar[i].get('MLA')
            if mla is None or (isinstance(mla, str) and mla.strip().lower() == "none"):
                raise ValueError(f'Emulator BAOSN: Missing extrapar MLA option')
            try:
                req_keys = _mla_requirements[mla]
            except KeyError:
                raise KeyError(f"Emulator BAOSN: Unknown MLA option: {mla}")
            miss = [k for k in req_keys if k not in self.extrapar[i]]
            if miss:
                raise KeyError(f"Emulator BAOSN: Missing extrapar keys for {mla}: {miss}")
        # BASIC CHECKS ENDS ----------------------------------------------------

        # H(Z) -----------------------------------------------------------------
        i = 1
        self.info[i] = np.load(path.join(RT,self.extra_args.get("extra")[i]),allow_pickle=True)
        self.z[i]    = np.load(path.join(RT,self.extrapar[i]['ZLIN']),allow_pickle=True)
        self.tmat[i] = np.load(path.join(RT,self.extrapar[i]["TMAT"]),allow_pickle=True)

        self.ord[i] = self.extra_args['ord'][i]  
        self.offset[i] = self.extrapar[i]['offset']
        
        if self.extrapar[i]['MLA'].strip().lower() == 'resmlp':
            self.M[i] = ResMLP(input_dim = len(self.ord[i]), 
                               output_dim = len(self.tmat[i]), 
                               int_dim = self.extrapar[i]['INTDIM'], 
                               N_layer = self.extrapar[i]['NLAYER'])
        self.M[i] = self.M[i].to(self.device)
        self.M[i] = nn.DataParallel(self.M[i])
        self.M[i].load_state_dict(torch.load(path.join(RT,self.extra_args.get("file")[i]),map_location=self.device))
        self.M[i] = self.M[i].module.to(self.device)
        self.M[i].eval()
        # SN(Z) ----------------------------------------------------------------
        i = 0
        if self.extrapar[i]['MLA'].strip().lower() == 'int':
            zmin = self.extrapar[i]['ZMIN']
            zmax = self.extrapar[i]['ZMAX']
            if (zmax > self.z[1][-1]):
                zmax = self.z[1][-1]
            NZ = self.extrapar[i]['NZ']
            self.z[i] = np.linspace(zmin, zmax, NZ)
            self.ord[i] = self.extra_args.get('ord')[1] # Same as H(z)

    def predict(self, model, X, extrainfo, transform_matrix, offset):
        X_mean   = torch.Tensor(extrainfo.item()['X_mean']).to(self.device)
        X_std    = torch.Tensor(extrainfo.item()['X_std']).to(self.device)
        Y_mean   = extrainfo.item()['Y_mean']
        Y_std    = extrainfo.item()['Y_std']
        Y_mean_2 = torch.Tensor(extrainfo.item()['Y_mean_2']).to(self.device)
        Y_std_2  = torch.Tensor(extrainfo.item()['Y_std_2']).to(self.device)
        X = torch.Tensor(X).to(self.device)
        with torch.no_grad():
            X_norm = ((X - X_mean) / X_std)
            X_norm.to(self.device)
            pred = model(X_norm)
            M_pred = pred.to(self.device)
            y_pred = (M_pred.float() *Y_std_2.float()+Y_mean_2.float()).cpu().numpy()
            y_pred = np.matmul(y_pred,transform_matrix)*Y_std+Y_mean
            y_pred = np.exp(y_pred) - offset
        return y_pred[0]

    def cumulative_simpson(self, z, y):
        
        n = len(z)
        if n < 3 or (n - 1) % 2 != 0:
            raise ValueError("Need an odd number of points (even number of intervals).")
        dz = z[1] - z[0]
        
        # Simpson contributions on each pair of intervals [z[2m], z[2m+2]]
        # there are (n-1)/2 such chunks
        f0 = y[:-2:2]    # y[0], y[2], y[4], …
        f1 = y[1:-1:2]   # y[1], y[3], y[5], …
        f2 = y[2::2]     # y[2], y[4], y[6], …
        chunks = dz/3 * (f0 + 4*f1 + f2)
        
        # cumulative sum of the full-chunk integrals at even indices
        cum_even = np.concatenate(([0.0], np.cumsum(chunks)))
        
        # build the full cumulative array
        C = np.empty_like(y)
        C[0] = 0.0
        C[2::2] = cum_even[1:]               # at z[2], z[4], …
        for i in range(1, n, 2):
            C[i] = C[i - 1] + dz / 6 * (y[i - 1] + 4 * y[i] + y[i + 1])
        return C

    def calculate(self, par):  
        state = {}
        out = ["dl","H","z"]
        # z ------------------------------------------------------------
        i = 2
        state[out[i]] = self.z[0]
        # H(z) ------------------------------------------------------------
        i = 1
        params = self.extra_args.get('ord')[i]
        p = np.array([par[key] for key in params])
        Hz = self.predict(self.M[i],p,self.info[i],self.tmat[i],self.offset[i])
        func = interpolate.interp1d(self.z[1], Hz,
                                    kind='cubic',
                                    assume_sorted=True,
                                    fill_value="extrapolate")
        state[out[i]] = func(self.z[0])
        # SN ------------------------------------------------------------
        i = 0
        func = interpolate.interp1d(self.z[1], 2.99792458e5/Hz,
                            kind='cubic',
                            assume_sorted=True,
                            fill_value="extrapolate")
        zstep = np.linspace(0.0, self.z[0][-1], 2*len(self.z[0])+1)
        chi   = self.cumulative_simpson(zstep,func(zstep))#*(1 + zstep)
        zhigh = 1200 #redshift to which we are going to extend by numerical integration
        NZEXT = 4501 #number of z bins we are going to numerically integrate in the extended region
        zext = np.linspace(zstep[-1], zhigh, NZEXT) #create the extended z array
        zfinal = np.concatenate((zstep, zext[1:]))
        h = params['H0']/100
        omegar = 3.612711417813115e-05/h/h
        H_ext = params['H0']*np.sqrt(params['omegam']*(1+zext)**3+omegar*(1+zext)**4) #this is an approximation

        chi_ext = self.cumulative_simpson(zext, 2.99792458e5/H_ext)+chi[-1]
        chi_final = np.concatenate((chi, chi_ext[1:]))
        if 'omk' in params:
            K_abs = abs(params['omk'])*(params['H0']/2.99792458e5)**2
            if np.isclose(params['omk'], 0.0, atol=1e-12):
                dl = chi_final*(1 + zfinal)
            elif params['omk']>0:
                dl = np.sinh(chi_final*K_abs)/K_abs*(1 + zfinal)
            else:
                dl = np.sin(chi_final*K_abs)/K_abs*(1 + zfinal)
        else:
            dl = chi_final*(1 + zfinal)
        
        state[out[i]] = interpolate.interp1d(zfinal, dl, 
                                             kind='cubic',
                                             assume_sorted=True,
                                             fill_value="extrapolate")(self.z[0])
        return state