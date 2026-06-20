import numpy as np
import os
import os.path as path
from cobaya.theory import Theory
from cobaya.typing import InfoDict
import joblib
from typing import Mapping, Iterable
from cobaya.typing import empty_dict, InfoDict

class emulrdrag(Theory):
    renames: Mapping[str, str] = empty_dict
    extra_args: InfoDict = { }
    _must_provide: dict
    path: str
    
    def initialize(self):
        super().initialize()
        RT = os.environ.get("ROOTDIR")
        self.imax = 1
        for name in ("M", "info", "ord"):
            setattr(self, name, [None] * self.imax)
        self.req    = [] 
        
        # BASIC CHECKS BEGINS ------------------------------------------------
        _required_lists = [
            ("extra", "Emulator Theta: Missing emulator file (extra) option"),
            ("ord", "Emulator Theta: Missing ord (parameter ordering) option"),
            ("file", "Emulator Theta: Missing emulator file option")
        ]
        for key, msg in _required_lists:
            if (tmp := self.extra_args.get(key)) is None:
                raise ValueError(msg)        
            if any(x is None or (isinstance(x, str) and 
                   x.strip().lower() == "none") for x in tmp[:self.imax]):
                raise ValueError(msg)
        # BASIC CHECKS ENDS ------------------------------------------------

        self.info[0] = np.load(path.join(RT,self.extra_args.get("extra")[0]),allow_pickle=True)        
        self.M[0]    = joblib.load(path.join(RT,self.extra_args.get("file")[0]))
        self.ord[0]  = self.extra_args.get('ord')[0]
        
        self.req.extend(self.ord[0])
        self.req = list(set(self.req))
        d = {}
        for i in self.req:
            d[i] = None
        self.req = d

    def get_requirements(self):
        return self.req

    def calculate(self, state, want_derived=False, **params):
        par = params.copy()

        params = self.ord[0]
        X_mean = self.info[0].item()['X_mean']
        Y_mean = self.info[0].item()['Y_mean']
        X_std  = self.info[0].item()['X_std']
        Y_std  = self.info[0].item()['Y_std']
        p =  np.array([par[key] for key in params]) - X_mean     
        rd = self.M[0].predict(p/X_std)[0]*Y_std[0] + Y_mean[0]
        state.update({'rdrag':rd})
        if state["derived"] is not None:
            state["derived"].update({'rdrag':rd})
        return True

    def get_rdrag(self):
        state = self.current_state.copy()
        return state["rdrag"]
