import numpy as np
import cobaya
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import sys
import os
import platform
import yaml
from mpi4py import MPI
from scipy.stats import qmc
import copy
import functools, iminuit, copy, argparse, random, time 
import emcee, itertools
from schwimmbad import MPIPool
from cobaya.likelihood import Likelihood

class MyPkLikelihood(Likelihood):

    def initialize(self):
        self.kmax = 100.0
        self.linear_pk = None
        self.nonlinear_pk = None
        z1_mps = np.linspace(0,2,100,endpoint=False)
        z2_mps = np.linspace(2,10,10,endpoint=False)
        z3_mps = np.linspace(10,50,12)
        self.z_eval = np.concatenate((z1_mps,z2_mps,z3_mps),axis=0)
    def get_requirements(self):
        return {"omegabh2": None, "Pk_interpolator": {
        "z":self.z_eval,
        "k_max": 200,
        "nonlinear": (True,False),
        "vars_pairs": ([("delta_tot", "delta_tot")])
        }, "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
        'tt': 0
        },"omegach2":None, "H0": None, "ns": None, "As": None, "tau": None}

    def logp(self, **params):
        # Get linear P(k, z)

        self.linear_pk = self.provider.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=False, extrap_kmax = self.kmax)

        # Get non-linear P(k, z)
        self.nonlinear_pk = self.provider.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=True, extrap_kmax = self.kmax
            )

        # Choose k range and evaluate both
        k = np.logspace(-4, np.log10(self.kmax), 2000)
        pk_lin = self.linear_pk.P(self.z_eval, k)
        pk_nonlin = self.nonlinear_pk.P(self.z_eval, k)



        # Return dummy log-likelihood
        return -0.5 
parser = argparse.ArgumentParser(prog='cos_uniform')


parser.add_argument("--N",
                    dest="N",
                    help="Number of points requested",
                    type=int,
                    nargs='?',
                    const=1,
                    default=100)

parser.add_argument("--mode",
                    dest="mode",
                    help="mode of points generated",
                    type=str,
                    nargs='?',
                    const=1,
                    default='train')

parser.add_argument("--data_path",
                    dest="data_path",
                    help="Directory for saving the data files",
                    type=str,
                    nargs='?',
                    const=1,
                    default='/data/')
parser.add_argument("--datavectors_file",
                    dest="datavectors_file",
                    help="Name for data vector files (no .npy included)",
                    type=str,
                    nargs='?',
                    const=1,
                    default='cos_uni')
parser.add_argument("--parameters_file",
                    dest="parameters_file",
                    help="Name for Parameter files (.npy included)",
                    type=str,
                    nargs='?',
                    const=1,
                    default='cos_uni_input.npy')
args, unknown = parser.parse_known_args()

####add yaml here, and make all input paratemers passing in
yaml_string=r"""

likelihood:
  dummy:
    class: MyPkLikelihood

params:
  tau:
    prior:
      min: 0.01
      max: 0.2
    ref:
      dist: norm
      loc: 0.055
      scale: 0.006
    proposal: 0.003
    latex: \tau_\mathrm{reio}

  logA:
    prior:
      min: 1.61
      max: 3.91
    ref:
      dist: norm
      loc: 3.0448
      scale: 0.05
    proposal: 3
    latex: \log(10^{10} A_\mathrm{s})
    drop: true
  
  omegabh2:
    prior:
      min: 0.0
      max: 0.04
    ref:
      dist: norm
      loc: 0.022383
      scale: 0.005
    proposal: 0.005
    latex: \Omega_\mathrm{b} h^2
  omegach2:
    prior:
      min: 0.0
      max: 0.5
    ref:
      dist: norm
      loc: 0.12011
      scale: 0.03
    proposal: 0.03
    latex: \Omega_\mathrm{c} h^2
  H0:
    prior:
      min: 20
      max: 120
    ref:
      dist: norm
      loc: 67
      scale: 2
    proposal: 0.001
    latex: H_0
  
  ns:
    prior:
      min: 0.6
      max: 1.3
    ref:
      dist: norm
      loc: 0.96605
      scale: 0.005
    proposal: 0.005
    latex: n_\mathrm{s}
  As:
    value: 'lambda logA: 1e-10*np.exp(logA)'
    latex: A_\mathrm{s}
  mnu:
    value: 0.06


  thetastar:
    derived: true
    latex: \Theta_\star
  rdrag:
    derived: True
    latex: r_\mathrm{drag}

theory:
  camb:
    path: ./external_modules/code/CAMB
    extra_args:
      halofit_version: mead2020
      #dark_energy_model: ppf
      lmax: 1000
      AccuracyBoost: 1.5
      lens_potential_accuracy: 8
      lens_k_eta_reference: 18000.0
      nonlinear: NonLinear_both
      recombination_model: CosmoRec
      Accuracy.AccurateBB: True




output: ./projects/axions/chains/EXAMPLE_EVALUATE0

"""


####add main function.


#===================================================================================================
# datavectors

def generate_parameters(N, u_bound, l_bound, mode, parameters_file, save=True):
    D = len(u_bound)
    if mode=='train':
        
        N_LHS = int(0.05*N)
        sampler = qmc.LatinHypercube(d=D)
        sample = sampler.random(n=N_LHS)
        sample_scaled = qmc.scale(sample, l_bound, u_bound)

        N_uni = N-N_LHS
        data = np.random.uniform(low=l_bound, high=u_bound, size=(N_uni, D))
        samples = np.concatenate((sample_scaled, data), axis=0)
    else:
        samples = np.random.uniform(low=l_bound, high=u_bound, size=(N, D))

    if save:
        np.save(parameters_file, samples)
        print('(Input Parameters) Saved!')
    return samples



if __name__ == '__main__':

    f = yaml_load(yaml_string)
    sys.modules["MyPkLikelihood"] = sys.modules[__name__]

    model = get_model(f)
    mode = args.mode
    N = args.N

    prior_params = list(model.parameterization.sampled_params())
    sampling_dim = len(prior_params)

    PATH = os.environ.get("ROOTDIR") + '/' + args.data_path
    datavectors_file_path = PATH + args.datavectors_file
    parameters_file  = PATH + args.parameters_file

    u_bound = model.prior.bounds()[:,1]
    l_bound = model.prior.bounds()[:,0]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print('rank',rank,'is at barrier')
        
    z1_mps = np.linspace(0,2,100,endpoint=False)
    z2_mps = np.linspace(2,10,10,endpoint=False)
    z3_mps = np.linspace(10,50,12)
    z_mps = np.concatenate((z1_mps,z2_mps,z3_mps),axis=0)
    len_z = len(z_mps)
    k = np.logspace(-4, 2, 2000)
    len_k = len(k)
    PK_LIN_DIR = datavectors_file_path + '_pklin.npy'
    PK_NONLIN_DIR = datavectors_file_path + '_pknonlin.npy'
    if rank == 0:
        samples = generate_parameters(N, u_bound, l_bound, mode, parameters_file)
        total_num_dvs = len(samples)

        param_info = samples[0:total_num_dvs:num_ranks]#reading for 0th rank input
        for i in range(1,num_ranks):#sending other ranks' data
            comm.send(
                samples[i:total_num_dvs:num_ranks], 
                dest = i, 
                tag  = 1
            )
                
    else:
            
        param_info = comm.recv(source = 0, tag = 1)

            
    num_datavector = len(param_info)


    PKLIN = np.zeros(
            (num_datavector, len_z, len_k), dtype = "float32"
        )
    PKNONLIN = np.zeros(
            (num_datavector, len_z, len_k), dtype = "float32"
        ) 

    for i in range(num_datavector):
        input_params = model.parameterization.to_input(param_info[i])
        print(input_params)
        input_params.pop("As", None)

        try:
            model.loglike(input_params)
            theory = list(model.theory.values())[1]
            lin_pk = theory.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=False, extrap_kmax = 200)
            nonlin_pk = theory.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=True, extrap_kmax = 200)

                
        except:
            print('fail')
        else:
            PKLIN[i] = lin_pk.P(z_mps, k)
            PKNONLIN[i] = nonlin_pk.P(z_mps, k)

    if rank == 0:
        result_pklin   = np.zeros((total_num_dvs, len_z, len_k), dtype="float32")
        result_pknonlin   = np.zeros((total_num_dvs, len_z, len_k), dtype="float32")
            
        result_pklin[0:total_num_dvs:num_ranks] = PKLIN
        result_pknonlin[0:total_num_dvs:num_ranks] = PKNONLIN

        for i in range(1,num_ranks):        
            result_pklin[i:total_num_dvs:num_ranks] = comm.recv(source = i, tag = 10)
            result_pknonlin[i:total_num_dvs:num_ranks] = comm.recv(source = i, tag = 11)

        np.save(PK_LIN_DIR, result_pklin)
        np.save(PK_NONLIN_DIR, result_pknonlin)
            
    else:    
        comm.send(PKLIN, dest = 0, tag = 10)
        comm.send(PKNONLIN, dest = 0, tag = 11)




#mpirun -n 5 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
#     --bind-to core --map-by core --report-bindings --mca mpi_yield_when_idle 1 \
#    python datageneratormps.py \
#    --data_path './trainingdata/' \
#    --datavectors_file 'dvfilename' \
#    --parameters_file 'paramfilename.npy' \
#    --N 100 \
#    --mode 'train' \
