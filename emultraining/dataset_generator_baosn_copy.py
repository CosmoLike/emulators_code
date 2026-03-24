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
from cobaya.likelihood import Likelihood
from schwimmbad import MPIPool


parser = argparse.ArgumentParser(prog='cos_uniform')


class SimpleBAODVLikelihood(Likelihood):
    # Example data can also be passed via YAML or hardcoded here
    def initialize(self):
        # Define data inside the class (or pass from outside if needed)
        z1=np.linspace(0,3,600, endpoint=False)
        z2=np.linspace(3,1200,200)
        self.z_data = np.concatenate((z1,z2),axis=0)  # example redshifts

    def get_requirements(self):
        # Tell Cobaya which theory quantities you need at these redshifts
        return {
            "angular_diameter_distance": {"z": self.z_data},
            "Hubble": {"z": self.z_data}
            
        }

    def logp(self, **params_values):
        # Get required theory quantities
        return -0.5

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
    class: SimpleBAODVLikelihood

params:
  
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
  tau:
    value: 0.06
    latex: \tau_\mathrm{reio}

  logA:
    value: 3.04
    latex: \log(10^{10} A_\mathrm{s})
    drop: true
  ns:
    value: 0.95
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
      #dark_energy_model: ppf
      lmax: 1000





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
mode = args.mode
N = args.N
PATH = os.environ.get("ROOTDIR") + '/' + args.data_path
parameters_file  = PATH + args.parameters_file

if __name__ == '__main__':
    f = yaml_load(yaml_string)
    sys.modules["SimpleBAODVLikelihood"] = sys.modules[__name__]
    model = get_model(f)
    
    prior_params = list(model.parameterization.sampled_params())
    sampling_dim = len(prior_params)
    datavectors_file_path = PATH + args.datavectors_file
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print('rank',rank,'is at barrier')
           
    z = model.likelihood['dummy'].z_data
    len_z = len(z)
    num_output = 2
    SN_DIR = datavectors_file_path + '_baosn.npy'
    u_bound = model.prior.bounds()[:,1]
    l_bound = model.prior.bounds()[:,0]
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

    BAOSN = np.zeros(
            (num_datavector, len_z, num_output), dtype = "float32"
        ) 

    for i in range(num_datavector):
        input_params = model.parameterization.to_input(param_info[i])  
        try:
            model.loglike(input_params)
            theory = list(model.theory.values())[1]
            H = theory.get_Hubble(z)
            Dl = theory.get_angular_diameter_distance(z)*(1+z)**2
                
        except:
            print('fail')
        else:
            BAOSN[i,:,0] = H
            BAOSN[i,:,1] = Dl
            

    if rank == 0:
        result_baosn   = np.zeros((total_num_dvs, len_z, num_output), dtype="float32")
        result_baosn[0:total_num_dvs:num_ranks] = BAOSN

        for i in range(1,num_ranks):        
            result_baosn[i:total_num_dvs:num_ranks] = comm.recv(source = i, tag = 10)

        np.save(SN_DIR, result_baosn)
            
    else:    
        comm.send(BAOSN, dest = 0, tag = 10)




#mpirun -n 5 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
#     --bind-to core --map-by core --report-bindings --mca mpi_yield_when_idle 1 \
#    python datageneratorbaosn.py \
#    --data_path './trainingdata/' \
#    --datavectors_file 'dvfilename' \
#    --parameters_file 'paramfilename.npy' \
#    --N 100 \
#    --mode 'train' \

