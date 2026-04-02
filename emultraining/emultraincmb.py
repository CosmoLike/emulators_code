import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from cobaya.theories.emulcmb.emulator import ResBlock, ResMLP, TRF, CNNMLP
from cobaya.likelihood import Likelihood
import cobaya
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import platform
import yaml
class MyPkLikelihood(Likelihood):

    def initialize(self):
        self.cov=np.load('covmat/cv_fid_cls.npy',allow_pickle=True)
    def get_requirements(self):
        return {"Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
        'tt': 0
        }}

    def logp(self, **params):
        
        return -0.5 
parser = argparse.ArgumentParser(prog='cos_uniform')

parser.add_argument("--mode",
                    dest="mode",
                    help="TT TE or EE",
                    type=str,
                    nargs='?',
                    const=1,
                    default='tt')



parser.add_argument("--batch",
                    dest="batch_size",
                    help="Number of samples per batch",
                    type=int,
                    nargs='?',
                    const=1,
                    default=512)
parser.add_argument("--epoch",
                    dest="n_epoch",
                    help="Number of epochs",
                    type=int,
                    nargs='?',
                    const=1,
                    default=700)

parser.add_argument("--Path",
                    dest="PATH",
                    help="Saving Directory, no .pt required",
                    type=str,
                    nargs='?',
                    const=1,
                    default='model')


args, unknown = parser.parse_known_args()

PATH = args.PATH
batch_size = args.batch_size
n_epoch = args.n_epoch


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
    drop: false
  
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


theory:
  emulcmb:
    path: ./cobaya/cobaya/theories/
    extra_args:
      # This version of the emul was not trained with CosmoRec
      eval: [True, True, True, True] #TT,TE,EE,PHIPHI
      device: "cuda"
      ord: [['tau','logA','omegabh2','omegach2','H0','ns'],
            ['tau','logA','omegabh2','omegach2','H0','ns'],
            ['tau','logA','omegabh2','omegach2','H0','ns'],
            ['tau','logA','omegabh2','omegach2','H0',ns']]
      file: ['external_modules/data/emultrf/CMB_TRF/emul_lcdm_CMBTT_CNN.pt',
             'external_modules/data/emultrf/CMB_TRF/emul_lcdm_CMBTE_CNN.pt',
             'external_modules/data/emultrf/CMB_TRF/emul_lcdm_CMBEE_CNN.pt', 
             'external_modules/data/emultrf/CMB_TRF/emul_lcdm_phi_ResMLP.pt']
      extra: ['external_modules/data/emultrf/CMB_TRF/extra_lcdm_CMBTT_CNN.npy',
              'external_modules/data/emultrf/CMB_TRF/extra_lcdm_CMBTE_CNN.npy',
              'external_modules/data/emultrf/CMB_TRF/extra_lcdm_CMBEE_CNN.npy', 
              'external_modules/data/emultrf/CMB_TRF/extra_lcdm_phi_ResMLP.npy']
      extrapar: [{'ellmax' : 5000, 'MLA': 'CNN', 'INTDIM': 4, 'INTCNN': 5120},
                 {'ellmax' : 5000, 'MLA': 'CNN', 'INTDIM': 4, 'INTCNN': 5120},
                 {'ellmax' : 5000, 'MLA': 'CNN', 'INTDIM': 4, 'INTCNN': 5120}, 
                 {'MLA': 'ResMLP', 'INTDIM': 4, 'NLAYER': 4, 
                  'TMAT': 'external_modules/data/emultrf/CMB_TRF/PCA_lcdm_phi.npy'}]
output: ./projects/axions/chains/EXAMPLE_EVALUATE0

"""
f = yaml_load(yaml_string)
sys.modules["MyPkLikelihood"] = sys.modules[__name__]

model = get_model(f)
mode = args.mode
if mode=='tt':
    ind = 0
elif mode=='te':
    ind = 1
elif mode=='ee':
    ind = 2
else:
    print('go to script for phiphi')
theory = list(model.theory.values())[0]
ell_max = theory.extrapar[ind]['ellmax']
int_dim = theory.extrapar[ind]['INTDIM']
cnn_dim = theory.extrapar[ind]['INTCNN']
extrainfo_file = theory.extra[ind]
output_file = theory.file[ind]

train_samples_file = PATH + '/train_params.npy'
valid_samples_file = PATH + '/valid_params.npy'

train_dv_file = PATH + '/train_dv.npy'
valid_dv_file = PATH + '/valid_dv.npy'

tau_pos = 0
As_pos = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':

    camb_ell_range        = ell_max  - 2
    extrainfo = np.load(extrainfo_file,allow_pickle=True)
    cl_fid = model.likelihood['dummy'].cov.item()[mode][:camb_ell_range]*2/np.exp(2*0.06)
    ell = np.arange(2,ell_max,1)
    covinv = 2/(2*ell+1)*cl_fid**2
    covinv = np.diag(covinv)
    train_samples = np.load(train_samples_file,allow_pickle=True)

    in_size=len(train_samples[0])
    out_size=camb_ell_range
    validation_samples = np.load(valid_samples_file,allow_pickle=True)

    train_data_vectors = np.load(train_dv_file,allow_pickle=True,mmap_mode='r+')
    # mmap_mode=r+ for huge files to be read from memory
    # not in RAM
    validation_data_vectors = np.load(valid_dv_file,allow_pickle=True)
    
    N_ave = int(0.1*N)
    X_mean = train_samples.mean(axis=0, keepdims=True)
    X_std = train_samples.std(axis=0, keepdims=True)
    Y_mean = train_data_vectors.mean(axis=0, keepdims=True)
    Y_std = train_data_vectors.std(axis=0, keepdims=True)
    extrainfo={'X_mean':X_mean,'X_std':X_std,'Y_mean':Y_mean,'Y_std':Y_std}
    np.save(extrainfo_file,extrainfo)
    


    X_mean = torch.Tensor(X_mean)#.to(device)
    X_std  = torch.Tensor(X_std)#.to(device)
    Y_mean = torch.Tensor(Y_mean).to(device)
    Y_std  = torch.Tensor(Y_std).to(device)


    covinv = torch.Tensor(covinv).to(device) #This is inverse of the Covariance Matrix

    train_samples=torch.Tensor(train_samples)#.to(device)
    train_data_vectors=torch.Tensor(train_data_vectors)#.to(device)
    validation_samples=torch.Tensor(validation_samples)#.to(device)
    validation_data_vectors=torch.Tensor(validation_data_vectors)#.to(device)


    X_train=(train_samples-X_mean)/X_std
    X_validation=(validation_samples-X_mean)/X_std
    X_train=X_train.to(torch.float32)
    X_validation=X_validation.to(torch.float32)

    X_mean=X_mean.to(device)
    X_std=X_std.to(device)

    trainset    = TensorDataset(X_train, train_data_vectors)
    validset    = TensorDataset(X_validation,validation_data_vectors)
    #testingset    = TensorDataset(X_testing, testing_data_vectors)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
    #testloader = DataLoader(testingset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

    model = CNNMLP(input_dim=in_size, output_dim=out_size, int_dim=intdim, cnn_dim=cnndim)

    model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=0)

    losses_train = []
    losses_vali = []
    losses_train_med = []
    losses_vali_med = []

    reduce_lr = True#reducing learning rate on plateau
    if reduce_lr==True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=15)

    for n in range(n_epoch):
        losses=[]
        for i, data in enumerate(trainloader):
            model.train()

            X = data[0].to(device)# send to device one by one
                
            Y_batch = data[1].to(device)# send to device one by one
            Y_pred  = model(X).to(device)
                
            As = torch.exp(X[:,As_pos]*X_std[0,As_pos]+X_mean[0,As_pos])
            exptau = torch.exp(2*X[:,tau_pos]*X_std[0,tau_pos]+2*X_mean[0,tau_pos])
            Y_pred = Y_pred*Y_std+Y_mean
            Y_batch = Y_batch/As[:,None]*exptau[:,None]
            diff = Y_pred - Y_batch
            
            loss1 = torch.diag(diff @ covinv @ torch.t(diff))# implement with torch.einsum
            loss1 = torch.sqrt(loss1)
            loss = torch.mean(loss1)
            losses.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses_train.append(np.mean(losses))# We take means since a loss function should return a single real number
        losses_train_med.append(np.median(losses))

        with torch.no_grad():
            model.eval()
            
            losses = []
            for i, data in enumerate(validloader):
                X_v       = data[0].to(device)
                    
                Y_v_batch = data[1].to(device)
                Y_v_pred = model(X_v).to(device)
                As=torch.exp(X_v[:,As_pos]*X_std[0,As_pos]+X_mean[0,As_pos])
                exptau=torch.exp(2*X_v[:,tau_pos]*X_std[0,tau_pos]+2*X_mean[0,tau_pos])
                Y_v_pred_back = Y_v_pred*Y_std+Y_mean
                Y_v_batch = Y_v_batch/As[:,None]*exptau[:,None]
                v_diff = (Y_v_batch - Y_v_pred_back)
                loss1 = torch.diag(v_diff @ covinv @ torch.t(v_diff))# implement with torch.einsum
                loss1 = torch.sqrt(loss1)
                loss_vali = torch.mean(loss1)
                losses.append(loss_vali.cpu().detach().numpy())

            losses_vali.append(np.mean(losses))
            losses_vali_med.append(np.median(losses))

            if reduce_lr == True:
                print('Reduce LR on plateu: ',reduce_lr)
                scheduler.step(losses_vali[n])
            

        print('epoch {}, loss={}, validation loss={}, lr={}, wd={})'.format(
                            n,
                            losses_train[-1],
                            losses_vali[-1],
                            optimizer.param_groups[0]['lr'],
                            optimizer.param_groups[0]['weight_decay']  
                        ))

    torch.save(model.state_dict(), PATH+'.pt')
#    CUDA_VISIBLE_DEVICES=0 python emultraincmb.py \
#    --mode 'tt' \
#    --batch 512 \
#    --epoch 700 \
#    --Path 'model' \