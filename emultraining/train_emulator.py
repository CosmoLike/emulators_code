import torch, os, sys, yaml
import numpy as np
from datetime import datetime
from emulator import ResTRF, ResCNN
import h5py as h5
import argparse
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#mpirun -n 2 --oversubscribe \
#  python external_modules/code/emulators/emultrf/emultraining/train_emulator.py \
#    --root projects/roman_real/  \
#    --fileroot emulators/nla_cosmic_shear/ \
#    --nparams 1000 \
#    --temp 128 \
#    --yaml 'w0wa_takahashi_cs_CNN.yaml' \
#    --datavsfile 'w0wa_takahashi_dvs_train' \
#    --paramfile 'w0wa_takahashi_params_train' \
#    --failfile  'w0wa_params_failed_train' \
#    --chain 1 \
#    --maxcorr 0.15 \
#    --loadchk 1
#    --freqchk 2000
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Command line args
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='train_emulator')
parser.add_argument("--yaml",
                    dest="yaml",
                    help="The training YAML containing the training_args block",
                    type=str,
                    required=True)
parser.add_argument("--root",
                    dest="root",
                    help="Project folder",
                    type=str,
                    required=True)
parser.add_argument("--fileroot",
                    dest="fileroot",
                    help="Subfolder of Project folder where we find yaml and fisher",
                    type=str,
                    required=True)
parser.add_argument("--epochs", 
                    dest="nepochs",
                    help="(int) number of training epochs",
                    type=int,
                    required=True)
parser.add_argument("--batchsize"
                    dest="bs",
                    help="batch size to use while training",
                    type=int,
                    required=True)
parser.add_argument("--learningrate", 
                    dest="lr",
                    help="learning rate to use while training. Default=1e-3",
                    type=float,
                    required=True)
parser.add_argument("--weight_decay",
                    dest="wd",
                    help="weight decay to use while training",
                    type=float,
                    default=0.0,
                    nargs='?')
parser.add_argument("--save_losses",
                    dest="sl",
                    help="save losses at each epoch. Default=False",
                    type=bool,
                    required=True)
parser.add_argument("--mask"
                    dest="mask",
                    help="mask the training data (mask define in dataset file)",
                    type=bool,
                    required=True)

args, unknown     = parser.parse_known_args()
cobaya_yaml       = args.yaml
probe             = args.probe
nepochs           = args.nepochs
batch_size        = args.bs
learning_rate     = args.lr
weight_decay      = args.wd
save_losses       = args.sl
training_masked   = args.mask

#===================================================================================================
# covariance matrix read from file specified in the .dataset file specified in YAML
    
def get_cov(train_yaml, masked=False):
    with open(train_yaml,'r') as stream:
        config_args = yaml.safe_load(stream)

    lkl_args = config_args['likelihood'] # dataset file with dv_fid, mask, etc.
    _lkl = lkl_args[list(lkl_args.keys())[0]] # get for dataset file
    path = _lkl['path']
    data_file = _lkl['data_file']
    data = open(path+'/'+data_file, 'r')
 
    for line in data.readlines():
        split = line.split()
        # need: dv_fid, cov, mask.
        if(split[0] == 'cov_file'):
            cov_file = split[2]

    full_cov = np.loadtxt(path+'/'+cov_file)
    cov_scenario = full_cov.shape[1]
    size = int(np.max(full_cov[:])+1)

    cov = np.zeros((size,size))

    for line in full_cov:
        i = int(line[0])
        j = int(line[1])

        if(cov_scenario == 3):
            cov_ij = line[2]
        elif(cov_scenario == 4):
            cov_ij = line[2]+line[3]
        elif(cov_scenario == 10):
            cov_ij = line[8]+line[9]

        cov[i,j] = cov_ij
        cov[j,i] = cov_ij

    cov_inv = np.linalg.inv(cov)

    if ( masked==False ):
        return cov

    else:
        mask = get_mask(train_yaml).astype(bool)
        cov_masked = cov[mask][:,mask]
        cov_inv_masked = np.zeros((size,size))
        

        for i in range(size):
            for j in range(i,size):
                if(i!=j):
                    mask_row    = mask[i]
                    mask_column = mask[j]

                    cov_inv_masked[i,j] = cov[i,j] * mask_row * mask_column
                    cov_inv_masked[j,i] = cov_inv_masked[i,j]
                else:
                    cov_inv_masked[i,j] = cov[i,j]

        cov_inv_masked = np.linalg.inv(cov_inv_masked)[mask][:,mask]

        return np.linalg.inv(cov_inv_masked)

def get_mask(train_yaml):
    with open(train_yaml,'r') as stream:
        config_args = yaml.safe_load(stream)

    lkl_args = config_args['likelihood'] # dataset file with dv_fid, mask, etc.
    _lkl = lkl_args[list(lkl_args.keys())[0]] # get for dataset file
    path = _lkl['path']
    data_file = _lkl['data_file']
    data = open(path+'/'+data_file, 'r')
 
    for line in data.readlines():
        split = line.split()
        # need: dv_fid, cov, mask.
        if(split[0] == 'mask_file'):
            mask_file = split[2]

    mask = np.loadtxt(path+'/'+mask_file)
    idxs = np.argsort(mask[:,0])

    return mask[:,1][idxs]

def get_datavector(train_yaml):
    with open(train_yaml,'r') as stream:
        config_args = yaml.safe_load(stream)

    lkl_args = config_args['likelihood'] # dataset file with dv_fid, mask, etc.
    _lkl = lkl_args[list(lkl_args.keys())[0]] # get for dataset file
    path = _lkl['path']
    data_file = _lkl['data_file']
    data = open(path+'/'+data_file, 'r')
 
    for line in data.readlines():
        split = line.split()
        # need: dv_fid, cov, mask.
        if(split[0] == 'data_file'):
            dv_file = split[2]

    datavector = np.loadtxt(path+'/'+dv_file)
    idxs = np.argsort(datavector[:,0])

    return datavector[:,1][idxs]

#===================================================================================================
# a progress bar to display while training. I find tqdm to be a little strange looking.

def progress_bar(train_loss, valid_loss, start_time, epoch, total_epochs, optim):
    ''' 
    simple progress bar for training the emulator
    '''

    elapsed_time = int((datetime.now() - start_time).total_seconds())
    lr = optim.param_groups[0]['lr']
    epoch=epoch+1

    width = 20
    factor = int( width * (epoch/total_epochs) )
    bar = '['
    for i in range(width):
        if i < factor:
            bar += '#'
        else:
            bar += ' '
    bar += ']'

    remaining_time = int((elapsed_time / (epoch)) * (total_epochs - (epoch)))

    print('\r' + bar + ' ' +                                \
          f'Epoch {epoch:3d}/{total_epochs:3d} | ' +        \
          f'loss={train_loss:1.3e}({valid_loss:1.3e}) | ' + \
          f'lr={lr:1.2e} | ' +                              \
          f'time elapsed={elapsed_time:7d} s; time remaining={remaining_time:7d} s',end='')
    sys.stdout.flush()

#===================================================================================================
# training routine

def train_emulator(train_yaml, probe,
            nepochs=250, batch_size=32, learning_rate=1e-3, weight_decay=0, 
            save_losses=False):
    '''
    routine to train an emulator. 

    string  train_yaml: the training YAML file. See 'projects/lsst_y1/EXAMPLE_TRAIN.yaml'
    boolean save_losses: save the training and validation loss to a text file 'losses_*.txt'
    '''
    print('')
    print('Probe =', probe)

    # TODO: get indices from cosmolike
    if probe=='galaxy_galaxy_lensing':
        start = 1080
        stop = 1995

    elif probe=='cosmic_shear':
        start = 0
        stop = 1080
        

    elif probe=='galaxy_clustering':
        start=1995
        stop = 2115

    else:
        raise NotImplementedError
    dv_idxs = np.array(range(start, stop))
    # if we want to train with a mask, get the mask, then we need to fix the sizes
    mask = np.ones(stop-start)
    
    # open the config file and get the arguments we want.
    with open(train_yaml,'r') as stream:
        args = yaml.safe_load(stream)

    # get training and validation_data
    if args['train_args']['training_data_path'][0] == '/':
        PATH = args['train_args']['training_data_path']
    else:
        PATH = os.environ.get("ROOTDIR") + '/' + args['train_args']['training_data_path']

    # get saving filenames      
    model_filename = args['train_args'][probe]['extra_args']['file'][0]
    extra_filename = args['train_args'][probe]['extra_args']['extra'][0]

    print('Model will be saved to:',model_filename,'and',extra_filename)
    print('')

    covmat_file      = PATH + args['train_args']['data_covmat_file']

    train_datavectors_file = PATH + args['train_args']['train_datavectors_file']
    train_parameters_file  = PATH + args['train_args']['train_parameters_file']

    valid_datavectors_file = PATH + args['train_args']['valid_datavectors_file']
    valid_parameters_file  = PATH + args['train_args']['valid_parameters_file']

    test_datavectors_file = PATH + args['train_args']['test_datavectors_file']
    test_parameters_file  = PATH + args['train_args']['test_parameters_file']

    print('Loading training data from:')
    print(train_datavectors_file)
    print(train_parameters_file)
    print('Loading validation data from:')
    print(valid_datavectors_file)
    print(valid_parameters_file)
    print('Loading testing data from:')
    print(test_datavectors_file)
    print(test_parameters_file)
    print('')

    # get the parameters
    sampled_params = args['train_args'][probe]['extra_args']['ord'][0]
    sampling_dim = len(sampled_params)

    print('Parameters are:')
    print(sampled_params)
    print('')

    # get device
    device = args['train_args'][probe]['extra_args']['device']

    # get model
    model_info = args['train_args'][probe]['extra_args']['extrapar'][0]

    if( 'TRF' == model_info['MLA'] ):
        model = ResTRF(sampling_dim, 
            int(np.sum(mask)),#model_info['OUTPUT_DIM'], 
            model_info['INT_DIM_RES'], 
            model_info['INT_DIM_TRF'],
            model_info['NC_TRF'])
    elif( 'MLP' == model_info['MLA'] ):
        model = ResMLP(sampling_dim,
            int(np.sum(mask)),#model_info['OUTPUT_DIM'],
            model_info['INT_DIM_RES'])
    elif( 'CNN' == model_info['MLA'] ):
        model = ResCNN(sampling_dim,
            int(np.sum(mask)),
            model_info['INT_DIM_RES'], 
            model_info['CNN_DIM'],
            model_info['KERNEL_DIM'])
    else:
        raise NotImplementedError
    print('Model output shape (# unmasked elements) =',int(np.sum(mask)))
    print('')

    # load and preprocess the data
    print('Processing the data. May take some time...')

    # read the header from the train_parameters:
    f_train = open(train_parameters_file)
    train_params = np.array(f_train.readline().split(' ')[1:])
    train_params[-1] = train_params[-1][:-1] # because the last param has a \n

    f_valid = open(valid_parameters_file)
    valid_params = np.array(f_valid.readline().split(' ')[1:])
    valid_params[-1] = valid_params[-1][:-1] # because the last param has a \n

    f_test = open(test_parameters_file)
    test_params = np.array(f_test.readline().split(' ')[1:])
    test_params[-1] = test_params[-1][:-1] # because the last param has a \n

    train_idxs = []
    valid_idxs = []
    test_idxs  = []
    
    for p in sampled_params:
        train_idxs.append(np.where(train_params==p)[0][0])
        valid_idxs.append(np.where(valid_params==p)[0][0])
        test_idxs.append(np.where(test_params==p)[0][0])

    # load the data of the given train_prefix and valid_prefix. Leave on cpu to save vram!
    # overwrite with my own manually so I can combine files. Add this functionality later!
    #print(f_train)
    #x_train = torch.as_tensor(f_train)

    x_train = torch.as_tensor(np.loadtxt(train_parameters_file)[:,train_idxs],dtype=torch.float32)
    y_train = torch.as_tensor(np.load(train_datavectors_file)[:,dv_idxs],dtype=torch.float32)

    x_valid = torch.as_tensor(np.loadtxt(valid_parameters_file)[:,valid_idxs],dtype=torch.float32)
    y_valid = torch.as_tensor(np.load(valid_datavectors_file)[:,dv_idxs],dtype=torch.float32)

    x_test = torch.as_tensor(np.loadtxt(test_parameters_file)[:,test_idxs],dtype=torch.float32)
    y_test = torch.as_tensor(np.load(test_datavectors_file)[:,dv_idxs],dtype=torch.float32)

    # convert data
    covmat = torch.as_tensor(get_cov(train_yaml,masked=training_masked)[start:stop, start:stop],dtype=torch.float32)
    #dv_fid = torch.as_tensor(torch.mean(y_train[dv_idxs],axis=0),dtype=torch.float64)
    dv_fid = torch.as_tensor(get_datavector(train_yaml)[dv_idxs],dtype=torch.float32)

    # normalize the input parameters
    samples_mean = torch.Tensor(x_train.mean(axis=0, keepdims=True))
    samples_std  = torch.Tensor(x_train.std(axis=0, keepdims=True))

    x_train = torch.div( (x_train - samples_mean), 5.0*samples_std)
    x_valid = torch.div( (x_valid - samples_mean), 5.0*samples_std)
    x_test  = torch.div( (x_test  - samples_mean), 5.0*samples_std)

    # samples_cov  = torch.Tensor(torch.cov(torch.t(x_train)))
    # s_evals, s_evecs = torch.linalg.eigh(samples_cov)
    # x_train = torch.div((x_train - samples_mean) @ s_evecs, 8.0 * torch.sqrt(s_evals))
    # x_valid = torch.div((x_valid - samples_mean) @ s_evecs, 8.0 * torch.sqrt(s_evals))
    # x_test  = torch.div((x_test  - samples_mean) @ s_evecs, 8.0 * torch.sqrt(s_evals))

    # x_scale = torch.max(x_train) - torch.min(x_train)
    # x_train = x_train/x_scale
    # x_valid = x_valid/x_scale
    # x_test  = x_test /x_scale



    # diagonalize the training datavectors
    dv_evals, dv_evecs = torch.linalg.eigh(covmat)
    #inv_covmat = torch.diag(1/dv_evals).type(torch.float32).to(device)

    y_train = torch.div( (y_train - dv_fid) @ dv_evecs, torch.sqrt(dv_evals))
    y_valid = torch.div( (y_valid - dv_fid) @ dv_evecs, torch.sqrt(dv_evals))
    y_test  = torch.div( (y_test  - dv_fid) @ dv_evecs, torch.sqrt(dv_evals))

    y_scale = 1.0#torch.max(y_train) - torch.min(y_train)
    #print(y_scale)

    # convert to float32
    x_train = torch.as_tensor(x_train,dtype=torch.float32)
    y_train = torch.as_tensor(y_train,dtype=torch.float32)
    x_valid = torch.as_tensor(x_valid,dtype=torch.float32)
    y_valid = torch.as_tensor(y_valid,dtype=torch.float32)
    x_test  = torch.as_tensor(x_test, dtype=torch.float32)
    y_test  = torch.as_tensor(y_test, dtype=torch.float32)

    # I'm going to try adding a new transform that just rescales everything to be between 0 and 1.
    #rescale = torch.max(y_train) - torch.min(y_train)
    #rescale = rescale.to(device)
    #print(rescale.get_device())
    output_dim = int(np.sum(mask))

    # setup ADAM optimizer and reduce_lr scheduler
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min',patience=10,factor=0.5)
     
    # load the data into loaders
    model.to(device)

    generator = torch.Generator(device=device)
    trainset    = torch.utils.data.TensorDataset(x_train, y_train)
    validset    = torch.utils.data.TensorDataset(x_valid, y_valid)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)#, generator=generator)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)#, generator=generator)

    # begin training
    print('\rBegin training...',end='')
    train_start_time = datetime.now()

    losses_train = []
    losses_valid = []
    loss = 100.

    tmp_evecs = dv_evecs.to(device)
    tmp_evals = dv_evals.to(device)
    tmp_dvfid = dv_fid.to(device)
    tmp_invcovmat = torch.linalg.inv(covmat).to(device)

    for e in range(nepochs):
        model.train()

        # training loss
        losses = []
        for i, data in enumerate(trainloader):    
            X       = data[0].to(device)
            Y_batch = data[1].to(device)
            Y_pred  = model(X)*y_scale

            # PCA part
            diff = Y_batch - Y_pred
            chi2 = torch.bmm(diff.view(batch_size,1,output_dim), diff.view(batch_size, output_dim, 1))
            # i give up, lets just change basis back.
            #Y_pred  = torch.mul(Y_pred,  torch.sqrt(tmp_evals)) @ torch.t(tmp_evecs) + tmp_dvfid
            #Y_batch = torch.mul(Y_batch, torch.sqrt(tmp_evals)) @ torch.t(tmp_evecs) + tmp_dvfid
            #chi2 = torch.diag((Y_pred - Y_batch) @ tmp_invcovmat @ torch.t(Y_pred - Y_batch))

            # loss = torch.mean(chi2)                      # ordinary chi2
            # loss = torch.mean((1+2*chi2)**(1/2))-1       # hyperbola
            loss = torch.mean(chi2**(1/2))                 # sqrt(chi2)

            losses.append(loss.cpu().detach().numpy())

            optim.zero_grad()
            loss.backward()
            optim.step()

        losses_train.append(np.mean(losses))

        ###validation loss
        losses=[]
        with torch.no_grad():
            model.eval()
            losses = []
            for i, data in enumerate(validloader):  
                X_v       = data[0].to(device)
                Y_v_batch = data[1].to(device)
                Y_v_pred = model(X_v)*y_scale

                diff_v = Y_v_batch - Y_v_pred
                #chi2_v = torch.diag(diff_v @ torch.t(diff_v))
                chi2_v = torch.bmm(diff_v.view(batch_size,1,output_dim), diff_v.view(batch_size, output_dim, 1))

                # Y_v_pred  = torch.mul(Y_v_pred,  torch.sqrt(tmp_evals)) @ torch.t(tmp_evecs) + tmp_dvfid
                # Y_v_batch = torch.mul(Y_v_batch, torch.sqrt(tmp_evals)) @ torch.t(tmp_evecs) + tmp_dvfid
                # chi2_v = torch.diag((Y_v_pred - Y_v_batch) @ tmp_invcovmat @ torch.t(Y_v_pred - Y_v_batch))

                # loss_vali = torch.mean(chi2_v)                      # ordinary chi2
                # loss_vali = torch.mean((1+2*chi2_v)**(1/2))-1       # hyperbola
                loss_vali = torch.mean(chi2_v**(1/2))                 # sqrt(chi2)

                losses.append(float(loss_vali.cpu().detach().numpy()))

            losses_valid.append(np.mean(losses))

            scheduler.step(losses_valid[e])
            optim.zero_grad()


        progress_bar(losses_train[-1],losses_valid[-1],train_start_time, e, nepochs, optim)
    
    if ( save_losses ):
        np.savetxt("losses.txt", np.array([losses_train,losses_valid],dtype=np.float32))

    # save the model
    torch.save(model.state_dict(), model_filename)
    with h5.File(extra_filename, 'w') as f:
        f['sample_mean']  = samples_mean
        #f['s_evecs']       = s_evecs
        #f['s_evals']       = s_evals
        f['sample_std']   = 5.0*samples_std
        f['dv_fid']        = dv_fid
        f['dv_evals']      = torch.sqrt(dv_evals)
        f['dv_evecs']      = dv_evecs
        #f['rescaling']     = rescale
        f['train_params']  = sampled_params
        #f['model_info']    = model_info


    # now lets test the model
    print('')
    print('Testing the model...')
    print('')
    # Reminder:
    # C = UDU^{-1}
    # dv_norm = D^{-1/2} U^{-1} dv
    # where D is diagonal and U is orthogonal, so
    # dv_norm.T = dv.T U D^{-1/2}
    # 
    # chi2 = dv.T @ C^{-1} @ dv
    #      = dv.T @ UD^{-1}U^{-1} @ dv
    #      = dv.T U @ D^{-1/2} D^{-1/2} @ U^{-1} dv
    #      = (dv.T U D^{-1/2}) @ (D^{-1/2} U^{-1} dv)
    #      = dv_norm.T @ dv_norm
    # and dv_norm is just the model output!
    with torch.no_grad():
        model.eval()

        Y_t = model(x_test.to(device))*y_scale
        diff = y_test.to(device) - Y_t
        delta_chi2 = torch.bmm(diff.view(len(diff),1,output_dim), diff.view(len(diff), output_dim, 1)).detach()[:,0,0]
        #Y_pred  = torch.mul(Y_t,  torch.sqrt(dv_evals).to(device)) @ torch.t(dv_evecs.to(device)) + dv_fid.to(device)
        #Y_batch = torch.mul(y_test, torch.sqrt(dv_evals)) @ torch.t(dv_evecs) + dv_fid
        #delta_chi2 = torch.diag((Y_pred.to('cpu') - Y_batch) @ torch.linalg.inv(covmat) @ torch.t(Y_pred.to('cpu') - Y_batch))

        # want to do a manual check of chi2 to see if they are the same? # THEY ARE! so why is my NN performing so bad on real data?
        # y_test_data_basis = torch.mul(y_test, torch.sqrt(dv_evals))
        # y_test_data_basis = y_test_data_basis @ torch.t(dv_evecs) + dv_fid

        # Y_t_data_basis = torch.mul(Y_t.to('cpu'), torch.sqrt(dv_evals)) 
        # Y_t_data_basis = Y_t_data_basis @ torch.t(dv_evecs) + dv_fid
        # diff_data_basis = Y_t_data_basis - y_test_data_basis
        # delta_chi2_data_basis = torch.diag(diff_data_basis @ torch.linalg.inv(covmat) @ torch.t(diff_data_basis))
        # print(delta_chi2.to('cpu') - delta_chi2_data_basis)
        # print(torch.mean(delta_chi2))
        # print(torch.mean(delta_chi2_data_basis))

        chi2_g_1  = 0
        chi2_g_p2 = 0

        for c in delta_chi2:
            if( c>0.2 ):
                chi2_g_p2 += 1
            if( c>=1 ):
                chi2_g_1 += 1

        print('Testing results.')
        print('Mean   Delta Chi2 = {:1.3e}'.format(torch.mean(delta_chi2).cpu().detach().numpy()))
        print('Median Delta Chi2 = {:1.3e}'.format(torch.median(delta_chi2).cpu().detach().numpy()))
        print('N points with Chi2 > 1  :', chi2_g_1)
        print('N points with Chi2 > 0.2:', chi2_g_p2)

        # Done :)
        print('\nDone!')

        # for debugging purposes, I'm going to look for the BEST point and check chi2 against cocoa
        Y_t = model(x_test.to(device))
        diff = y_test.to(device) - Y_t
        delta_chi2 = torch.bmm(diff.view(len(diff),1,output_dim), diff.view(len(diff), output_dim, 1)).detach()[:,0,0]
        delta_chi2_testing = torch.min(delta_chi2)
        idx_testing = torch.where(delta_chi2==delta_chi2_testing)
        print(delta_chi2_testing,idx_testing)

        Y_v = model(x_valid.to(device))
        diff = y_valid.to(device) - Y_v
        delta_chi2 = torch.bmm(diff.view(len(diff),1,output_dim), diff.view(len(diff), output_dim, 1)).detach()[:,0,0]
        delta_chi2_valid = torch.min(delta_chi2)
        idx_valid = torch.where(delta_chi2==delta_chi2_valid)
        print(delta_chi2_valid,idx_valid)

        Y_t = model(x_train.to(device))
        diff = y_train.to(device) - Y_t
        delta_chi2 = torch.bmm(diff.view(len(diff),1,output_dim), diff.view(len(diff), output_dim, 1)).detach()[:,0,0]
        delta_chi2_train = torch.min(delta_chi2)
        idx_train = torch.where(delta_chi2==delta_chi2_train)
        print(delta_chi2_train,idx_train)

    return

if __name__ == "__main__":
    train_emulator(cobaya_yaml, probe, 
        nepochs, batch_size, learning_rate, weight_decay, 
        save_losses)
