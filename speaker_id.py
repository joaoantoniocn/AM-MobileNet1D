# speaker_id.py
# ------------------
# Created by
# Mirco Ravanelli 
# Mila - University of Montreal 

# July 2018
# ------------------
# Modified by
# Joao Antonio Chagas Nunes
# Universidade Federal de Pernambuco

# Add AM-Softmax loss function
# December 2018
# ------------------
# Modified by
# Joao Antonio Chagas Nunes
# Universidade Federal de Pernambuco

# Add MobileNet1D and AM-MobileNet1D
# January 2020
# ------------------

# Description: 
# This code performs a speaker_id experiments with MobileNet1D and AM-MobileNet1D.
 
# How to run it:
# python speaker_id.py --cfg=cfg/AM_MobileNet1D_TIMIT.cfg

#

import os
import torch, torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import numpy as np
from data_io import ReadList,read_conf,str_to_bool
from mobilenet1d import MobileNetV2
from tqdm import tqdm

def create_batches_rnd(batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp):
    
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    #sig_batch_spec = np.zeros([batch_size, 1, 201, 17])
    sig_batch=np.zeros([batch_size, 1, wlen])
    lab_batch=np.zeros(batch_size)


    snt_id_arr=np.random.randint(N_snt, size=batch_size)
 
    rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)

    for i in range(batch_size):

        [signal, fs] = torchaudio.load(data_folder+wav_lst[snt_id_arr[i]])   # reading with torchaudio
        # -----

        snt_len = signal.shape[1]       # when reading audio from torchaudio
        snt_beg=np.random.randint(snt_len-wlen-1) #randint(0, snt_len-2*wlen-1)
        snt_end=snt_beg+wlen
  
        sig_batch[i,:]= signal[:, snt_beg:snt_end]*rand_amp_arr[i]        # when reading with psoundfile
        lab_batch[i]=lab_dict[wav_lst[snt_id_arr[i]]]


    inp=Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())

    lab= Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())


    return inp,lab


class AdditiveMarginSoftmax(nn.Module):
    # AMSoftmax
    def __init__(self, margin=0.35, s=30):
        super().__init__()

        self.m = margin #
        self.s = s
        self.epsilon = 0.000000000001
        print('AMSoftmax m = ' + str(margin))

    def forward(self, predicted, target):

        # ------------ AM Softmax ------------ #
        predicted = predicted / (predicted.norm(p=2, dim=0) + self.epsilon)
        indexes = range(predicted.size(0))
        cos_theta_y = predicted[indexes, target]
        cos_theta_y_m = cos_theta_y - self.m
        exp_s = np.e ** (self.s * cos_theta_y_m)

        sum_cos_theta_j = (np.e ** (predicted * self.s)).sum(dim=1) - (np.e ** (predicted[indexes, target] * self.s))

        log = -torch.log(exp_s/(exp_s+sum_cos_theta_j+self.epsilon)).mean()

        return log



# Reading cfg file
options=read_conf()


#[data]
tr_lst=options.tr_lst
te_lst=options.te_lst
pt_file=options.pt_file
class_dict_file=options.lab_dict
data_folder=options.data_folder+'/'
output_folder=options.output_folder

#[windowing]
fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)


#[class]
class_lay=list(map(int, options.class_lay.split(',')))
class_drop=list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act=list(map(str, options.class_act.split(',')))


#[optimization]
lr=float(options.lr)
batch_size=int(options.batch_size)
begin_epochs=int(options.begin_epochs)
N_epochs=int(options.N_epochs)
N_batches=int(options.N_batches)
N_eval_epoch=int(options.N_eval_epoch)
seed=int(options.seed)


# training list
wav_lst_tr=ReadList(tr_lst)
snt_tr=len(wav_lst_tr)

# test list
wav_lst_te=ReadList(te_lst)
snt_te=len(wav_lst_te)


# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder) 
    
    
# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
if (options.AMSoftmax == 'True'):
    print('Using AMSoftmax loss function...')
    cost = AdditiveMarginSoftmax(margin=float(options.AMSoftmax_m))

else:
    print('Using Softmax loss function...')
    cost = nn.NLLLoss()


  
# Converting context and shift in samples
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)

# Batch_dev
Batch_dev=batch_size


# Loading label dictionary
lab_dict=np.load(class_dict_file).item()


# ----- MobileNet -----
MOBILENET_net = MobileNetV2(num_classes=class_lay)
MOBILENET_net.cuda()
# ----- MobileNet -----

if pt_file!='none':
    checkpoint_load = torch.load(pt_file)
    MOBILENET_net.load_state_dict(checkpoint_load['MOBILENET_model_par'])

# ----- MobileNet -----
optimizer_MOBILENET = optim.RMSprop(MOBILENET_net.parameters(), lr=lr,alpha=0.95, eps=1e-8)
# ----- MobileNet -----

# recording time
begin = time.time()
time_batch = []

for epoch in range(begin_epochs, N_epochs):
  
    test_flag=0


    # ----- MobileNet -----
    MOBILENET_net.train()
    # ----- MobileNet -----
 
    loss_sum=0
    err_sum=0

    for i in tqdm(range(N_batches)):

        [inp,lab]=create_batches_rnd(batch_size,data_folder,wav_lst_tr,snt_tr,wlen,lab_dict,0.2)

        # ----- MobileNet -----
        pout = MOBILENET_net(inp)
        # ----- MobileNet -----
    
        pred=torch.max(pout,dim=1)[1]
        loss = cost(pout, lab.long())

        err = torch.mean((pred!=lab.long()).float())

        # ----- MobileNet -----
        optimizer_MOBILENET.zero_grad()
        # ----- MobileNet -----

        loss.backward()

        # ----- MobileNet -----
        optimizer_MOBILENET.step()
        # ----- MobileNet -----
    
        loss_sum=loss_sum+loss.detach()
        err_sum=err_sum+err.detach()
 

    loss_tot=loss_sum/N_batches
    err_tot=err_sum/N_batches
  
 
   
   
    # Full Validation  new
    if epoch%N_eval_epoch==0:

        # ----- MobileNet -----
        MOBILENET_net.eval()
        # ----- MobileNet -----
        test_flag=1
        loss_sum=0
        err_sum=0
        err_sum_snt=0
   
        with torch.no_grad():
            for i in range(snt_te):

                [signal, fs] = torchaudio.load(data_folder+wav_lst_te[i])  # reading with torchaudio

                lab_batch=lab_dict[wav_lst_te[i]]
    
                # split signals into chunks
                beg_samp=0
                end_samp=wlen

                N_fr=int((signal.shape[1]-wlen)/(wshift))           # when loading with torchaudio

                sig_arr=np.zeros([Batch_dev, 1, wlen])

                lab= Variable((torch.zeros(N_fr+1)+lab_batch).cuda().contiguous().long())
                pout=Variable(torch.zeros(N_fr+1,class_lay[-1]).float().cuda().contiguous())
                count_fr=0
                count_fr_tot=0
                while end_samp<signal.shape[1]:

                    sig_arr[count_fr, :] = signal[:, beg_samp:end_samp]

                    beg_samp=beg_samp+wshift
                    end_samp=beg_samp+wlen
                    count_fr=count_fr+1
                    count_fr_tot=count_fr_tot+1
                    if count_fr==Batch_dev:
                        inp=Variable(torch.from_numpy(sig_arr).float().cuda().contiguous())

                        # ----- MobileNet -----
                        time_begin = time.time()
                        pout[count_fr_tot - Batch_dev:count_fr_tot, :] = MOBILENET_net(inp)
                        time_end = time.time()
                        time_batch.append([inp.shape[0], time_end-time_begin])
                        # ----- MobileNet -----
                        count_fr=0
                        sig_arr=np.zeros([Batch_dev,1, wlen])
   
                if count_fr>0:
                    inp=Variable(torch.from_numpy(sig_arr[0:count_fr]).float().cuda().contiguous())

                    # ----- MobileNet -----
                    time_begin = time.time()
                    pout[count_fr_tot - count_fr:count_fr_tot, :] = MOBILENET_net(inp)
                    time_end = time.time()
                    time_batch.append([inp.shape[0], time_end - time_begin])
                    # ----- MobileNet -----
                pred=torch.max(pout,dim=1)[1]
                loss = cost(pout, lab.long())

                err = torch.mean((pred!=lab.long()).float())
    
                [val,best_class]=torch.max(torch.sum(pout,dim=0),0)
                err_sum_snt=err_sum_snt+(best_class!=lab[0]).float()
    
    
                loss_sum=loss_sum+loss.detach()
                err_sum=err_sum+err.detach()
    
            err_tot_dev_snt=err_sum_snt/snt_te
            loss_tot_dev=loss_sum/snt_te
            err_tot_dev=err_sum/snt_te

        final = time.time()
        print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f time=%f" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt, final-begin))
  
        with open(output_folder+"/res.res", "a") as res_file:
            res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f time=%f\n" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt,final-begin))


        # ----- MobileNet -----
        checkpoint = {'MOBILENET_model_par': MOBILENET_net.state_dict()}
        # ----- MobileNet -----
        torch.save(checkpoint,output_folder+'/model_raw_'+ str(epoch) +'.pkl')
  
    else:
        final = time.time()
        print("epoch %i / %i, loss_tr=%f err_tr=%f time=%f" % (epoch, N_epochs, loss_tot,err_tot, final-begin))
