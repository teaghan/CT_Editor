import numpy as np
import os
import h5py
import argparse
from scipy import ndimage
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("model_name", help="Name of model.", type=str)

    # Optional arguments
    
    # How often to display the losses
    parser.add_argument("-v", "--verbose_iters", 
                        help="Number of batch  iters after which to evaluate val set and display output.", 
                        type=int, default=10000)
    
    # How often to display save the model
    parser.add_argument("-ct", "--cp_time", 
                        help="Number of minutes after which to save a checkpoint.", 
                        type=float, default=15)
    # Alternate data directory than cycgan/data/
    parser.add_argument("-dd", "--data_dir", 
                        help="Different data directory from ct dir.", 
                        type=str, default=None)
    
    # Parse arguments
    args = parser.parse_args()
    
    return args

def batch_to_cuda(batch):
    for k in batch.keys():
        batch[k] = batch[k].cuda()
    return batch

def normalize(image, min_bound=-1000., max_bound=1500.):
    '''Normalize between -1 and 1'''
    image = 2*(image - min_bound) / (max_bound - min_bound) - 1
    image = np.clip(image, -1, 1)
    return image

def denormalize(image, min_bound=-1000., max_bound=1500.):
    '''Normalize between -1 and 1'''
    image = ((image + 1) * (max_bound - min_bound) / 2 ) + min_bound
    return image

def random_perturb(Xbatch,rotate=False): 
    # Apply some random transformations...
    swaps = np.random.choice([-1,1],size=(3,))
    Xcpy = Xbatch.copy()
    Xcpy = Xbatch[:,::swaps[0],::swaps[1],::swaps[2]]
    txpose = np.random.permutation([2,3])
    Xcpy = np.transpose(Xcpy, tuple([0,1] + list(txpose)))
    if rotate:
        # Arbitrary rotation is composition of two
        Xcpy[0] = ndimage.interpolation.rotate(Xcpy[0], 
                                               np.random.uniform(-5, 5), 
                                               axes=(1,0), order=1,
                                               reshape=False, cval=-1000,
                                               mode='nearest')
        Xcpy[0] = ndimage.interpolation.rotate(Xcpy[0], 
                                               np.random.uniform(-5, 5), 
                                               axes=(2,1), order=1,
                                               reshape=False, cval=-1000,
                                               mode='nearest')
            
    return Xcpy

class CTDataset(Dataset):
    
    """
    Data loader for the CT segments
    """

    def __init__(self, h5_file):
        self.h5_file = h5_file
        
    def __len__(self):
        with h5py.File(self.h5_file, "r") as f:
            num_samples =  len(f['Segment'])
        return num_samples
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, "r") as f:
            # Load segment
            seg = f['Segment'][idx].reshape(1, *f['Shape'][idx]).astype(np.float32)
            # Apply random augmentation
            seg = random_perturb(seg, rotate=True)
            # Convert to torch tensor
            seg = torch.from_numpy(seg.copy())
            
        # Normalize between -1 and 1
        seg = normalize(seg)
            
        return {'x':seg} 
    
def run_reg_iter(refiner, fake_sample, self_regularization_loss):
    ref_sample = refiner(fake_sample)
    reg_loss = self_regularization_loss(ref_sample, fake_sample)
    return reg_loss

def run_dis_iter(refiner, discriminator, fake_sample, true_sample, local_adversarial_loss, use_cuda):
    # Refine sample
    ref_sample = refiner(fake_sample)
    # Predict discriminator on refined sample
    cla_fake = discriminator(ref_sample.detach())
    # Predict discriminator on true sample
    cla_true = discriminator(true_sample)
    
    # Discriminator targets
    tgt_fake = torch.zeros(cla_fake.size()[0], dtype=torch.long)
    tgt_true = torch.ones(cla_true.size()[0], dtype=torch.long)
    
    # Switch to GPU
    if use_cuda:
        tgt_fake = tgt_fake.cuda()
        tgt_true = tgt_true.cuda()
        
    # Evaluate loss
    dis_loss_fake = local_adversarial_loss(cla_fake, tgt_fake)
    dis_loss_true = local_adversarial_loss(cla_true, tgt_true)
    dis_acc_fake = torch.mean(cla_fake[:,0])
    dis_acc_true = torch.mean(cla_true[:,1])
    
    return dis_loss_fake, dis_loss_true, dis_acc_fake, dis_acc_true

def run_gen_iter(refiner, discriminator, fake_sample, local_adversarial_loss, self_regularization_loss, use_cuda):
    
    # Refine sample
    ref_sample = refiner(fake_sample)
    # Predict discriminator on refined sample
    cla_fake = discriminator(ref_sample)
    
    # Discriminator targets
    tgt_fake = torch.ones(cla_fake.size()[0], dtype=torch.long)
    # Switch to GPU
    if use_cuda:
        tgt_fake = tgt_fake.cuda()
        
    # Evaluate regularization loss
    reg_loss = self_regularization_loss(ref_sample, fake_sample)
    
    # Evaluate adversarial loss
    gen_loss = local_adversarial_loss(cla_fake, tgt_fake)
    gen_acc = torch.mean(cla_fake[:,1])
    
    return gen_loss, reg_loss, gen_acc

def plot_samples(refiner, plot_fake_samples, plot_true_samples, model_name, 
                 val_results_dir, cur_iter, batchsize, ds):
    # Plot 5 samples
    plot_ref_samples = []
    for i, sample in enumerate(plot_fake_samples):
        plot_ref_samples.append(refiner(sample).cpu().data.numpy()[0,0])
        plot_fake_samples[i] = sample.cpu().data.numpy()[0,0]

    fig = plt.figure(figsize=(20, 12))
    outer = gridspec.GridSpec(3, 5, wspace=0.2, hspace=0.2)
    for i, (orig_seg, ref_seg, true_seg) in enumerate(zip(plot_fake_samples, plot_ref_samples, plot_true_samples)):
        # Plot original segment
        inner_orig = gridspec.GridSpecFromSubplotSpec(4, 4, 
                                                      subplot_spec=outer[0, i], 
                                                      wspace=0., hspace=0.)
        for cur_slice, _ in enumerate(inner_orig):
            ax = plt.Subplot(fig, inner_orig[cur_slice])
            if cur_slice<len(orig_seg):
                ax.imshow(orig_seg[cur_slice], cmap=plt.cm.bone, vmin=-1, vmax=0.4)
            ax.tick_params(axis='both', which='both',
                           bottom=False, top=False, labelbottom=False,
                           left=False, right=False, labelleft=False)
            ax.axis('off')
            if ((i==2) & (cur_slice==2)):
                ax.set_title('Original Segments', fontsize=25)

            fig.add_subplot(ax)

        # Plot refined segment
        inner_ref = gridspec.GridSpecFromSubplotSpec(4, 4, 
                                                      subplot_spec=outer[1, i], 
                                                      wspace=0., hspace=0.)
        for cur_slice, _ in enumerate(inner_ref):
            ax = plt.Subplot(fig, inner_ref[cur_slice])
            if cur_slice<len(ref_seg):
                ax.imshow(ref_seg[cur_slice], cmap=plt.cm.bone, vmin=-1, vmax=0.4)
            ax.tick_params(axis='both', which='both',
                           bottom=False, top=False, labelbottom=False,
                           left=False, right=False, labelleft=False)
            ax.axis('off')
            if ((i==2) & (cur_slice==2)):
                ax.set_title('Edited Segments', fontsize=25)
            fig.add_subplot(ax)

        # Plot true segment
        inner_true = gridspec.GridSpecFromSubplotSpec(4, 4, 
                                                      subplot_spec=outer[2, i], 
                                                      wspace=0., hspace=0.)
        for cur_slice, _ in enumerate(inner_true):
            ax = plt.Subplot(fig, inner_true[cur_slice])
            if cur_slice<len(true_seg):
                ax.imshow(true_seg[cur_slice], cmap=plt.cm.bone, vmin=-1, vmax=0.4)
            ax.tick_params(axis='both', which='both',
                           bottom=False, top=False, labelbottom=False,
                           left=False, right=False, labelleft=False)
            ax.axis('off')
            if ((i==2) & (cur_slice==2)):
                ax.set_title('True Segments', fontsize=25)
            fig.add_subplot(ax)
    fig.text(0,0.9, '%s, %s, batch iter: %s'% (model_name, ds, int(cur_iter//batchsize)), fontsize=20)
    plt.savefig(os.path.join(val_results_dir, '%s_%s_batchiter_%s.png'%(model_name, ds, int(cur_iter//batchsize))), 
                transparent=True)
    plt.close('all')