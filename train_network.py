# added data augmentation
# import packages
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import h5py
from collections import defaultdict
import sys
import os
import glob
import time
import configparser
from distutils import util

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from network import Refiner, EdgeKernel, Discriminator
from training_fns import (CTDataset, parseArguments, batch_to_cuda, run_reg_iter, 
                          run_dis_iter, run_gen_iter, plot_samples)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(1)
torch.manual_seed(1)

# Check for GPU
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using GPU!')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed(1)
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Collect the command line arguments
args = parseArguments()
model_name = args.model_name
verbose_iters = args.verbose_iters
cp_time = args.cp_time
data_dir = args.data_dir

# Directories
cur_dir = os.path.dirname(__file__)
config_dir = os.path.join(cur_dir, 'configs/')
model_dir = os.path.join(cur_dir, 'models/')
progress_dir = os.path.join(cur_dir, 'progress/')
train_results_dir = os.path.join(cur_dir, 'results/', model_name+'/train/')
val_results_dir = os.path.join(cur_dir, 'results/', model_name+'/val/')
if not os.path.exists(val_results_dir):
    os.makedirs(val_results_dir)
    os.makedirs(train_results_dir)

if args.data_dir is None:
    data_dir = os.path.join(cur_dir, 'data/')

# Model configuration
config = configparser.ConfigParser()
config.read(config_dir+model_name+'.ini')
architecture_config = config['ARCHITECTURE']
print('\nCreating model: %s'%model_name)
print('\nConfiguration:')
for key_head in config.keys():
    if key_head=='DEFAULT':
        continue
    print('  %s' % key_head)
    for key in config[key_head].keys():
        print('    %s: %s'%(key, config[key_head][key]))
        
# DATA FILES
real_train_file = os.path.join(data_dir, config['DATA']['nod_train_file'])
real_val_file = os.path.join(data_dir, config['DATA']['nod_val_file'])
fake_train_file = os.path.join(data_dir, config['DATA']['hlt_train_file'])
fake_val_file = os.path.join(data_dir, config['DATA']['hlt_val_file'])

# TRAINING PARAMETERS
batchsize = int(config['TRAINING']['batchsize'])
learning_rate_refiner = float(config['TRAINING']['learning_rate_refiner'])
learning_rate_discriminator = float(config['TRAINING']['learning_rate_discriminator'])
lr_decay_iters = float(config['TRAINING']['lr_decay_iters'])
lr_decay_rate = float(config['TRAINING']['lr_decay_rate'])
min_lr_ref = float(config['TRAINING']['min_lr_ref'])
min_lr_dis = float(config['TRAINING']['min_lr_dis'])
loss_weight_reg = float(config['TRAINING']['loss_weight_reg'])
init_reg_iters = float(config['TRAINING']['init_reg_iters'])
init_dis_iters = float(config['TRAINING']['init_dis_iters'])
gen_to_dis_iters = float(config['TRAINING']['gen_to_dis_iters'])
total_batch_iters = float(config['TRAINING']['total_batch_iters'])

# ARCHITECTURE PARAMETERS
refiner_num_blocks = int(config['ARCHITECTURE']['refiner_num_blocks'])
refiner_num_filters = int(config['ARCHITECTURE']['refiner_num_filters'])
refiner_filter_len = int(config['ARCHITECTURE']['refiner_filter_len'])
discrim_num_filters = eval(config['ARCHITECTURE']['discrim_num_filters'])
discrim_stride_len = eval(config['ARCHITECTURE']['discrim_stride_len'])
discrim_filter_len = int(config['ARCHITECTURE']['discrim_filter_len'])
fade_perc = float(config['ARCHITECTURE']['fade_perc'])

# Collect all possible shapes in the training set
with h5py.File(real_train_file, "r") as f:
    shapes = f['Shape'][:]
    shapes = np.unique(shapes,axis=0)
# Create Edge Kernel to be applied to output of refiner
edgekernel = EdgeKernel(shapes, fade_perc=fade_perc, use_cuda=use_cuda)

# BUILD THE NETWORKS

print('\nBuilding networks...')
refiner = Refiner(num_blocks=refiner_num_blocks, in_features=1, 
                  nb_features=refiner_num_filters, 
                  filter_len=refiner_filter_len, 
                  init=True, edge_kernel=edgekernel, use_cuda=use_cuda)
discriminator = Discriminator(nb_features=discrim_num_filters, 
                              stride_len=discrim_stride_len,
                              filter_len=discrim_filter_len, 
                              init=True, use_cuda=use_cuda)

# Display model architectures
print('\n\nREFINER ARCHITECTURE:\n')
print(refiner)
print('\n\nDISCRIMINATOR ARCHITECTURE:\n')
print(discriminator)

# Construct optimizers
optimizer_ref = torch.optim.SGD(refiner.parameters(), 
                                 lr=learning_rate_refiner, 
                                 momentum=0.9)
optimizer_dis = torch.optim.SGD(discriminator.parameters(), 
                                 lr=learning_rate_discriminator, 
                                 momentum=0.9)

# Learning rate decay
def make_lr_decay(init_lr, step_size, decay_rate, min_lr):
    ''' Step Learning Rate decay with a minimum lr '''
    def decay_fn(cur_iter):
        decayed_lr = init_lr*decay_rate**(cur_iter//step_size)
        return max(min_lr, decayed_lr)
    return decay_fn
decay_fn_ref = make_lr_decay(learning_rate_refiner, lr_decay_iters, lr_decay_rate, min_lr_ref)
decay_fn_dis = make_lr_decay(learning_rate_discriminator, lr_decay_iters, lr_decay_rate, min_lr_dis)
lr_scheduler_ref = torch.optim.lr_scheduler.LambdaLR(optimizer_ref, 
                                                     lr_lambda=decay_fn_ref)
lr_scheduler_dis = torch.optim.lr_scheduler.LambdaLR(optimizer_dis, 
                                                     lr_lambda=decay_fn_dis)

# Loss functions
self_regularization_loss = nn.L1Loss()
local_adversarial_loss = nn.CrossEntropyLoss()

# Check for pre-trained weights
model_filename =  os.path.join(model_dir,model_name+'.pth.tar')
if os.path.exists(model_filename):
    fresh_model = False
else:
    fresh_model = True
    
# Load pretrained model
if fresh_model:
    print('\nStarting fresh model to train...')
    cur_iter = 1
    losses = defaultdict(list)
    init_reg_complete = False
    init_dis_complete = False
else:
    print('\nLoading saved model to continue training...')
    # Load model info
    checkpoint = torch.load(model_filename, map_location=lambda storage, loc: storage)
    init_reg_complete = checkpoint['init_reg_complete']
    init_dis_complete = checkpoint['init_dis_complete']
    losses = defaultdict(list, dict(checkpoint['losses']))
    try:
        cur_iter = losses['batch_iters'][-1] * batchsize + 1
    except IndexError:
        cur_iter = 1
    
    # Load optimizer states
    optimizer_ref.load_state_dict(checkpoint['optimizer_ref'])
    optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
    lr_scheduler_ref.load_state_dict(checkpoint['lr_scheduler_ref'])
    lr_scheduler_dis.load_state_dict(checkpoint['lr_scheduler_dis'])
    
    # Load model weights
    refiner.load_state_dict(checkpoint['refiner'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    
# Data

# Training datasets to loop through
real_train_dataset = CTDataset(real_train_file)
real_train_dataloader = DataLoader(real_train_dataset, batch_size=1, shuffle=True, num_workers=6)
fake_train_dataset = CTDataset(fake_train_file)
fake_train_dataloader = DataLoader(fake_train_dataset, batch_size=1, shuffle=True, num_workers=6)
# Validation datasets to loop through
real_val_dataset = CTDataset(real_val_file)
real_val_dataloader = DataLoader(real_val_dataset, batch_size=1, shuffle=True, num_workers=6)
fake_val_dataset = CTDataset(fake_val_file)
fake_val_dataloader = DataLoader(fake_val_dataset, batch_size=1, shuffle=True, num_workers=6)

# Initial training of refiner using regularization loss only
if not init_reg_complete:
    refiner.train()
    discriminator.eval()
    
    losses_cp = defaultdict(list)
    
    print('Initial training of refiner using the self-regularization loss for %i iterations...' % init_reg_iters)
    cur_reg_iter = 1
    while cur_reg_iter <= (init_reg_iters*batchsize):
        for fake_sample in fake_train_dataloader:
            
            if use_cuda:
                fake_sample = batch_to_cuda(fake_sample)
                        
            # Evaluate self-regularization
            reg_loss = run_reg_iter(refiner, fake_sample['x'], self_regularization_loss)
            
            # Backpropogate loss
            reg_loss.backward()
            
            # Save loss value
            losses_cp['init_reg_loss_train'].append(reg_loss.cpu().data.numpy())
            
            if (cur_reg_iter%batchsize==0):
                # Adjust network weights
                optimizer_ref.step()
                optimizer_ref.zero_grad()            
            cur_reg_iter += 1
            
            if cur_reg_iter % (verbose_iters*batchsize) == 0: 
                # Run network on validation set
                refiner.eval()
                for fake_sample in fake_val_dataloader:
                    if use_cuda:
                        fake_sample = batch_to_cuda(fake_sample)
                    
                    # Evaluate self-regularization
                    reg_loss = run_reg_iter(refiner, fake_sample['x'], self_regularization_loss)
                    losses_cp['init_reg_loss_val'].append(reg_loss.cpu().data.numpy())
                
                # Calculate averages
                for k in losses_cp.keys():
                    losses[k].append(np.mean(losses_cp[k]))
                losses['init_reg_batch_iters'].append(cur_reg_iter/batchsize)

                # Print current status
                print('\nBatch Iterations: %i/%i ' % (cur_reg_iter/batchsize, init_reg_iters))
                print('\n\t|     Self-Reg    |')
                print('Train   |     %0.5f     |\nVal     |     %0.5f     |' % 
                      (losses['init_reg_loss_train'][-1], losses['init_reg_loss_val'][-1]))
                print('\n') 
                
                refiner.train()
                
                # Save losses to file to analyze throughout training. 
                np.save(os.path.join(progress_dir, model_name+'_losses.npy'), losses) 
                # Reset checkpoint loss dict
                losses_cp = defaultdict(list)
            
            if not (cur_reg_iter <= (init_reg_iters*batchsize)):
                break
    
    # Save model            
    init_reg_complete = True            
    torch.save({'losses': losses,
                'init_reg_complete': init_reg_complete,
                'init_dis_complete': init_dis_complete,
                'optimizer_ref' : optimizer_ref.state_dict(),
                'optimizer_dis' : optimizer_dis.state_dict(),
                'refiner' : refiner.state_dict(),
                'discriminator' : discriminator.state_dict(),
                'lr_scheduler_ref' : lr_scheduler_ref.state_dict(),
                'lr_scheduler_dis' : lr_scheduler_dis.state_dict()}, 
               model_filename)                
else:
    print('Initial training of refiner using the self-regularization loss already complete.')

# Initial training of discriminator using adversarial loss only
if not init_dis_complete:
    refiner.eval()
    discriminator.train()
    
    losses_cp = defaultdict(list)
    
    print('Initial training of discriminator using the adversarial loss for %i iterations...' % init_dis_iters)
    cur_dis_iter = 1
    while cur_dis_iter <= (init_dis_iters*batchsize):
        # Iterate through both datasets simultaneously
        realdataloader_iterator = iter(real_train_dataloader)
        for fake_sample in fake_train_dataloader:
            try:
                true_sample = next(realdataloader_iterator)
            except StopIteration:
                realdataloader_iterator = iter(real_train_dataloader)
                true_sample = next(realdataloader_iterator)
                
            if use_cuda:
                fake_sample = batch_to_cuda(fake_sample)
                true_sample = batch_to_cuda(true_sample)
            
            # Evaluate adversarial loss
            (dis_loss_fake, dis_loss_true, 
             dis_acc_fake, dis_acc_true) = run_dis_iter(refiner, discriminator, 
                                                        fake_sample['x'], true_sample['x'], 
                                                        local_adversarial_loss, use_cuda)
            
            # Backpropogate loss
            dis_loss_fake.backward()
            dis_loss_true.backward()
            
            # Save loss values
            losses_cp['init_dis_fake_loss_train'].append(dis_loss_fake.cpu().data.numpy())
            losses_cp['init_dis_true_loss_train'].append(dis_loss_true.cpu().data.numpy())
            losses_cp['init_dis_fake_acc_train'].append(dis_acc_fake.cpu().data.numpy())
            losses_cp['init_dis_true_acc_train'].append(dis_acc_true.cpu().data.numpy())
            
            if (cur_dis_iter%batchsize==0):
                # Adjust network weights
                optimizer_dis.step()
                optimizer_dis.zero_grad()            
            cur_dis_iter += 1
            
            if cur_dis_iter % (verbose_iters*batchsize) == 0: 
                # Run network on validation set
                discriminator.eval()
                # Iterate through both datasets simultaneously
                realdataloader_iterator = iter(real_val_dataloader)
                for fake_sample in fake_val_dataloader:
                    try:
                        true_sample = next(realdataloader_iterator)
                    except StopIteration:
                        realdataloader_iterator = iter(real_val_dataloader)
                        true_sample = next(realdataloader_iterator)
                        
                    if use_cuda:
                        fake_sample = batch_to_cuda(fake_sample)
                        true_sample = batch_to_cuda(true_sample)
                
                    # Evaluate adversarial loss
                    (dis_loss_fake, dis_loss_true, 
                     dis_acc_fake, dis_acc_true) = run_dis_iter(refiner, discriminator, 
                                                        fake_sample['x'], true_sample['x'], 
                                                        local_adversarial_loss, use_cuda)
                    losses_cp['init_dis_fake_loss_val'].append(dis_loss_fake.cpu().data.numpy())
                    losses_cp['init_dis_true_loss_val'].append(dis_loss_true.cpu().data.numpy())
                    losses_cp['init_dis_fake_acc_val'].append(dis_acc_fake.cpu().data.numpy())
                    losses_cp['init_dis_true_acc_val'].append(dis_acc_true.cpu().data.numpy())
                
                # Calculate averages
                for k in losses_cp.keys():
                    losses[k].append(np.mean(losses_cp[k]))
                losses['init_dis_batch_iters'].append(cur_dis_iter/batchsize)

                # Print current status
                print('\nBatch Iterations: %i/%i ' % (losses['init_dis_batch_iters'][-1], init_dis_iters))
                print('\n\t|     Dis-Real    |     Dis-Fake    |     Acc-Real    |     Acc-Fake    |')
                print('Train   |     %0.5f     |     %0.5f     |     %0.5f     |     %0.5f     |' % 
                      (losses['init_dis_true_loss_train'][-1], losses['init_dis_fake_loss_train'][-1],
                       losses['init_dis_true_acc_train'][-1], losses['init_dis_fake_acc_train'][-1]))
                print('Val     |     %0.5f     |     %0.5f     |     %0.5f     |     %0.5f     |' % 
                      (losses['init_dis_true_loss_val'][-1], losses['init_dis_fake_loss_val'][-1],
                       losses['init_dis_true_acc_val'][-1], losses['init_dis_fake_acc_val'][-1]))
                print('\n') 
                
                # Save losses to file to analyze throughout training. 
                np.save(os.path.join(progress_dir, model_name+'_losses.npy'), losses) 
                # Reset checkpoint loss dict
                losses_cp = defaultdict(list)
                
                discriminator.train()
                
            if not (cur_dis_iter <= (init_dis_iters*batchsize)):
                break
    
    # Save model
    init_dis_complete = True            
    torch.save({'losses': losses,
                'init_reg_complete': init_reg_complete,
                'init_dis_complete': init_dis_complete,
                'optimizer_ref' : optimizer_ref.state_dict(),
                'optimizer_dis' : optimizer_dis.state_dict(),
                'refiner' : refiner.state_dict(),
                'discriminator' : discriminator.state_dict(),
                'lr_scheduler_ref' : lr_scheduler_ref.state_dict(),
                'lr_scheduler_dis' : lr_scheduler_dis.state_dict()}, 
               model_filename)
else:
    print('Initial training of discriminator using the adversarial loss already complete.')    

# Main training of the refiner and discriminator
print('Training the network...')
print('Progress will be displayed every %i iterations and the model will be saved every %i minutes.'%
      (verbose_iters,cp_time))

losses_cp = defaultdict(list)
cp_start_time = time.time()

while cur_iter <= (total_batch_iters*batchsize):
    # Iterate through both datasets simultaneously
    realdataloader_iterator = iter(real_train_dataloader)
    for fake_sample in fake_train_dataloader:
        try:
            true_sample = next(realdataloader_iterator)
        except StopIteration:
            realdataloader_iterator = iter(real_train_dataloader)
            true_sample = next(realdataloader_iterator)
            
        if use_cuda:
            fake_sample = batch_to_cuda(fake_sample)
            true_sample = batch_to_cuda(true_sample)

        # Train a discriminator iteration
        train_dis = False
        if (gen_to_dis_iters>1):
            if ((cur_iter//batchsize)%gen_to_dis_iters==0):
                train_dis = True
        else:
            train_dis = True
        if train_dis:
            refiner.eval()
            discriminator.train()
            (dis_loss_fake, dis_loss_true, 
             dis_acc_fake, dis_acc_true) = run_dis_iter(refiner, discriminator, 
                                                        fake_sample['x'], true_sample['x'], 
                                                        local_adversarial_loss, use_cuda)

            # Backpropogate loss
            dis_loss_fake.backward()
            dis_loss_true.backward()

            # Save loss values
            losses_cp['dis_fake_loss_train'].append(dis_loss_fake.cpu().data.numpy())
            losses_cp['dis_true_loss_train'].append(dis_loss_true.cpu().data.numpy())
            losses_cp['dis_fake_acc_train'].append(dis_acc_fake.cpu().data.numpy())
            losses_cp['dis_true_acc_train'].append(dis_acc_true.cpu().data.numpy())

            if (cur_iter%batchsize==0):
                # Adjust network weights
                optimizer_dis.step()
                optimizer_dis.zero_grad()  
                # Adjust learning rate
                lr_scheduler_dis.step()
        
        # Train a refiner iteration
        train_ref = False
        if (gen_to_dis_iters<1):
            if ((cur_iter//batchsize)%np.round((1/gen_to_dis_iters),0)==0):
                train_ref = True
        else:
            train_ref = True
        if train_ref:
            refiner.train()
            discriminator.eval()
            gen_loss, reg_loss, gen_acc = run_gen_iter(refiner, discriminator, 
                                                         fake_sample['x'], 
                                                         local_adversarial_loss, 
                                                         self_regularization_loss, 
                                                         use_cuda)

            # Backpropogate loss
            total_ref_loss = gen_loss + loss_weight_reg*reg_loss
            total_ref_loss.backward()
            
            # Save loss values
            losses_cp['gen_loss_loss_train'].append(gen_loss.cpu().data.numpy())
            losses_cp['reg_loss_loss_train'].append(reg_loss.cpu().data.numpy())
            losses_cp['gen_acc_train'].append(gen_acc.cpu().data.numpy())

            if (cur_iter%batchsize==0):
                # Adjust network weights
                optimizer_ref.step()
                optimizer_ref.zero_grad() 
                # Adjust learning rate
                lr_scheduler_ref.step()
          

        if cur_iter % (verbose_iters*batchsize) == 0: 
            # Run network on validation set
            refiner.eval()
            discriminator.eval()
            plot_fake_samples = []
            plot_true_samples = []
            # Iterate through both datasets simultaneously
            realdataloader_iterator = iter(real_val_dataloader)
            num_val = 0
            for fake_sample in fake_val_dataloader:
                try:
                    true_sample = next(realdataloader_iterator)
                except StopIteration:
                    realdataloader_iterator = iter(real_val_dataloader)
                    true_sample = next(realdataloader_iterator)
                    
                if use_cuda:
                    fake_sample = batch_to_cuda(fake_sample)
                    true_sample = batch_to_cuda(true_sample)
                # Collect first 5 samples to plot    
                if len(plot_fake_samples)<5:    
                    plot_fake_samples.append(fake_sample['x'])
                    plot_true_samples.append(true_sample['x'].cpu().data.numpy()[0,0])
                # Only evaluate first 200 samples
                if num_val>=200:
                    break
                else:
                    num_val+=1
                # Evaluate discriminator
                (dis_loss_fake, dis_loss_true, 
                 dis_acc_fake, dis_acc_true) = run_dis_iter(refiner, discriminator, 
                                                            fake_sample['x'], true_sample['x'], 
                                                            local_adversarial_loss, use_cuda)
                losses_cp['dis_fake_loss_val'].append(dis_loss_fake.cpu().data.numpy())
                losses_cp['dis_true_loss_val'].append(dis_loss_true.cpu().data.numpy())
                losses_cp['dis_fake_acc_val'].append(dis_acc_fake.cpu().data.numpy())
                losses_cp['dis_true_acc_val'].append(dis_acc_true.cpu().data.numpy())
                
                # Evaluate refiner
                gen_loss, reg_loss, gen_acc = run_gen_iter(refiner, discriminator, 
                                                           fake_sample['x'], 
                                                           local_adversarial_loss, 
                                                           self_regularization_loss, 
                                                           use_cuda)
                losses_cp['gen_loss_loss_val'].append(gen_loss.cpu().data.numpy())
                losses_cp['reg_loss_loss_val'].append(reg_loss.cpu().data.numpy())
                losses_cp['gen_acc_val'].append(gen_acc.cpu().data.numpy())

            # Calculate averages
            for k in losses_cp.keys():
                losses[k].append(np.mean(losses_cp[k]))
            losses['batch_iters'].append(cur_iter/batchsize)

            # Print current status
            print('\nBatch Iterations: %i/%i ' % (losses['batch_iters'][-1], total_batch_iters))
            print('\n\t|   Dis-Real  |   Dis-Fake  | Dis-Acc-Real| Dis-Acc-Fake|  Generator  | Gen-Acc-Fake|   Self-Reg  |')
            print('Train   |   %0.5f   |   %0.5f   |   %0.5f   |   %0.5f   |   %0.5f   |   %0.5f   |   %0.5f   |' % 
                  (losses['dis_true_loss_train'][-1], losses['dis_fake_loss_train'][-1],
                   losses['dis_true_acc_train'][-1], losses['dis_fake_acc_train'][-1],
                   losses['gen_loss_loss_train'][-1], losses['gen_acc_train'][-1],
                   losses['reg_loss_loss_train'][-1]))
            print('Val     |   %0.5f   |   %0.5f   |   %0.5f   |   %0.5f   |   %0.5f   |   %0.5f   |   %0.5f   |' % 
                  (losses['dis_true_loss_val'][-1], losses['dis_fake_loss_val'][-1],
                   losses['dis_true_acc_val'][-1], losses['dis_fake_acc_val'][-1],
                   losses['gen_loss_loss_val'][-1], losses['gen_acc_val'][-1],
                   losses['reg_loss_loss_val'][-1]))
            print('\n') 
            
            # Save losses to file to analyze throughout training. 
            np.save(os.path.join(progress_dir, model_name+'_losses.npy'), losses) 
            # Reset checkpoint loss dict
            losses_cp = defaultdict(list)
            
            # Plot 5 validation samples
            plot_samples(refiner, plot_fake_samples, plot_true_samples, model_name, 
                         val_results_dir, cur_iter, batchsize, 'val')
            
            # Plot 5 training samples as well
            plot_fake_samples = []
            plot_true_samples = []
            # Iterate through both datasets simultaneously
            realdataloader_iterator = iter(real_train_dataloader)
            for fake_sample in fake_train_dataloader:
                try:
                    true_sample = next(realdataloader_iterator)
                except StopIteration:
                    realdataloader_iterator = iter(real_train_dataloader)
                    true_sample = next(realdataloader_iterator)
                if use_cuda:
                    fake_sample = batch_to_cuda(fake_sample)
                    true_sample = batch_to_cuda(true_sample)
                # Collect first 5 samples to plot    
                if len(plot_fake_samples)<5:    
                    plot_fake_samples.append(fake_sample['x'])
                    plot_true_samples.append(true_sample['x'].cpu().data.numpy()[0,0])
                    
            plot_samples(refiner, plot_fake_samples, plot_true_samples, model_name, 
                         train_results_dir, cur_iter, batchsize, 'train')
            
        # Save periodically
        if time.time() - cp_start_time >= cp_time*60:
            print('Saving network...')

            torch.save({'losses': losses,
                        'init_reg_complete': init_reg_complete,
                        'init_dis_complete': init_dis_complete,
                        'optimizer_ref' : optimizer_ref.state_dict(),
                        'optimizer_dis' : optimizer_dis.state_dict(),
                        'refiner' : refiner.state_dict(),
                        'discriminator' : discriminator.state_dict(),
                        'lr_scheduler_ref' : lr_scheduler_ref.state_dict(),
                        'lr_scheduler_dis' : lr_scheduler_dis.state_dict()}, 
                       model_filename)

            cp_start_time = time.time()
        cur_iter += 1

        if not (cur_iter <= (total_batch_iters*batchsize)):
            break
            
torch.save({'losses': losses,
                        'init_reg_complete': init_reg_complete,
                        'init_dis_complete': init_dis_complete,
                        'optimizer_ref' : optimizer_ref.state_dict(),
                        'optimizer_dis' : optimizer_dis.state_dict(),
                        'refiner' : refiner.state_dict(),
                        'discriminator' : discriminator.state_dict(),
                        'lr_scheduler_ref' : lr_scheduler_ref.state_dict(),
                        'lr_scheduler_dis' : lr_scheduler_dis.state_dict()}, 
                       model_filename)
