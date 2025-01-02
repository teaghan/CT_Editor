import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
from scipy.ndimage import gaussian_filter

def init_weights(m):
    """
    Glorot uniform initialization for network.
    """
    if 'conv' in m.__class__.__name__.lower():
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def compute_out_size(in_size, mod):
    """
    Compute output size of Module `mod` given an input with size `in_size`.
    """
    
    f = mod.forward(autograd.Variable(torch.Tensor(1, *in_size)))
    return f.size()[1:]

class ResnetBlock(nn.Module):
    def __init__(self, input_features, nb_features=64, filter_len=3, stride=1, padding=0):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv3d(input_features, nb_features, 
                               kernel_size=filter_len, stride=stride, 
                               padding=padding)
        self.conv2 = nn.Conv3d(nb_features, nb_features, 
                               kernel_size=filter_len, stride=stride, 
                               padding=padding)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        residual = x   
        out = self.leakyrelu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.leakyrelu(out)
        return out


class Refiner(nn.Module):
    def __init__(self, num_blocks=4, in_features=1, nb_features=64, filter_len=3, 
                 init=True, edge_kernel=None, use_cuda=True):
        super(Refiner, self).__init__()
        
        self.edge_kernel = edge_kernel

        # Input conv layer
        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels=in_features, out_channels=nb_features, 
                      kernel_size=filter_len, stride=1, padding=1),
            nn.LeakyReLU()
        )

        # ResNet blocks
        blocks = []
        for i in range(num_blocks):
            blocks.append(ResnetBlock(nb_features, nb_features, 
                                      filter_len=filter_len, padding=1))
        self.resnet_blocks = nn.Sequential(*blocks)

        # Output conv layer
        self.conv_2 = nn.Sequential(
            nn.Conv3d(in_channels=nb_features, 
                                             out_channels=in_features, 
                                             kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
        
        # Initialize weights and biases
        if init:
            self.conv_1.apply(init_weights)
            self.resnet_blocks.apply(init_weights)
            self.conv_2.apply(init_weights)

        # Switch to GPU
        if use_cuda:
            self.conv_1 = self.conv_1.cuda()
            self.resnet_blocks = self.resnet_blocks.cuda()
            self.conv_2 = self.conv_2.cuda()

    def train_mode(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag

    def forward(self, x):
        out = self.conv_1(x)
        out = self.resnet_blocks(out)
        out = self.conv_2(out)
        if self.edge_kernel is not None:
            out = self.edge_kernel.apply_mask(x, out)
        return out

class EdgeKernel(nn.Module):
    def __init__(self, shapes, fade_perc=0.2, use_cuda=True):
        super(EdgeKernel, self).__init__()

        self.shapes = torch.tensor(shapes).type(torch.IntTensor)
        if use_cuda:
            self.shapes = self.shapes.cuda()
        
        self.masks = []
        for cnn_input_size in shapes:

            z_size = int(np.rint(fade_perc*cnn_input_size[0]))
            y_size = int(np.rint(fade_perc*cnn_input_size[1]))
            x_size = int(np.rint(fade_perc*cnn_input_size[2]))

            # Create the fade along each axis
            fade_z = np.linspace(0, 1, z_size)[...,np.newaxis,np.newaxis]
            fade_y = np.linspace(0, 1, y_size)[np.newaxis,...,np.newaxis]
            fade_x = np.linspace(0, 1, x_size)[np.newaxis,np.newaxis,...]
            
            # Turn this into a mask that can be applied to the scan during pre-processing.
            mask = np.ones((1,*cnn_input_size))
            mask[0,:z_size] *= fade_z
            mask[0,-z_size:] *= np.flip(fade_z)
            mask[0,:,:y_size] *= fade_y
            mask[0,:,-y_size:] *= np.flip(fade_y)
            mask[0,:,:,:x_size] *= fade_x
            mask[0,:,:,-x_size:] *= np.flip(fade_x)
            
            # Convolve with Gaussian
            mask = gaussian_filter(mask, sigma=0.6)
            
            mask = torch.Tensor(mask)
            if use_cuda:
                mask = mask.cuda()
            self.masks.append(mask)
    
    def apply_mask(self, orig_sample, edit_sample):
        # Locate correct mask
        shape = orig_sample.shape[-3:]
        shape = torch.tensor(shape, dtype=torch.int).view(1,3)
        indx = torch.nonzero(torch.sum(self.shapes==shape, dim=1)==3)
        mask = self.masks[indx]
        # Apply mask
        comb_sample = mask*edit_sample + (1-mask)*orig_sample
        return comb_sample
    
class Discriminator(nn.Module):
    def __init__(self, in_features=1, nb_features=[96,64,32,32,2], stride_len=[2,2,1,1,1], 
                 filter_len=3, init=True, use_cuda=True):
        super(Discriminator, self).__init__()

        paddings = []
        filter_lens = []
        for stride in stride_len:
            if stride>1:
                filter_lens.append(filter_len)
                paddings.append(1)
            else:
                filter_lens.append(1)
                paddings.append(0)
        
        # Convolutional layers
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels=in_features, out_channels=nb_features[0], 
                      kernel_size=filter_lens[0], stride=stride_len[0], padding=paddings[0]),
            nn.LeakyReLU(),

            nn.Conv3d(in_channels=nb_features[0], out_channels=nb_features[1], 
                      kernel_size=filter_lens[1], stride=stride_len[1], padding=paddings[1]),
            nn.LeakyReLU(),

            nn.MaxPool3d(filter_len, 1, 1),

            nn.Conv3d(in_channels=nb_features[1], out_channels=nb_features[2], 
                      kernel_size=filter_lens[2], stride=stride_len[2], padding=paddings[2]),
            nn.LeakyReLU(),

            nn.Conv3d(in_channels=nb_features[2], out_channels=nb_features[3], 
                      kernel_size=filter_lens[3], stride=stride_len[3], padding=paddings[3]),
            nn.LeakyReLU(),

            nn.Conv3d(in_channels=nb_features[3], out_channels=nb_features[4], 
                      kernel_size=filter_lens[4], stride=stride_len[4], padding=paddings[4]),
            
        )
        
        # Initialize weights and biases
        if init:
            self.convs.apply(init_weights)

        # Switch to GPU
        if use_cuda:
            self.convs = self.convs.cuda()

    def train_mode(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag

    def forward(self, x):
        out = self.convs(x)
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, 2)
        out = nn.Softmax(dim=1)(out)
        return out