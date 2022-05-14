import tensorflow as tf
from copy import deepcopy
import numpy as np
import timeit
from tensorflow.python.platform import flags
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import time
from multiprocessing import Process
from tqdm import tqdm

from data import Cifar10, CelebAHQ, Mnist, ImageNet, LSUNBed, STLDataset
from models import CelebAModel, MNISTModel, ImagenetModel
from ResNetModel import ResNetModel
import os.path as osp
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#from logger import TensorBoardOutputFormat
from utils import ReplayBuffer, ReservoirBuffer
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import time as time
from io import StringIO
from tensorflow.core.util import event_pb2

import torch
torch.backends.cudnn.enabled = False

import numpy as np
#from scipy.misc import imsave
import matplotlib.pyplot as plt
from easydict import EasyDict

from utils import ReplayBuffer
from torch.optim import Adam, SGD
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

FLAGS = flags.FLAGS

# Distributed training hyperparameters
flags.DEFINE_integer('nodes', 1,
    'number of nodes for training')
flags.DEFINE_integer('gpus', 1,
    'number of gpus per nodes')
flags.DEFINE_integer('node_rank', 0,
    'rank of node')

# Configurations for distributed training
flags.DEFINE_string('master_addr', '8.8.8.8',
    'address of communicating server')
flags.DEFINE_string('port', '10002',
    'port of training')
flags.DEFINE_bool('slurm', False,
    'whether we are on slurm')
flags.DEFINE_bool('repel_im', True,
    'maximize entropy by repeling images from each other')
flags.DEFINE_bool('hmc', False,
    'use the hamiltonian monte carlo sampler')
flags.DEFINE_bool('square_energy', False,
    'make the energy square')
flags.DEFINE_bool('alias', False,
    'make the energy square')

flags.DEFINE_string('dataset','cifar10','default set as cifar10')
flags.DEFINE_integer('batch_size',256, 'batch size during training')
flags.DEFINE_bool('multiscale', False, 'A multiscale EBM')
flags.DEFINE_bool('self_attn', True, 'Use self attention in models')
flags.DEFINE_bool('sigmoid', False, 'Apply sigmoid on energy (can improve the stability)')
flags.DEFINE_bool('anneal', False, 'Decrease noise over Langevin steps')
flags.DEFINE_integer('data_workers', 0,
    'Number of different data workers to load data in parallel')
flags.DEFINE_integer('buffer_size',1200, 'Size of inputs')
flags.DEFINE_integer('divi_buffer_size',1, 'Divide buffer to update')

# General Experiment Settings
flags.DEFINE_string('logdir', 'cachedir',
    'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 150,'save outputs every so many batches')
flags.DEFINE_integer('test_interval', 1000,'evaluate outputs every so many batches')
flags.DEFINE_integer('resume_iter', 0, 'iteration to resume training from')
flags.DEFINE_bool('train', True, 'whether to train or test')
flags.DEFINE_bool('transform', True, 'apply data augmentation when sampling from the replay buffer')
flags.DEFINE_bool('kl', True, 'apply a KL term to loss')
flags.DEFINE_bool('cuda', True, 'move device on cuda')
flags.DEFINE_integer('epoch_num', 10000, 'Number of Epochs to train on')
flags.DEFINE_integer('ensembles', 1, 'Number of ensembles to train models with')
flags.DEFINE_float('lr', 2e-4, 'Learning for training')
flags.DEFINE_float('kl_coeff', 1.0, 'coefficient for kl')

# EBM Specific Experiments Settings
flags.DEFINE_string('objective', 'cd', 'use the cd objective')

# Setting for MCMC sampling
flags.DEFINE_integer('num_steps', 40, 'Steps of gradient descent for training')
flags.DEFINE_float('step_lr', 100.0, 'Size of steps for gradient descent')
flags.DEFINE_bool('replay_batch', True, 'Use MCMC chains initialized from a replay buffer.')
flags.DEFINE_bool('reservoir', True, 'Use a reservoir of past entires')
flags.DEFINE_float('noise_scale', 1.,'Relative amount of noise for MCMC')

# Architecture Settings
flags.DEFINE_integer('filter_dim', 64, 'number of filters for conv nets')
flags.DEFINE_integer('im_size', 32, 'size of images')
flags.DEFINE_bool('spec_norm', False, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('norm', True, 'Use group norm in models norm in models')

# Conditional settings
flags.DEFINE_bool('cond', False, 'conditional generation with the model')
flags.DEFINE_bool('all_step', False, 'backprop through all langevin steps')
flags.DEFINE_bool('log_grad', False, 'log the gradient norm of the kl term')
flags.DEFINE_integer('cond_idx', 0, 'conditioned index')

def compress_x_mod(x_mod):
    x_mod = (255 * np.clip(x_mod, 0, 1)).astype(np.uint8)
    return x_mod


def decompress_x_mod(x_mod):
    x_mod = x_mod / 256  + \
        np.random.uniform(0, 1 / 256, x_mod.shape)
    return x_mod

def gen_image(FLAGS, model, im_neg, num_steps, sample=False):
    im_noise = torch.randn_like(im_neg).detach()

    im_negs_samples = []

    for i in range(num_steps):
        print(i)
        im_noise.normal_()

        if FLAGS.anneal:
            im_neg = im_neg + 0.001 * (num_steps - i - 1) / num_steps * im_noise
        else:
            im_neg = im_neg + 0.001 * im_noise

        im_neg.requires_grad_(requires_grad=True)
        energy = model.forward(im_neg)

        if FLAGS.all_step:
            im_grad = torch.autograd.grad([energy.sum()], [im_neg], create_graph=True)[0]
        else:
            im_grad = torch.autograd.grad([energy.sum()], [im_neg])[0]

        if i == num_steps - 1:
            im_neg_orig = im_neg
            im_neg = im_neg - FLAGS.step_lr * im_grad

            if FLAGS.dataset == "cifar10":
                n = 128
            elif FLAGS.dataset == "celeba":
                # Save space
                n = 128
            elif FLAGS.dataset == "lsun":
                # Save space
                n = 32
            elif FLAGS.dataset == "object":
                # Save space
                n = 32
            elif FLAGS.dataset == "mnist":
                n = 32
            elif FLAGS.dataset == "imagenet":
                n = 32
            elif FLAGS.dataset == "stl":
                n = 32

            im_neg_kl = im_neg_orig[:n]
            if sample:
                pass
            else:
                energy = model.forward(im_neg_kl)
                im_grad = torch.autograd.grad([energy.sum()], [im_neg_kl], create_graph=True)[0]

            im_neg_kl = im_neg_kl - FLAGS.step_lr * im_grad[:n]
            im_neg_kl = torch.clamp(im_neg_kl, 0, 1)
        else:
            im_neg = im_neg - FLAGS.step_lr * im_grad

        im_neg = im_neg.detach()

        if sample:
            im_negs_samples.append(im_neg)

        im_neg = torch.clamp(im_neg, 0, 1)

    if sample:
        return im_neg, im_neg_kl, im_negs_samples, im_grad
    else:
        return im_neg, im_neg_kl, im_grad

rank_idx = 1
pthfile = 'cachedir/default/model_9960.pth'
net = torch.load(pthfile)
model = net['model_state_dict_0']
init_ranndom = torch.Tensor(np.random.uniform(0.0, 1.0, (256, 3, 32, 32))).cuda(rank_idx)
data_corrupt = gen_image(FLAGS, model, init_ranndom,3)[0]
picture_img = torch.tensor(compress_x_mod(data_corrupt.cpu().detach().numpy()))
data_corrupt2 = picture_img.permute(0, 2, 3, 1).float().contiguous()
k = data_corrupt2.cpu().detach().numpy()
for i in range(256):
    from PIL import Image
    import os
    res = k[i] 
    image = Image.fromarray(np.uint8(res)).convert('RGB')
    if not os.path.exists('Sample_Data'):
        os.makedirs('Sample_Data')
    savepath = 'Sample_Data/' +'IGEBM'+'_'+str(i)+'.jpg'
    image.save(savepath)