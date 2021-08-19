from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle
import traceback

import opts
import models
from dataloader import *
import skimage.io
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
from misc.loss_wrapper import LossWrapper
from torch.nn.utils import clip_grad_norm_ 
import pickle as pkl
from unlabdataloader import BuDataLoader
from misc.gumbel_functions import gumbel_softmax
import torch.nn.functional as F

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    model_id = 'teacher'
    predname = 'pseudo_labels.txt'
    # Deal with feature things before anything
    opt.use_fc, opt.use_att = utils.if_use_feat(opt.caption_model)
    if opt.use_box:
        opt.att_feat_size = opt.att_feat_size + 5

    acc_steps = getattr(opt, 'acc_steps', 1)
    lab_loader = DataLoader(opt, opt.fraction)
    unlab_loader = BuDataLoader(opt)
    opt.vocab_size = lab_loader.vocab_size
    opt.seq_length = lab_loader.seq_length
    cov = False
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    torch.set_grad_enabled(False)
    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
                histories = utils.pickle_load(f)
    else:
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['iterators'] = lab_loader.iterators
        infos['split_ix'] = lab_loader.split_ix
        infos['vocab'] = lab_loader.get_vocab()
    
    infos['opt'] = opt

    iteration = infos.get('iter', 0)

    lab_loader.iterators = infos.get('iterators', lab_loader.iterators)
    lab_loader.split_ix = infos.get('split_ix', lab_loader.split_ix)
    
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    opt.vocab = lab_loader.get_vocab()
    
    teacher = models.setup(opt).cuda()
    
    epoch_done = True
    teacher.eval()
    f = open(predname, 'w')

    teacher.load_state_dict(torch.load('log/log_'+model_id+'/model-best.pth'))
    
    sc_flag = False
    while True:
        unlab_data = unlab_loader.get_batch('train')
        tmp = [unlab_data['fc_feats'], unlab_data['att_feats'], unlab_data['att_masks']]
        tmp = [i if i is None else i.cuda() for i in tmp]
        fc_aug_feats, att_aug_feats, att_aug_masks = tmp
        del tmp
        aug_unlab_logprobs = teacher(fc_feats=fc_aug_feats, att_feats=att_aug_feats, att_masks=att_aug_masks, mode='sample', opt=vars(opt))[0].data
        inp_idx = aug_unlab_logprobs.cpu().numpy()
        BK, T = inp_idx.shape
        for i in range(BK):
            f.write(" ".join([str(i) for i in list(inp_idx[i])]))
            f.write("\n")
        if unlab_data['bounds']['wrapped']:
            f.close()
            break
            

opt = opts.parse_opt()
train(opt)
