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
from labdataloader import *
import skimage.io
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
from misc.loss_wrapper import LossWrapper
from torch.nn.utils import clip_grad_norm_ 
import pickle as pkl
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

class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, gamma=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        target = target[:, :input.size(1)]
        input = input.contiguous().view(-1, input.size(-1))
        target = target.contiguous().view(-1)
        """expects log probs as inp, and target values are class nums"""
        indexed_logprobs = inputs[range(targets.shape[0]), targets.long()]
        probs = torch.exp(indexed_logprobs)
        F_loss = -(1-probs)**self.gamma * indexed_logprobs
        return F_loss

def train(opt):
    # Deal with feature things before anything
    opt.use_fc, opt.use_att = utils.if_use_feat(opt.caption_model)
    if opt.use_box:
        opt.att_feat_size = opt.att_feat_size + 5

    acc_steps = getattr(opt, 'acc_steps', 1)
    lab_loader = DataLoader(opt, opt.fraction)
    opt.vocab_size = lab_loader.vocab_size
    opt.seq_length = lab_loader.seq_length
    cov = False
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'-best.pkl'), 'rb') as f:
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
    epoch = infos.get('epoch', 0)
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    lab_loader.iterators = infos.get('iterators', lab_loader.iterators)
    lab_loader.split_ix = infos.get('split_ix', lab_loader.split_ix)
    
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    opt.vocab = lab_loader.get_vocab()
    
    model = models.setup(opt).cuda()
    lw_model = LossWrapper(model, opt)
#     criterion_CE = utils.LabelSmoothing(smoothing=0.1)
    criterion_label = utils.LabelSmoothingCrossEntropy(smoothing=opt.label_smoothing)
    
    epoch_done = True
    model.train()

    if opt.noamopt:
        assert opt.caption_model in ['transformer','aoa'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        optimizer._step = iteration
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
#     optimizer = utils.build_optimizer(model.parameters(), opt)    
    
#     Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-best.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer-best.pth')))
    
    
    
    def save_checkpoint(model, infos, optimizer, histories=None, append=''):
        if len(append) > 0:
            append = '-' + append
        # if checkpoint_path doesn't exist
        if not os.path.isdir(opt.checkpoint_path):
            os.makedirs(opt.checkpoint_path)

        model_ckpt_path = os.path.join(opt.checkpoint_path, 'model%s.pth' %(append))
        torch.save(model.state_dict(), model_ckpt_path)
        print("model saved to {}".format(model_ckpt_path))

        optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
        torch.save(optimizer.state_dict(), optimizer_path)

        with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
            utils.pickle_dump(infos, f)
        if histories:
            with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
                utils.pickle_dump(histories, f)

    try:
        sc_flag = False
        while True:
            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate  ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob

                # If start self critical training
                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    sc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    sc_flag = False
                epoch_done = False
                
                if (opt.use_warmup == 1) and (iteration < opt.noamopt_warmup):
                    opt.current_lr = opt.learning_rate * (iteration+1) / opt.noamopt_warmup
                    utils.set_lr(optimizer, opt.current_lr)

            data = lab_loader.get_batch('train')
            if (iteration % acc_steps == 0):
                optimizer.zero_grad()
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['att_masks']]
            tmp = [i if i is None else i.cuda() for i in tmp]
            fc_feats, att_feats, labels, att_masks = tmp
            del tmp
            org_unlab_logprobs = model(fc_feats, att_feats, labels, att_masks)
            loss = criterion_label(org_unlab_logprobs, labels[:, 1:]).mean()
            labeled_loss = loss / acc_steps
#             labeled_loss = 0
            train_loss = labeled_loss
            train_loss.backward()

            if ((iteration+1) % acc_steps == 0):
                clip_grad_norm_(model.parameters(), opt.grad_clip)
                optimizer.step()

            end = time.time()

            if not sc_flag:
                print("iter {} (epoch {}), label_loss = {:.3f}" \
                .format(iteration, epoch, labeled_loss), flush=True)
                
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}" \
                    .format(iteration, epoch, model_out['reward'].mean()), flush=True)

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
                if opt.noamopt:
                    opt.current_lr = optimizer.rate()
                elif opt.reduce_on_plateau:
                    opt.current_lr = optimizer.current_lr
                add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    add_summary_value(tb_summary_writer, 'avg_reward', model_out['reward'].mean(), iteration)

                loss_history[iteration] = train_loss.item() if not sc_flag else model_out['reward'].mean().item()
                lr_history[iteration] = opt.current_lr
                ss_prob_history[iteration] = model.ss_prob


            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = lab_loader.iterators
            infos['split_ix'] = lab_loader.split_ix
            
            # make evaluation on validation set, and save model
            if data['bounds']['wrapped']:
#                 eval model
                eval_kwargs = {'split': 'val',
                                'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats = eval_utils.eval_split(
                    model, lw_model.crit, lab_loader, eval_kwargs)
                

                if opt.reduce_on_plateau:
                    if 'CIDEr' in lang_stats:
                        optimizer.scheduler_step(-lang_stats['CIDEr'])
                    else:
                        optimizer.scheduler_step(val_loss)

                add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        add_summary_value(tb_summary_writer, k, v, iteration)
                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss

                best_flag = False

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                infos['best_val_score'] = best_val_score
                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['ss_prob_history'] = ss_prob_history

                save_checkpoint(model, infos, optimizer, histories)
#                 if opt.save_history_ckpt:
#                     save_checkpoint(model, infos, optimizer, append=str(iteration))

                if best_flag:
                    save_checkpoint(model, infos, optimizer, append='best')

            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break
    except:
        print('Save ckpt on exception ...')
        save_checkpoint(model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)    

opt = opts.parse_opt()
print(opt)
train(opt)
