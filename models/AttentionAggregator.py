from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import argparse
import json
import random 
from random import choice
from .custom_lstms import StackedGRU2, BidirGRULayer, StackedGRU, GRULayer, LayerNormGRUCell

def init_lstm_wt(lstm):
    for names in lstm.GRU._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm.GRU, name)
                wt.data.uniform_(-0.02, 0.02)
            elif name.startswith('bias_'):
                bias = getattr(lstm.GRU, name)
                n = bias.size(0)
                bias.data.fill_(0.)
                
def init_linear_wt(linear):
    linear.weight.data.normal_(std=1e-4)
    if linear.bias is not None:
        linear.bias.data.normal_(std=1e-4)
        
        
class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout, n_aug, bi):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_aug= n_aug
        self.embedding = nn.Embedding(9488, 300).from_pretrained(torch.load('/home/arjit/spacetime/allembedding_weights300.pt').float())
        self.rnn = StackedGRU2(self.n_layers, BidirGRULayer, 
                        first_layer_args=[LayerNormGRUCell, emb_dim, hid_dim, False],
                      other_layer_args=[LayerNormGRUCell, hid_dim * 2, hid_dim, False])
        self.hidden0 = nn.Parameter(1e-2*torch.randn((2, 2, 1, self.hid_dim)).cuda())
        
        
    def forward(self, src, lengths):
        BK, T = src.shape
        embedded = self.embedding(src)
        
        hidden = self.rnn(embedded.permute(1, 0, 2).contiguous(), self.hidden0.repeat(1, 1, BK, 1)).view(self.n_layers, T, BK, 2, self.hid_dim).mean(0).permute(1, 0, 2, 3).contiguous()
        return hidden.mean(2), (hidden[:, 0, -1, :] + hidden[torch.arange(BK).long(), lengths, 0])/2
        
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, bi):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(9488, 300).from_pretrained(torch.load('/home/arjit/spacetime/allembedding_weights300.pt').float())
        self.emb_dim= emb_dim
        self.rnn = GRULayer(LayerNormGRUCell, emb_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim*2, vocab_size)
        self.softmax= nn.Softmax(dim=1)
        self.vocab_size= vocab_size
        self.p_gen_linear = nn.Linear(self.hid_dim + self.emb_dim, 1)
        self.x_context = nn.Linear(self.hid_dim + self.emb_dim, self.emb_dim)
        init_linear_wt(self.fc_out)
        init_linear_wt(self.p_gen_linear)

    def forward(self, input, hidden, context):
        
        #input = [B]
        #hidden = [n layers * n directions, B, hid dim]
        #cell = [n layers * n directions, B, hid dim]
        B = input.shape[0]
        input = input.unsqueeze(0)
        
        embedded = self.embedding(input)
        
        #embedded = [1, B, emb dim]
        inp = torch.cat([embedded, context], 2)
        inp = F.relu(self.x_context(inp))
        p_gen = torch.sigmoid(self.p_gen_linear(torch.cat([hidden.view(1, B, -1), inp], 2)))
        hidden = self.rnn(inp, hidden.view(B, self.hid_dim))
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        outp = self.softmax(self.fc_out(torch.cat([hidden, context], -1).squeeze(0)))
        
        #prediction = [batch size, output dim]
        
        return outp, hidden, p_gen.squeeze(0)

def nucleus(probs, top_p=0.5):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs >= top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = torch.zeros_like(probs, dtype=sorted_indices_to_remove.dtype).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    probs[indices_to_remove] = 1e-10

    return torch.multinomial(F.normalize(probs, p=1, dim=-1), num_samples=1).view(-1)
    
class Seq2Seq(nn.Module):
    def __init__(self, opt, cov = False):
        super().__init__()
        self.cov = cov
        self.vocab = opt.vocab
        self.vocab_size = len(opt.vocab)+1 # <PAD>
        EMB_DIM = opt.custom_decoder_emb_dim
        HID_DIM = opt.custom_decoder_hid_dim
        N_LAYERS = opt.custom_decoder_num_hidden
        ENC_DROPOUT = opt.custom_decoder_enc_drop
        DEC_DROPOUT = opt.custom_decoder_dec_drop
        N_AUG= opt.n_augs
        self.H = HID_DIM
        self.pad_idx= 0
        self.BIDIRECTIONAL= opt.custom_decoder_bi
        self.encoder= Encoder(EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, N_AUG, self.BIDIRECTIONAL)
        self.decoder= Decoder(self.vocab_size, EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, self.BIDIRECTIONAL)
        self.layers = N_LAYERS
        self.attn = nn.Linear(HID_DIM, HID_DIM)
        self.W_h = nn.Linear(HID_DIM, HID_DIM, bias=False)
        self.v = nn.Linear(HID_DIM, 1, bias = False)
        if cov:
            self.W_c = nn.Linear(1, HID_DIM, bias=False)
        init_linear_wt(self.W_h)
        self.tforcing = None
    
        
        
    def forward(self, src, tgt_idx):

        B, K, T = src.shape
        src = src.view(B*K, T)
        V = self.vocab_size
        
        mask = (src != self.pad_idx).float()
        src_len = torch.mul(mask, torch.arange(T).unsqueeze(0).to(src.device).float()).argmax(1)
        
        outputs = torch.zeros(T, B, V).to(src.device)
#         outputs_identity = torch.zeros(T, B*K, V).to(src.device)
        encodings, enc_hidden = self.encoder(src, src_len.long())
        attn_enc = self.W_h(encodings)
        hidden = enc_hidden.unsqueeze(0)
#         inp = torch.zeros(B*K).to(src.device).long() + self.pad_idx
#         if self.cov:
#             coverage_identity = torch.zeros_like(src).float()
#             coverage_identity_loss = 0
#         for t in range(T):
#             first_level_attn = self.attn(hidden.permute(1, 0, 2).contiguous().view(B*K, 1, -1)).expand(B*K, T, self.H) + attn_enc
#             if self.cov:
#                 first_level_attn = first_level_attn + self.W_c(coverage_identity.unsqueeze(-1))
#             first_level_attn = self.v(torch.tanh(first_level_attn)).squeeze(2)
#             first_level_attn = first_level_attn.masked_fill(mask == 0, -1e10)
#             first_level_attn = F.softmax(first_level_attn, dim=1)
#             if self.cov:
#                 coverage_identity_loss = coverage_identity_loss + torch.sum(mask*torch.min(coverage_identity, first_level_attn))
#                 coverage_identity = coverage_identity + first_level_attn
#             attn_vec = torch.mul(first_level_attn.unsqueeze(-1), encodings).sum(1)
#             outp, hidden, _ = self.decoder(inp, hidden, attn_vec.unsqueeze(0))
# #             outp = p_gen * outp
# #             outp = outp.scatter_add(1, src, (1 - p_gen) * first_level_attn)
#             outputs_identity[t] = outp
#             if random.random() < self.tforcing and (src_len>=t).byte().all():
#                 inp = src[:, t].detach()
#             else:
#                 inp = outp.max(-1)[1].detach()
            
        inp = torch.zeros(B).to(src.device).long() + self.pad_idx
        hidden = enc_hidden.view(1, B, K, -1).mean(2)
        '''
        src BK, T
        mask BK, T
        hidden L, B, H
        encodings BK, T, 2*H
        first_level_attn BK, T
        '''
        if self.cov:
            coverage_agg = torch.zeros_like(src).float()
            coverage_agg_loss = 0
#         for t in range(T):
#             attn_feat = self.attn(hidden.permute(1, 0, 2).contiguous().view(B, 1, -1))
#             first_level_attn = attn_feat.repeat(1, K, 1).view(B*K, 1, -1).expand(B*K, T, self.H) + attn_enc
#             if self.cov:
#                 first_level_attn = first_level_attn + self.W_c(coverage_agg.unsqueeze(-1))
#             first_level_attn = self.v(torch.tanh(first_level_attn)).squeeze(2)
#             first_level_attn = first_level_attn.masked_fill(mask == 0, -1e10)
#             first_level_attn = F.softmax(first_level_attn, dim=1)
#             if self.cov:
#                 coverage_agg_loss = coverage_agg_loss + torch.sum(mask*torch.min(coverage_agg, first_level_attn))
#                 coverage_agg = coverage_agg + first_level_attn
#             attn_vec = torch.mul(first_level_attn.unsqueeze(2), encodings).sum(1).view(B, K, -1)#.max(1)[0] # BxKx2H
#             second_level_attn = self.v(torch.tanh(attn_feat.expand(B, K, self.H) + self.W_h(attn_vec))).squeeze(2)
#             second_level_attn = F.softmax(second_level_attn, dim=1)
#             attn_vec = torch.mul(second_level_attn.unsqueeze(2), attn_vec).sum(1)
# #             attn_vec = attn_vec.mean(1)
#             outp, hidden, p_gen = self.decoder(inp, hidden, attn_vec.unsqueeze(0))
#             outp = p_gen * outp
#             outp = outp.scatter_add(1, src.view(B, -1), (1 - p_gen) * (second_level_attn.view(-1, 1) * first_level_attn).view(B, -1))
# #             outp = outp.scatter_add(1, src.view(B, -1), ((1 - p_gen)/K)*first_level_attn.view(B, -1))
#             outputs[t] = outp
#             if random.random() < self.tforcing and (tgt_idx[:, t]!=0).byte().all():
#                 inp = tgt_idx[:, t].detach()
#             else:
#                 inp = outp.max(-1)[1].detach()
        for t in range(T):
            attn_feat = self.attn(hidden.permute(1, 0, 2).contiguous().view(B, 1, -1))
            first_level_attn = attn_feat.repeat(1, K, 1).view(B*K, 1, -1).expand(B*K, T, self.H) + attn_enc
            if self.cov:
                first_level_attn = first_level_attn + self.W_c(coverage_agg.unsqueeze(-1))
            first_level_attn = self.v(torch.tanh(first_level_attn)).squeeze(2)
            first_level_attn = first_level_attn.masked_fill(mask == 0, -1e10).view(B, K*T)
            first_level_attn = F.softmax(first_level_attn, dim=1)
            if self.cov:
                coverage_agg_loss = coverage_agg_loss + torch.sum(mask*torch.min(coverage_agg, first_level_attn.view(coverage_agg.shape)))
                coverage_agg = coverage_agg + first_level_attn.view(coverage_agg.shape)
            attn_vec = torch.mul(first_level_attn.unsqueeze(2), encodings.view(B, K*T, -1)).sum(1).view(B, -1)#.max(1)[0] # BxKx2H
#             second_level_attn = self.v(torch.tanh(attn_feat.expand(B, K, self.H) + self.W_h(attn_vec))).squeeze(2)
#             second_level_attn = F.softmax(second_level_attn, dim=1)
#             attn_vec = torch.mul(second_level_attn.unsqueeze(2), attn_vec).sum(1)
#             attn_vec = attn_vec.mean(1)
            outp, hidden, p_gen = self.decoder(inp, hidden, attn_vec.unsqueeze(0))
            outp = p_gen * outp
            outp = outp.scatter_add(1, src.view(B, -1), (1 - p_gen) * first_level_attn.view(B, -1))
#             outp = outp.scatter_add(1, src.view(B, -1), ((1 - p_gen)/K)*first_level_attn.view(B, -1))
            outputs[t] = outp
            if random.random() < self.tforcing and (tgt_idx[:, t]!=0).byte().all():
                inp = tgt_idx[:, t].detach()
            else:
                inp = outp.max(-1)[1].detach()
        if self.cov:        
            return torch.log(outputs), coverage_agg_loss/(B*K*T)
#             return 100*torch.log(outputs), 100*torch.log(outputs_identity), coverage_agg_loss/(B*K*T), coverage_identity_loss/(B*K*T)
        return torch.log(outputs)
