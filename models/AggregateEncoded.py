from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import argparse
import json

# def parse_encoder_decoder_opt():
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--input_json', type=str, default='data/cocotalk.json',
# 		help='path to the json file containing additional info and vocab')
# 	parser.add_argument('--checkpoint', type=str, default=None,
# 		help='Initialize model from checkpoint')
# 	parser.add_argument('--custom_decoder_emb_dim', type=int, default=256,
# 		help='Embedding dimension for encoder decoder')
# 	parser.add_argument('--custom_decoder_hid_dim', type=int, default=512,
# 		help='Dimension of hidden layer')
# 	parser.add_argument('--custom_decoder_num_hidden', type=int, default=2,
# 		help='Number of hidden layers')
# 	parser.add_argument('--custom_decoder_enc_drop', type=float, default=0.5,
# 		help='Dropout of encoder')
# 	parser.add_argument('--custom_decoder_dec_drop', type=float, default=0.5,
# 		help='Dropout of decoder')
# 	parser.add_argument('--n_augs', type=int, default=2,
# 		help='Number of augmented images')
# 	parser.add_argument('--preproc', help='Are inputs preprocessed',
# 		action='store_true')

# 	args = parser.parse_args()
# 	return args


class Encoder(nn.Module):
	def __init__(self, embedding_layer, emb_dim, hid_dim, n_layers, dropout, n_aug, bi):
		super().__init__()
		
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		self.n_aug= n_aug
		self.embedding = embedding_layer
		self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, bidirectional=bi)
		
	def forward(self, src, src_len):
		
		#src = [src len, batch size, n_aug]
		embedded = self.embedding(src)
		#embedded = [src len, batch size, emb dim*n_aug]
		pack_embedded= pack(embedded, src_len, enforce_sorted=False)

		outputs, (hidden, cell) = self.rnn(pack_embedded)
		#outputs = [src len, batch size, hid dim ]
		#hidden = [n layers * n directions, batch size, hid dim]
		#cell = [n layers * n directions, batch size, hid dim]
		
		#outputs are always from the top hidden layer
		
		return hidden, cell


class Decoder(nn.Module):
	def __init__(self, vocab_size, embedding_layer, emb_dim, hid_dim, n_layers, dropout, bi):
		super().__init__()
		
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		self.embedding = embedding_layer
		self.emb_dim= emb_dim
		if bi:
			self.n_layers*=2
		self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers, dropout = dropout)
		self.fc_out = nn.Linear(hid_dim, vocab_size)
		self.softmax= nn.LogSoftmax(dim=1)
		self.vocab_size= vocab_size


	def forward(self, input, hidden, cell):
		
		#input = [batch size]
		#hidden = [n layers * n directions, batch size, hid dim]
		#cell = [n layers * n directions, batch size, hid dim]
		
		#n directions in the decoder will both always be 1, therefore:
		#hidden = [n layers, batch size, hid dim]
		#context = [n layers, batch size, hid dim]
		
		input = input.unsqueeze(0)
		
		#input = [1, batch size]
		
		embedded = self.embedding(input)
		
		#embedded = [1, batch size, emb dim]
		output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
		
		#output = [seq len, batch size, hid dim * n directions]
		#hidden = [n layers * n directions, batch size, hid dim]
		#cell = [n layers * n directions, batch size, hid dim]
		
		#seq len and n directions will always be 1 in the decoder, therefore:
		#output = [1, batch size, hid dim]
		#hidden = [n layers, batch size, hid dim]
		#cell = [n layers, batch size, hid dim]
		
		prediction = self.softmax(self.fc_out(output.squeeze(0)))

		
		#prediction = [batch size, output dim]
		
		return prediction, hidden, cell


class Seq2Seq(nn.Module):
	def __init__(self, opt):
		super().__init__()
		
		self.vocab_size = len(opt.vocab)+1 # <PAD>
		EMB_DIM = opt.custom_decoder_emb_dim
		HID_DIM = opt.custom_decoder_hid_dim
		N_LAYERS = opt.custom_decoder_num_hidden
		ENC_DROPOUT = opt.custom_decoder_enc_drop
		DEC_DROPOUT = opt.custom_decoder_dec_drop
		N_AUG= opt.n_augs
		self.pad_idx= 0
		MAX_LEN=8
		EMBEDDING_LAYER= nn.Linear(self.vocab_size, EMB_DIM, bias=False)
		BIDIRECTIONAL= opt.custom_decoder_bi
		self.encoder= Encoder(EMBEDDING_LAYER, EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, N_AUG, BIDIRECTIONAL)
		self.decoder= Decoder(self.vocab_size, EMBEDDING_LAYER, EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, BIDIRECTIONAL)
		# self.device = None
		# assert self.encoder.hid_dim == self.decoder.hid_dim, \
		# 	"Hidden dimensions of encoder and decoder must be equal!"
		# assert self.encoder.n_layers == self.decoder.n_layers, \
		# 	"Encoder and decoder must have equal number of layers!"

		self.apply(init_weights)
		
	def forward(self, src):

		############# New input being passed is [T, B, K, V]
		#teacher_forcing_ratio is probability to use teacher forcing
		#e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
		# self.device= src.device
		
		batch_size = src.shape[1]
		trg_len = src.shape[0]
		trg_vocab_size = self.vocab_size

		src_idx= src.argmax(dim=3)
		src_len= torch.LongTensor(src.shape[1], src.shape[2]).zero_().to(src.device)
		for j in range(src.shape[1]):
			for k in range(src.shape[2]):
				for i in range(src.shape[0]):
					if src_idx[i,j,k]==self.pad_idx:
						src_len[j,k]=i+1
						break
		
		src_len[(src_len==0)]=trg_len
		# src_len= torch.argsort(src_len, dim=0)[0,:,:].squeeze() # [B,K]
		# mask= torch.arange(src.shape[0]).to(src.device)\
		# 	.unsqueeze(1).unsqueeze(2)\
		# 	.expand(src.shape[0], src.shape[1], src.shape[2]) < src_len.unsqueeze(0)

		# mask= mask.to(src.device)
		# src_len= src_len.max(dim=1)[0].contiguous()
		#tensor to store decoder outputs
		outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(src.device)
		
		#last hidden state of the encoder is used as the initial hidden state of the decoder
		encodings= [self.encoder(src[:,:,i,:], src_len[:,i]) for i in range(self.encoder.n_aug)]
		hidden= torch.mean(torch.stack([t[0] for t in encodings]), dim=0)
		cell= torch.mean(torch.stack([t[1] for t in encodings]), dim=0)
		#first input to the decoder is the <sos> tokens
		inp = torch.FloatTensor(batch_size, self.vocab_size).to(src.device)
		inp.fill_(0)
		inp[:,self.pad_idx]=1
		for t in range(trg_len):
			
			#insert input token embedding, previous hidden and previous cell states
			#receive output tensor (predictions) and new hidden and cell states
			outp, hidden, cell = self.decoder(inp, hidden, cell)
			
			#place predictions in a tensor holding predictions for each token
			outputs[t] = outp
			inp= torch.FloatTensor(batch_size, self.vocab_size).to(src.device).zero_()
			inp.scatter_(1,outp.argmax(dim=1).unsqueeze(1),1)			
			#decide if we are going to use teacher forcing or not
		
		return outputs

def init_weights(m):
	for name, param in m.named_parameters():
		nn.init.uniform_(param.data, -0.08, 0.08)


# def train_model(opt):
# 		#####################
# 	# Needs to be changed
# 	info = json.load(open(opt.input_json))
# 	ix_to_word = info['ix_to_word']
# 	ix_to_word = {int(k):v for k, v in ix_to_word.items()} 
# 	vocab = {v:k for k, v in ix_to_word.items()}
# 	VOCAB_SIZE = len(ix_to_word)+2 # <EOS> & <PAD>
# 	EMB_DIM = opt.emb_dim
# 	HID_DIM = opt.hid_dim
# 	N_LAYERS = opt.num_hidden
# 	ENC_DROPOUT = opt.enc_drop
# 	DEC_DROPOUT = opt.dec_drop
# 	N_AUG= opt.n_aug
# 	PAD_IDX= 0
# 	EOS_IDX= len(ix_to_word)+1
# 	MAX_LEN=8
# 	N_EPOCHS=30
# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 	criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
# 	####################
# 	if not opt.preproc:
# 		######### Input of the model
# 		caps= [["bike", "is", "red"], ["a", "boy", "on", "bike"], ["green", "forest"], ["dense", "forest"]]
# 		######### Simple list of split captions
# 		##########
# 		########## Outputs
# 		out_caps=[["a", "boy", "on", "red", "bike"], ["green", "dense", "forest"]]
# 		##########
# 		cap_len= len(caps)
# 		trn= torch.LongTensor(MAX_LEN, cap_len//N_AUG,N_AUG).fill_(PAD_IDX)	
# 		trn_len= torch.LongTensor(cap_len//N_AUG).fill_(0)	
# 		for i in range(0,cap_len,N_AUG):
# 			l=0
# 			for j in range(N_AUG):
# 				for t,w in enumerate(caps[i+j]):
# 					trn[t, i//N_AUG, j]= vocab[w]
# 				l= max(l, len(caps[i+j]))
# 			trn_len[i//N_AUG]= l
# 		tgt= torch.LongTensor(MAX_LEN, cap_len//N_AUG).fill_(0)
# 		for cap_num, cap in enumerate(out_caps):
# 			for word_num, word in enumerate(cap):
# 				tgt[word_num+1, cap_num]=vocab[word]
# 			tgt[len(cap)+1, cap_num]=EOS_IDX

# 	else:
# 		trn= torch.LongTensor([[[1,1],[2,3],[3,2]], [[1,1],[2,3],[3,2]], [[0,1],[2,0],[0,0]], [[0,0],[0,0],[0,0]]]) # time x batch x aug 
# 		trn_len= torch.LongTensor([3,3,2])
# 		tgt=torch.LongTensor([[0,0,0], [1,2,3], [1,2,0], [0,0,0]])
# 	EMBEDDING_LAYER= nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD_IDX)
# 	enc = Encoder(EMBEDDING_LAYER, EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, N_AUG)
# 	dec = Decoder(VOCAB_SIZE, EMBEDDING_LAYER, EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
# 	model = Seq2Seq(enc, dec, device).to(device)
# 	optimizer = optim.Adam(model.parameters())
# 	if opt.checkpoint:
# 		checkpoint= torch.load(opt.checkpoint)
# 		model.load_state_dict(checkpoint["model_state_dict"])
# 		model.train()
# 		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# 	else:
# 		model.train()
# 		model.apply(init_weights)
	
# 	for i in range(N_EPOCHS): ####### Test code just train on same batch 30 times
# 		optimizer.zero_grad()
# 		output= model(trn, trn_len, tgt)
# 		output_dim= output.shape[-1]
# 		output= output[1:].view(-1, output_dim)
# 		tgt_temp= tgt[1:].view(-1)
# 		loss= criterion(output, tgt_temp)
# 		loss.backward()
# 		nn.utils.clip_grad_norm_(model.parameters(), 0.5)
# 		optimizer.step()
# 		print("Iter i: ", i)
# 		print("Loss: ", loss.item())
# 		print("Training input shape: ", trn.shape)
# 		print("Target shape: ", tgt.shape)	

# 	with torch.no_grad():
# 		output= model(trn, trn_len, tgt)
# 		output_dim= output.shape[-1]
# 		output= output[1:].view(-1, output_dim)
# 		tgt_temp= tgt[1:].view(-1)
# 		loss= criterion(output, tgt_temp)
# 		print("Loss: ", loss.item())

# 	torch.save({
# 		'model_state_dict': model.state_dict(),
# 		'optimizer_state_dict': optimizer.state_dict()
# 	}, "checkpoint.pt")

# 	########################### Code to get final captions
# 	with torch.no_grad():
# 		cap= model.get_captions(trn, trn_len,MAX_LEN, PAD_IDX)
# 		print("Captions Shape: ", cap.shape)
# 		print("Captions are : ")
# 		for i in range(cap.shape[1]):
# 			tp=[]
# 			for j in range(cap.shape[0]):
# 				if j==0:
# 					continue
# 				if cap[j,i].item()==EOS_IDX:
# 					break
# 				tp.append(ix_to_word[cap[j,i].item()])
# 			print(tp)
# 	########################### 
		
# # opt={"checkpoint": "checkpoint.pt"}
# opt=parse_encoder_decoder_opt()
# train_model(opt)

######################### TESTING FOR SIMPLE ERRORS
# i=0
# while True:
# 	optimizer.zero_grad()
# 	output= model(trn, trn_len, tgt)
# 	output_dim= output.shape[-1]
# 	output= output[1:].view(-1, output_dim)
# 	tgt_temp= tgt[1:].view(-1)
# 	loss= criterion(output, tgt_temp)
# 	loss.backward()
# 	nn.utils.clip_grad_norm_(model.parameters(), 0.5)
# 	optimizer.step()
# 	epoch_loss+=loss.item()
# 	i+=1
# 	print("Iter i: ", i)
# 	print("Loss: ", loss.item())
# 	print("Training input shape: ", trn.shape)
# 	print("Target shape: ", tgt.shape)


###############################

# <SOS>, <EOS> in both output and input

# def train(model, iterator, optimizer, criterion, clip):
	
#     model.train()
	
#     epoch_loss = 0
	
#     for i, batch in enumerate(iterator):
		
#         src = batch.src
#         trg = batch.trg
		
#         optimizer.zero_grad()
		
#         output = model(src, trg)
		
#         #trg = [trg len, batch size]
#         #output = [trg len, batch size, output dim]
		
#         output_dim = output.shape[-1]
		
#         output = output[1:].view(-1, output_dim)
#         trg = trg[1:].view(-1)
		
#         #trg = [(trg len - 1) * batch size]
#         #output = [(trg len - 1) * batch size, output dim]
		
#         loss = criterion(output, trg)
		
#         loss.backward()
		
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		
#         optimizer.step()
		
#         epoch_loss += loss.item()
		
#     return epoch_loss / len(iterator)

# def get_captions(model, src, eos_idx, max_len):
#     model.eval()
#     epoch_loss = 0
#     with torch.no_grad():
#         output = model(src, eos_idx, max_len)         
#     return output


# def epoch_time(start_time, end_time):
#     elapsed_time = end_time - start_time
#     elapsed_mins = int(elapsed_time / 60)
#     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#     return elapsed_mins, elapsed_secs


# N_EPOCHS = 10
# CLIP = 1

# best_valid_loss = float('inf')

# for epoch in range(N_EPOCHS):
	
#     start_time = time.time()
	
#     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
#     # valid_loss = evaluate(model, valid_iterator, criterion)
	
#     end_time = time.time()
	
#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
	
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'decoder.pt')
	
#     print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
