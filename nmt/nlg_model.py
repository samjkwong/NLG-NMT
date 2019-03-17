#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from nmt_model import NMT
import os

class NLG(nn.Module):
    # Natural Language Generation model using a Neural Machine Translation context 
    # https://arxiv.org/abs/1702.07826

    def __init__(self, speakers, embed_size, hidden_size, dropout_rate, vocab, no_char_decoder, lr, clip_grad, lr_decay):
        # Need: NMT Model for each speaker (ex: translate from "person speaking to Michael" to "Michael")
        # Model for determining who speaks after who
        super(NLG, self).__init__()
        self.NMT_speakers = []
        self.NMT_models = []
        self.NMT_optimizers = []
        self.clip_grad = clip_grad
        self.lrs = []
        self.lr_decay = lr_decay
        # find a way to not have to hard-code speakers?
        for speaker in speakers:
            model = NMT(embed_size=embed_size,
                hidden_size=hidden_size,
                dropout_rate=dropout_rate,
                vocab=vocab, no_char_decoder=no_char_decoder)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            self.NMT_speakers.append(speaker.replace("/", "-").replace(" ","-"))
            self.NMT_models.append(model)
            self.NMT_optimizers.append(optimizer)
            self.lrs.append(lr)

    # change for double training?
    # def forward(self, speaker: str, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
    def forward(self, speaker: str, source: List[str], target: List[str]):
        if speaker in self.NMT_speakers:
            i = self.NMT_speakers.index(speaker)
            model = self.NMT_models[i]
            optimizer = self.NMT_optimizers[i]
            lr = self.lrs[i]

            optimizer.zero_grad()

            batch_size = 1

            example_losses = -model([source], [target]) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)

            optimizer.step()

            lr = optimizer.param_groups[0]['lr'] * self.lr_decay
            self.lrs[i] = lr

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return batch_loss
        return 0

    def beam_search(self, speaker, src_sent, beam_size=50, max_decoding_time_step=70):
        i = self.NMT_speakers.index(speaker)
        model = self.NMT_models[i]

        was_training = model.training
        model.eval()

        example_hyps = []
        with torch.no_grad():
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

        if was_training: model.train(was_training)

        return example_hyps

    '''
    @staticmethod
    def load(directory: str, no_char_decoder=False):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        args = params['args']
        model = NMT(vocab=params['vocab'], no_char_decoder=no_char_decoder, **args)
        model.load_state_dict(params['state_dict'])
        for filename in os.listdir(directory):
            NMT_model = NMT.load(directory + "/" + filename, no_char_decoder=no_char_decoder)
        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embed_size,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate,
                no_char_decoder=self.no_char_decoder),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

        for i in range(len(self.NMT_speakers)):
            self.NMT_models[i].save("models/" + self.NMT_speakers[i] + "_" + path)
    '''