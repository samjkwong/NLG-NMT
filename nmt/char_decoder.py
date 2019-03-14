#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()

        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size, bidirectional=False)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id), bias=True)
        pad_token_idx = self.target_vocab.char2id['<pad>']
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size, padding_idx=pad_token_idx)

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        embeddings = self.decoderCharEmb(input)
        dec_hidden, dec_cell = self.charDecoder(embeddings, dec_hidden)
        scores = self.char_output_projection(dec_hidden)

        return (scores, dec_cell)
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        #print(char_sequence)
        #print(char_sequence.size())
        inpt = char_sequence.narrow(0, 0, char_sequence.size()[0] - 1)
        #print(inpt)
        #print(inpt.size())
        target = char_sequence.narrow(0, 1, char_sequence.size()[0] - 1)
        #print(target)
        #print(target.size())
        scores, dec_hidden = self.forward(inpt, dec_hidden)
        #target = self.decoderCharEmb(target)
        #print(target.size())
        #print(scores.size())
        target = target.contiguous()
        target = target.view(target.size()[0] * target.size()[1])
        scores = scores.view(scores.size()[0] * scores.size()[1], scores.size()[2])
        #print(target.size())
        #print(scores.size())
        loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.target_vocab.char2id['<pad>'])
        return loss(scores, target)

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        output_chars = []
        #print(initialStates[0].size()[1])
        for i in range(initialStates[0].size()[1]):
            output_chars.append([])
        #print(output_chars)
        #output_chars = [[] * initialStates[0].size()[1]]
        current_char = [[self.target_vocab.start_of_word] * initialStates[0].size()[1]]
        dec_hidden = initialStates
        for i in range(max_length):
            current_char = torch.tensor(current_char, device=device)
            scores, dec_hidden = self.forward(current_char, dec_hidden=dec_hidden)
            #embedding = self.decoderCharEmb(current_char)
            #print(embedding.size())
            #dec_hidden = self.charDecoder(embedding, dec_hidden)
            #print('hi')
            #scores = self.char_output_projection(dec_hidden[0])
            softmax = nn.Softmax(dim=2)
            p = softmax(scores)
            current_char = torch.argmax(p, dim=2)
            #print(current_char.size())
            chars = current_char.tolist()[0]
            #print(output_chars)
            #print(len(chars))
            for i in range(len(chars)):
                output_chars[i].append(chars[i])
        #print(output_chars)
        '''
        output_words = []
        idx = 0
        i = 0
        word = ""
        while i < len(output_chars):
            if idx < 21:
                print(self.target_vocab.id2char[self.target_vocab.end_of_word])
                if output_chars[i] != self.target_vocab.id2char[self.target_vocab.end_of_word]:
                    word += output_chars[i]
                    idx += 1
                else:
                    i += 21 - idx
                    idx = 0
                    output_words.append(word)
                    word = output_chars[i]
            else:
                idx = 0
                output_words.append(word)
                word = output_chars[i]
            i += 1
        output_words.append(word)
        print(output_words)
        '''
        #print(len(output_chars))
        '''
        output_chars_list = []
        idx = 0
        chars_list = []
        for i in range(len(output_chars)):
            if idx < 21:
                idx += 1
                chars_list.append(output_chars[i])
            else:
                output_chars_list.append(chars_list)
                idx = 0
                chars_list = [output_chars[i]]
        output_chars_list.append(chars_list)
        '''

        output_words = []
        for chars_list in output_chars:
            word = ""
            for c in chars_list:
                if c != self.target_vocab.end_of_word and c!= self.target_vocab.char2id['<pad>']:
                    word += self.target_vocab.id2char[c]
                else:
                    break
            output_words.append(word)

        return output_words
        
        ### END YOUR CODE

