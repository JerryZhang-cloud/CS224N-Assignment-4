#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2022-23: Homework 4
utils.py: Utility Functions
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
Siyan Li <siyanli@stanford.edu>
"""

import math
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import sentencepiece as spm
print("start punkt downloading...")


def ensure_punkt_downloaded():
    if not nltk.data.find('tokenizers/punkt'):
        print("Downloading 'punkt' tokenizer...")
        nltk.download('punkt')
    else:
        print("'punkt' tokenizer already downloaded.")
 
# 在程序开始时调用
ensure_punkt_downloaded()

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    """
    Jerry: use input pad_token to compensate to each sentence so all sentences are equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    # Find the length of the longest sentence in the batch    
    max_len = max(len(sentence) for sentence in sents)    
    
    sents_padded = []    
    for sentence in sents:        # Pad the sentence with pad_token to match the length of the longest sentence        
        padding = [pad_token] * (max_len - len(sentence))        
        sents_padded.append(sentence + padding)

    ### END YOUR CODE

    return sents_padded


def read_corpus(file_path, source, vocab_size=2500):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    @param vocab_size (int): number of unique subwords in
        vocabulary when reading and tokenizing
    """
    print("Current directory in util.py:", os.getcwd())

    #. This code is using the SentencePiece library for subword tokenization.
    #. SentencePieceProcessor() creates an instance of a tokenizer that can encode and decode text using a pre-trained SentencePiece model.

    data = []
    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(source))  #. Jerry:'{}.model'.format(source) dynamically formats the string by replacing {} with source. so it equals sp.load('src.model')

    with open(file_path, 'r', encoding='utf8') as f:  #. Jerry: 'r' means in read mode; with ... as f: The with statement ensures that the file is automatically closed after reading, even if an error occurs.f is a file object that allows you to read the file’s content.
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            # only append <s> and </s> to the target sentence
            if source == 'tgt':
                subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)

    return data


def autograder_read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size) # Jerry:math.ceil() is used to round up the result. This ensures that if there are any leftover items
    index_array = list(range(len(data))) # Jerry:range(len(data)) generates a sequence of integers from 0 to len(data)-1. So, if len(data) = 5, this will produce [0, 1, 2, 3, 4].

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size] # Jerry: i * batch_size gives the start index for the current batch. after colon (i + 1) * batch_size gives the end index (non-inclusive) for the current batch.
        examples = [data[idx] for idx in indices] # what is examples? examples is a list of pairs, where each pair corresponds to a source sentence(e[0]) and a target sentencee[1]. The pair e would be a tuple (or list) containing two elements. 

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples] #. Jerry: e[0] is list of source sentences
        tgt_sents = [e[1] for e in examples] #. Jerry: e[1] is list of target sentences

        yield src_sents, tgt_sents

