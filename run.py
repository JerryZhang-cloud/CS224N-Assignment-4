#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2022-23: Homework 4
run.py: Run Script for Simple NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
Siyan Li <siyanli@stanford.edu>

Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""
import math
import sys
import pickle
import time
import os


from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import sacrebleu
from nmt_model import Hypothesis, NMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import read_corpus, batch_iter  #. Jerry: it means from utils.py import method read_coprpous and batch_iter
from vocab import Vocab, VocabEntry

import torch
import torch.nn.utils
from torch.utils.tensorboard import SummaryWriter


def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """

    """ JERRY
    Purpose: This line saves the current training mode of the model (model.training) into the variable was_training.

    Explanation:

    In PyTorch, models can be in one of two modes: training mode (model.train()) or evaluation mode (model.eval()).

    The model.training attribute is True if the model is in training mode and False if it is in evaluation mode.

    We save this state so that we can restore it later after evaluating the model."""
    
    was_training = model.training

    """ JERRY
    Purpose: This switches the model to evaluation mode.

    Explanation:

    In evaluation mode, certain layers (e.g., dropout, batch normalization) behave differently than in training mode.

    For example:

    Dropout: In training mode, dropout randomly zeros some neurons to prevent overfitting. In evaluation mode, dropout is turned off, and all neurons are used.

    Batch Normalization: In training mode, batch normalization uses the statistics of the current batch. In evaluation mode, it uses running averages of statistics computed during training.

    By calling model.eval(), we ensure that the model behaves consistently during evaluation."""
    model.eval()  # Jerry, swtich model to evaluation stage

    cum_loss = 0.
    cum_tgt_words = 0.

    """with torch.no_grad():
    Purpose: This context manager disables gradient computation.

    Explanation:

    During evaluation, we don't need to compute gradients because we're not updating the model's parameters (no backpropagation is performed).

    Disabling gradient computation reduces memory usage and speeds up computation.

    Inside the with torch.no_grad() block, all operations on tensors will not track gradients."""
   
    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item() # JERRY: loss is a tensor containing the loss value for the current batch. loss.item() extracts the scalar value from the tensor and converts it to a Python float. cum_loss is a running total of the loss across all batches. By adding loss.item() to cum_loss, we keep track of the total loss for the entire dataset.
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:  #JERRY: TRUE: Training Stage; FALSE: Evaluation Stage
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    # remove the start and end tokens
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    
    # detokenize the subword pieces to get full sentences
    detokened_refs = [''.join(pieces).replace('▁', ' ') for pieces in references]
    detokened_hyps = [''.join(hyp.value).replace('▁', ' ') for hyp in hypotheses]

    # sacreBLEU can take multiple references (golden example per sentence) but we only feed it one
    bleu = sacrebleu.corpus_bleu(detokened_hyps, [detokened_refs])

    return bleu.score


def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    train_data_src = read_corpus(args['--train-src'], source='src', vocab_size=21000)       # EDIT: NEW VOCAB SIZE
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt', vocab_size=8000)

    dev_data_src = read_corpus(args['--dev-src'], source='src', vocab_size=3000)
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt', vocab_size=2000)

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    #. It opens the file specified by args['--vocab',It reads the vocabulary (likely stored in JSON or text format).It creates and returns a Vocab object.Would create a Vocab object that maps words to their corresponding IDs.
    #. vocab result is a dictionary word2ID, here the word refers to subword(check CS224N)
    #. content of json file is not readable,but content of vocab file is readable
    vocab = Vocab.load(args['--vocab'])  # vocab.json file

    # model = NMT(embed_size=int(args['--embed-size']),                                 # EDIT: 4X EMBED AND HIDDEN SIZES 
    #             hidden_size=int(args['--hidden-size']),
    #             dropout_rate=float(args['--dropout']),
    #             vocab=vocab)

    model = NMT(embed_size=1024,
                hidden_size=768,
                dropout_rate=float(args['--dropout']),
                vocab=vocab)
    
    tensorboard_path = "nmt" if args['--cuda'] else "nmt_local"
    writer = SummaryWriter(log_dir=f"./runs/{tensorboard_path}")

    """ 
    Jerry: 
    1) the train() method in model.train() is inheritated from PyTorch's nn.Module class. 
    2) train() is from parent class that's why you don't see from NMT implementation.
    3) the purpose of train() is to switch to the training mode, so not the training itself
    4) training mode means it affects layers like dropout(?) and batchnorm(?), and put the model in a state where it expects to be trained with backpropagation. 
    """
    model.train() 
    

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)  #. Jerry:initalize parameter weights -> The uniform_() function fills the tensor with random values sampled from range (-uniform_init, uniform_init)

    #. How to understand mask? -> "masking" in machine learning, especially in sequence models, comes from the idea of hiding or excluding certain elements from computations. 
    vocab_mask = torch.ones(len(vocab.tgt)) # Jerry:This creates a 1D tensor vocab_mask filled with ones. The length of the tensor is equal to the size of the target vocabulary (len(vocab.tgt)).
    vocab_mask[vocab.tgt['<pad>']] = 0 # Jerry: Masking: This operation creates a mask where every element in vocab_mask is 1, except for the <pad> token, which is set to 0. The mask is commonly used to exclude the <pad> token during computations like attention in sequence models (e.g., transformers), where you don’t want the padding tokens to influence the model's behavior.

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)                       # EDIT: SMALLER LEARNING RATE

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1
        print("epoch : ",epoch)
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            example_losses = -model(src_sents, tgt_sents) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            # Jerry: here you normally run into the confusion that how come backword() can be applied to a num directly?  actually you can now understand this is as special PyTorch mechnism
            # Jerry: instead of treaing loss as a single number, you understand PyTorch traverse the full calculation formula that finally get num loss, and then backpropogate this full process to calculate gradient w.r.t all parameters.
            loss.backward() # Jerry: Purpose: Performs backpropagation, which calculates the gradients of the loss with respect to the model's parameters (weights).

            # clip gradient
            # Jerry: why clipping? because the caluclated gradient maybe too large which will be used later to to optimization on the parameters. this would cause too big adjustment to the parameters and causing model behave worse instead of better
            # Jerry: so the idea is to normalize gradient value to a given range (check more details in CS224N learning notes), and only use normalized grandient to optimzie model parameters
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step() #Jerry: TODO: put the parameter optmization formula here....

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                writer.add_scalar("loss/train", report_loss / report_examples, train_iter)
                writer.add_scalar("perplexity/train", math.exp(report_loss / report_tgt_words), train_iter)
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                writer.add_scalar("loss/val", cum_loss / cum_examples, train_iter)
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                writer.add_scalar("perplexity/val", dev_ppl, train_iter)
                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

            if epoch == int(args['--max-epoch']):
                print('reached maximum number of epochs!', file=sys.stderr)
                exit(0)


def decode(args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src', vocab_size=3000)
    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt', vocab_size=2000)

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src,
                            #  beam_size=int(args['--beam-size']),                      EDIT: BEAM SIZE USED TO BE 5
                             beam_size=10,
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score), file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ''.join(top_hyp.value).replace('▁', ' ')
            f.write(hyp_sent + '\n')


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def main():
    
    print("just to be sure i am already in the main code")
    """ Main func.
    """
    # args = docopt(__doc__)  Jerry: temporarily disable to enable the debug mode from vs code

    args = {'--batch-size': '32',
            '--beam-size': '5',
            '--clip-grad': '5.0',
            '--cuda': False,
            '--dev-src': './zh_en_data/dev.zh',
            '--dev-tgt': './zh_en_data/dev.en',
            '--dropout': '0.3',
            '--embed-size': '256',
            '--help': False,
            '--hidden-size': '256',
            '--input-feed': False,
            '--log-every': '10',
            '--lr': '5e-5',
            '--lr-decay': '0.5',
            '--max-decoding-time-step': '70',
            '--max-epoch': '30',
            '--max-num-trial': '5',
            '--patience': '5',
            '--sample-size': '5',
            '--save-to': 'model.bin',
            '--seed': '0',
            '--train-src': './zh_en_data/train_debug.zh',
            '--train-tgt': './zh_en_data/train_debug.en',
            '--uniform-init': '0.1',
            '--valid-niter': '2000',
            '--vocab': 'vocab.json',
            'MODEL_PATH': None,
            'OUTPUT_FILE': None,
            'TEST_SOURCE_FILE': None,
            'TEST_TARGET_FILE': None,
            'decode': False,
            'train': True}
    
    print(args)

    # Check pytorch version
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')

# Jerry: _name_ is special variable, when a python file is run directly (i.e. you execute it as python script.py), the Python interpreter sets _name_ to "_main_"
# Jerry: if _name_ == '_main_' ensures that the code inside the block only runs when the script is executed directly, not when it is imported as a module.
if __name__ == '__main__':
    print ("before main()")
    print("Current directory:", os.getcwd())
    main()
