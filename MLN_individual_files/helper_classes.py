from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from MLN_individual_files.helper_classes import *
import numpy as np
import pickle
from transformers import get_scheduler
from transformers import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class AbstractDataset(Dataset):
    def __init__(self, inputs, outputs, filter='none'):
        if filter == "none":
            self.filter = lambda x: True
        elif filter == "alnum":
            def replace(word): return word.replace("'", "").replace("-", "").replace(" ", "")
            self.filter = lambda x: len(replace(x)) == 0 or replace(x).isalnum()

class Indexer:
    def __init__(self, valid_indices):
        self.valid_indices = valid_indices
        lengths = [len(sentence_indices) for sentence_indices in self.valid_indices]
        self.cumsum = torch.LongTensor([0] + lengths).cumsum(dim=0)

    def get_indices(self, index):
        sentence_index = torch.searchsorted(self.cumsum, index, right=True).item() - 1
        word_index = index - self.cumsum[sentence_index]
        word_index = self.valid_indices[sentence_index][word_index]

        return sentence_index, word_index

    def __len__(self):
        return self.cumsum[-1].item()

class MultilexnormDataset(AbstractDataset):
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs)
        self.inputs = inputs
        self.outputs = outputs

        valid_indices = [[i for i, word in enumerate(sentence) if self.filter(word)] for sentence in inputs]
        self.indexer = Indexer(valid_indices)

    def __getitem__(self, index):
        sentence_index, word_index = self.indexer.get_indices(index)

        out = self.outputs[sentence_index][word_index]
        raw = self.inputs[sentence_index]

        raw = raw[:word_index] + ["<extra_id_0>", raw[word_index], "<extra_id_1>"] + raw[word_index+1:]
        raw = ' '.join(raw)

        return raw, out, sentence_index, word_index

    def __len__(self):
        return len(self.indexer)

from typing import List
class MultiPlexDataset(Dataset):

    def __init__(self,
                 X,
                 y,
                 only_include_corrections: bool = True):
        """

        :param path_to_files: List of paths to the files with data
        :param only_include_corrections: Whether to only include samples where there are corrections
        """

        self.only_include_corrections = only_include_corrections
        self.dataset_counter = 0
        self.data = {}


        for norms, refs in zip(X,y):
          self.create_samples(norms,refs)

        print("Dataset initialized...")

    def create_samples(self, norms, refs):
        if norms and refs:
            for i, word in enumerate(norms):

                if self.only_include_corrections and word == refs[i]:
                    continue

                if i == 0:
                    sample_input = "<extra_id_0>" + word + "<extra_id_1> " + " ".join(norms[i + 1:])
                elif i == len(norms) - 1:
                    sample_input = " ".join(norms[:i]) + " <extra_id_0>" + word + "<extra_id_1>"
                else:
                    sample_input = " ".join(norms[:i]) + " <extra_id_0>" + word + "<extra_id_1> " + " ".join(
                        norms[i + 1:])

                self.data[self.dataset_counter] = {"input_sample": sample_input, "expected_output": refs[i]}
                self.dataset_counter += 1

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data.keys())

def open_dataset(path, load_outputs=True):
    with open(path) as f:
        sentences = f.read().split("\n\n")[:-1]
    sentences = [s.split('\n') for s in sentences]
    inputs = [[w.split('\t')[0] for w in s] for s in sentences]

    if not load_outputs:
        return inputs

    outputs = [[w.split('\t')[1] for w in s] for s in sentences]
    return inputs, outputs


class CollateFunctor_Train:
    def __init__(self, tokenizer, encoder_max_length=320, decoder_max_length=32):
        self.tokenizer = tokenizer
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length

    def __call__(self, samples):
        inputs = list(map(lambda x: x["input_sample"], samples))

        inputs = self.tokenizer(
            inputs, padding=True, truncation=True, pad_to_multiple_of=8,
            max_length=self.encoder_max_length, return_attention_mask=True, return_tensors='pt'
        )

        outputs = list(map(lambda x: x["expected_output"], samples))

        outputs = self.tokenizer(
            outputs, padding=True, truncation=True, pad_to_multiple_of=8,
            max_length=self.decoder_max_length, return_attention_mask=True, return_tensors='pt'
        )

        batch = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": outputs.input_ids,
            "decoder_attention_mask": outputs.attention_mask
        }
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100  # used to mask the loss in T5
        return batch

import os
import os.path

class OutputAssembler:
    def __init__(self, directory, dataset):
        self.directory = directory
        self.dataset = dataset
        self.postprocessing = {
            "none": NonePostprocessor,
            "alnum": AlnumPostprocessor,
        }['alnum'](1.0)

        self.cache = {}

    def step(self, output_dict):
        output_dict = (output_dict["predictions"], output_dict["scores"], output_dict["sentence_ids"], output_dict["word_ids"])
        for word_preds, scores, sent_id, word_id in zip(*output_dict):
            word_preds = [w.replace('\n', '').replace('\t', ' ') for w in word_preds]
            pairs = list(zip(word_preds, scores))

            self.cache.setdefault(sent_id, {})[word_id] = pairs

    def flush(self):
        predictions = self.assemble(self.cache)
        inputs = self.dataset.inputs

        raw_path = f"{self.directory}/raw_outputs_mln_base.txt"
        postprocessed_path = f"{self.directory}/outputs_mln_base.txt"

        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        with open(raw_path, "w") as f:
            for i, input_sentence in enumerate(inputs):
                for j, input_word in enumerate(input_sentence):
                    try:
                        prediction_string = '\t'.join([f"{w}\t{s}" for w, s in predictions[i][j]])
                    except:
                        print(i, j, len(predictions[i]))
                        for k, p in enumerate(predictions[i]):
                            print(k, p)
                        print(flush=True)
                        exit()
                    line = f"{input_word}\t{prediction_string}"
                    f.write(f"{line}\n")
                f.write("\n")

        self.postprocessing.process_file(raw_path, postprocessed_path)

    def assemble(self, prediction_dict):
        prediction_list = []
        for sent_id, raw_sentence in enumerate(self.dataset.inputs):
            prediction_list.append(
                [prediction_dict.get(sent_id, {}).get(word_id, [(raw_word, 0.0)]) for word_id, raw_word in enumerate(raw_sentence)]
            )

        return prediction_list

class AbstractPostprocessor:
    def __init__(self, bias=1.0):
        self.bias = bias

    def __call__(self, raw, predictions):
        pass

    def process_file(self, input_path, output_path):
        with open(input_path, "r") as f:
            sentences = f.read().split("\n\n")[:-1]
            sentences = [s.split('\n') for s in sentences]

        with open(output_path, "w") as f:
            for sentence in sentences:
                for word in sentence:
                    raw, *predictions = word.split('\t')
                    predictions = [(word, float(score)) for word, score in zip(predictions[::2], predictions[1::2])]
                    prediction = self(raw, predictions)
                    f.write(f"{raw}\t{prediction}\n")
                f.write("\n")

    def rebalance(self, raw, predictions):
        predictions = [(w, s) if w != raw else (w, s*self.bias) for w, s in predictions]
        predictions = sorted(predictions, key=lambda item: item[1], reverse=True)
        return predictions


class NonePostprocessor(AbstractPostprocessor):
    def __call__(self, raw, predictions):
        predictions = self.rebalance(raw, predictions)
        return predictions[0][0]


class AlnumPostprocessor(AbstractPostprocessor):
    def __call__(self, raw, predictions):
        if raw.isdigit() and len(raw) > 1:
            return raw
        if not raw.replace("'", "").isalnum():
            return raw
        predictions = self.rebalance(raw, predictions)
        return predictions[0][0]

class CollateFunctor:
    def __init__(self, tokenizer, encoder_max_length, decoder_max_length):
        self.tokenizer = tokenizer
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length

    def __call__(self, samples):
        inputs, outputs, sentence_indices, word_indices = map(list, zip(*samples))
        
        inputs = self.tokenizer(
            inputs, padding=True, truncation=True, pad_to_multiple_of=8,
            max_length=self.encoder_max_length, return_attention_mask=True, return_tensors='pt'
        )
        outputs = self.tokenizer(
            outputs, padding=True, truncation=True, pad_to_multiple_of=8,
            max_length=self.decoder_max_length, return_attention_mask=True, return_tensors='pt'
        )
        

        batch = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": outputs.input_ids,
            "decoder_attention_mask": outputs.attention_mask,
            "word_ids": word_indices,
            "sentence_ids": sentence_indices
        }
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100  # used to mask the loss in T5
        return batch

from torch.utils.data import DataLoader
from torch.optim import AdamW
def get_train_dataloader(dataset,tokenizer):
    collate_fn = CollateFunctor(tokenizer, 320, 32)

    return DataLoader(
        dataset, batch_size=8, shuffle=False, drop_last=True,
        num_workers=0, collate_fn=collate_fn
    )



import Levenshtein as Lev
def wer(s1, s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """

    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    return Lev.distance(''.join(w1), ''.join(w2))

def wer_normalized(s1, s2):
  return wer(s1.lower(),s2.lower()) / len(s2.split(" "))
def calculate_wer(df):
  return wer_normalized(df['corrected'],df['reference_text'])

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu


def calculate_bleu_baseline_normalized(df):
  total_bleu = 0
  for i in range(len(df)):
    ref = [df['reference_text'].iloc[i].split(" ")]
    hyp = df['transcription'].iloc[i].split(" ")
    sentence_bleu_score = sentence_bleu(ref, hyp)
    total_bleu += sentence_bleu_score
  return total_bleu / len(df)

def calculate_gleu_baseline_normalized(df):
  total_gleu = 0
  for i in range(len(df)):
    ref = [df['reference_text'].iloc[i].split(" ")]
    hyp = df['transcription'].iloc[i].split(" ")
    sentence_gleu_score = sentence_gleu(ref, hyp)
    total_gleu += sentence_gleu_score
  return total_gleu / len(df)

def calculate_bleu_normalized(df):
  total_bleu = 0
  for i in range(len(df)):
    ref = [df['reference_text'].iloc[i].split(" ")]
    hyp = df['corrected'].iloc[i].split(" ")
    sentence_bleu_score = sentence_bleu(ref, hyp)
    total_bleu += sentence_bleu_score
  return total_bleu / len(df)

def calculate_gleu_normalized(df):
  total_gleu = 0
  for i in range(len(df)):
    ref = [df['reference_text'].iloc[i].split(" ")]
    hyp = df['corrected'].iloc[i].split(" ")
    sentence_gleu_score = sentence_gleu(ref, hyp)
    total_gleu += sentence_gleu_score
  return total_gleu / len(df)


  def open_dataset(path, load_outputs=True):
    with open(path) as f:
        sentences = f.read().split("\n\n")[:-1]
    sentences = [s.split('\n') for s in sentences]
    inputs = [[w.split('\t')[0] for w in s] for s in sentences]

    if not load_outputs:
        return inputs

    outputs = [[w.split('\t')[1] for w in s] for s in sentences]
    return inputs, outputs