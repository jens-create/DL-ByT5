from torch.utils.data import Dataset


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
