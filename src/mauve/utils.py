# Author: Krishna Pillutla
# License: GPLv3
import json
import os
import time
from tqdm.auto import tqdm as tqdm_original

import torch
from transformers import AutoModel, AutoTokenizer


CPU_DEVICE = torch.device('cpu')
tqdm = lambda *args, **kwargs: tqdm_original(
    *args, **kwargs, disable=os.environ.get("DISABLE_TQDM", False))


def get_device_from_arg(device_id):
    if (device_id is not None and
            torch.cuda.is_available() and
            0 <= device_id < torch.cuda.device_count()):
        return torch.device(f'cuda:{device_id}')
    else:
        return CPU_DEVICE

def get_model(model_name, tokenizer, device_id):
    device = get_device_from_arg(device_id)
    if 'gpt2' in model_name:
        model = AutoModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).to(device)
        model = model.eval()
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return model

def get_tokenizer(model_name='gpt2'):
    if 'gpt2' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return tokenizer

def load_json_dataset(data_path, max_num_data):
    texts = []
    for i, line in enumerate(open(data_path)):
        if i >= max_num_data:
            break
        texts.append(json.loads(line)['text'])
    return texts

def load_and_tokenize_json_data(tokenizer, data_path, max_len=1024, max_num_data=float('inf')):
    """ Load and tokenize the data in a jsonl format

    :param tokenizer:  HF tokenizer object
    :param data_path: jsonl file to read. Read the "text" field of each line
    :param max_len: maximum length of tokenized data
    :param max_num_data: maximum number of lines to load
    :return: list of `torch.LongTensor`s of shape (1, num_tokens), one for each input line
    """
    assert max_len <= 1024 and max_num_data >= 2000, f"max_len={max_len}, max_num_data={max_num_data} are insufficent"
    t1 = time.time()
    texts = load_json_dataset(data_path, max_num_data=max_num_data)
    t2 = time.time()
    print(f'dataset load time: {round(t2-t1, 2)} sec')
    t1 = time.time()
    tokenized_texts = [tokenizer.encode(sen, return_tensors='pt', truncation=True, max_length=max_len)
                      for sen in texts]
    t2 = time.time()
    print(f'tokenizing time: {round(t2-t1, 2)} sec')
    return tokenized_texts

def decode_samples_from_lst(tokenizer, tokenized_texts):
    """ Decode from tokens to string

    :param tokenizer: HF tokenizer
    :param tokenized_texts: list of list of tokens
    :return: decoded output as a list of strings of the same length as tokenized_text_list
    """
    t1 = time.time()
    output = []
    for l in tokenized_texts:
        o = tokenizer.decode(torch.LongTensor(l), skip_special_tokens=True)
        output.append(o)
    t2 = time.time()
    print(f'de-tokenizing time: {round(t2-t1, 2)}')
    return output

@torch.no_grad()
def featurize_tokens_from_model(model, tokenized_texts, batch_size, name="", is_mean=False, verbose=False):
    """Featurize tokenized texts using models, support batchify

    :param model: HF Transformers model
    :param batch_size: Batch size used during forward pass
    :param tokenized_texts: list of torch.LongTensor of shape (1, length)
    :param verbose: If True, print status and time
    :return:
    """
    device = next(model.parameters()).device
    t1 = time.time()

    feats, chunks, chunk_sent_lengths = [], [], []
    chunk_idx = 0

    while chunk_idx * batch_size < len(tokenized_texts):
        _chunk = [_t.view(-1) for _t in tokenized_texts[chunk_idx * batch_size: (chunk_idx + 1) * batch_size]]
        chunks.append(_chunk)
        chunk_sent_lengths.append([len(_c) for _c in _chunk])
        chunk_idx += 1

    with tqdm(total=len(tokenized_texts), desc=f"Featurizing {name}") as pbar:
        for batch, batch_sent_length in zip(chunks, chunk_sent_lengths):
            h = featurize_batch(model, batch, batch_sent_length, device, is_mean=is_mean)

            feats.append(h.cpu())
            pbar.update(h.shape[0])

    t2 = time.time()
    if verbose:
        print(f'Featurize time: {round(t2-t1, 2)}')
    return torch.cat(feats)


def featurize_batch(model, batch, batch_sent_lengths, device, is_mean=False):
    padded_batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0).to(device)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(sent_length).long() for sent_length in batch_sent_lengths],
        batch_first=True, padding_value=0).to(device)

    outs = model(input_ids=padded_batch, attention_mask=attention_mask,
                 past_key_values=None, output_hidden_states=True,
                 return_dict=True)
    hidden_states = outs.hidden_states[-1]

    if is_mean:
        hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        h = hidden_states.sum(1) / attention_mask.sum(-1).unsqueeze(-1)
    else:
        _, _, hidden_size = hidden_states.size()
        gather_mask = (attention_mask.sum(-1) - 1).reshape(-1, 1, 1).repeat(1, 1, hidden_size)
        h = torch.gather(hidden_states, 1, gather_mask).squeeze(1)

    return h
