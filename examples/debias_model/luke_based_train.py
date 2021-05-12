
import os, sys
import joblib
sys.path.append('../../')
currentdir = os.path.dirname(os.path.realpath('reading_comprehension'))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from collections import defaultdict as ddict 
import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace 
import collections
from utils import convert_examples_to_features_debias

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
from torch.nn import CrossEntropyLoss

from transformers import RobertaTokenizer
from transformers import AutoTokenizer
from model import BertModel
from torch.optim import AdamW
import pickle
import json
from tqdm import tqdm

from reading_comprehension.utils.dataset import debiasProcessor
from reading_comprehension.utils.wiki_link_db import WikiLinkDB
from luke.model import LukeEntityAwareAttentionModel
from luke.utils.model_utils import ModelArchive
from reading_comprehension.utils.feature import convert_examples_to_features

def seed_everything(seed=11747):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default= "data/gender_train.json")
    parser.add_argument("--seed", type=int, default= 1)
    parser.add_argument("--batch_size", type=int, default= 80)
    parser.add_argument("--epochs", type=int, default= 5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cuda", type=str,   default= '0')
    parser.add_argument("--dev", type=str,   default= 'data/gender_dev.json')
    args = parser.parse_args()
    print(f"RUN: {vars(args)}")
    return args

def compute_f1(gold_toks, pred_toks):
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def collate_fn(batch):
        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o[1], attr_name), dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        ret = dict(
            word_ids=create_padded_sequence("word_ids", AutoTokenizer.from_pretrained('roberta-base').pad_token_id),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            entity_ids=create_padded_sequence("entity_ids", 0)[:, : 128],
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0)[:, : 128],
            entity_position_ids=create_padded_sequence("entity_position_ids", -1)[:, : 128, :],
            entity_segment_ids=create_padded_sequence("entity_segment_ids", 0)[:, : 128],
        )
        # if args.no_entity:
        #     ret["entity_attention_mask"].fill_(0)

        # if evaluate:
        #     ret["example_indices"] = torch.tensor([o[0] for o in batch], dtype=torch.long)
        
        ret["start_positions"] = torch.tensor([o[1].start_positions[0] for o in batch], dtype=torch.long)
        ret["end_positions"] = torch.tensor([o[1].end_positions[0] for o in batch], dtype=torch.long)

        return ret

def model_eval(loader,model,use_cuda,encoder):
        model.eval()
        f1=0
        with tqdm(total=len(train_dataloader)) as pbar:
            for step, batch in enumerate(loader):
                b_ids, b_mask, b_start, b_end    =  batch['word_ids'], batch['word_attention_mask'], batch['start_positions'],batch['end_positions']
                inputs = {k: v.cuda() for k, v in _create_model_arguments(batch).items() if k not in ['end_positions','start_positions']}
                
                hidden_seq = encoder(**inputs)[0][:, : b_ids.size(1), :]
                if use_cuda:
                    b_ids = b_ids.cuda()
                    b_mask = b_mask.cuda()
                    b_start = b_start.cuda()
                    b_end = b_end.cuda()

                logits =  model.luke_forward(hidden_seq,b_mask,downstream=True)
                start_logits, end_logits = logits.split(1, dim=-1)
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)
                f1_start=compute_f1(start_logits.argmax(1).cpu().tolist(),b_start.cpu().tolist())
                f1_end=compute_f1(end_logits.argmax(1).cpu().tolist(),b_end.cpu().tolist())
                f1+=(f1_start+f1_end)/2
                pbar.set_description(f'evaluation')
                pbar.update()

        return f1/len(loader)
def _create_model_arguments(batch):
        return batch

if __name__ == "__main__":
    args = get_args()

    seed_everything(args.seed)
    processor = debiasProcessor()
    examples_train = processor.get_train_examples('data')
    examples_dev = processor.get_dev_examples('data')

    wiki_link_db=WikiLinkDB('../../enwiki_20160305.pkl')
    model_redirect_mappings = joblib.load('../../enwiki_20181220_redirects.pkl')
    link_redirect_mappings = joblib.load('../../enwiki_20160305_redirects.pkl')
    model_archive=ModelArchive.load("../../luke_large_500k.tar.gz")
    encoder=LukeEntityAwareAttentionModel(model_archive.config)
    encoder.load_state_dict(model_archive.state_dict, strict=False)
    encoder.cuda()
    examples_train = processor.get_train_examples('data')
    examples_dev = processor.get_dev_examples('data')
    train_features= convert_examples_to_features(
        examples=examples_train,
        tokenizer=AutoTokenizer.from_pretrained('roberta-base'),
        entity_vocab=model_archive.entity_vocab,
        wiki_link_db=wiki_link_db,
        model_redirect_mappings=model_redirect_mappings,
        link_redirect_mappings=link_redirect_mappings,
        max_seq_length=200,
        max_mention_length=model_archive.max_mention_length,
        doc_stride=1,
        max_query_length=100,
        min_mention_link_prob=0.01,
        segment_b_id=0,
        add_extra_sep_token=True,
        is_training=True
    )
    dev_features= convert_examples_to_features(
        examples=examples_train,
        tokenizer=AutoTokenizer.from_pretrained('roberta-base'),
        entity_vocab=model_archive.entity_vocab,
        wiki_link_db=wiki_link_db,
        model_redirect_mappings=model_redirect_mappings,
        link_redirect_mappings=link_redirect_mappings,
        max_seq_length=200,
        max_mention_length=model_archive.max_mention_length,
        doc_stride=1,
        max_query_length=100,
        min_mention_link_prob=0.01,
        segment_b_id=0,
        add_extra_sep_token=True,
        is_training=True
    )
    train_dataloader=DataLoader(list(enumerate(train_features)), batch_size=args.batch_size, collate_fn=collate_fn)
    dev_dataloader=DataLoader(list(enumerate(dev_features)), batch_size=args.batch_size, collate_fn=collate_fn)
    model = BertModel()
    use_cuda = False

    if int(args.cuda)>= 0:
        use_cuda = True
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
        model.cuda()
        print('----use cuda----')

    lr = args.lr

    ## specify the optimizer 
    optimizer   = AdamW(model.parameters(), lr = lr)
    best_model= None
    best_dev_f1= 0
    filepath = 'luke_based_model.pt'
    for epoch in range(args.epochs):
        with tqdm(total=len(train_dataloader)) as pbar:
            model.train()
            # print(epoch)
            train_loss  = 0
            num_batches = 0

            for step, batch in enumerate(train_dataloader):
                b_ids, b_mask, b_start, b_end    =  batch['word_ids'], batch['word_attention_mask'], batch['start_positions'],batch['end_positions']
                inputs = {k: v.cuda() for k, v in _create_model_arguments(batch).items() if k not in ['end_positions','start_positions']}
                
                hidden_seq = encoder(**inputs)[0][:, : b_ids.size(1), :]
                # print(hidden_seq.shape)

                if use_cuda:
                    b_ids = b_ids.cuda()
                    b_mask = b_mask.cuda()
                    b_start = b_start.cuda()
                    b_end = b_end.cuda()

                optimizer.zero_grad()
                logits=model.luke_forward(hidden_seq,b_mask,downstream=True)
                # logits =  model(b_ids, b_mask,downstream=True)
                start_logits, end_logits = logits.split(1, dim=-1)
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)


                loss_fct = CrossEntropyLoss()
                start_loss = loss_fct(start_logits, b_start)
                end_loss = loss_fct(end_logits, b_end)
                total_loss = (start_loss + end_loss) / 2


                total_loss.backward()
                optimizer.step()

                train_loss += total_loss.item()
                num_batches+= 1
                pbar.set_description(f'eoch {epoch}, loss {total_loss.item()}')
                pbar.update()

        train_f1 =  model_eval(train_dataloader, model,use_cuda,encoder)
        # print('train f1:',train_f1)
        dev_f1 = model_eval(dev_dataloader, model,use_cuda,encoder)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_model = model
            torch.save(best_model.state_dict(), filepath)
    print(f'f1 score: {best_dev_f1}')




