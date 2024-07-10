#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import os.path

from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam

from src.data.helpers import get_data_loaders
from src.models import get_model
from src.utils.logger import create_logger
from src.utils.utils import *
import numpy as np
recoreds=[]

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=128)
    parser.add_argument("--bert_model", type=str, default="./bert-base-uncased")#, choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str, default="/path/to/data_dir/")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--glove_path", type=str, default="datasets/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="bow", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt","latefusion","latefusion_pdf"])
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/path/to/save_dir/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="food101", choices=["food101","MVSA_Single"])
    parser.add_argument("--task_type", type=str, default="classification", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument("--df", type=bool, default=True)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--noise_type", type=str, default="Salt")
    parser.add_argument("--du", type=bool,default=True)

def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer(model, args):
    if args.model in ["bert", "concatbert", "mmbt"]:
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def model_eval(i_epoch, data, model, args, criterion):
    with torch.no_grad():
        losses, preds,txt_preds,img_preds, tgts = [], [], [],[],[]
        for batch in tqdm(data, total=len(data)):
            loss,out,txt_logits, img_logits,tgt = model_forward(
                i_epoch, model, args, criterion, batch, mode='eval')
            losses.append(loss.item())


            if args.task_type == "multilabel":
                pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
                txt_pred = torch.sigmoid(txt_logits).cpu().detach().numpy() > 0.5
                img_pred = torch.sigmoid(img_logits).cpu().detach().numpy() > 0.5
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
                txt_pred = torch.nn.functional.softmax(txt_logits, dim=1).argmax(dim=1).cpu().detach().numpy()
                img_pred = torch.nn.functional.softmax(img_logits, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            txt_preds.append(txt_pred)
            img_preds.append(img_pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)
    
    metrics = {"loss": np.mean(losses)}
    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        txt_preds = [l for sl in txt_preds for l in sl]
        img_preds = [l for sl in img_preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)
        metrics["txt_acc"] = accuracy_score(tgts, txt_preds)
        metrics["img_acc"] = accuracy_score(tgts, img_preds)
    return metrics

def model_forward(i_epoch, model, args, criterion, batch,mode='eval'):
    txt, segment, mask, img, tgt,idx = batch
    freeze_img = i_epoch < args.freeze_img
    freeze_txt = i_epoch < args.freeze_txt

    if args.model == "bow":
        txt = txt.cuda()
        out = model(txt)
    elif args.model == "img":
        img = img.cuda()
        out = model(img)
    elif args.model == "concatbow":
        txt, img = txt.cuda(), img.cuda()
        out = model(txt, img)
    elif args.model == "bert":
        txt, mask, segment = txt.cuda(), mask.cuda(), segment.cuda()
        out = model(txt, mask, segment)
    elif args.model == "concatbert":
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        out = model(txt, mask, segment, img)
    elif args.model == "latefusion_pdf":
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        if args.du==True:
          out, txt_logits, img_logits,txt_tcp_pred,img_tcp_pred,_,_ = model(txt, mask, segment, img, 'pdf_test')
        else:
          out, txt_logits, img_logits,txt_tcp_pred,img_tcp_pred = model(txt, mask, segment, img, 'pdf_trian')

    else:
        assert args.model == "mmbt"
        for param in model.enc.img_encoder.parameters():
            param.requires_grad = not freeze_img
        for param in model.enc.encoder.parameters():
            param.requires_grad = not freeze_txt

        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        out = model(txt, mask, segment, img)

    tgt = tgt.cuda()

    txt_clf_loss = nn.CrossEntropyLoss()(txt_logits, tgt)
    img_clf_loss = nn.CrossEntropyLoss()(img_logits, tgt)
    clf_loss=txt_clf_loss+img_clf_loss
    
    label = F.one_hot(tgt, num_classes=args.n_classes)  # [b,c]

    if args.task_type == "multilabel":
        txt_pred = torch.sigmoid(txt_logits)
        img_pred = torch.sigmoid(img_logits)
    else:
        txt_pred = torch.nn.functional.softmax(txt_logits, dim=1)
        img_pred = torch.nn.functional.softmax(img_logits, dim=1)
    txt_tcp, _ = torch.max(txt_pred * label, dim=1,keepdim=True)
    img_tcp, _ = torch.max(img_pred * label, dim=1,keepdim=True)
    maeloss = nn.L1Loss(reduction='mean')
    tcp_pred_loss = maeloss(txt_tcp_pred, txt_tcp.detach()) + maeloss(img_tcp_pred, img_tcp.detach())
    
    txt_clf_loss = nn.CrossEntropyLoss()(txt_logits, tgt)
    img_clf_loss = nn.CrossEntropyLoss()(img_logits, tgt)
    clf_loss=txt_clf_loss+img_clf_loss+nn.CrossEntropyLoss()(out,tgt)

    if mode=='train':
        loss = torch.mean(clf_loss)+torch.mean(tcp_pred_loss)
        return loss,out,tgt
    else:
        loss= torch.mean(clf_loss)+torch.mean(tcp_pred_loss)
        return loss,out,txt_logits, img_logits,tgt
        
def train(args):
 
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loaders = get_data_loaders(args)
    model = get_model(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger("%s/eval_logfile.log" % args.savedir, args)
    model.cuda()
    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()

    accList=[]
    txtaccList = []
    imgaccList = []
    for test_name, test_loader in test_loaders.items():
        test_metrics = model_eval(
            np.inf, test_loader, model, args, criterion
        )
        log_metrics(f"Test - {test_name}", test_metrics, args, logger)
        accList.append(test_metrics['acc'])
        txtaccList.append(test_metrics['txt_acc'])
        imgaccList.append(test_metrics['img_acc'])

    info = f"name:{args.name} seed:{args.seed} noise:{args.noise} test_txt_acc: {txtaccList[0]:0.5f} test_img_acc: {imgaccList[0]:0.5f} test_acc: {accList[0]:0.5f}\n"
    print(info)
    file_path = f"eval_data/{args.name}_result_info_du_{args.du}.txt"
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            pass
    with open(file_path, "a+") as f:
        f.write(info)

    result_json={
        'name':args.name,
        'method': args.model+'_pdf',
        'seed':args.seed,
        'noise':args.noise,
        'test_acc':accList[0],
    }

    path = f"eval_data/{args.task}_result_{args.noise_type}.json"
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            exist_json = json.load(f)
    else:
        exist_json = []

    if not result_json in exist_json:
        exist_json.append(result_json)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(exist_json, f, ensure_ascii=False, indent=2)
            
def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    cli_main()
