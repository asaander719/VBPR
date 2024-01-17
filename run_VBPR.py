import os
import time
import numpy as np
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from util import config
from loader_vbpr import Load_Data
import csv
from torch.optim import Adam
import json
from torch.nn import *
from Models.VBPR import VBPR

def get_parser(): 
    parser = argparse.ArgumentParser(description='Modified VBPR for Compatibility Modeling')
    parser.add_argument('--config', type=str, default='config/IQON3000_RB.yaml', help='config file')
    parser.add_argument('opts', help='see config/IQON3000_RB.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 8
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.val_loss_min = np.Inf
        self.val_auc_max = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_auc, model):
        score = val_auc
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        '''Saves model when validation auc increase.'''
        if self.verbose:
            self.trace_func(f'Validation auc increase ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        # torch.save(model, data_config['model_file'])
        self.val_auc_max = val_auc

def load_csv_data(train_data_path):
    result = []
    with open(train_data_path,'r') as fp:
        for line in fp:
            t = line.strip().split(',')
            t = [int(i) for i in t]
            result.append(t)
    return result

def training(model, train_data_loader, device, optimizer, epoch):
    model.train()
    loss_scalar = 0.
    max_iter = args.epochs * len(train_data_loader)
    for iteration, aBatch in enumerate(train_data_loader):
        aBatch = [x.to(device) for x in aBatch]
        loss = model.fit(aBatch) 
        iteration +=1 
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_scalar += loss.detach().cpu()

    end = time.time()
    logger.info('Epoch: [{}/{}][{}/{}] '
                'Loss {loss_meter:.4f} '.format(epoch+1, args.epochs, iteration + 1, len(train_data_loader),
                                                    loss_meter=loss_scalar/iteration))
    return loss_scalar/iteration

def evaluating(model, testData, device, t_len):
    model.eval()
    pos = 0
    for i, aBatch in enumerate(testData):
        aBatch = [x.to(device) for x in aBatch]
        output = model.inference(aBatch)        
        pos += float(torch.sum(output.ge(0)))
    AUC = pos/t_len
    return AUC

def main(args, logger):
    visual_features_tensor = torch.load(args.visual_features_tensor, map_location= lambda a,b:a.cpu())#torch.Size([142737, 2048])

    user_map = json.load(open(args.user_map))
    item_map = json.load(open(args.item_map)) 
    args.user_num = len(user_map)
    args.item_num = len(item_map)
    
    model = VBPR(args.item_num, args.hidden_dim, args.visual_feature_dim, visual_features_tensor.to(args.device), args.with_Nor)
    model.to(args.device)
    logger.info(model)
    optimizer = Adam([{'params': model.parameters(),'lr': args.base_lr, "weight_decay": args.wd}])
 
    train_data_ori = load_csv_data(args.train_data)
    train_data_ori  = torch.LongTensor(train_data_ori)
    train_data = Load_Data(train_data_ori)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    if args.evaluate:
        valid_data_ori = load_csv_data(args.valid_data)
        v_len = len(valid_data_ori)
        valid_data_ori = torch.LongTensor(valid_data_ori)
        valid_data = Load_Data(valid_data_ori)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size_val, shuffle=False)
        
    test_data_ori = load_csv_data(args.test_data)
    t_len = len(test_data_ori)
    test_data_ori  = torch.LongTensor(test_data_ori)
    test_data = Load_Data(test_data_ori)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_val, shuffle=False)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.97 ** epoch) 

    best_auc = 0
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        loss_train = training(model, train_loader, args.device, optimizer, epoch)
        scheduler.step()
        if args.evaluate:
            valid_auc = evaluating(model, valid_loader, args.device, v_len)
            end = time.time()
            logger.info('Valid: [{}] '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, accuracy=valid_auc))
        test_auc = evaluating(model, test_loader, args.device, t_len) 
        end = time.time()
        logger.info('Test: [{}] '
                    'Accuracy {accuracy:.4f}.'.format(epoch +1, accuracy=test_auc))
        if test_auc > best_auc:
            best_auc = test_auc
            filename = args.save_path + '/VBPR' + '.pth.tar'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save(model, filename)   
        early_stopping(valid_auc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break 

if __name__ == '__main__':
    args = get_parser()
    logger = get_logger()
    args.gpu = 0
    args.device = torch.device("cuda:%s"%args.gpu if torch.cuda.is_available() else "cpu")
    main(args, logger)