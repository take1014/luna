#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import sys
import os
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from luna_model import LunaModel
from luna_dataset import LunaDataset
from utils import enumerateWithEstimate
from logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX  = 1
METRICS_LOSS_NDX  = 2
METRICS_SIZE      = 3

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        self.use_cuda  = torch.cuda.is_available()
        self.device    = torch.device('cuda' if self.use_cuda else 'cpu')
        if sys_argv is None:
            # get command line string
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers', help='Number of worker processes for background data loading'      , default =  8      , type=int,)
        parser.add_argument('--batch-size' , help='Batch size to use for training'                              , default = 32      , type=int,)
        parser.add_argument('--epochs'     , help='Number of epochs to train for'                               , default = 20      , type=int,)
        parser.add_argument('--tb-prefix'  , help='Data prefix to use for Tensorboard run. Defaults to chapter.', default = 'p2ch11', type=str,)
        parser.add_argument('comment'      , help='Comment suffix for Tensorboard run.'                         , default = 'dwlpt' , nargs='?',)

        # Analyze argument
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.train_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.model     = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info('Using CUDA; {} devices'.format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initDataLoader(self, isValSet_bool=False):
        ds = LunaDataset(val_stride=10, isValSet_bool=isValSet_bool,)
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        return DataLoader(ds, batch_size=batch_size, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda,)

    def initOptimizer(self):
        return optim.SGD(self.model.parameters(), lr = 0.001, momentum=0.99)

    def doTraining(self, epoch_ndx, train_dl):
        # Change training mode
        self.model.train()

        # Metricsの初期化
        trainMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)

        batch_iter = enumerateWithEstimate(train_dl, 'E{} Training'.format(epoch_ndx), start_ndx=train_dl.num_workers,)

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trainMetrics_g)
            loss_var.backward()
            self.optimizer.step()
        self.totalTrainingSamples_count += len(train_dl.dataset)
        return trainMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        # Change training mode
        with torch.no_grad():
            self.model.eval()

            # Metricsの初期化
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)

            batch_iter = enumerateWithEstimate(val_dl, 'E{} Validation'.format(epoch_ndx), start_ndx=val_dl.num_workers,)

            # Validationなので推論のみ。ロスの計算と逆伝搬は行わない
            for batch_ndx, batch_tup in batch_iter:
                # 平均損失は使わない
                _ = self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')


    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g    = loss_func(logits_g, label_g[:,1],)

        start_ndx = batch_ndx * batch_size
        end_ndx   = start_ndx + label_t.size(0) # label_t.size(0) = batch_size

        # Metricsの更新
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:,1].detach()
        metrics_g[METRICS_PRED_NDX , start_ndx:end_ndx] = probability_g[:,1].detach()
        metrics_g[METRICS_LOSS_NDX , start_ndx:end_ndx] = loss_g.detach()

        #ワンバッチごとに平均化
        return loss_g.mean()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t, classificationThreshold=0.5):
        self.initTensorboardWriters()

        log.info('E{} {}'.format( epoch_ndx, type(self).__name__, ))

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask  = metrics_t[METRICS_PRED_NDX]  <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask  = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        trueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum())
        truePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum())

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100
        precision = metrics_dict['pr/precision'] = truePos_count / np.float32(truePos_count + falsePos_count)
        recall    = metrics_dict['pr/recall']    = truePos_count / np.float32(truePos_count + falseNeg_count)

        # F1 Score
        metrics_dict['pr/f1_score'] = 2 * precision * recall / (precision + recall)

        # logger
        log.info(("E{} {:8} {loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score").format(epoch_ndx, mode_str, **metrics_dict,) )
        log.info(("E{} {:8} {loss/neg:.4f} loss, "
                 + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})").format(epoch_ndx, mode_str + '_neg', neg_correct=neg_correct, neg_count=neg_count, **metrics_dict,))
        log.info(("E{} {:8} {loss/pos:.4f} loss, "+ "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
                 ).format(epoch_ndx, mode_str + '_pos', pos_correct=pos_correct, pos_count=pos_count, **metrics_dict,))

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve( 'pr', metrics_t[METRICS_LABEL_NDX], metrics_t[METRICS_PRED_NDX], self.totalTrainingSamples_count, )

        bins = [x/50.0 for x in range(51)]

        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram( 'is_neg', metrics_t[METRICS_PRED_NDX, negHist_mask],
                                  self.totalTrainingSamples_count, bins=bins, )
        if posHist_mask.any():
            writer.add_histogram( 'is_pos', metrics_t[METRICS_PRED_NDX, posHist_mask],
                                  self.totalTrainingSamples_count, bins=bins, )

    # Tensorboard writers
    def initTensorboardWriters(self):
        if self.train_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.train_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def main(self):
        log.info('Starting{}, {}'.format(type(self).__name__, self.cli_args))
        # set data loader
        train_dl = self.initDataLoader(isValSet_bool=False)
        val_dl   = self.initDataLoader(isValSet_bool=True)

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info('Epoch {} of {}, {}/{} batches of size {}*{}'.format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))
            # train
            trainMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'train', trainMetrics_t)

            # validation
            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)
            if hasattr(self, 'train_writer'):
                self.train_writer.close()
                self.val_writer.close()

if __name__ == '__main__':
    LunaTrainingApp().main()
