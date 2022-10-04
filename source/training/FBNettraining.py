from source.utils import accuracy, isfloat
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data, mixup_cluster_loss, inner_loss, intra_loss
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
from .training import Train


class FBNetTrain(Train):

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        super().__init__(cfg, model, optimizers, lr_schedulers, dataloaders, logger)
        self.group_loss = cfg.model.group_loss
        self.sparsity_loss = cfg.model.sparsity_loss
        self.sparsity_loss_weight = cfg.model.sparsity_loss_weight

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()

        for time_series, node_feature, label in self.train_dataloader:
            self.current_step += 1
            label = label.float()

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()

            if self.config.preprocess.continus:
                time_series, node_feature, label = continus_mixup_data(
                    time_series, node_feature, y=label)

            predict, learnable_matrix = self.model(time_series, node_feature)

            loss = 2 * self.loss_fn(predict, label)

            if self.group_loss:
                if self.config.preprocess.continus:
                    loss += mixup_cluster_loss(learnable_matrix,
                                               label)
                else:
                    loss += 2 * intra_loss(label[:, 1], learnable_matrix) + \
                        inner_loss(label[:, 1], learnable_matrix)

            if self.sparsity_loss:
                sparsity_loss = self.sparsity_loss_weight * \
                    torch.norm(learnable_matrix, p=1)
                loss += sparsity_loss

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
            # wandb.log({"LR": lr_scheduler.lr,
            #            "Iter loss": loss.item()})

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for time_series, node_feature, label in dataloader:
            label = label.float()
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            output, _ = self.model(time_series, node_feature)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label[:, 1])[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')

        report = classification_report(
            labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall
