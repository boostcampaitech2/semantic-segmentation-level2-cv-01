# # https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

class Metrics:

    def __init__(self, classes, len_loader):
        self.classes = {idx:label for idx, label in enumerate(classes)}
        self.n_class = len(self.classes)
        self.len_loader = len_loader
        self.mean_acc = 0
        self.mean_acc_each_cls = np.zeros(self.n_class)
        self.mean_mean_acc_cls = 0
        self.mean_mIoU = 0
        self.mean_loss = 0
        self.mean_fwavacc = 0
        self.mean_IoU = np.zeros(self.n_class)
        self.hist = np.zeros((self.n_class, self.n_class))
        self.mIoU = 0
        self.loss = 0
        self.best_mIoU = 0
        self.best_epoch = 0

    def label_accuracy_score(self):
        """
        Returns accuracy score evaluation result.
        - [acc]: overall accuracy
        - [acc_cls]: mean accuracy
        - [mean_iu]: mean IU
        - [fwavacc]: fwavacc
        """
        hist = self.hist

        acc = np.diag(hist).sum() / hist.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_cls = np.diag(hist) / hist.sum(axis=1)
        mean_acc_cls = np.nanmean(acc_cls)

        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        self.mIoU = mean_iu

        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_acc_cls, mean_iu, fwavacc, iu


    def add_hist(self, label_trues, label_preds):
        """
            stack hist(confusion matrix)
        """

        for lt, lp in zip(label_trues, label_preds):
            self.hist += self._fast_hist(lt.flatten(), lp.flatten())

        # return hist

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_class)
        hist = np.bincount(
            self.n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_class ** 2).reshape(self.n_class, self.n_class)
        return hist

    def accumulate_loss(self, loss):
        self.loss = loss.item()
        self.mean_loss += loss.item()

    def update_loss(self):
        self.mean_loss /= self.len_loader

    def accumulate(self, loss):
        acc, acc_cls, mean_acc_cls, mIoU, fwavacc, IoU = self.label_accuracy_score()
        self.mIoU = mIoU
        self.loss = loss.item()

        self.mean_acc += acc
        self.mean_acc_each_cls += np.array(acc_cls)
        self.mean_mean_acc_cls += mean_acc_cls
        self.mean_mIoU += mIoU
        self.mean_loss += loss.item()
        self.mean_fwavacc += fwavacc
        self.mean_IoU += np.array(IoU)

    def update(self):
        self.mean_acc /= self.len_loader
        self.mean_acc_each_cls /= self.len_loader
        self.mean_mean_acc_cls /= self.len_loader
        self.mean_mIoU /= self.len_loader
        self.mean_loss /= self.len_loader
        self.mean_fwavacc /= self.len_loader
        self.mean_IoU /= self.len_loader

    def init_metrics(self):
        self.mean_acc = 0
        self.mean_acc_each_cls = np.zeros(self.n_class)
        self.mean_mean_acc_cls = 0
        self.mean_mIoU = 0
        self.mean_loss = 0
        self.mean_fwavacc = 0
        self.mean_IoU = np.zeros(self.n_class)
        self.hist = np.zeros((self.n_class, self.n_class))