import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from autolearn.utils import log


class Callback:
    def __init__(self): pass
    def on_train_begin(self, *args, **kwargs): pass
    def on_train_end(self, *args, **kwargs): pass
    def on_epoch_begin(self, *args, **kwargs): pass
    def on_epoch_end(self, *args, **kwargs): pass
    def on_batch_begin(self, *args, **kwargs): pass
    def on_batch_end(self, *args, **kwargs): pass
    def on_loss_begin(self, *args, **kwargs): pass
    def on_loss_end(self, *args, **kwargs): pass
    def on_step_begin(self, *args, **kwargs): pass
    def on_step_end(self, *args, **kwargs): pass


class EarlyStopping(Callback):
    def __init__(self, patience=5, tol=0.001, min_epochs=1):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.tol = tol
        self.best = -np.inf
        self.best_epoch = -1
        self.wait = 0
        self.stopped_epoch = -1
        self.min_epochs = min_epochs
        self.updated = False
        self.begin_use = False

    def on_epoch_begin(self, begin_use):
        self.begin_use = begin_use

    def on_epoch_end(self, epoch, val_acc, epoch_loss):
        if not self.begin_use:
            return False

        self.updated = False
        val_loss = min(1.0, val_acc + self.tol)

        if val_acc > self.best and self.best < 0.999:
            self.best = max(val_loss - self.tol, self.best)
            self.best_epoch = epoch
            self.wait = 0
            self.updated = True
        else:
            self.wait += 1
            if self.wait >= self.patience and epoch > self.min_epochs:
                self.stopped_epoch = epoch
                log(
                    f"Early stopping conditioned on val_acc patience {self.patience} "
                    f"in epoch {self.stopped_epoch}. "
                    f"Metric is {val_acc}, best {self.best} in epoch {self.best_epoch}"
                )
                return True
        return False


class Checkpoint(Callback):
    def __init__(self, ckp_path, earlystop_cb, cur_iter=0):
        super(Checkpoint, self).__init__()
        self.ckp_path = None
        if ckp_path is not None:
            self.ckp_path = Path(ckp_path)
            if not os.path.exists(ckp_path):
                os.makedirs(ckp_path)
        self.earlystop_cb = earlystop_cb
        self.num_ckp = 0
        self.best_model = None
        self.cur_iter = cur_iter

    def _save_model(self, model):
        if self.ckp_path is None:
            return

        save_path = self.ckp_path / f"{self.cur_iter}_nfo_ckp_{self.num_ckp}_epoch_{self.earlystop_cb.best_epoch}" \
                                    f"_{self.earlystop_cb.best: .7f}.ckp"
        torch.save(
            model.state_dict(),
            save_path
        )
        self.num_ckp += 1
        log(f"save {self.num_ckp}th checkpoint in Path {save_path}")

    def on_epoch_end(self, model: torch.nn.Module):
        earlystop = self.earlystop_cb
        if earlystop.updated:
            self.best_model = model
        if not earlystop.updated or earlystop.wait <= earlystop.patience * 0.2:
            return
        self._save_model(model)

    def on_train_end(self, model):
        self._save_model(self.best_model)


class LossTradeOff(Callback):
    def __init__(self, trade_off_epoch, default_trade_off=1.0, trade_off=None):
        super(LossTradeOff, self).__init__()
        self.trade_off = trade_off
        self.default = default_trade_off
        self.trade_off_epoch = trade_off_epoch

    def on_epoch_begin(self):
        return self.default if self.trade_off is None else self.trade_off
    
    def on_epoch_end(self, epoch, loss_1, loss_2):
        if self.trade_off is None and epoch >= self.trade_off_epoch:
            # balance loss1 and loss2
            self.trade_off = loss_2 / loss_1
            log(f"use trade off factor {self.trade_off} to balance loss")


class ValidLoss(Callback):
    def __init__(self, valid_set, forward_func, loss_1, loss_2, device):
        super(ValidLoss, self).__init__()
        self.valid_set = valid_set
        self.forward_func = forward_func
        self.device = device
        self.loss_1 = loss_1
        self.loss_2 = loss_2
        self.trade_off = None

    def on_epoch_begin(self, trade_off, default):
        if not default:
            self.trade_off = trade_off

    def on_epoch_end(self, model, train_loss):
        if self.valid_set is None or self.trade_off is None:
            return train_loss

        data_loader = DataLoader(self.valid_set, batch_size=len(self.valid_set), shuffle=False)
        total = 0
        with torch.no_grad():
            for x, y1, x2, y2 in data_loader:
                x, y1 = x.to(self.device), y1.to(self.device)
                y_hat_1, y_hat_2 = model(x)
                loss_1 = self.loss_1(y_hat_1.squeeze(), y1.float())
                y2 = x.reshape(-1)
                y_hat_2 = y_hat_2.reshape(y2.size()[0], -1)
                loss_2 = self.loss_2(y_hat_2.squeeze(), y2)
                loss_1 = loss_1.sum()
                loss_2 = loss_2.sum()
                total += self.trade_off * loss_1.item() + loss_2.item()
        return total
