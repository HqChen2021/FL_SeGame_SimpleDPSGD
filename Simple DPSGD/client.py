#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import torch
import copy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from utils import Initialize_Model
from collections import OrderedDict
from DPOptimizerClass import DPOptimizer


class client(object):
    def __init__(self, cid, args, global_model, train_set, test_set):
        self.cid = cid
        self.args = args
        self.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        self.criterion = nn.NLLLoss().to(self.device)
        self.DELTA = 1 / (1.1 * len(train_set))
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.local_bs, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.local_bs, shuffle=True)
        self.acc, self.loss, self.eps = [], [], {}
        self.target_acc = args.target_acc
        self.model = copy.deepcopy(global_model)
        self.noise_multiplier = self.args.noise_multiplier

    # c #actual nosie added into gards is N(0,z^2*c^2)
    def load_weights(self, model, weights):
        for (k1, _), (k2, _) in zip(weights.items(), model.state_dict().items()):
            if k1 == k2:
                model.load_state_dict(weights)
            elif "_module" in k2:
                model._module.load_state_dict(weights)
            else:
                new_global_weights = OrderedDict()
                for k, v in weights.items():
                    name = k[8:]
                    new_global_weights[name] = v
                model.load_state_dict(new_global_weights)
            return model

    def train(self, model, optimizer, dataloader):
        epoch_loss, epoch_acc = [], []
        for epoch in range(self.args.local_ep):
            batch_loss, batch_acc = [], []
            for batch_idx, (images, target) in enumerate(dataloader):
                optimizer.zero_grad()
                images, target = images.to(self.device), target.to(self.device)
                output = model(images)
                loss = self.criterion(output, target)
                batch_loss.append(loss.item())
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                batch_acc.append((preds == labels).mean().item())
                loss.backward()  # at this step the model should have stored the gradients
                optimizer.step()
            epoch_loss.append(np.mean(batch_loss))
            epoch_acc.append(np.mean(batch_acc))
        return model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)

    def draw_new_z(self,current_z, z_min):
        # pdf f(z)=-2*z/(z_max-current_z)^2+2/(z_max-current_z)
        # using inversion method, starts with u~U(0,1), gengerate z=F^(-1)(u)
        # invers func: lambda y: (z_max - current_z) * (1 - np.sqrt(1 - y))
        return (current_z - z_min) * (1 - np.sqrt(1 - np.random.uniform(0, 1)))

    def update_model(self, global_round, model_weights):
        self.model = self.load_weights(self.model, model_weights)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.5)
        # game part, evaluate
        if self.args.is_dp:
            if self.args.se_game and len(self.eps) > self.args.start_time * self.args.epochs * self.args.frac:
                acc, _ = self.inference(model_weights)
                if acc < self.target_acc:
                    self.noise_multiplier -= self.draw_new_z(current_z=self.noise_multiplier, z_min=self.args.min_z)
                    last_round = list(self.eps.keys())[-1]
                    print(f'client {self.cid} participated in {len(self.eps)} times,\n '
                          f'now increase z from {self.eps[last_round]} to {self.noise_multiplier}')
                else:
                    print(f'client {self.cid} get accuracy: {acc} > target_accuracy: {self.args.target_acc} ')

            self.model.train()
            dp_optimizer = DPOptimizer(optimizer=optimizer,
                                     noise_multiplier=self.noise_multiplier,
                                     max_grad_norm=self.args.max_grad_norm,
                                     expected_batch_size=self.args.local_bs)
            train_results = self.train(self.model, dp_optimizer, self.train_loader)
            self.eps[global_round] = self.noise_multiplier
        else:
            self.model.train()
            train_results = self.train(self.model, optimizer, self.train_loader)
        return train_results

    def inference(self, model_weights):
        model = Initialize_Model(self.args)
        model = self.load_weights(model, model_weights)
        model.eval()
        batch_acc, batch_loss = [], []
        for batch_idx, (images, target) in enumerate(self.test_loader):
            images, target = images.to(self.device), target.to(self.device)
            output = model(images)
            loss = self.criterion(output, target)
            batch_loss.append(loss.item())
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            batch_acc.append((preds == labels).mean().item())
        self.acc.append(np.mean(batch_acc))
        self.loss.append(np.mean(batch_acc))
        return np.mean(batch_acc), np.mean(batch_loss)
