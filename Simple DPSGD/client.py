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
from opacus.validators import ModuleValidator
from opacus import GradSampleModule


def creat_model(args):
    if args.is_dp:
        model = GradSampleModule(Initialize_Model(args))
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            model = ModuleValidator.fix(model)
    else:
        model = Initialize_Model(args)
    return model


class client(object):
    def __init__(self, cid, args, train_set, test_set):
        self.cid = cid
        self.args = args
        self.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        self.criterion = nn.NLLLoss().to(self.device)
        self.DELTA = 1 / (1.1 * len(train_set))
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.local_bs, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.local_bs, shuffle=True)
        self.acc, self.loss, self.z = [], [], {}
        self.target_acc = args.target_acc
        self.model = creat_model(args)
        hasattr(next(self.model.parameters()), 'grad_sample')
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

    def train(self, model, dp_optimizer, optimizer, dataloader):
        epoch_loss, epoch_acc = [], []
        for epoch in range(self.args.local_ep):
            batch_loss, batch_acc = [], []
            for batch_idx, (images, target) in enumerate(dataloader):
                if epoch != self.args.local_ep - 1:
                    optimizer.zero_grad()
                else:
                    dp_optimizer.zero_grad()
                images, target = images.to(self.device), target.to(self.device)
                output = model(images)
                loss = self.criterion(output, target)
                batch_loss.append(loss.item())
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                batch_acc.append((preds == labels).mean().item())
                loss.backward()  # at this step the model should have stored the gradients
                if epoch != self.args.local_ep - 1:
                    optimizer.step()
                else:
                    dp_optimizer.step()
            epoch_loss.append(np.mean(batch_loss))
            epoch_acc.append(np.mean(batch_acc))
        return model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)

    def draw_new_z(self, current_z, z_min):
        # pdf f(z)=-2*z/(z_max-current_z)^2+2/(z_max-current_z)
        # using inversion method, starts with u~U(0,1), gengerate z=F^(-1)(u)
        # invers func: lambda y: (z_max - current_z) * (1 - np.sqrt(1 - y))
        return (current_z - z_min) * (1 - np.sqrt(1 - np.random.uniform(0, 1)))

    def whether_change(self, prob):
        """
        args
        prob \in (0,1) : the probability of decrease noise_multiplier,
        determined by acc and participating times

        return
        p(x<prob) where x~U(0,1)
        """
        return True if np.random.uniform(0, 1) < prob else False

    def calculate_prob(self, times, acc):
        """
        args
        x negetively related to participated times, the bigger x is, client is less participated
        in training, thus the smaller prob to decrease noise_multiplier. Since it is more likely
        client's data has not been seen by the model yet, and is not appropriate to decrease noise
        too early-->p(x)=(- 2 * x / expect_times**2 + 2 / expect_times)
        acc is the testing accuracy, the smaller acc is, the more likey to decrease noise_multiplier
        --> p(y)=(- 2 * acc / self.args.target_acc**2 + 2 / self.args.target_acc)

        return: rescaled p(x)*p(y), i.e., p(x)/p(x)_max * p(y)/p(y)_max
        """
        expect_times = self.args.epochs * self.args.frac
        x = expect_times - times
        return (-x / expect_times + 1) * np.abs(
            -acc / self.args.target_acc + 1)  # abs account for case where acc > target_acc

    def update_model(self, global_round, model_weights):
        self.model = self.load_weights(self.model, model_weights)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.5)
        # game part
        if self.args.is_dp:
            if self.args.se_game:
                acc, _ = self.inference(model_weights)
                prob = self.calculate_prob(len(self.z), acc)
                assert 0 <= prob < 1, "probability is not right"
                if self.whether_change(prob) and acc < self.target_acc:
                    self.noise_multiplier -= self.draw_new_z(current_z=self.noise_multiplier, z_min=self.args.min_z)
                    last_round = list(self.z.keys())[-1]
                    print(f'client {self.cid} participated in {len(self.z)} times,\n '
                          f'now decrease z from {self.z[last_round]} to {self.noise_multiplier}')
                elif acc > self.target_acc:
                    print(f'client {self.cid} get accuracy: {acc} > target_accuracy: {self.args.target_acc} ')
                else:
                    pass
            self.model.train()
            dp_optimizer = DPOptimizer(optimizer=optimizer,
                                       noise_multiplier=self.noise_multiplier,
                                       max_grad_norm=self.args.max_grad_norm,
                                       expected_batch_size=self.args.local_bs)
            train_results = self.train(self.model, dp_optimizer, optimizer, self.train_loader)
            self.z[global_round] = self.noise_multiplier
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
