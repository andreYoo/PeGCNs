#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict
import platform
import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from model.Autoregression import *
import pdb

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed()
    random.seed()
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')

    parser.add_argument(
        '--ar_weights',
        default=None,
        help='the weights for network initialization')


    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    parser.add_argument(
        '--code_size',
        type=int,
        default=256,
        help='the code size for Autoregressive RNN model')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')

    parser.add_argument(
        '--drop_list',
        type=int,
        default=0,
        help='weight decay for optimizer')

    parser.add_argument(
        '--modf',
        type=str,
        default='normal',
        help='weight decay for optimizer')

    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.drop_list = self.arg.drop_list  # Added
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.weight_for_celoss = 0.1

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        self.ar_model = Autoregressive_RNN(self.arg.code_size,self.arg.code_size,self.arg.batch_size).cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        self.coding_loss=nn.BCEWithLogitsLoss().cuda(output_device)

        if self.arg.weights:  # only test phase and resuming training
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            self.print_log('Load weights from {}.'.format(self.arg.ar_weights))

            if self.arg.phase == 'train':
                self.print_log('Resuming training')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)
                ar_weights = torch.load(self.arg.ar_weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            ar_weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in ar_weights.items()])

            keys = list(weights.keys())

            if isinstance(self.drop_list, list):
                if len(self.drop_list) and self.arg.noise_mode == 'drop':  # Added
                    print('noise mode: ', self.arg.noise_mode, 'drop list:', self.drop_list)
                    tmp = ['data_bn.weight', 'data_bn.bias', 'data_bn.running_mean', 'data_bn.running_var']
                    mask = np.ones(150, dtype=np.uint8)
                    for vidx in self.drop_list:
                        mask[vidx * 3: (vidx + 1) * 3] = 0
                        mask[vidx * 3 + 75: (vidx + 1) * 3 + 75] = 0

                    _mask = torch.from_numpy(mask.astype(np.uint8))
                    for t in tmp:
                        mask = _mask.to(weights[t].device)
                        weights[t] = torch.masked_select(weights[t], mask)

                    mask = np.ones((25, 25), dtype=np.uint8)
                    for vidx in self.drop_list:
                        mask[vidx, :] = 0
                        mask[:, vidx] = 0

                    _mask = torch.from_numpy(mask.astype(np.uint8))
                    size = 25 - len(self.drop_list)
                    for i in range(1, 11):
                        key = f'l{i}.gcn1.PA'
                        mask = _mask.to(weights[key].device)
                        b = weights[key][:, mask]
                        weights[key] = b.view(-1, size, size)

            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
                self.ar_model.load_state_dict(ar_weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)
                self.ar_model = nn.DataParallel(
                    self.ar_model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        params = list(self.model.parameters()) + list(self.ar_model.parameters())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log_x(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ];' + str
        print(str)
        if self.arg.print_log:
            with open('{}/logaa.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/{platform.node()}_log.txt', 'a') as f:
                print(str, file=f)

    def print_log_cusmtom(self, msg):

        if self.arg.phase == 'test':
            bb = [self.arg.test_feeder_args['modf'],
                  self.arg.weights,
                  self.arg.model,
                  str(self.arg.test_feeder_args['drop_num']),
                  ]
        else:
            bb = [self.arg.train_feeder_args['modf'],
                  str(self.arg.train_feeder_args['drop_num']),
                  self.arg.model_saved_name,
                  'warm:',
                  str(self.arg.warm_up_epoch),
                  str(self.arg.only_train_part),
                  str(self.arg.only_train_epoch)
                  ]

        localtime = time.asctime(time.localtime(time.time()))

        msg = "[ " + localtime + ' ] ' + ';'.join(bb) + msg
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/{platform.node()}_summary_log.txt', 'a') as f:
                print(msg, file=f)


    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def contrastive_coding_loss(self,encoded,predicted):
        eye_shape = encoded.shape[0]
        target = torch.eye(eye_shape).reshape(1,eye_shape,eye_shape).repeat(encoded.shape[1],1,1)
        target = target.to("cuda")
        prod=torch.bmm(encoded.contiguous().view(encoded.shape[1],eye_shape,-1),predicted.view(predicted.shape[1],-1,eye_shape))
        return self.coding_loss(prod,target)

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
                        # print(key + '-require grad')
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
                        # print(key + '-not require grad')
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1

            # ----------------------------------------------------------> debugging
            # print('original data shape', data.shape)
            # first_smaple = data[0] # (2, 3, 300, 25, 2)
            # orginal_x = first_smaple[0] # (3,300,25,2)
            # noised_x = first_smaple[1] # (3,300,25,2)
            # diff = orginal_x - noised_x
            # print('difference:', diff.sum())

            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            if self.arg.train_feeder_args['modf'] == 'mutual':
                original_data = data[:, 0, :, :, :, :]
                noised_data = data[:, 1, :, :, :, :]

                #----------------------------------------------------------> debugging
                # print('original data shape', original_data.shape)
                # print('noised data shape', noised_data.shape)
                
                #generate Target data for semi-supervised learning
                original_data = Variable(original_data.float().cuda(self.output_device), requires_grad=False)
                original_before_fc = self.model(original_data)
                """
                original_before_fc: latent feature of original data x, dim=256 
                original_after_fc: fc feature of original data x, dim=60 
                """

                #generate trainig data
                #pdb.set_trace()
                noised_data = Variable(noised_data.float().cuda(self.output_device), requires_grad=False)
                noise_before_fc  = self.model(noised_data)
                """
                noise_before_fc: latent feature of noised data x', dim=256
                noised_after_fc: fc feature of noised data x', dim=60                
                """
                prediction_results = self.ar_model(noise_before_fc)
                output = self.model.module.classify(prediction_results)
                output2 = self.model.module.classify(original_before_fc)
                output3 = self.model.module.classify(noise_before_fc)
                
                # ----------------------------------------------------------> debugging
                # print('before FC: ' , noise_before_fc.size() , ' after FC', noised_after_fc.size())
                """
                output: final prediction. fix here
                """


            else:
                original_data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                original_before_fc, original_after_fc = self.model(original_data)
                output = original_after_fc

            # if batch_idx == 0 and epoch == 0:
            #     self.train_writer.add_graph(self.model, output)
            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0

            # Contrastive Encoding Loss----------------------------------->
            self.celoss= self.contrastive_coding_loss(prediction_results,original_before_fc)
            
            # ------------------------------------------------------------> loss-function, jm
            self.entropy_loss = self.loss(output, label)+self.loss(output2,label)+self.loss(output3,label)
            
            loss = self.entropy_loss + self.weight_for_celoss*self.celoss
           
            # ------------------------------------------------------------> loss-function, jm

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('Cross entropy loss', self.entropy_loss.data.item(), self.global_step)
            self.train_writer.add_scalar('Coding loss', self.celoss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_l1', l1, self.global_step)
            # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            # if self.global_step % self.arg.log_interval == 0:
            #     self.print_log(
            #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
            #             batch_idx, len(loader), loss.data[0], lr))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-main-' + str(epoch).zfill(2) + '-' + str(int(self.global_step)).zfill(5) + '.pt')
            
            ar_state_dict = self.ar_model.state_dict()
            ar_weights = OrderedDict([[k.split('module.')[-1],v.cpu()] for k, v in ar_state_dict.items()])
            torch.save(ar_weights,self.arg.model_saved_name + '-regression-' + str(epoch).zfill(2) + '-' + str(int(self.global_step)).zfill(5) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        ckpt = time.time()
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            step = 0
            process = tqdm(self.data_loader[ln])

            for batch_idx, (data, label, index) in enumerate(process):
                # data = (batch, 3, 300, 25, 2)
                # data = data[0:300,:,:,:,:]
                # print('eval, for-loop' , type(data), data.size())

                if self.arg.test_feeder_args['modf'] == 'mutual':
                    #data = data[:, 0, :, :, :, :] # pure
                    data = data[:, 1, :, :, :, :] # noise
                
                with torch.no_grad():
                    data = Variable(
                        # tmp_data.float().cuda(self.output_device),
                        data.float().cuda(self.output_device),
                        requires_grad=False)
                    label = Variable(
                        label.long().cuda(self.output_device),
                        requires_grad=False)
                    latents = self.model(data)
                    predicts = self.ar_model(latents)
                    output = self.model.module.classify(predicts)
                    
                    celoss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(celoss.data.item())
                    # print('input data, 0b, 0t, 20j, 0m', data[0,:,0,20,0])
                    # print('result index and fc', index[0], 'fc size ', output.data[0,:].size(), ' val: ' ,output.data[0,:])
                    # quit()

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
            # self.lr_scheduler.step(loss)
            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
            # tmp_log_str =f'{self.arg.drop_list}, {self.arg[labeling_mode}'
            self.print_log_cusmtom(f';{100 * self.data_loader[ln].dataset.top_k(score, 1):.2f}%;{100 * self.data_loader[ln].dataset.top_k(score, 5):.2f}%')

            # print(self.arg.drop_list, str(self.arg.drop_list))
            # print(self.arg.model_args['graph_args']['labeling_mode'])

            # print('It took ', time.time() -ckpt , 's')
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

    def start(self):

        if self.arg.phase == 'train':
            aa = [self.arg.model_saved_name,
                  self.arg.modf,
                  self.arg.train_feeder_args['modf'],
                  self.arg.train_feeder_args['drop_num'],
                  self.arg.train_feeder_args['ntype'],
                  self.arg.train_feeder_args['ttype']]

            bb = [self.arg.test_feeder_args['modf'],
                  self.arg.test_feeder_args['drop_num'],
                  self.arg.test_feeder_args['ntype'],
                  self.arg.test_feeder_args['ttype']]

            self.print_log(f'Training: {aa}')
            self.print_log(f'Testing: {bb}')
            # self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            start_epoch = self.arg.start_epoch
            if self.arg.weights:
                self.global_step = int(arg.weights[:-3].split('-')[-1])
                start_epoch = int(arg.weights[:-3].split('-')[-2])

            for epoch in range(start_epoch, self.arg.num_epoch):
                if self.lr < 1e-3:
                    break
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=save_model)

                if epoch % 10 == 0:
                    self.eval(
                        epoch,
                        save_score=self.arg.save_score,
                        loader_name=['test'])

            print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':

            bb = [self.arg.test_feeder_args['modf'],
                  self.arg.test_feeder_args['drop_num'],
                  self.arg.test_feeder_args['ntype'],
                  self.arg.test_feeder_args['ttype']]

            self.print_log(f'Testing: {bb}')

            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = True
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)

        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    # print(arg)
    # quit()
    processor = Processor(arg)
    processor.start()
    # WOWOWOW
