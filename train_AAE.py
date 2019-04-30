# Copyright 2018 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import print_function

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image

from net import *
from utils import input_helper

use_cuda = torch.cuda.is_available()

FloatTensor = torch.FloatTensor
IntTensor = torch.IntTensor
LongTensor = torch.LongTensor
torch.set_default_tensor_type('torch.FloatTensor')

# If zd_merge true, will use zd discriminator that looks at entire batch.
zd_merge = False

if use_cuda:
    device = torch.cuda.current_device()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FloatTensor = torch.cuda.FloatTensor
    IntTensor = torch.cuda.IntTensor
    LongTensor = torch.cuda.LongTensor
    print("Running on ", torch.cuda.get_device_name(device))


def setup(x):
    if use_cuda:
        return x.cuda()
    else:
        return x.cpu()


def numpy2torch(x):
    return setup(torch.from_numpy(x))


def extract_batch(data, it, batch_size):
    x = numpy2torch(data[it * batch_size:(it + 1) * batch_size, :, :]) / 255.0
    # x.sub_(0.5).div_(0.5)
    return Variable(x)


def main(dataset_name, inliner_class):
    batch_size = 128
    zsize = 32

    dataset = input_helper.keras_inbuilt_dataset(dataset=dataset_name, normal_class_label=inliner_class,
                                                 batch_size=batch_size)

    if dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset == 'catdog':
        channels = 3
    elif dataset_name == 'fashion_mnist':
        channels = 1
    else:
        raise AttributeError

    G = Generator(zsize, channels=channels)
    setup(G)
    G.weight_init(mean=0, std=0.02)

    D = Discriminator(channels=channels)
    setup(D)
    D.weight_init(mean=0, std=0.02)

    E = Encoder(zsize, channels=channels)
    setup(E)
    E.weight_init(mean=0, std=0.02)

    if zd_merge:
        ZD = ZDiscriminator_mergebatch(zsize, batch_size).to(device)
    else:
        ZD = ZDiscriminator(zsize, batch_size).to(device)

    setup(ZD)
    ZD.weight_init(mean=0, std=0.02)

    lr = 0.002

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    E_optimizer = optim.Adam(E.parameters(), lr=lr, betas=(0.5, 0.999))
    GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.999))
    ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.999))

    train_epoch = 80

    BCE_loss = torch.nn.BCELoss()
    y_real_ = torch.ones(batch_size)
    y_fake_ = torch.zeros(batch_size)

    y_real_z = torch.ones(1 if zd_merge else batch_size)
    y_fake_z = torch.zeros(1 if zd_merge else batch_size)

    sample = torch.randn(64, zsize).view(-1, zsize, 1, 1)

    for epoch in range(train_epoch):
        G.train()
        D.train()
        E.train()
        ZD.train()

        Gtrain_loss = 0
        Dtrain_loss = 0
        Etrain_loss = 0
        GEtrain_loss = 0
        ZDtrain_loss = 0

        epoch_start_time = time.time()

        if (epoch + 1) % 30 == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            D_optimizer.param_groups[0]['lr'] /= 4
            GE_optimizer.param_groups[0]['lr'] /= 4
            E_optimizer.param_groups[0]['lr'] /= 4
            ZD_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        for it in range(len(dataset) // batch_size):
            # [None, H, W, C] -> [None, C, H, W]
            x, _ = dataset.get_next_batch()
            x = np.moveaxis(x, source=3, destination=1).astype(np.float32)
            x = torch.from_numpy(x).to(device)

            #############################################

            D.zero_grad()

            D_result = D(x).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            z = torch.randn((batch_size, zsize)).view(-1, zsize, 1, 1)
            z = Variable(z)

            x_fake = G(z).detach()
            D_result = D(x_fake).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()

            D_optimizer.step()

            Dtrain_loss += D_train_loss.item()

            #############################################

            G.zero_grad()

            z = torch.randn((batch_size, zsize)).view(-1, zsize, 1, 1)
            z = Variable(z)

            x_fake = G(z)
            D_result = D(x_fake).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)

            G_train_loss.backward()
            G_optimizer.step()

            Gtrain_loss += G_train_loss.item()

            #############################################

            ZD.zero_grad()

            z = torch.randn((batch_size, zsize)).view(-1, zsize)
            z = Variable(z)

            ZD_result = ZD(z).squeeze()
            ZD_real_loss = BCE_loss(ZD_result, y_real_z)

            z = E(x).squeeze().detach()

            ZD_result = ZD(z).squeeze()
            ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

            ZD_train_loss = ZD_real_loss + ZD_fake_loss
            ZD_train_loss.backward()

            ZD_optimizer.step()

            ZDtrain_loss += ZD_train_loss.item()

            #############################################

            E.zero_grad()
            G.zero_grad()

            z = E(x)
            x_d = G(z)

            ZD_result = ZD(z.squeeze()).squeeze()

            E_loss = BCE_loss(ZD_result, y_real_z) * 2.0

            Recon_loss = F.binary_cross_entropy(x_d, x)

            (Recon_loss + E_loss).backward()

            GE_optimizer.step()

            GEtrain_loss += Recon_loss.item()
            Etrain_loss += E_loss.item()

            if it == 0:
                directory = 'results' + str(inliner_class)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                comparison = torch.cat([x[:64], x_d[:64]])
                save_image(comparison.cpu(),
                           'results' + str(inliner_class) + '/reconstruction_' + str(epoch) + '.png', nrow=64)

        Gtrain_loss /= (len(dataset))
        Dtrain_loss /= (len(dataset))
        ZDtrain_loss /= (len(dataset))
        GEtrain_loss /= (len(dataset))
        Etrain_loss /= (len(dataset))

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, Gloss: %.3f, Dloss: %.3f, ZDloss: %.3f, GEloss: %.3f, Eloss: %.3f' % (
        (epoch + 1), train_epoch, per_epoch_ptime, Gtrain_loss, Dtrain_loss, ZDtrain_loss, GEtrain_loss, Etrain_loss))

        with torch.no_grad():
            resultsample = G(sample).cpu()
            directory = 'results' + str(inliner_class)
            os.makedirs(directory, exist_ok=True)
            save_image(resultsample.view(64, 1, 32, 32),
                       'results' + str(inliner_class) + '/sample_' + str(epoch) + '.png')

    print("Training finish!... save training results")
    torch.save(G.state_dict(), "Gmodel.pkl")
    torch.save(E.state_dict(), "Emodel.pkl")
    torch.save(D.state_dict(), "Dmodel.pkl")
    torch.save(ZD.state_dict(), "ZDmodel.pkl")


if __name__ == '__main__':
    main(dataset_name='cifar10', inliner_class=0)
