import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from unet.unet_model import *
from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from dataset.time_dataset import NewTsDataset
from pathlib import Path

from options.train_options import TrainOptions


# def sample_images(batches_done):
#     """Saves a generated sample from the test set"""
#     imgs = next(iter(val_dataloader))
#     G_AB.eval()
#     G_BA.eval()
#     real_A = Variable(imgs["A"].type(Tensor))
#     fake_B = G_AB(real_A)
#     real_B = Variable(imgs["B"].type(Tensor))
#     fake_A = G_BA(real_B)
#     # Arange images along x-axis
#     real_A = make_grid(real_A, nrow=5, normalize=True)
#     real_B = make_grid(real_B, nrow=5, normalize=True)
#     fake_A = make_grid(fake_A, nrow=5, normalize=True)
#     fake_B = make_grid(fake_B, nrow=5, normalize=True)
#     # Arange images along y-axis
#     image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
#     save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)

if __name__ == '__main__':

    opt = TrainOptions().parse()
    print(opt)

    device = torch.device('cuda') if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    print(f"Device: {device}")

    # # Create sample and checkpoint directories
    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    # Dataset loader
    datapath = Path('data')
    dataset = NewTsDataset(datapath/'CaseI-Attacks without any change.csv')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=4)
    print("The size of the dataset is: ", dataset.data_normal.size(), dataset.data_abnormal.size())

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    cuda = torch.cuda.is_available()

    normal_feature_len = dataset.data_normal.size(2)
    abnormal_feature_len = dataset.data_abnormal.size(2)
    normal_seq_len = dataset.data_normal.size(1)
    abnormal_seq_len = dataset.data_abnormal.size(1)

    G_AB = LSTMUnetGenerator(opt.batch_size, normal_seq_len, normal_feature_len, device=device)
    G_BA = LSTMUnetGenerator(opt.batch_size, abnormal_seq_len, abnormal_feature_len, device=device)
    D_A = LSTMDiscriminator(normal_feature_len)
    D_B = LSTMDiscriminator(abnormal_feature_len)

    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
        G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
        D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
        D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        
        # Initialize weights
       ''' G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)'''

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Image transformations
    # transforms_ = [
    #     transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    #     transforms.RandomCrop((opt.img_height, opt.img_width)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ]

    # Training data loader
    # dataloader = DataLoader(
    #     ImageDataset("./data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    #     num_workers=opt.n_cpu,
    # )
    
    # # Test data loader
    # val_dataloader = DataLoader(
    #     ImageDataset("./data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    #     batch_size=5,
    #     shuffle=True,
    #     num_workers=1,
    # )


    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            # real_A = Variable(batch["A"].type(Tensor))
            # real_B = Variable(batch["B"].type(Tensor))
            real_A = batch["Normal"].clone().detach().to(device)
            real_B = batch["Abnormal"].clone().detach().to(device)

            # Adversarial ground truths
            # valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            # fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            valid = torch.tensor(np.ones((real_A.size(0), D_A.output_shape)), device=device, requires_grad=False, dtype=torch.float32)
            fake = torch.tensor(np.zeros((real_A.size(0), D_A.output_shape)), device=device, requires_grad=False, dtype=torch.float32)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            print(f"\033[92m{real_A.size()}, {real_B.size()}, {fake_B.size()}\033[0m")

            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # # If at sample interval save image
            # if batches_done % opt.sample_interval == 0:
            #     sample_images(batches_done)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
            torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
