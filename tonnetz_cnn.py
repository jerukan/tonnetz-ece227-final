"""
Every utility related to Tonnetz graph CNN model training.
"""

from pathlib import Path
from typing import List

import hexagdly
import lightning as L
import music21 as m21
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2

import tonnetz_util as tnzu


class MidiTonnetzDataset(Dataset):
    """
    General dataset for single piano song data loading.

    Represents as a series of finite Tonentz graphs for each measure in a song.
    """
    def __init__(self, midi_path, nprev=1, transform=None, interval="quarter", midioffset=0):
        self.midi_path = Path(midi_path)
        if not self.midi_path.exists():
            raise FileNotFoundError(f"{midi_path} does not exist")
        self.score = m21.converter.parse(self.midi_path)
        self.tonnetzmaps: List[tnzu.TonnetzMap] = tnzu.midi_to_tonnetzmaps(self.midi_path, interval=interval, midioffset=midioffset)
        self.oddqgrids = np.array([tm.to_oddq_grid().astype(np.float32) for tm in self.tonnetzmaps])
        if transform is None:
            self.transform = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.transform = transform
        self.nprev = nprev

    def __len__(self):
        return len(self.tonnetzmaps) - self.nprev

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.__len__() + idx
        tonnetzprev = self.oddqgrids[idx:idx + self.nprev]
        tonnetznext = self.tonnetzmaps[idx + self.nprev].to_oddq_grid().astype(np.float32)
        tonnetznext_unsq = np.array([tonnetznext])
        return torch.from_numpy(tonnetzprev), torch.from_numpy(tonnetznext_unsq)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            hexagdly.Conv2d(in_channels, mid_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            hexagdly.Conv2d(mid_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            hexagdly.MaxPool2d(kernel_size=1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetModel(L.LightningModule):
    def __init__(self, nchannels, bilinear=False, pos_weight=4):
        super().__init__()
        self.n_channels = nchannels
        self.bilinear = bilinear

        self.inc = DoubleConv(nchannels, 64)
        self.down1 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down2 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.outc = hexagdly.Conv2d(64, 1, kernel_size=1)

        self.lossfunc = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]).to("mps:0"))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.lossfunc(out, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.lossfunc(out, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class CrapModel(L.LightningModule):

    def __init__(self, nchannels=1, pos_weight=4):
        super().__init__()
        self.conv1 = hexagdly.Conv2d(nchannels, 64, kernel_size=2)
        self.conv2 = hexagdly.Conv2d(64, 128, kernel_size=2)
        self.conv3 = hexagdly.Conv2d(128, 64, kernel_size=1)
        self.finalconv = hexagdly.Conv2d(64, 1, kernel_size=1)
        self.lossfunc = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]).to("mps:0"))

    def encoder(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.finalconv(x)

        return x
    
    def forward(self, x):
        return self.encoder(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.lossfunc(out, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.lossfunc(out, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
