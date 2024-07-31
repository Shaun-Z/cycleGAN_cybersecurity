""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class UNet_1D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_1D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv_1D(n_channels, 32))
        self.down1 = (Down_1D(32, 64))
        self.down2 = (Down_1D(64, 128))
        factor = 2 if bilinear else 1
        self.down3 = (Down_1D(128, 256 // factor))
        self.up1 = (Up_1D(256, 128 // 1, bilinear))
        self.up2 = (Up_1D(128, 64 // 1, bilinear))
        self.up3 = (Up_1D(64, 32 // 1, bilinear))
        self.outc = (OutConv_1D(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.outc = torch.utils.checkpoint(self.outc)

class LSTMUnetGenerator(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.

    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(self, batch_size, seq_len, feature_len, kernel_size=7, n_layers=1, hidden_dim=256, device='cpu'):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.inc = (DoubleConv_1D(1, 32))
        self.h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.lstm = nn.LSTM(input_size=feature_len-kernel_size*2+2, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(self.hidden_dim, feature_len), nn.Tanh())

        self.down1 = (Down_1D(32, 64))
        self.h_1 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.c_1 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.lstm1 = nn.LSTM(input_size=feature_len//2-kernel_size*2+2, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.linear1 = nn.Sequential(nn.Linear(self.hidden_dim, feature_len//2), nn.Tanh())

        self.down2 = (Down_1D(64, 128))
        self.h_2 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.c_2 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.lstm2 = nn.LSTM(input_size=feature_len//4-kernel_size*2+2, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.linear2 = nn.Sequential(nn.Linear(self.hidden_dim, feature_len//4), nn.Tanh())

        self.down3 = (Down_1D(128, 256 // 1))
        self.h_3 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.c_3 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.lstm3 = nn.LSTM(input_size=feature_len//8-kernel_size*2+2, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.linear3 = nn.Sequential(nn.Linear(self.hidden_dim, feature_len//8), nn.Tanh())

        self.up1 = (Up_1D(256, 128 // 1, bilinear=False))
        self.h_1U = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.c_1U = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.lstm1U = nn.LSTM(input_size=feature_len//8*2-kernel_size*2+2, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.linear1U = nn.Sequential(nn.Linear(self.hidden_dim, feature_len//8*2), nn.Tanh())

        self.up2 = (Up_1D(128, 64 // 1, bilinear=False))
        self.h_2U = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.c_2U = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.lstm2U = nn.LSTM(input_size=feature_len//8*4-kernel_size*2+2, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.linear2U = nn.Sequential(nn.Linear(self.hidden_dim, feature_len//8*4), nn.Tanh())

        self.up3 = (Up_1D(64, 32 // 1, bilinear=False))
        self.h_3U = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.c_3U = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.lstm3U = nn.LSTM(input_size=feature_len//8*8-kernel_size*2+2, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.linear3U = nn.Sequential(nn.Linear(self.hidden_dim, feature_len//8*8), nn.Tanh())

        self.outc = (OutConv_1D(32, 1))
        self.h_4U = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.c_4U = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        self.lstm4U = nn.LSTM(input_size=feature_len//8*8, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.linear4U = nn.Sequential(nn.Linear(self.hidden_dim, feature_len//8*8), nn.Tanh())

    def forward(self, x):
        # device = x.device # Get input data's device
        x_mid = self.inc(x)
        batch_size, seq_len, feature_len  = x_mid.size(0), x_mid.size(1), x_mid.size(2)
        recurrent_features, _ = self.lstm(x_mid, (self.h_0, self.c_0))
        x1 = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim)).view(batch_size, seq_len, feature_len+(self.kernel_size-1)*2)
        
        x1_mid = self.down1(x1)
        batch_size, seq_len, feature_len  = x1_mid.size(0), x1_mid.size(1), x1_mid.size(2)
        recurrent_features, _ = self.lstm1(x1_mid, (self.h_1, self.c_1))
        x2 = self.linear1(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim)).view(batch_size, seq_len, feature_len+(self.kernel_size-1)*2)

        x2_mid = self.down2(x2)
        batch_size, seq_len, feature_len  = x2_mid.size(0), x2_mid.size(1), x2_mid.size(2)
        recurrent_features, _ = self.lstm2(x2_mid, (self.h_2, self.c_2))
        x3 = self.linear2(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim)).view(batch_size, seq_len, feature_len+(self.kernel_size-1)*2)

        x3_mid = self.down3(x3)
        batch_size, seq_len, feature_len  = x3_mid.size(0), x3_mid.size(1), x3_mid.size(2)
        recurrent_features, _ = self.lstm3(x3_mid, (self.h_3, self.c_3))
        x4 = self.linear3(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim)).view(batch_size, seq_len, feature_len+(self.kernel_size-1)*2)

        # print(x1.shape, x2.shape, x3.shape, x4.shape)

        out_mid = self.up1(x4, x3)
        batch_size, seq_len, feature_len  = out_mid.size(0), out_mid.size(1), out_mid.size(2)
        recurrent_features, _ = self.lstm1U(out_mid, (self.h_1U, self.c_1U))
        out = self.linear1U(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        out = out.view(batch_size, seq_len, feature_len+(self.kernel_size-1)*2)

        out_mid = self.up2(out, x2)
        batch_size, seq_len, feature_len  = out_mid.size(0), out_mid.size(1), out_mid.size(2)
        recurrent_features, _ = self.lstm2U(out_mid, (self.h_2U, self.c_2U))
        out = self.linear2U(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        out = out.view(batch_size, seq_len, feature_len+(self.kernel_size-1)*2)

        out_mid = self.up3(out, x1)
        batch_size, seq_len, feature_len  = out_mid.size(0), out_mid.size(1), out_mid.size(2)
        recurrent_features, _ = self.lstm3U(out_mid, (self.h_3U, self.c_3U))
        out = self.linear3U(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        out = out.view(batch_size, seq_len, feature_len+(self.kernel_size-1)*2)

        out_mid = self.outc(out)
        batch_size, seq_len, feature_len  = out_mid.size(0), out_mid.size(1), out_mid.size(2)
        recurrent_features, _ = self.lstm4U(out_mid, (self.h_4U, self.c_4U))
        out = self.linear4U(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        out = out.view(batch_size, seq_len, feature_len)

        return out