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

    def __init__(self, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.inc = (DoubleConv_1D(1, 32))
        self.down1 = (Down_1D(32, 64))
        self.down2 = (Down_1D(64, 128))
        self.down3 = (Down_1D(128, 256 // 1))
        self.up1 = (Up_1D(256, 128 // 1, bilinear=False))
        self.up2 = (Up_1D(128, 64 // 1, bilinear=False))
        self.up3 = (Up_1D(64, 32 // 1, bilinear=False))
        self.outc = (OutConv_1D(32, 1))

        # self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        # self.linear = nn.Sequential(nn.Linear(hidden_dim, feature_len), nn.Tanh())

    def LSTMCalculation(self, x):
        batch_size, seq_len, feature_len  = x.size(0), x.size(1), x.size(2)

        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        lstm = nn.LSTM(input_size=feature_len, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        linear = nn.Sequential(nn.Linear(self.hidden_dim, feature_len+6*2), nn.Tanh())

        recurrent_features, _ = lstm(x, (h_0, c_0))
        outputs = linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, feature_len+6*2)

        return outputs

    def forward(self, x):
        # device = x.device # Get input data's device
        
        # x1=self.inc(x)
        # x2=self.down1(x1)
        # x3=self.down2(x2)
        # x4=self.down3(x3)
        # batch_size, seq_len, feature_len  = x1.size(0), x1.size(1), x1.size(2)
        # h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        # c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        # lstm = nn.LSTM(input_size=feature_len, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        # linear = nn.Sequential(nn.Linear(self.hidden_dim, feature_len), nn.Tanh())

        # recurrent_features, _ = lstm(x1, (h_0, c_0))
        # outputs = linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        # outputs = outputs.view(batch_size, seq_len, feature_len)
        x1 = self.LSTMCalculation(self.inc(x))
        x2 = self.LSTMCalculation(self.down1(x1))
        x3 = self.LSTMCalculation(self.down2(x2))
        x4 = self.LSTMCalculation(self.down3(x3))

        print(x1.shape, x2.shape, x3.shape, x4.shape)

        out = self.LSTMCalculation(self.up1(x4, x3))
        print(out.size())
        out = self.LSTMCalculation(self.up2(out, x2))
        print(out.size())
        out = self.LSTMCalculation(self.up3(out, x1))
        print(out.size())
        out = self.LSTMCalculation(self.outc(out))
        
        return out