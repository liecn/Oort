# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class classificationModel(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim_in=1,conv_dim_out=5,conv_dim_conv=299):
        super(classificationModel, self).__init__()

        self.out_size=conv_dim_out
        self.conv1 = nn.Conv1d(conv_dim_in, 8, 5, stride=3)

        self.lstm = nn.LSTM(
            # 8 * opts.image_height + opts.embedding,
            # 8 * int(opts.stft_nfft/2+1),
            conv_dim_conv,
            conv_dim_out*32,
            batch_first=True,
            bidirectional=False)

        self.dense = nn.Linear(conv_dim_conv*8, conv_dim_out*32)


        self.fcn1 = nn.Linear(conv_dim_out*32, conv_dim_out*8)
        # self.fcn1 = nn.Linear(conv_dim_lstm*4, conv_dim_out*4)
        self.fcn2 = nn.Linear(8 * conv_dim_out, conv_dim_out)
        
        self.softmax=nn.Softmax(dim=1)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.conv1(x.float().unsqueeze(1)))
        out = out.reshape(out.size(0), -1)
        out=self.act(self.dense(out))
        # out,_ = self.lstm(out)
        out = self.act(out)
        # out=self.drop2(out)

        out = self.act(self.fcn1(out))
        # out=self.drop1(out)

        # out = self.softmax(self.fcn2(out))
        out = self.fcn2(out)
        return out
