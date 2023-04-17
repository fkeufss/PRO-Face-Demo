import torch.optim
import torch.nn as nn
from hinet import Hinet
import modules.Unet_common as common
import proface.config.config as c


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dwt = common.DWT().to(device)
iwt = common.IWT().to(device)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = Hinet()

    def forward(self, input1, input2, rev=False):
        if not rev:
            secret_img, cover_img = input1, input2
            # cover_input = dwt(cover_img)  # torch.Size([8, 12, W, H])
            # secret_input = dwt(secret_img)
            # input_img = torch.cat((cover_input, secret_input), 1)
            input_img = torch.cat((cover_img, secret_img), 1)
            output = self.model(input_img)  # torch.Size([8, 24, W, H])
            # output_steg = output.narrow(1, 0, 4 * c.channels_in)  # 取前半部分通道
            # output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)  # 取后半部分通道
            output_steg = output.narrow(1, 0, c.channels_in)  # 取前半部分通道
            output_z = output.narrow(1, c.channels_in, output.shape[1] - c.channels_in)  # 取后半部分通道
            # output_steg_img = iwt(output_steg)
            # return output_z, output_steg, output_steg_img
            # return output_z, output_steg, output_steg_img
            return output_z, output_steg
        else:
            output_z, output_steg = input1, input2
            output_rev = torch.cat((output_steg, output_z), 1)
            output_image = self.model(output_rev, rev=True)
            # secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            # secret_rev_img = iwt(secret_rev)
            cover_rev_img = output_image.narrow(1, 0, c.channels_in)
            secret_rev_img = output_image.narrow(1, c.channels_in, output_image.shape[1] - c.channels_in)
            return secret_rev_img, cover_rev_img


class ModelDWT(nn.Module):
    def __init__(self):
        super(ModelDWT, self).__init__()
        self.model = Hinet()

    def forward(self, input1, input2, password, rev=False):
        if not rev:
            secret_img, cover_img = input1, input2
            cover_dwt = dwt(cover_img)  # torch.Size([Batch, 12, W, H])
            secret_dwt = dwt(secret_img)
            input_dwt = torch.cat((cover_dwt, secret_dwt), 1)
            output = self.model(input_dwt, password)  # torch.Size([Batch, 24, W, H])
            output_steg_dwt = output.narrow(1, 0, 4 * c.channels_in)  # 取前半部分通道
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)  # 取后半部分通道
            # output_steg = output.narrow(1, 0, c.channels_in)  # 取前半部分通道
            # output_z = output.narrow(1, c.channels_in, output.shape[1] - c.channels_in)  # 取后半部分通道
            output_steg_img = iwt(output_steg_dwt)
            # return output_z, output_steg, output_steg_img
            # return output_z, output_steg, output_steg_img
            return output_z, output_steg_img
        else:
            output_z, output_steg_img = input1, input2
            output_steg_dwt = dwt(output_steg_img)
            output_rev = torch.cat((output_steg_dwt, output_z), 1)
            output_dwt = self.model(output_rev, password, rev=True)
            secret_rev_dwt = output_dwt.narrow(1, 4 * c.channels_in, output_dwt.shape[1] - 4 * c.channels_in)
            secret_rev_img = iwt(secret_rev_dwt)
            cover_rev_dwt = output_dwt.narrow(1, 0, 4 * c.channels_in)
            cover_rev_img = iwt(cover_rev_dwt)
            return secret_rev_img, cover_rev_img



def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            # param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            param.data = c.init_scale * torch.randn(param.data.shape).to(device)
            if split[-2] == 'conv5':
                param.data.fill_(0.)