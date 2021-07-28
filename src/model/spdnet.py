import torch
import torch.nn as nn
from torch.nn import functional as F
from model import common
from pytorch_wavelets import DWTForward, DWTInverse

def make_model(args, parent=False):
    return Rainnet(args)

def get_residue(tensor , r_dim = 1):
    """
    return residue_channle (RGB)
    """
    # res_channel = []
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel

class convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        self.relu = nn.ReLU()
        self.padding = nn.ReflectionPad2d(kernel_size//2)
        self.conv = nn.Conv2d(inputchannel, outchannel, kernel_size, stride)
        self.ins = nn.InstanceNorm2d(outchannel, affine=True)
    def forward(self, x):
        x = self.conv(self.padding(x))
        # x= self.ins(x)
        x = self.relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(Upsample, self).__init__()
      reflection_padding = kernel_size // 2
      self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
      self.relu = nn.ReLU()

    def forward(self, x, y):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.relu(out)
        out = F.interpolate(out, y.size()[2:])
        return out

class RB(nn.Module):
    def __init__(self, n_feats, nm='in'):
        super(RB, self).__init__()
        module_body = []
        for i in range(2):
            module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
            module_body.append(nn.ReLU())
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.ReLU()
        self.se = common.SELayer(n_feats, 1)

    def forward(self, x):
        res = self.module_body(x)
        res = self.se(res)
        res += x
        return res

class RIR(nn.Module):
    def __init__(self, n_feats, n_blocks, nm='in'):
        super(RIR, self).__init__()
        module_body = [
            RB(n_feats) for _ in range(n_blocks)
        ]
        module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return self.relu(res)

class res_ch(nn.Module):
    def __init__(self, n_feats, blocks=2):
        super(res_ch,self).__init__()
        self.conv_init1 = convd(3, n_feats//2, 3, 1)
        self.conv_init2 = convd(n_feats//2, n_feats, 3, 1)
        self.extra = RIR(n_feats, n_blocks=blocks)

    def forward(self,x):
        x = self.conv_init2(self.conv_init1(x))
        x = self.extra(x)
        return x

class Fuse(nn.Module):
    def __init__(self, inchannel=64, outchannel=64):
        super(Fuse, self).__init__()
        self.up = Upsample(inchannel, outchannel, 3, 2)
        self.conv = convd(outchannel, outchannel, 3, 1)
        self.rb = RB(outchannel)
        self.relu = nn.ReLU()

    def forward(self, x, y):

        x = self.up(x, y)
        # x = F.interpolate(x, y.size()[2:])
        # y1 = torch.cat((x, y), dim=1)
        y = x+y

        # y = self.pf(y1) + y

        return self.relu(self.rb(y))

class Prior_Sp(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim=32):
        super(Prior_Sp, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        # self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        # self.softmax  = nn.Softmax(dim=-1)
        self.sig = nn.Sigmoid()
    def forward(self,x, prior):
        
        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        attention = self.sig(energy)
        # print(attention.size(),x.size())
        attention_x = x * attention
        attention_p = prior * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x),dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        p_gamma = self.gamma2(torch.cat((prior, attention_p),dim=1))
        prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

        return x_out, prior_out

class subnet(nn.Module):
    def __init__(self, n_feats, blocks=2):
        super(subnet,self).__init__()

        self.DWT = DWTForward(J=1, wave='haar').cuda()
        self.IDWT = DWTInverse(wave='haar').cuda()

        self.DWT_2 = convd(n_feats*4, n_feats, 3, 1)
        self.DWT_3 = convd(n_feats*4, n_feats, 3, 1)
        self.IDWT_2 = convd(n_feats, n_feats*4, 3, 1)
        self.IDWT_3 = convd(n_feats, n_feats*4, 3, 1)
        # fuse res
        self.prior = Prior_Sp()
        self.fuse_res = convd(n_feats*2, n_feats, 3, 1)
        # branch1
        self.branch1_1 = RIR(n_feats=n_feats, n_blocks=blocks)
        self.branch1_1_d = convd(n_feats, n_feats, 3, 2)
        self.branch1_2 = RIR(n_feats=n_feats, n_blocks=blocks)
        self.branch1_2_d = convd(n_feats, n_feats, 3, 2)
        self.branch1_3 = RIR(n_feats=n_feats, n_blocks=blocks)

        # branch2
        self.down2 = convd(n_feats, n_feats, 3, 2)
        self.branch2_1 = RIR(n_feats=n_feats, n_blocks=blocks)
        self.branch2_1_u = Upsample(n_feats, n_feats, 3, 2)
        self.branch2_1_d = convd(n_feats, n_feats, 3, 2)
        self.branch2_2 = RIR(n_feats=n_feats, n_blocks=blocks)
        self.branch2_2_u = Upsample(n_feats, n_feats, 3, 2)
        self.branch2_2_d = convd(n_feats, n_feats, 3, 2)
        self.branch2_3 = RIR(n_feats=n_feats, n_blocks=blocks)

        # branch3
        self.down3 = convd(n_feats, n_feats, 3, 2)
        self.branch3_1 = RIR(n_feats=n_feats, n_blocks=blocks)
        self.branch3_1_u = Upsample(n_feats, n_feats, 3, 2)
        self.branch3_2 = RIR(n_feats=n_feats, n_blocks=blocks)
        self.branch3_2_u = Upsample(n_feats, n_feats, 3, 2)
        self.branch3_3 = RIR(n_feats=n_feats, n_blocks=blocks)

        #Fuse
        self.fuse12=Fuse(n_feats, n_feats)
        self.fuse23=Fuse(n_feats,n_feats)

    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:, :, i, :, :])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self, out):
        yh = []
        C = int(out.shape[1] / 4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:, :, 0].contiguous()
        yh.append(y[:, :, 1:].contiguous())

        return yl, yh

    def forward(self, x, res_feats):
        x_p, res_feats_p = self.prior(x, res_feats)
        x_s = torch.cat((x_p, res_feats_p),dim=1)
        x1_init = self.fuse_res(x_s)
        DMT2_yl,DMT2_yh = self.DWT(x1_init)
        DMT2 = self._transformer(DMT2_yl, DMT2_yh)# 32*4=128
        x2_init = self.DWT_2(DMT2)
        DMT3_yl,DMT3_yh = self.DWT(x2_init)
        DMT3 = self._transformer(DMT3_yl, DMT3_yh)# 32*4=128
        x3_init = self.DWT_3(DMT3)

        # column 1
        x1_1 = self.branch1_1(x1_init)
        x2_1 = self.branch2_1(x2_init)
        x3_1 = self.branch3_1(x3_init)
        # cross conection
        x1_i = x1_1
        x2_i = x2_1
        x3_i = x3_1
 
        # column 2
        x1_2 = self.branch1_2(x1_i)
        x2_2 = self.branch2_2(x2_i)
        x3_2 = self.branch3_2(x3_i)
        # cross conection
        x1_i = x1_2
        x2_i = x2_2
        x3_i = x3_2

        # column 3
        x1_3 = self.branch1_3(x1_i)
        x2_3 = self.branch2_3(x2_i)
        x3_3 = self.branch3_3(x3_i)
        # # cross conection
        x1_i = x1_3
        x2_i = x2_3
        x3_i = x3_3

        x3_i = self.IDWT_3(x3_i)#32-->128
        D_3 =self._Itransformer(x3_i)
        IDMT3 =self.IDWT(D_3)#128-->32

        x2_i = self.fuse23(IDMT3, x2_i)
        x2_i = self.IDWT_2(x2_i)#32-->128
        D_2 =self._Itransformer(x2_i)
        IDMT2 =self.IDWT(D_2)#128-->32

        x1_i = self.fuse12(IDMT2, x1_i)

        return x1_i

class Rainnet(nn.Module):
    def __init__(self,args):
        super(Rainnet,self).__init__()
        n_feats = args.n_feats
        blocks = args.n_resblocks
        
        self.conv_init1 = convd(3, n_feats//2, 3, 1)
        self.conv_init2 = convd(n_feats//2, n_feats, 3, 1)
        self.res_extra1 = res_ch(n_feats, blocks)
        self.sub1 = subnet(n_feats, blocks)
        self.res_extra2 = res_ch(n_feats, blocks)
        self.sub2 = subnet(n_feats, blocks)
        self.res_extra3 = res_ch(n_feats, blocks)
        self.sub3 = subnet(n_feats, blocks)

        self.ag1 = convd(n_feats*2,n_feats,3,1)
        self.ag2 = convd(n_feats*3,n_feats,3,1)
        self.ag2_en = convd(n_feats*2, n_feats, 3, 1)
        self.ag_en = convd(n_feats*3, n_feats, 3, 1)

        self.output1 = nn.Conv2d(n_feats, 3, 3, 1, padding=1)
        self.output2 = nn.Conv2d(n_feats, 3, 3, 1, padding=1)
        self.output3 = nn.Conv2d(n_feats, 3, 3, 1, padding=1)
        
        # self._initialize_weights()

    def forward(self,x):

        res_x = get_residue(x)
        x_init = self.conv_init2(self.conv_init1(x))
        x1 = self.sub1(x_init, self.res_extra1(torch.cat((res_x, res_x, res_x), dim=1))) #+ x   # 1
        out1 = self.output1(x1)
        res_out1 = get_residue(out1)
        x2 = self.sub2(self.ag1(torch.cat((x1,x_init),dim=1)), self.res_extra2(torch.cat((res_out1, res_out1, res_out1), dim=1))) #+ x1 # 2
        x2_ = self.ag2_en(torch.cat([x2,x1], dim=1))
        out2 = self.output2(x2_)
        res_out2 = get_residue(out2)
        x3 = self.sub3(self.ag2(torch.cat((x2,x1,x_init),dim=1)), self.res_extra3(torch.cat((res_out2, res_out2, res_out2), dim=1))) #+ x2 # 3
        x3 = self.ag_en(torch.cat([x3,x2,x1],dim=1))
        out3 = self.output3(x3)

        return out3, out2, out1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# sum(param.numel() for param in net.parameters())
