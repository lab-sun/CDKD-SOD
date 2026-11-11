import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Linear Embedding: 
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):
        
        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        
        self.in_channels = in_channels
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim


        # RGB decoder

        self.linear_c4_rgb = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3_rgb = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2_rgb = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1_rgb = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        
        self.linear_fuse_rgb = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)
                            )
                            
        self.mlp_bu = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)
                            )

        self.linear_pred_rgb = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)


        self.cbr_bu_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.rgb_comple_fusion_conv1 = nn.Conv2d(in_channels=self.num_classes*2,out_channels=self.num_classes,kernel_size=1,stride=1,padding=0)

        #self.linear_pred_fusion = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)                


       
    def forward(self, rgb):
        # len=4, 1/4,1/8,1/16,1/32

        c1_r, c2_r, c3_r, c4_r = rgb
        verbose = False

        if verbose: print("c1_r size",c1_r.size())
        if verbose: print("c2_r size",c2_r.size())
        if verbose: print("c3_r size",c3_r.size())
        if verbose: print("c4_r size",c4_r.size())


        if verbose: print("c1_d size",c1_d.size())
        if verbose: print("c2_d size",c2_d.size())
        if verbose: print("c3_d size",c3_d.size())
        if verbose: print("c4_d size",c4_d.size())
        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_r.shape

        _c4_r = self.linear_c4_rgb(c4_r).permute(0,2,1).reshape(n, -1, c4_r.shape[2], c4_r.shape[3])
        if verbose: print("_c4_r size after linear_c4_rgb",_c4_r.size())


        _c4_r = F.interpolate(_c4_r, size=c1_r.size()[2:],mode='bilinear',align_corners=self.align_corners)
        if verbose: print("_c4_r size after interpolate",_c4_r.size())



        _c3_r = self.linear_c3_rgb(c3_r).permute(0,2,1).reshape(n, -1, c3_r.shape[2], c3_r.shape[3])
        if verbose: print("_c3_r size after linear_c3_rgb",_c3_r.size())


        _c3_r = F.interpolate(_c3_r, size=c1_r.size()[2:],mode='bilinear',align_corners=self.align_corners)
        if verbose: print("_c3_r size after interpolate",_c3_r.size())



        _c2_r = self.linear_c2_rgb(c2_r).permute(0,2,1).reshape(n, -1, c2_r.shape[2], c2_r.shape[3])
        if verbose: print("_c2_r size after linear_c2_rgb",_c2_r.size())


        _c2_r = F.interpolate(_c2_r, size=c1_r.size()[2:],mode='bilinear',align_corners=self.align_corners)
        if verbose: print("_c2_r size after interpolate",_c2_r.size())



        _c1_r = self.linear_c1_rgb(c1_r).permute(0,2,1).reshape(n, -1, c1_r.shape[2], c1_r.shape[3])
        if verbose: print("_c1_r size after linear_c1_rgb",_c1_r.size())




        _c_r = self.linear_fuse_rgb(torch.cat([_c4_r, _c3_r, _c2_r, _c1_r], dim=1))


        cbr_bu = self.mlp_bu(torch.cat([_c4_r, _c3_r, _c2_r, _c1_r], dim=1))
        fusion_bu = cbr_bu + _c_r



        x_r = self.dropout(_c_r)
        x_r = self.linear_pred_rgb(x_r)


        bu_pred = self.cbr_bu_pred(fusion_bu)



        bu_fusion = self.rgb_comple_fusion_conv1(torch.cat((x_r,bu_pred),dim=1))



        # rgb_comple,depth_comple,rgb_fusion,depth_fusion = self.DRC(fusion,x_r,x_d)


        return x_r, bu_fusion, fusion_bu

