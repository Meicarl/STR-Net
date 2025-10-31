import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

class get_conv_layer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        act=nn.ReLU,
        norm=nn.BatchNorm3d,
        padding=1,
        dilation=1,
        bias=False,
        conv_only=False,
        is_transposed=False,
    ):
        super().__init__()
        
        conv_op = nn.ConvTranspose3d if is_transposed else nn.Conv3d

        self.conv = conv_op(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        self.conv_only = conv_only
        self.norm = norm(out_channels) if not conv_only else None
        self.act = act() if not conv_only else None

    def forward(self, x):
        x = self.conv(x)
        if not self.conv_only:
            x = self.norm(x)
            x = self.act(x)
        return x

class UnetBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            conv_only=False,
        )
        self.conv2 = get_conv_layer(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_only=False,
        )

        self.lrelu = torch.nn.LeakyReLU(inplace=True, negative_slope=0.01)
        self.norm1 = torch.nn.InstanceNorm3d(num_features=out_channels)
        self.norm2 = torch.nn.InstanceNorm3d(num_features=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class  UnetResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            conv_only=False,
        )
        self.conv2 = get_conv_layer(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_only=False,
        )
        self.lrelu = torch.nn.LeakyReLU(inplace=True, negative_slope=0.01)
        self.norm1 = torch.nn.InstanceNorm3d(num_features=out_channels)
        self.norm2 = torch.nn.InstanceNorm3d(num_features=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                conv_only=False,
            )
            self.norm3 = torch.nn.InstanceNorm3d(num_features=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        # if out.shape != residual.shape:
        #     residual = F.interpolate(residual, size=out.shape[2:], mode='trilinear', align_corners=True)
        out += residual
        out = self.lrelu(out)
        return out


class Uper(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        upsample_kernel_size=2,
        res_block: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.out_channels = out_channels    
        self.transp_conv = get_conv_layer(
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
            padding=0,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        # skip = F.interpolate(skip, size=out.shape[2:], mode='trilinear', align_corners=True)
        out = F.interpolate(out, size=skip.shape[2:], mode='trilinear', align_corners=True)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out



class Down(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        res_block: bool = False,
    ):
        super().__init__()

        if res_block:
            self.layer = UnetResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            )
        else:
            self.layer = UnetBasicBlock(  # type: ignore
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            )

    def forward(self, inp):
        return self.layer(inp)

class Out(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
        ):
            super().__init__()
            self.conv = get_conv_layer(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_only=False,
            )

    def forward(self, inp):
        return self.conv(inp)

class Usp(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        upsample_kernel_size=2,
        target_size=None  # 添加目标大小参数
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.out_channels = out_channels    
        self.target_size = target_size  # 保存目标大小
        self.transp_conv = get_conv_layer(
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
            padding=0,
        )

        self.conv_block = UnetBasicBlock(  # type: ignore
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
        )

    def forward(self, inp):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = self.conv_block(out)
        
        # 如果设置了目标大小，则进行插值
        if self.target_size is not None:
            out = F.interpolate(out, size=self.target_size, mode='trilinear', align_corners=False)
        
        return out



def plot_training_curves(train_loss_epochs_list, val_loss_epochs_list, 
                         train_psnr_epochs_list, val_psnr_epochs_list, 
                         train_ssim_epochs_list, val_ssim_epochs_list, 
                         train_out, best_model_path_psnr, 
                         best_model_path_loss, best_model_path_ssim):
    # 绘制训练集与验证集损失曲线
    plt.figure()
    plt.plot(train_loss_epochs_list, label='Train Loss')
    plt.plot(val_loss_epochs_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Val Loss')
    plt.savefig(os.path.join(train_out, 'train_val_loss_curve.png'))
    plt.show()

    # 绘制训练集与验证集PSNR曲线
    plt.figure()
    plt.plot(train_psnr_epochs_list, label='Train PSNR')
    plt.plot(val_psnr_epochs_list, label='Val PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()
    plt.title('Train and Val PSNR')
    plt.savefig(os.path.join(train_out, 'train_val_psnr_curve.png'))
    plt.show()

    # 绘制训练集与验证集SSIM曲线
    plt.figure()
    plt.plot(train_ssim_epochs_list, label='Train SSIM')
    plt.plot(val_ssim_epochs_list, label='Val SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.title('Train and Val SSIM')
    plt.savefig(os.path.join(train_out, 'train_val_ssim_curve.png'))
    plt.show()

    print(f"Best PSNR model saved at: {best_model_path_psnr}")
    print(f"Best Loss model saved at: {best_model_path_loss}")
    print(f"Best SSIM model saved at: {best_model_path_ssim}")

class oneConv3D(nn.Module):
    # 3D 卷积 + ReLU 激活函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations, bias=False)
        )

    def forward(self, x):
        return self.conv(x)

class MSFblock3D(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(MSFblock3D, self).__init__()
        
        # 输入的每个特征图的通道数列表
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        # 对每个特征图进行GAP和权重计算的层 (1x1x1卷积)
        self.gap_layers = nn.ModuleList([nn.AdaptiveAvgPool3d(1) for _ in in_channels_list])
        
        # 统一通道数的卷积层，确保每个特征图的通道数一致
        # self.conv_layers = nn.ModuleList([oneConv3D(in_channels, out_channels, kernel_sizes=1, paddings=0, dilations=1) for in_channels in in_channels_list])
        self.conv_layers = oneConv3D(self.in_channels_list[3], self.in_channels_list[3], kernel_sizes=1, paddings=0, dilations=1) 
        # 目标大小
        target_size = (25, 25, 49)
        self.usp_list = nn.ModuleList([
            Usp(in_channels=self.in_channels_list[0], out_channels=self.in_channels_list[3], kernel_size=3, upsample_kernel_size=16, target_size=target_size),
            Usp(in_channels=self.in_channels_list[1], out_channels=self.in_channels_list[3], kernel_size=3, upsample_kernel_size=8, target_size=target_size),
            Usp(in_channels=self.in_channels_list[2], out_channels=self.in_channels_list[3], kernel_size=3, upsample_kernel_size=4, target_size=target_size),
            Usp(in_channels=self.in_channels_list[3], out_channels=self.in_channels_list[3], kernel_size=3, upsample_kernel_size=2, target_size=target_size)
        ])

        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.project = nn.Sequential(
            nn.Conv3d(self.out_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm3d(self.out_channels),
            nn.ReLU(),)

    def forward(self, feature_list):
        assert len(feature_list) == len(self.in_channels_list), "特征图数量与通道数列表不匹配"
       
        # 第一步：对每个特征图进行通道调整，并应用 GAP 和权重计算
        gap_weights = []
        processed_features = []
        for idx, feature in enumerate(feature_list):
            feature = self.usp_list[idx](feature)

            # 使用 1x1x1 卷积将每个特征图的通道数调整为相同的大小
            # feature = self.conv_layers[idx](feature)
            processed_features.append(feature)

            # avg_pool = self.gap_layers[idx](feature)  # GAP操作
            weights = self.conv_layers(self.gap_layers[idx](feature))  # GAP操作
            gap_weights.append(weights)
            if idx == 3:
                break
        
        # 第二步：将权重拼接并跨尺度进行归一化
        gap_weights_concat = torch.cat(gap_weights, dim=2)  # 在通道维度拼接权重
        normalized_weights = self.softmax(self.sigmoid(gap_weights_concat))  # 先Sigmoid后Softmax归一化

        # 第三步：将归一化的权重应用于各个特征图
        weighted_features = []
        for idx, feature in enumerate(processed_features):
            weight = torch.unsqueeze(normalized_weights[:, :, idx], 2)  # 获取对应的权重
            weighted_features.append(weight * feature)  # 将权重应用于特征图
            if idx == 3:
                break

        # 第四步：将加权后的特征图相加得到输出
        fused_feature = sum(weighted_features)
        
        fused_feature = self.project(fused_feature+feature_list[-1])

        return fused_feature
        
        # return fused_feature+feature_list[-1]


class EMA3D(nn.Module):
    def __init__(self, channels, factor=9):
        super(EMA3D, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv3d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv3d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, d, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, d, h, w)  # b*g, c//g, d, h, w
        x_d = self.pool_d(group_x)
        x_h = self.pool_h(group_x).permute(0, 1, 3, 2, 4)
        x_w = self.pool_w(group_x).permute(0, 1, 4, 3, 2)       
        dhw = self.conv1x1(torch.cat([x_d, x_h, x_w], dim=2))
        x_d, x_h, x_w = torch.split(dhw, [d, h, w], dim=2)
        x1 = self.gn(group_x * x_d.sigmoid() * x_h.sigmoid() * x_w.permute(0, 1, 4, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, dhw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, dhw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, d, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, d, h, w)
    
