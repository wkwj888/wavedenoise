"""
小波变换模块

本模块实现了基于卷积神经网络的小波变换，用于图像的多分辨率分析。
小波变换是一种时频分析工具，可以将图像分解为不同频率和位置的子带，
常用于图像压缩、去噪和特征提取等应用。
"""
import torch.nn as nn
import torch
import math
import pickle


class WaveletTransform(nn.Module):
    """
    小波变换模块
    
    该类实现了可学习的小波变换和逆变换，支持图像的分解和重构。
    使用卷积和转置卷积操作来实现小波变换的前向和逆向过程。
    """
    
    def __init__(self, scale=1, dec=True, params_path='./models/wavelet_weights_c2.pkl',
                 transpose=True):
        """
        初始化小波变换模块
        
        Args:
            scale (int): 小波变换的尺度参数，决定变换的分辨率级别
                        scale=1 表示一级变换，scale=2 表示二级变换，以此类推
            dec (bool): 变换方向标志
                       True: 执行小波分解（前向变换），将图像分解为子带
                       False: 执行小波重构（逆变换），将子带重构为图像
            params_path (str): 预训练小波滤波器权重文件的路径
                              包含小波基函数的参数，用于初始化卷积层权重
            transpose (bool): 是否对输出进行维度转置操作
                             用于调整子带的排列顺序，便于后续处理
        """
        super(WaveletTransform, self).__init__()

        # 保存初始化参数
        self.scale = scale          # 小波变换尺度
        self.dec = dec             # 变换方向（分解/重构）
        self.transpose = transpose  # 是否进行维度转置

        # 计算卷积核大小：2^scale
        # 例如：scale=1时，ks=2；scale=2时，ks=4
        ks = int(math.pow(2, self.scale))
        
        # 计算输出通道数：3个颜色通道 × 小波子带数量
        # 对于2D小波变换，每个尺度会产生ks×ks个子带
        nc = 3 * ks * ks        # 根据变换方向创建相应的卷积层
        if dec:
            # 小波分解：使用标准卷积层
            # - 输入：3通道RGB图像
            # - 输出：nc个子带（低频+高频分量）
            # - 核大小=步长=ks，实现下采样
            # - groups=3：每个颜色通道独立处理
            # - bias=False：不使用偏置项
            self.conv = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=3,
                                  bias=False)
        else:
            # 小波重构：使用转置卷积层
            # - 输入：nc个子带
            # - 输出：3通道RGB图像
            # - 核大小=步长=ks，实现上采样
            # - groups=3：每个颜色通道独立处理
            # - bias=False：不使用偏置项
            self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0,
                                           groups=3, bias=False)        # 加载预训练的小波滤波器权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # 打开权重文件（二进制模式）
                f = open(params_path, 'rb')
                
                # 创建pickle反序列化器，设置编码为latin1
                # 这是为了兼容旧版本Python保存的pickle文件
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                
                # 加载权重字典
                dct = u.load()
                # 备用方法：dct = pickle.load(f)
                f.close()
                
                # 根据核大小从字典中选择对应的权重
                # 'rec%d' % ks：例如'rec2', 'rec4'等，对应不同尺度的小波基
                m.weight.data = torch.from_numpy(dct['rec%d' % ks])
                
                # 冻结权重，不参与梯度更新
                # 这保证了小波基函数的数学性质不被破坏
                m.weight.requires_grad = False    
    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x (torch.Tensor): 输入张量
                             对于分解模式：形状为[B, 3, H, W]的RGB图像
                             对于重构模式：形状为[B, nc, H', W']的小波系数
        
        Returns:
            torch.Tensor: 输出张量
                         对于分解模式：形状为[B, nc, H//ks, W//ks]的小波系数
                         对于重构模式：形状为[B, 3, H*ks, W*ks]的重构图像
        """
        if self.dec:
            # 小波分解模式
            # 通过卷积操作将图像分解为小波子带
            output = self.conv(x)
            
            if self.transpose:
                # 重新排列小波系数的维度
                # 目的：将不同频率的子带分组，便于后续处理
                osz = output.size()  # 获取输出张量的尺寸 [B, nc, H', W']
                
                # 维度变换过程：
                # 1. view(osz[0], 3, -1, osz[2], osz[3])：重塑为[B, 3, ks*ks, H', W']
                #    将nc个通道重新组织为3个颜色通道，每个包含ks*ks个子带
                # 2. transpose(1, 2)：交换维度1和2，得到[B, ks*ks, 3, H', W']
                #    将子带维度移到前面
                # 3. contiguous().view(osz)：重新整理内存布局并恢复原始形状
                output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1, 2).contiguous().view(osz)
        else:
            # 小波重构模式
            if self.transpose:
                # 逆转分解时的维度变换
                # 将重新排列的小波系数恢复为卷积层期望的格式
                xx = x
                xsz = xx.size()  # 获取输入张量的尺寸
                
                # 维度变换过程（与分解时相反）：
                # 1. view(xsz[0], -1, 3, xsz[2], xsz[3])：重塑为[B, ks*ks, 3, H', W']
                # 2. transpose(1, 2)：交换维度，得到[B, 3, ks*ks, H', W']
                # 3. contiguous().view(xsz)：恢复为[B, nc, H', W']格式
                xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1, 2).contiguous().view(xsz)
            
            # 通过转置卷积将小波系数重构为图像
            output = self.conv(xx)
        
        return output