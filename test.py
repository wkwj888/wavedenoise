import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion,DenoisingDiffusion_Wavelet, DiffusiveRestoration
# from models import DenoisingDiffusion_Dual
import torch.distributed as dist
import time
import psutil
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score
from PIL import Image
import cv2


def parse_args_and_config():
    """解析命令行参数和配置文件"""
    parser = argparse.ArgumentParser(description='使用基于块的去噪扩散模型进行天气复原')
    parser.add_argument("--config", type=str, required=True,
                        help="配置文件路径")
    parser.add_argument('--resume', default='', type=str,
                        help='用于评估的扩散模型检查点路径')
    parser.add_argument("--grid_r", type=int, default=16,
                        help="网格单元宽度r，定义块之间的重叠")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="隐式采样步数")
    parser.add_argument("--test_set", type=str, default='raindrop',
                        help="复原测试集选项: ['raindrop', 'snow', 'rainfog', 'dpd', 'buildings']")
    parser.add_argument("--image_folder", default='results/images', type=str,
                        help="保存复原图像的位置")
    parser.add_argument("--metrics_folder", default='results/metrics', type=str,
                        help="保存评估指标的位置")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='初始化训练的种子 (默认: 61)')
    parser.add_argument('--calculate_fid', action='store_true',
                        help='是否计算FID指标（需要大量计算资源）')
    
    # 分布式训练参数通常由分布式启动器设置（如torchrun）
    # 最好从环境变量获取，或提供合理的默认值
    parser.add_argument('--init_method', type=str, default='env://',
                        help='Torch分布式初始化方法')

    args = parser.parse_args()

    # 如果可用，从环境变量更新rank、world_size、local_rank（由torchrun使用）
    if 'RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
    else:
        args.rank = 0 # 非分布式或未设置时的默认值

    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
    else:
        args.world_size = 1 # 非分布式或未设置时的默认值

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 如果WORLD_SIZE为1，local_rank应该为0
        # 否则可能表示分布式训练存在问题但LOCAL_RANK未设置
        args.local_rank = 0

    # 读取配置文件
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    """将字典转换为命名空间对象"""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def calculate_ssim(img1, img2):
    """计算SSIM指标"""
    # 转换为numpy格式 (H, W, C)
    img1_np = img1.cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.cpu().numpy().transpose(1, 2, 0)
    
    # 如果是RGB图像
    if img1_np.shape[2] == 3:
        return ssim(img1_np, img2_np, multichannel=True, channel_axis=2, data_range=1.0)
    else:
        return ssim(img1_np.squeeze(), img2_np.squeeze(), data_range=1.0)


def save_images_for_fid(images, labels, folder_path, prefix="img"):
    """保存图像用于FID计算"""
    os.makedirs(folder_path, exist_ok=True)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        # 确保图像在[0,1]范围内
        img = torch.clamp(img, 0, 1)
        # 转换为PIL图像并保存
        img_pil = Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        img_pil.save(os.path.join(folder_path, f"{prefix}_{label}.png"))


def calculate_fid_score(restored_folder, gt_folder, device):
    """计算FID指标"""
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [gt_folder, restored_folder],
            batch_size=32,
            device=device,
            dims=2048
        )
        return fid_value
    except Exception as e:
        print(f"FID计算失败: {e}")
        return None


def get_gpu_memory_usage():
    """获取GPU显存使用情况(GB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def main():
    """主函数"""
    args, config = parse_args_and_config()
    
    # 检测是否是通过分布式启动器启动
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    if is_distributed and args.world_size > 1:
        # 分布式设置
        print("检测到分布式环境，初始化分布式进程组...")
        
        # 如果使用env://且未由启动器设置，确保MASTER_ADDR和MASTER_PORT已设置
        if args.init_method == 'env://' and 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if args.init_method == 'env://' and 'MASTER_PORT' not in os.environ:
             # 如果未设置，查找空闲端口或使用默认端口
            try:
                sock = socket.socket()
                sock.bind(('', 0))
                os.environ['MASTER_PORT'] = str(sock.getsockname()[1])
                sock.close()
            except socket.error:
                 os.environ['MASTER_PORT'] = '5678' # 默认备用端口
        
        # 初始化分布式进程组
        dist.init_process_group(backend='nccl', init_method=args.init_method,
                                world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.local_rank)

        device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        # 单GPU或CPU设置
        if torch.cuda.is_available():
            # 强制使用CUDA_VISIBLE_DEVICES指定的GPU
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)
            print(f"单GPU模式，使用设备: {device}")
        else:
            device = torch.device("cpu")
            print("CUDA不可用，使用CPU")
    
    print(f"最终使用设备: {device}")
    config.device = device

    # 打印运行模式信息
    if torch.cuda.is_available() and args.world_size == 1:
        print('注意：在单GPU上运行评估。')
    elif torch.cuda.is_available() and args.world_size > 1:
        print('注意：使用分布式设置运行评估。')
    else:
        print('注意：在CPU上运行评估。')

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # 创建metrics文件夹
    os.makedirs(args.metrics_folder, exist_ok=True)

    # 数据加载
    print("=> 使用数据集 '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](args, config)
    _, val_loader = DATASET.get_loaders(parse_patches=False, validation=args.test_set)

    # 创建模型
    print("=> 创建带包装器的去噪扩散模型...")
    if config.data.wavelet:
        # 使用小波变换版本的扩散模型
        diffusion = DenoisingDiffusion_Wavelet(args, config)
    elif config.data.dataset=="DPD_Dual":
        # 使用双重扩散模型（代码中被注释）
        diffusion = DenoisingDiffusion_Dual(args, config)
    else:
        # 使用标准扩散模型
        diffusion = DenoisingDiffusion(args, config)
    
    # 创建扩散复原模型包装器
    model = DiffusiveRestoration(diffusion, args, config)
    
    # 模型加载逻辑
    if args.resume and os.path.isfile(args.resume):
        print(f"从 {args.resume} 加载检查点")
        model.diffusion.load_ddm_ckpt(args.resume, ema=True)
        model.diffusion.model.eval()
    else:
        print('Pre-trained diffusion model path is missing!')

    # 执行图像复原并计算评估指标
    print("\n开始评估...")
    start_time = time.time()
    
    # 传递评估参数给restore方法
    model.restore(val_loader, validation=args.test_set, r=args.grid_r, 
                 calculate_metrics=True, metrics_folder=args.metrics_folder,
                 calculate_fid=args.calculate_fid)
    
    total_time = time.time() - start_time
    print(f"\n评估完成，总耗时: {total_time:.2f}秒")


if __name__ == '__main__':
    main()