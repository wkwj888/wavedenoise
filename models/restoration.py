import torch
import torch.nn as nn
import utils
import torchvision
import os
import numpy as np
import time
import yaml
from skimage.metrics import structural_similarity as ssim
from PIL import Image


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


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
    
    for img, label in zip(images, labels):
        # 确保图像在[0,1]范围内
        img = torch.clamp(img, 0, 1)
        # 转换为PIL图像并保存
        img_pil = Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        img_pil.save(os.path.join(folder_path, f"{prefix}_{label}.png"))


def get_gpu_memory_usage():
    """获取GPU显存使用情况(GB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            # self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore_lap_dec(self,x_cond):
        lap_pyr = self.diffusion.lap.pyramid_decom(x_cond)
        x_cond = lap_pyr[-1]
        x_gt_lowf = x_cond[:, 3:, :, :]  # low frequency

        self.diffusion.lap_high_trans.eval()
        lap_pyr_inp = []
        for level in range(self.diffusion.lap.num_high+1):
            lap_pyr_inp.append(lap_pyr[level][:, :3, :, :])
        pyr_inp_trans = self.diffusion.lap_high_trans(lap_pyr_inp)

        return x_cond,x_gt_lowf,lap_pyr,pyr_inp_trans

    def restore_lap_rec(self,lap_pyr,x_output,x_gt_lowf,pyr_inp_trans):
        lap_pyr_output = []
        for level in range(self.diffusion.lap.num_high):
            lap_pyr_output.append(lap_pyr[level])
        lap_pyr_output.append(torch.cat([x_output.to(self.diffusion.device), x_gt_lowf], dim=1))

        # for check trans
        lap_pyr_check_trans = []
        for level in range(self.diffusion.lap.num_high-1):
            lap_pyr_check_trans.append(torch.cat([pyr_inp_trans[level], pyr_inp_trans[level]], dim=1))
        lap_pyr_check_trans.append(torch.cat([x_gt_lowf, x_output.to(self.diffusion.device)], dim=1))
        x_check = self.diffusion.lap.pyramid_recons(lap_pyr_check_trans)
        x_check1 = x_check[:, :3, :, :]
        x_check2 = x_check[:, 3:, :, :]

        x_output = self.diffusion.lap.pyramid_recons(lap_pyr_output)
        x_cond = self.diffusion.lap.pyramid_recons(lap_pyr)  # total frequency
        x_cond = x_cond[:, :3, :, :]
        x_output = x_output[:, :3, :, :]
        return x_output, x_cond, x_check1, x_check2

    def restore(self, val_loader, validation='snow', r=None, calculate_metrics=False, 
                metrics_folder=None, calculate_fid=False):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)
        
        # 原有的PSNR列表
        psnr_list_torch = []
        psnr_list_np = []
        psnr_list_GPU = []
        psnr_list_wdnet = []
        
        # 新增评估指标列表
        if calculate_metrics:
            ssim_list = []
            inference_times = []
            memory_usage = []
            
            # FID相关
            if calculate_fid:
                restored_images_for_fid = []
                gt_images_for_fid = []
                image_labels = []
                fid_restored_folder = os.path.join(metrics_folder, "fid_restored")
                fid_gt_folder = os.path.join(metrics_folder, "fid_gt")
        
        with torch.no_grad():
            for i, (x, y, total) in enumerate(val_loader):
                print(f"starting processing from image {y}")
                
                # 记录开始时间和显存
                if calculate_metrics:
                    start_time = time.time()
                    torch.cuda.empty_cache()  # 清空缓存以获得更准确的显存测量
                    start_memory = get_gpu_memory_usage()
                
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_all = x.to(self.diffusion.device)
                x_all = data_transform(x_all)
                if self.config.data.global_attn == True:
                    total = total.to(self.diffusion.device)
                    total = data_transform(total)
                if self.config.data.lap:
                    x_cond, x_gt_lowf, lap_pyr, pyr_inp_trans = self.restore_lap_dec(x_cond)

                if self.config.data.dataset == "DPD_Dual":
                    x_cond = x_all[:, :6, :, :]
                    x_gt = x_all[:, 6:, :, :]
                else:
                    x_cond = x_all[:, :3, :, :]
                    x_gt = x_all[:, 3:, :, :]
                if self.config.data.wavelet and not self.config.data.wavelet_in_unet:
                    x_cond = self.diffusion.wavelet_dec(x_cond)
                    x_gt = self.diffusion.wavelet_dec(x_gt)
                    if self.config.model.pred_channels < self.config.model.in_channels:
                        x_output_wdnet = self.diffusion.generator(x[:, :3, :, :].to(self.diffusion.device))
                        x_output_wdnet_norm = data_transform(x_output_wdnet) #中心化-1-1
                        x_output_wdnet_wav = self.diffusion.wavelet_dec(x_output_wdnet_norm)

                if self.config.data.wavelet and not self.config.data.wavelet_in_unet and self.config.model.use_other_channels:
                    if 0:
                        x_other = x_gt[:, self.config.model.pred_channels:, :, :]
                    else:
                        x_other = x_output_wdnet_wav[:,self.config.model.other_channels_begin:,:,:]
                else:
                    x_other = None

                x_output_list = self.diffusive_restoration(x_cond,x_other=x_other, r=r,last=False,total=total,use_global=self.config.data.global_attn,use_other=self.config.model.use_other_channels)

                x_output = x_output_list[1][-5]
                if self.config.data.lap:
                    x_output, x_cond, x_check1, x_check2 = self.restore_lap_rec(lap_pyr,x_output,x_gt_lowf,pyr_inp_trans)
                if self.config.data.wavelet and not self.config.data.wavelet_in_unet and self.config.model.pred_channels < self.config.model.in_channels:
                    x_output_hrgt_cat = torch.cat([x_output.to(self.diffusion.device)[:, :self.config.model.pred_channels],
                                          x_gt[:, self.config.model.pred_channels:]], dim=1)
                    x_output = torch.cat([x_output.to(self.diffusion.device)[:, :self.config.model.pred_channels],
                                          x_output_wdnet_wav[:, self.config.model.pred_channels:]], dim=1)
                    x_output_lrgt_cat = torch.cat(
                        [x_gt.to(self.diffusion.device)[:, :self.config.model.pred_channels],
                         x_output_wdnet_wav[:, self.config.model.pred_channels:]], dim=1)
                    x_output_lrgt_hrcond_cat = torch.cat(
                        [x_gt.to(self.diffusion.device)[:, :self.config.model.pred_channels],
                         x_cond[:, self.config.model.pred_channels:]], dim=1)

                if self.config.data.wavelet and not self.config.data.wavelet_in_unet:
                    x_output = self.diffusion.wavelet_rec(x_output.to(self.diffusion.device))
                    x_cond = self.diffusion.wavelet_rec(x_cond.to(self.diffusion.device))
                    if self.config.model.pred_channels < self.config.model.in_channels and self.config.model.use_other_channels:
                        x_output_hrgt_cat = self.diffusion.wavelet_rec(x_output_hrgt_cat.to(self.diffusion.device))
                        x_output_hrgt_cat = inverse_data_transform(x_output_hrgt_cat)
                        x_output_lrgt_cat = self.diffusion.wavelet_rec(x_output_lrgt_cat.to(self.diffusion.device))
                        x_output_lrgt_cat = inverse_data_transform(x_output_lrgt_cat)
                        x_output_lrgt_hrcond_cat = self.diffusion.wavelet_rec(x_output_lrgt_hrcond_cat.to(self.diffusion.device))
                        x_output_lrgt_hrcond_cat = inverse_data_transform(x_output_lrgt_hrcond_cat)

                x_output = inverse_data_transform(x_output)
                x_cond = inverse_data_transform(x_cond)
                if self.config.data.dataset == "DPD_Dual":
                    gt = x[:, 6:, :, :]
                    x_cond = x_cond[:, :3, :, :]
                else:
                    gt = x[:, 3:, :, :]
                
                # 记录推理时间和显存使用
                if calculate_metrics:
                    end_time = time.time()
                    inference_time = end_time - start_time
                    max_memory = get_gpu_memory_usage()
                    
                    inference_times.append(inference_time)
                    memory_usage.append(max_memory - start_memory)
                
                # 计算PSNR (原有逻辑)
                psnr_this1 = utils.torchPSNR(gt, x_output.cpu())
                psnr_cond1 = utils.torchPSNR(gt, x_cond.cpu())
                psnr_this_GPU = utils.calculate_psnr_in_GPU(gt.to(x_output.device), x_output,True)
                psnr_this_np = utils.calculate_psnr(torch.clamp(gt[0]*255,0,255).numpy().transpose((1,2,0)), torch.clamp(x_output[0]*255,0,255).cpu().numpy().transpose((1,2,0)),True)
                
                if self.config.data.wavelet and self.config.model.pred_channels < self.config.model.in_channels:
                    psnr_this_wdnet = utils.calculate_psnr(torch.clamp(gt[0] * 255, 0, 255).numpy().transpose((1, 2, 0)),torch.clamp(x_output_wdnet[0] * 255, 0, 255).cpu().numpy().transpose((1, 2, 0)), True)
                    psnr_list_wdnet.append(psnr_this_wdnet)
                
                psnr_list_torch.append(psnr_this1)
                psnr_list_np.append(psnr_this_np)
                psnr_list_GPU.append(psnr_this_GPU)
                
                # 计算SSIM
                if calculate_metrics:
                    ssim_value = calculate_ssim(gt[0], x_output[0])
                    ssim_list.append(ssim_value)
                    
                    # 收集FID数据
                    if calculate_fid:
                        restored_images_for_fid.append(x_output[0])
                        gt_images_for_fid.append(gt[0])
                        image_labels.append(y[0] if isinstance(y, list) else str(y))

                print("psnr this",psnr_this1)
                print("psnr cond", psnr_cond1)
                if calculate_metrics:
                    print(f"ssim: {ssim_value:.4f}")
                    print(f"inference time: {inference_time:.4f}s")
                    print(f"memory usage: {memory_usage[-1]:.2f}GB")
                
                if self.config.data.wavelet and not self.config.data.wavelet_in_unet and self.config.model.use_other_channels:
                    if self.config.model.pred_channels < self.config.model.in_channels:
                        utils.logging.save_image(x_output_lrgt_cat, os.path.join(image_folder, f"{y}_lrgt_hrwdnet.png"))
                        utils.logging.save_image(x_output_wdnet, os.path.join(image_folder, f"{y}_all_wdnet.png"))
                        utils.logging.save_image(x_output_lrgt_hrcond_cat,os.path.join(image_folder, f"{y}_lrgt_hrcond.png"))
                        utils.logging.save_image(x_output_hrgt_cat, os.path.join(image_folder, f"{y}_lrdiff_hrgt.png"))

                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y}_output.png"))
                utils.logging.save_image(x_cond, os.path.join(image_folder, f"{y}_cond.png"))
                utils.logging.save_image(gt, os.path.join(image_folder, f"{y}_gt.png"))
        
        # 打印原有结果
        print("psnr all torch",np.mean(psnr_list_torch))
        print("psnr all np", np.mean(psnr_list_np))
        print("psnr all GPU", np.mean(psnr_list_GPU))
        if self.config.data.wavelet and self.config.model.pred_channels < self.config.model.in_channels :
            print("psnr all wdnet", np.mean(psnr_list_wdnet))
        
        # 计算并保存新的评估指标
        if calculate_metrics:
            # 计算FID
            fid_score = None
            if calculate_fid and len(restored_images_for_fid) > 0:
                print("正在保存图像用于FID计算...")
                save_images_for_fid(restored_images_for_fid, image_labels, fid_restored_folder, "restored")
                save_images_for_fid(gt_images_for_fid, image_labels, fid_gt_folder, "gt")
                
                print("正在计算FID指标...")
                try:
                    from pytorch_fid import fid_score as fid_calc
                    fid_score = fid_calc.calculate_fid_given_paths(
                        [fid_gt_folder, fid_restored_folder],
                        batch_size=32,
                        device=self.diffusion.device,
                        dims=2048
                    )
                    print(f"FID Score: {fid_score:.4f}")
                except Exception as e:
                    print(f"FID计算失败: {e}")
                    fid_score = None
            
            # 汇总所有指标
            metrics_summary = {
                'PSNR': {
                    'torch_mean': float(np.mean(psnr_list_torch)),
                    'torch_std': float(np.std(psnr_list_torch)),
                    'np_mean': float(np.mean(psnr_list_np)),
                    'gpu_mean': float(np.mean(psnr_list_GPU)),
                    'all_values': [float(x) for x in psnr_list_torch]
                },
                'SSIM': {
                    'mean': float(np.mean(ssim_list)),
                    'std': float(np.std(ssim_list)),
                    'min': float(np.min(ssim_list)),
                    'max': float(np.max(ssim_list)),
                    'all_values': [float(x) for x in ssim_list]
                },
                'Inference_Time': {
                    'mean': float(np.mean(inference_times)),
                    'std': float(np.std(inference_times)),
                    'total': float(np.sum(inference_times)),
                    'all_values': [float(x) for x in inference_times]
                },
                'Memory_Usage_GB': {
                    'mean': float(np.mean(memory_usage)),
                    'std': float(np.std(memory_usage)),
                    'max': float(np.max(memory_usage)),
                    'all_values': [float(x) for x in memory_usage]
                }
            }
            
            if fid_score is not None:
                metrics_summary['FID'] = float(fid_score)
            
            if self.config.data.wavelet and self.config.model.pred_channels < self.config.model.in_channels:
                metrics_summary['PSNR_WDNet'] = {
                    'mean': float(np.mean(psnr_list_wdnet)),
                    'all_values': [float(x) for x in psnr_list_wdnet]
                }
            
            # 保存详细结果
            results_file = os.path.join(metrics_folder, f"evaluation_results_{validation}.yaml")
            with open(results_file, 'w') as f:
                yaml.dump(metrics_summary, f, default_flow_style=False)
            
            print(f"\n=== 评估结果摘要 ===")
            print(f"PSNR: {metrics_summary['PSNR']['torch_mean']:.4f} ± {metrics_summary['PSNR']['torch_std']:.4f}")
            print(f"SSIM: {metrics_summary['SSIM']['mean']:.4f} ± {metrics_summary['SSIM']['std']:.4f}")
            print(f"平均推理时间: {metrics_summary['Inference_Time']['mean']:.4f}s")
            print(f"总推理时间: {metrics_summary['Inference_Time']['total']:.2f}s")
            print(f"平均显存占用: {metrics_summary['Memory_Usage_GB']['mean']:.2f}GB")
            print(f"最大显存占用: {metrics_summary['Memory_Usage_GB']['max']:.2f}GB")
            if fid_score is not None:
                print(f"FID: {fid_score:.4f}")
            
            print(f"\n详细结果已保存到: {results_file}")

    def diffusive_restoration(self, x_cond, x_other=None, r=None,last =True, total=None,use_global=False,use_other=False):
        if self.config.data.wavelet_in_unet:
            p_size = self.config.data.patch_size
        else:
            p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn((x_cond.shape[0],self.config.model.pred_channels,x_cond.shape[2],x_cond.shape[3]), device=self.diffusion.device)
        if self.config.data.wavelet:
            x_output = self.diffusion.sample_image(x_cond, x, x_other=x_other,last=last,patch_locs=corners, patch_size=p_size,
                                               total=total,use_global=use_global,use_other=use_other)
        else:
            x_output = self.diffusion.sample_image(x_cond, x, last=last, patch_locs=corners,
                                                   patch_size=p_size,
                                                   total=total, use_global=use_global)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        if h_list[-1]+output_size<h:
            h_list.append(h-output_size)
        if w_list[-1]+output_size<w:
            w_list.append(w-output_size)
        return h_list, w_list