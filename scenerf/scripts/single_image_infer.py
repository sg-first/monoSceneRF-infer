import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scenerf.models.scenerf_bf import SceneRF
from scenerf.models.utils import depth2disp
import gc
import math
import inspect
import matplotlib as mpl
import matplotlib.cm as cm

torch.cuda.set_per_process_memory_fraction(0.7)
torch.cuda.empty_cache()

def create_orbit_transform(theta, phi, radius):
    """
    以正面视角为基准的小角度旋转
    """
    # 正面视角
    if abs(theta) < 1e-6 and abs(phi - math.pi/2) < 1e-6:
        return torch.eye(4, dtype=torch.float32).cuda()
    
    # 构建旋转矩阵
    transform = torch.eye(4, dtype=torch.float32).cuda()
    
    # 先绕y轴旋转theta角度（水平旋转）
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    transform[0, 0] = cos_theta
    transform[0, 2] = sin_theta
    transform[2, 0] = -sin_theta
    transform[2, 2] = cos_theta
    
    # 再绕x轴旋转(phi-pi/2)角度（垂直旋转）
    phi_offset = phi - math.pi/2
    cos_phi = math.cos(phi_offset)
    sin_phi = math.sin(phi_offset)
    temp = transform.clone()
    transform[1, 1] = cos_phi
    transform[1, 2] = -sin_phi
    transform[2, 1] = sin_phi
    transform[2, 2] = cos_phi * temp[2, 2]
    transform[2, 0] = cos_phi * temp[2, 0]
    
    # 设置相机位置（在z=radius平面上）
    transform[0:3, 3] = transform[0:3, 2] * radius
    
    print(f"\n角度: theta={math.degrees(theta):.1f}度, phi={math.degrees(phi):.1f}度")
    print("Transform:")
    print(transform)
    
    return transform

def clear_gpu_memory():
    """手动清理GPU显存"""
    torch.cuda.empty_cache()
    gc.collect()

def disparity_normalization_vis(disparity):
    """
    :param disparity: Bx1xHxW, pytorch tensor of float32
    :return:
    """
    assert len(disparity.size()) == 4 and disparity.size(1) == 1
    disp_min = torch.amin(disparity, (1, 2, 3), keepdim=True)
    disp_max = torch.amax(disparity, (1, 2, 3), keepdim=True)
    disparity_syn_scaled = (disparity - disp_min) / (disp_max - disp_min)
    disparity_syn_scaled = torch.clip(disparity_syn_scaled, 0.0, 1.0)
    return disparity_syn_scaled

def main():
    # 设置参数
    model_path = "/home/ubuntu/scenerf_bundlefusion.ckpt"  # 模型路径
    img_path = "/home/ubuntu/monoSceneRF-infer/scenerf/scripts/building.jpg"     # 输入图像路径
    save_dir = "/home/ubuntu/monoSceneRF-infer/results/reconstruction/colors"          # 保存结果路径
    
    # 清理显存
    clear_gpu_memory()
    
    # 加载模型
    model = SceneRF.load_from_checkpoint(model_path)
    model.cuda()
    model.eval()
    
    # 清理显存
    clear_gpu_memory()
    
    # 加载图像并确保数据类型为float32
    img = Image.open(img_path)
    original_width, original_height = img.size
    # 调整图像大小
    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    
    # 清理显存
    clear_gpu_memory()
    
    # 设置相机参数
    cam_K = torch.tensor([[262.5, 0, 160], [0, 262.5, 120], [0, 0, 1]], dtype=torch.float32).cuda()  # 调整相机内参
    inv_K = torch.inverse(cam_K)
    
    # 清理显存
    clear_gpu_memory()
    
    # 获取像素坐标
    pix_coords, out_pix_coords, _ = model.spherical_mapping.from_pixels(inv_K=inv_K)
    
    # 清理显存
    clear_gpu_memory()
    
    # 获取RGB特征
    x_rgbs = model.net_rgb(img, pix=pix_coords, pix_sphere=out_pix_coords)
    x_rgb = {}
    for k in x_rgbs:
        x_rgb[k] = x_rgbs[k][0]  # 取第一个batch的结果
    
    # 清理显存
    clear_gpu_memory()
    
    # 设置渲染参数
    img_size = (original_width, original_height)  # 调整图像大小
    scale = 2
    xs = torch.arange(start=0, end=img_size[0], step=scale, dtype=torch.float32).type_as(cam_K)
    ys = torch.arange(start=0, end=img_size[1], step=scale, dtype=torch.float32).type_as(cam_K)
    grid_x, grid_y = torch.meshgrid(xs, ys)
    rendered_im_size = grid_x.shape
    
    sampled_pixels = torch.cat([
        grid_x.unsqueeze(-1),
        grid_y.unsqueeze(-1)
    ], dim=2).reshape(-1, 2)
    
    # 清理显存
    clear_gpu_memory()
    
    # 渲染
    with torch.no_grad():
        # 尝试不同的视角
        # test_angles = [
        #     (0, math.pi/2),          # 正面平视
        #     (0, math.pi/3),          # 正面略微俯视 (30度)
        #     (math.pi/6, math.pi/2),  # 右前方30度平视
        #     (-math.pi/6, math.pi/2), # 左前方30度平视
        #     (math.pi/4, math.pi/3),  # 右前方45度俯视
        #     (-math.pi/4, math.pi/3), # 左前方45度俯视
        # ]
        test_angles = [
            (0, math.pi/2),                          # 正面平视（基准）
            (math.radians(5), math.pi/2),            # 向右转5度
            (-math.radians(5), math.pi/2),           # 向左转5度
            (0, math.pi/2 - math.radians(5)),        # 向下看5度
            (0, math.pi/2 + math.radians(5)),        # 向上看5度
            (math.radians(5), math.pi/2 - math.radians(5))  # 右转5度同时向下看5度
        ]

        # 为测试创建保存目录
        test_dir = os.path.join(save_dir, "angle_test")
        os.makedirs(test_dir, exist_ok=True)

        for idx, (theta, phi) in enumerate(test_angles):
            print(f"\n测试视角 {idx+1}: theta={theta:.2f}, phi={phi:.2f}")
            transform = create_orbit_transform(theta, phi, radius=1.0)  # 保持radius=1.0不变

            render_out_dict = model.render_rays_batch(
                cam_K,
                transform,
                x_rgb,
                ray_batch_size=200,
                sampled_pixels=sampled_pixels
            )
            print(inspect.signature(model.render_rays_batch))
        
            # 清理显存
            clear_gpu_memory()
        
            # 获取深度和颜色
            depth_rendered = render_out_dict['depth'].reshape(rendered_im_size[0], rendered_im_size[1])
            color_rendered = render_out_dict['color'].reshape(rendered_im_size[0], rendered_im_size[1], 3)

            print(f"深度范围: {depth_rendered.min().item():.2f} - {depth_rendered.max().item():.2f}")
            # 清理显存
            clear_gpu_memory()
        
            # 上采样
            depth_rendered = F.interpolate(
                depth_rendered.T.unsqueeze(0).unsqueeze(0),
                scale_factor=scale,
                mode="bilinear"
            )
            color_rendered = F.interpolate(
                color_rendered.permute(2, 1, 0).unsqueeze(0),
                scale_factor=scale,
                mode="bilinear"
            )
        
            # 清理显存
            clear_gpu_memory()
            
            # 转换为numpy
            color_rendered_np = color_rendered.clamp(0, 1).detach().cpu().numpy().squeeze()
            color_rendered_np = np.transpose(color_rendered_np, (1, 2, 0))
        
            # 清理显存
            clear_gpu_memory()
        
            # 保存结果
            os.makedirs(test_dir, exist_ok=True)
            plt.imsave(os.path.join(test_dir, f"angle_{idx:02d}_rgb.png"), color_rendered_np)
        
            # 保存深度图
            disp = depth2disp(depth_rendered, min_depth=0.1, max_depth=12.0).squeeze()
            disp_np = disp.detach().cpu().numpy()
            img = Image.fromarray((disp_np * 255.0).astype(np.uint8))
            img.save(os.path.join(test_dir, f"angle_{idx:02d}_depth.png"))

            # 保存深度可视化图
            vmax = disp_np.max()
            vmin = disp_np.min()
            normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
            im = Image.fromarray(colormapped_im)
            im.save(os.path.join(test_dir, f"angle_{idx:02d}_depth_visual.png"))
        
            print("Results saved to", save_dir)

if __name__ == "__main__":
    main() 
    # 在渲染之前添加这段代码来对比两个transform
    # original_transform = torch.eye(4, dtype=torch.float32).cuda()
    
    # our_transform = create_orbit_transform(theta=0, phi=math.pi/2, radius=1.0)

    # print("Original transform:")
    # print(original_transform)
    # print("\nOur transform:")
    # print(our_transform)
    # test_angles = [
    #     (0, math.pi/2),                          # 正面平视
    #     (math.radians(10), math.pi/2),           # 水平偏移10度
    #     (0, math.pi/2 - math.radians(10))        # 俯视10度
    # ]
    # for theta, phi in test_angles:
    #     transform = create_orbit_transform(theta=theta, phi=phi, radius=1.0)
    #     print(f"\n角度: theta={theta:.3f} ({theta*180/math.pi:.1f}度), "
    #           f"phi={phi:.3f} ({phi*180/math.pi:.1f}度)")
    #     print(transform)