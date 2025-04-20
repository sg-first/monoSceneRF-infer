import torch
import numpy as np
import os
from scenerf.models.scenerf_bf import SceneRF
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from scenerf.models.utils import sample_rel_poses_bf

torch.set_grad_enabled(False)

def main():
    """对单张图片生成新视角"""
    # 设置参数
    model_path = "C:/Users/Bruce Wayne/Downloads/scenerf_kitti.ckpt"  # 替换为你的模型路径
    img_path = "C:/Users/Bruce Wayne/Downloads/building.jpg"  # 替换为你的图片路径
    save_dir = "D:/workspace/SceneRF-main/results/reconstruction"  # 替换为你想保存结果的目录
    
    # 新视角参数
    angle = 30  # 旋转角度
    step = 0.2  # 步长
    max_distance = 2.1  # 最大距离
    
    # 1. 加载模型
    model = SceneRF.load_from_checkpoint(model_path)
    model.cuda()
    model.eval()
    
    # 2. 加载图片和相机参数
    image = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((480, 640)),  # 调整到模型需要的尺寸
        T.ToTensor(),
    ])
    img_inputs = transform(image).unsqueeze(0).cuda()  # shape: [1, 3, H, W]
    
    cam_K = torch.tensor([[525.0, 0, 320], [0, 525.0, 240], [0, 0, 1]]).cuda()
    inv_K = torch.inverse(cam_K)

    # 3. 获取球面映射
    pix_coords, out_pix_coords, _ = model.spherical_mapping.from_pixels(inv_K=inv_K)
    x_rgbs = model.net_rgb(img_inputs, pix=pix_coords, pix_sphere=out_pix_coords)

    # 4. 处理特征
    x_rgb = {}
    for k in x_rgbs:
        x_rgb[k] = x_rgbs[k][0]  # 取第一个batch

    # 5. 生成像素坐标
    img_size = (640, 480)
    scale = 2
    xs = torch.arange(start=0, end=img_size[0], step=scale).type_as(cam_K)
    ys = torch.arange(start=0, end=img_size[1], step=scale).type_as(cam_K)
    grid_x, grid_y = torch.meshgrid(xs, ys)
    rendered_im_size = grid_x.shape

    sampled_pixels = torch.cat([
        grid_x.unsqueeze(-1),
        grid_y.unsqueeze(-1)
    ], dim=2).reshape(-1, 2)

    # 6. 生成新视角位姿
    rel_poses = sample_rel_poses_bf(angle, max_distance, step)

    # 7. 对每个新视角进行渲染
    with torch.no_grad():
        for (step, angle), rel_pose in rel_poses.items():
            T_source2infer = rel_pose
            
            # 渲染新视角
            render_out_dict = model.render_rays_batch(
                cam_K,
                T_source2infer.type_as(cam_K),
                x_rgb,
                ray_batch_size=2000,
                sampled_pixels=sampled_pixels
            )
            
            # 获取渲染结果
            depth_rendered = render_out_dict['depth'].reshape(rendered_im_size[0], rendered_im_size[1])
            color_rendered = render_out_dict['color'].reshape(rendered_im_size[0], rendered_im_size[1], 3)

            # 上采样到原始分辨率
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

            # 保存结果
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存深度图
                depth_np = depth_rendered.squeeze().detach().cpu().numpy()
                np.save(os.path.join(save_dir, f"depth_{step:.2f}_{angle:.2f}.npy"), depth_np)
                
                # 保存RGB图
                color_np = color_rendered.clamp(0, 1).detach().cpu().numpy().squeeze()
                color_np = np.transpose(color_np, (1, 2, 0))
                np.save(os.path.join(save_dir, f"color_{step:.2f}_{angle:.2f}.npy"), color_np)
                
                print(f"Saved results for step={step:.2f}, angle={angle:.2f}")
    
    # 清理GPU内存
    del model, img_inputs, depth_rendered, color_rendered, pix_coords, out_pix_coords, x_rgb
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 