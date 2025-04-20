import torch
import numpy as np
import os
from scenerf.data.utils import fusion
from scenerf.models.utils import sample_rel_poses_bf
from PIL import Image
import pickle
from scenerf.loss.depth_metrics import compute_depth_errors

torch.set_grad_enabled(False)

def tsdf2occ(tsdf, th=0.25, max_th=0.2, voxel_size=0.04):
    occ = np.zeros(tsdf.shape)
    th_indivi = (voxel_size/2 + np.arange(96).reshape(1, 1, 96) * voxel_size) * th
    th_indivi[th_indivi < voxel_size] = voxel_size
    th_indivi[th_indivi > max_th] = max_th
    occ[(np.abs(tsdf) < th_indivi) & (np.abs(tsdf) != 255)] = 1
    return occ

def read_rgb(path):
    """读取RGB图像并归一化"""
    img = Image.open(path).convert("RGB")
    img = np.array(img, dtype=np.float32, copy=False) / 255.0
    return img

def evaluate_depth(gt_depth, pred_depth):
    depth_errors = []
    depth_error = compute_depth_errors(
        gt=gt_depth.reshape(-1).detach().cpu().numpy(),
        pred=pred_depth.reshape(-1).detach().cpu().numpy(),
    )
    depth_errors.append(depth_error)
    agg_depth_errors = np.array(depth_errors).sum(0)
    return agg_depth_errors

def main():
    # 硬编码输入参数
    base_dir = "D:/workspace/SceneRF-main/results/reconstruction"
    rgb_path = "C:/Users/Bruce Wayne/Downloads/building.jpg"
    save_dir = os.path.join(base_dir, "tsdf")
    
    print(f"Base directory: {base_dir}")
    print(f"RGB path: {rgb_path}")
    print(f"Save directory: {save_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} does not exist!")
        return
        
    # 列出所有文件
    print("\nFiles in directory:")
    for f in os.listdir(base_dir):
        print(f)
    
    # 相机参数
    cam_K = np.array([[525.0, 0, 320], [0, 525.0, 240], [0, 0, 1]])
    
    # 1. 设置TSDF体积参数
    voxel_size = 0.04
    sx, sy, sz = 4.8, 4.8, 3.84
    scene_size = (sx, sy, sz)
    vox_origin = (-sx / 2, -sy / 2, 0)
    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = vox_origin
    vol_bnds[:,1] = vox_origin + np.array([scene_size[0], scene_size[1], scene_size[2]])

    # 2. 遍历所有深度图
    depth_files = [f for f in os.listdir(base_dir) if f.startswith('depth_') and f.endswith('.npy')]
    print(f"\nFound {len(depth_files)} depth files to process")
    
    for depth_file in depth_files:
        depth_path = os.path.join(base_dir, depth_file)
        print(f"\nProcessing {depth_file}")
        
        # 为每个深度图创建一个新的TSDF体积
        tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size, trunc_margin=10)
        
        # 加载深度图和RGB图
        print("Loading depth and RGB images...")
        depth = np.load(depth_path)
        rgb = read_rgb(rgb_path) * 255
        
        print("Integrating into TSDF volume...")
        # 3. 集成到TSDF体积中
        tsdf_vol.integrate(rgb, depth, cam_K, np.eye(4), obs_weight=1.)
   
        print("Getting mesh and volume data...")
        # 4. 获取网格和体积数据
        verts, faces, norms, colors = tsdf_vol.get_mesh()
        tsdf_grid, _ = tsdf_vol.get_volume() 

        # 5. 保存结果
        data = {
            "tsdf_grid": tsdf_grid,
            "verts": verts,
            "faces": faces,
            "norms": norms,
            "colors": colors,
        }
        
        os.makedirs(save_dir, exist_ok=True)
        # 使用深度图文件名作为TSDF文件名
        save_filename = depth_file.replace('.npy', '.pkl')
        save_filepath = os.path.join(save_dir, save_filename)
        print(f"Saving to {save_filepath}")
        with open(save_filepath, "wb") as handle:
            pickle.dump(data, handle)
            print(f"Saved TSDF to {save_filepath}")

if __name__ == "__main__":
    main() 