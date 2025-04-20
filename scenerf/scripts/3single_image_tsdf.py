import torch
import numpy as np
import os
from scenerf.data.utils import fusion
import pickle

torch.set_grad_enabled(False)

def main():
    # 硬编码输入参数
    input_dir = "D:/workspace/SceneRF-main/results/reconstruction/tsdf"
    output_dir = "D:/workspace/SceneRF-main/results/reconstruction/sc_gt"
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist!")
        return
        
    # 列出所有文件
    print("\nFiles in directory:")
    for f in os.listdir(input_dir):
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

    # 2. 遍历所有TSDF文件
    tsdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    print(f"\nFound {len(tsdf_files)} TSDF files to process")
    
    for tsdf_file in tsdf_files:
        tsdf_path = os.path.join(input_dir, tsdf_file)
        print(f"\nProcessing {tsdf_file}")
        
        # 加载TSDF数据
        with open(tsdf_path, "rb") as f:
            data = pickle.load(f)
            tsdf_grid = data["tsdf_grid"]
        
        # 生成占用网格
        occ = np.zeros_like(tsdf_grid) + 255
        occ[(tsdf_grid > voxel_size) & (tsdf_grid != 255)] = 0  # 未知体素值为255
        occ[(abs(tsdf_grid) < voxel_size) & (tsdf_grid != 255)] = 1

        # 保存结果
        data = {
            "tsdf_grid": tsdf_grid,
            "occ": occ.astype(np.uint8),
        }
        
        os.makedirs(output_dir, exist_ok=True)
        save_filepath = os.path.join(output_dir, tsdf_file)
        print(f"Saving to {save_filepath}")
        with open(save_filepath, "wb") as handle:
            pickle.dump(data, handle)
            print(f"Saved to {save_filepath}")

if __name__ == "__main__":
    main() 