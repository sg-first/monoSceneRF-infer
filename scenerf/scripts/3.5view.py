import pickle
import open3d as o3d
import numpy as np
from scenerf.data.utils import fusion

# 加载TSDF数据
with open("D:/workspace/SceneRF-main/results/reconstruction/sc_gt/depth_2.00_30.00.pkl", "rb") as f:
    data = pickle.load(f)
    print("Data keys:", data.keys())  # 打印数据中的所有键
    tsdf_grid = data["tsdf_grid"]

# 设置体素参数
voxel_size = 0.04
sx, sy, sz = 4.8, 4.8, 3.84
scene_size = (sx, sy, sz)
vox_origin = (-sx / 2, -sy / 2, 0)

# 创建TSDF体积
vol_bnds = np.zeros((3,2))
vol_bnds[:,0] = vox_origin
vol_bnds[:,1] = vox_origin + np.array([scene_size[0], scene_size[1], scene_size[2]])
tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size, trunc_margin=10, use_gpu=False)

# 设置TSDF网格
tsdf_vol._tsdf_vol_cpu = tsdf_grid

# 获取网格
verts, faces, norms, _ = tsdf_vol.get_mesh()

# 创建并显示网格
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)
# 设置默认颜色为灰色
mesh.paint_uniform_color([0.7, 0.7, 0.7])
mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh])
