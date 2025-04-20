# Sherlock, this is the LOW-MEMORY SceneRF Inference Script (Optimized for 4GB GPUs)

import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from scenerf.models.scenerf import SceneRF as scenerf
from scenerf.models.utils import compute_direction_from_pixels

def load_image(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((24, 80)),  # further reduced res
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0).cuda()  # [1, 3, H, W], keep in float32

def run_inference(img_path, model_path):
    torch.cuda.empty_cache()
    model = scenerf.load_from_checkpoint(model_path).cuda().eval()  # float32

    image = load_image(img_path)  # already on cuda, float32

    # Inverse in float32
    cam_K = torch.tensor([[125.0, 0, 40], [0, 125.0, 12], [0, 0, 1]], device="cuda", dtype=torch.float32)
    inv_K = torch.inverse(cam_K)

    # Generate sampled_pixels [N, 3]
    H, W = 24, 80
    ys = torch.arange(0, H, dtype=torch.float32, device="cuda")
    xs = torch.arange(0, W, dtype=torch.float32, device="cuda")
    grid_y, grid_x = torch.meshgrid(ys, xs)
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # [N, 2]
    ones = torch.ones((coords.shape[0], 1), device="cuda", dtype=torch.float32)
    sampled_pixels = torch.cat([coords, ones], dim=-1).contiguous()  # [N, 3]

    with torch.no_grad():
        pix_coords, out_pix_coords, _ = model.spherical_mapping.from_pixels(inv_K=inv_K)
        x_rgb = model.net_rgb(image, pix=pix_coords, pix_sphere=out_pix_coords)

        ray_batch_size = 50  # super small batch
        depth_predictions = []
        for i in range(0, sampled_pixels.shape[0], ray_batch_size):
            chunk_pixels = sampled_pixels[i:i+ray_batch_size].contiguous().T  # [3, N]
            dirs = (inv_K[:3, :3] @ chunk_pixels).T  # manual replacement of compute_direction_from_pixels

            chunk_pixels = chunk_pixels.T.contiguous()  # [N, 3]
            chunk_x_rgb = {k: x_rgb[k] for k in x_rgb}
            out = model.render_rays_batch(
                cam_K,
                torch.eye(4, device="cuda", dtype=torch.float32),
                chunk_x_rgb,
                ray_batch_size=ray_batch_size,
                sampled_pixels=chunk_pixels,
            )
            depth_predictions.append(out['depth'])  # no .half()

    depth = torch.cat(depth_predictions, dim=0).reshape(H, W)
    depth_np = depth.detach().cpu().numpy()

    np.save("depth_output.npy", depth_np)
    plt.imsave("depth_output.png", depth_np, cmap='plasma')
    print("[✔] Saved depth_output.png and .npy")
    plt.imshow(depth_np, cmap='plasma')
    plt.colorbar()
    plt.title("Predicted Depth")
    plt.show()

if __name__ == "__main__":
    run_inference(
        "C:/Users/Bruce Wayne/Downloads/building.jpg",
        "C:/Users/Bruce Wayne/Downloads/scenerf_kitti.ckpt"
    )
