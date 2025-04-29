# MonoSceneRF: A Pipeline for NeRF Reconstruction from a Single Image

[Qianqian Xiong](https://www.researchgate.net/profile/Qianqian-Xiong-5)&nbsp;&nbsp;&nbsp;
[Tingzhen Liu](https://github.com/sg-first)&nbsp;&nbsp;&nbsp;
[Anh-Quan Cao](https://anhquancao.github.io)&nbsp;&nbsp;&nbsp;
[Raoul de Charette](https://team.inria.fr/rits/membres/raoul-de-charette/)  
Canberra, Australia / Paris, France

[![arXiv](https://img.shields.io/badge/arXiv%20%2B%20supp-2212.02501-purple)](https://arxiv.org/abs/2212.02501) 
[![Project page](https://img.shields.io/badge/Project%20Page-SceneRF-red)](https://astra-vision.github.io/SceneRF/)

# Teaser
<!-- ![](./teaser/method.png) -->
<img style="width:500px;max-width:100%" src="./teaser/teaser.png">
<table>
<tr>
    <td align="center"><b>Outdoor</b> scenes</td>
    <td align="center"><b>Indoor</b> scenes</td>
</tr>
<tr>
    <td style="width:50%!important">
        <img style="width:100%" src="./teaser/outdoor.gif">
        <!-- <img style="width:100%" src="./teaser/outdoor2.gif"> -->
        <!-- <img style="width:100%" src="./teaser/outdoor3.gif"> -->
    </td>
    <td style="width:50%!important">
        <img style="width:100%" src="./teaser/indoor.gif" />
        <!-- <img style="width:100%" src="./teaser/indoor2.gif" /> -->
    </td>
</tr>
</table>


# Run

## Using Conda
1. Create conda environment:
```
$ conda create -y -n scenerf python=3.7
$ conda activate scenerf
```
2. This code was implemented with python 3.7, pytorch 1.7.1 and CUDA 10.2. Please install [PyTorch](https://pytorch.org/): 

```
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```
3. Install the dependencies:

```
$ cd scenerf/
$ pip install -r requirements.txt
```
4. Install tbb
```
$ conda install -c bioconda tbb=2020.2
```
5. Downgrade torchmetrics

```
$ pip install torchmetrics==0.6.0
```
6. Finally, install scenerf:
```
$ pip install -e ./
```

## Download Pretrained model on Bundlefusion
Please download the [pretrained model](https://www.rocq.inria.fr/rits_files/computer-vision/scenerf/scenerf_bundlefusion.ckpt).

## Infere RGB Image and Depth Info

Inside the `./scenerf/scripts/` directory, two single-image inference scripts are provided to render scenes from different viewpoints:

- `mono_infer_angles.py`: Renders the scene from six different angles, including front view, ±5° horizontally, ±5° vertically, and a combined 5° right + 5° down;
- `mono_infer_distance.py`: Moves the camera forward and backward along the viewing direction by sampling 21 evenly spaced positions from `r = -1.0` to `1.0`.

To run the scripts, simply modify the following path parameters:

```python
model_path = "path/to/your/checkpoint.ckpt"
img_path = "path/to/your/input_image.jpg"
```

The rendered results will be saved to the directory specified by `save_dir`.


# Bundlefusion dataset

1. Please download 8 scenes from [Bundlefusion website](https://graphics.stanford.edu/projects/bundlefusion/) and unzip them to `/gpfsdswork/dataset/bundlefusion` (change to your dataset directory).
2. Store paths in environment variables for faster access:    
    ```
    $ export BF_ROOT=/gpfsdswork/dataset/bundlefusion
    ```

# Training Bundlefusion

1. Create folders to store training logs at **/gpfsscratch/rech/kvd/uyl37fq/logs/monoscene2/bundlefusion** (Change to your directory).

2. Store in an environment variable:

    ```
    $ export BF_LOG=/gpfsscratch/rech/kvd/uyl37fq/logs/monoscene2/bundlefusion
    ```

3. Train scenerf using 4 v100-32g GPUs with batch_size of 4 (1 item per GPU):

    ```
    $ cd scenerf/
    $ python scenerf/scripts/train_bundlefusion.py --bs=4 --n_gpus=4 \
        --n_rays=2048 --lr=2e-5 \
        --enable_log=True \
        --root=$BF_ROOT \
        --logdir=$BF_LOG
    ```

# Evaluate Bundlefusion

Create folders to store intermediate evaluation data at `/gpfsscratch/rech/kvd/uyl37fq/to_delete/eval` and reconstruction data at `/gpfsscratch/rech/kvd/uyl37fq/to_delete/recon`.

```
$ export EVAL_SAVE_DIR=/gpfsscratch/rech/kvd/uyl37fq/to_delete/eval
$ export RECON_SAVE_DIR=/gpfsscratch/rech/kvd/uyl37fq/to_delete/recon
```
    
## Novel depths synthesis on Bundlefusion
Supposed we obtain the model from the training step at `/gpfsscratch/rech/kvd/uyl37fq/to_delete/last.ckpt` (Change to your location). We follow the steps below to evaluate the novel depths synthesis performance. 
1. Compute the depth metrics on all frames in each sequence, additionally grouped by the distance to the input frame.

```
$ cd scenerf/
$ python scenerf/scripts/evaluation/save_depth_metrics_bf.py \
    --eval_save_dir=$EVAL_SAVE_DIR \
    --root=$BF_ROOT \
    --model_path=/gpfsscratch/rech/kvd/uyl37fq/to_delete/last.ckpt
```
2. Aggregate the depth metrics from all sequences.
```
$ cd scenerf/
$ python scenerf/scripts/evaluation/agg_depth_metrics_bf.py \
    --eval_save_dir=$EVAL_SAVE_DIR \
    --root=$BF_ROOT
```

## Novel views synthesis on Bundlefusion
Given the trained model at `/gpfsscratch/rech/kvd/uyl37fq/to_delete/last.ckpt`, the novel views synthesis performance is obtained as followed:
1. Render an RGB image for every frame in each sequence.
```
$ cd scenerf/
$ python scenerf/scripts/evaluation/render_colors_bf.py \
    --eval_save_dir=$EVAL_SAVE_DIR \
    --root=$BF_ROOT \
    --model_path=/gpfsscratch/rech/kvd/uyl37fq/to_delete/last.ckpt
```
2. Compute the metrics, additionally grouped by the distance to the input frame.
```
$ cd scenerf/
$ python scenerf/scripts/evaluation/eval_color_bf.py --eval_save_dir=$EVAL_SAVE_DIR
```


## Scene reconstruction on Bundlefusion
1. Generate novel views/depths for reconstructing scene.
```
$ cd scenerf/
$ python scenerf/scripts/reconstruction/generate_novel_depths_bf.py \
    --recon_save_dir=$RECON_SAVE_DIR \
    --root=$BF_ROOT \
    --model_path=/gpfsscratch/rech/kvd/uyl37fq/to_delete/last.ckpt \
    --angle=30 --step=0.2 --max_distance=2.1
```

2. Convert the novel views/depths to TSDF volume. **Note: the angle, step, and max_distance should match the previous step.**
```
$ cd scenerf/
$ python scenerf/scripts/reconstruction/depth2tsdf_bf.py \
    --recon_save_dir=$RECON_SAVE_DIR \
    --root=$BF_ROOT \
    --angle=30 --step=0.2 --max_distance=2.1
```
3. Generate the voxel ground-truth for evaluation.
```
$ cd scenerf/
$ python scenerf/scripts/reconstruction/generate_sc_gt_bf.py \
    --recon_save_dir=$RECON_SAVE_DIR \
    --root=$BF_ROOT
```

4. Compute scene reconstruction metrics using the generated TSDF volumes.
```
$ cd scenerf/
$ python scenerf/scripts/evaluation/eval_sc_bf.py \
    --recon_save_dir=$RECON_SAVE_DIR \
    --root=$BF_ROOT
```

## Optional: Infer 3d mesh from a Single Image

I also explored the complete 3d mesh pipeline from a single RGB image. Although the final rendering quality was limited and not included in the main report, the implementation process may still be of interest for reference or further development.

The following scripts, located in `./scenerf/scripts/`, form the step-by-step process:

- `1generate_novel_depths_bf_single.py`  
- `2depth2tsdf_bf_single.py`  
- `3single_image_tsdf.py`  
- `3.5view.py`

Each script contains inline parameter instructions. Modify the paths and values as needed before execution.


# Acknowledgment
This project was partly supported by the French project SIGHT (ANR-20-CE23-0016) and conducted in the SAMBA collaborative project, co-funded by BpiFrance in the Investissement d’Avenir Program, while also being funded by the ANU College of Engineering, Computing & Cybernetics under the course ENGN4528 Computer Vision. The work was performed using HPC resources from GENCI–IDRIS (Grant 2021-AD011012808, 2022-AD011012808R1, and 2023-AD011014102) and Tencent Cloud. We would like to thank Fabio Pizzati and Ivan Lopes for their kind proofreading, as well as all members of the Astra-vision group at Inria Paris for their insightful discussions.
