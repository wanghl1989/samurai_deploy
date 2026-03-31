<div align="center">
<img align="left" width="100" height="100" src="https://github.com/user-attachments/assets/1834fc25-42ef-4237-9feb-53a01c137e83" alt="">

# SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory

[Cheng-Yen Yang](https://yangchris11.github.io), [Hsiang-Wei Huang](https://hsiangwei0903.github.io/), [Wenhao Chai](https://rese1f.github.io/), [Zhongyu Jiang](https://zhyjiang.github.io/#/), [Jenq-Neng Hwang](https://people.ece.uw.edu/hwang/)

[Information Processing Lab, University of Washington](https://ipl-uw.github.io/) 
</div>


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-lasot-ext)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot-ext?p=samurai-adapting-segment-anything-model-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-got-10k)](https://paperswithcode.com/sota/visual-object-tracking-on-got-10k?p=samurai-adapting-segment-anything-model-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-needforspeed)](https://paperswithcode.com/sota/visual-object-tracking-on-needforspeed?p=samurai-adapting-segment-anything-model-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-lasot)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot?p=samurai-adapting-segment-anything-model-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-otb-2015)](https://paperswithcode.com/sota/visual-object-tracking-on-otb-2015?p=samurai-adapting-segment-anything-model-for-1)

[[Arxiv]](https://arxiv.org/abs/2411.11922) [[Project Page]](https://yangchris11.github.io/samurai/) [[Raw Results]](https://drive.google.com/drive/folders/1ssiDmsC7mw5AiItYQG4poiR1JgRq305y?usp=sharing) 

This repository is the official implementation of SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory

https://github.com/user-attachments/assets/9d368ca7-2e9b-4fed-9da0-d2efbf620d88

All rights are reserved to the copyright owners (TM & © Universal (2019)). This clip is not intended for commercial use and is solely for academic demonstration in a research paper. Original source can be found [here](https://www.youtube.com/watch?v=cwUzUzpG8aM&t=4s).

## Getting Started

#### SAMURAI Installation 

SAM 2 needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`. Please follow the instructions [here](https://github.com/facebookresearch/sam2?tab=readme-ov-file) to install both PyTorch and TorchVision dependencies. You can install **the SAMURAI version** of SAM 2 on a GPU machine using:
```
cd sam2
pip install -e .
pip install -e ".[notebooks]"
```

Please see [INSTALL.md](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) from the original SAM 2 repository for FAQs on potential issues and solutions.

Install other requirements:
```
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru
```

#### SAM 2.1 Checkpoint Download

```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

#### ONNX 导出（export_onnx.py，新增）

我们在本仓库新增了 `scripts/export_onnx.py`，用于把 samurai（sam2）模型拆分并导出为 ONNX 子模块，便于后续在以下项目中使用：

- [samurai-onnxruntime](https://github.com/wp133716/samurai-onnxruntime)：使用 ONNX Runtime 做推理的实现；
- [samurai-ascendcl](https://github.com/wp133716/samurai-ascendcl)：面向 Ascend/华为 AI 芯片的部署适配；
- [samurai-tensorrt](https://github.com/wp133716/samurai-tensorrt)：用于 TensorRT 的推理加速与优化。

主要功能与说明：
- 支持导出的子模块包括 image_encoder、memory_attention、memory_encoder、mask_decoder；
- 支持可选的 FP16 转换、动态轴、简化与动态量化（脚本内可配置）；
- 导出过程中会保存部分中间位置编码文件（如 `om_model/vision_pos_embeds.npy`、`om_model/maskmem_pos_enc.npy`、`om_model/maskmem_tpos_enc.npy`），这些文件在部分硬件后端（Ascend 等）上会用到。

依赖（导出时需安装，示例）：

```
pip install onnx==1.17.0
pip install onnxruntime-gpu==1.20.0   # 如需 GPU 运行 ONNX Runtime
pip install onnxconverter_common # onnxconverter_common 用于自动混合精度转换
pip install onnxsim # 用于简化 ONNX 模型
```

示例用法：

```
python scripts/export_onnx.py --model_path sam2/checkpoints/sam2.1_hiera_tiny.pt
```

输出位置与注意事项：
- 导出文件默认写入脚本中指定的 `onnx_path` 路径下，例如 `onnx_model/mask_decoder.onnx`；
- 如果启用 FP16，会生成后缀 `_FP16.onnx` 的文件；若启用量化，会生成 `_INT8.onnx`（需谨慎验证准确率）；
- 导出与后续部署依赖的具体工具链（TensorRT/Ascend ACL）会有额外要求，请参见对应 downstream 项目的 README。


相关项目：
- [samurai-onnxruntime](https://github.com/wp133716/samurai-onnxruntime)（ONNX Runtime 推理）
- [samurai-ascendcl](https://github.com/wp133716/samurai-ascendcl)（Ascend/华为芯片部署）
- [samurai-tensorrt](https://github.com/wp133716/samurai-tensorrt)（TensorRT 加速）

#### Data Preparation

Please prepare the data in the following format:
```
data/LaSOT
├── airplane/
│   ├── airplane-1/
│   │   ├── full_occlusion.txt
│   │   ├── groundtruth.txt
│   │   ├── img
│   │   ├── nlp.txt
│   │   └── out_of_view.txt
│   ├── airplane-2/
│   ├── airplane-3/
│   ├── ...
├── basketball
├── bear
├── bicycle
...
├── training_set.txt
└── testing_set.txt
```

#### Main Inference
```
python scripts/main_inference.py 
```

## Demo on Custom Video

To run the demo with your custom video or frame directory, use the following examples:

**Note:** The `.txt` file contains a single line with the bounding box of the first frame in `x,y,w,h` format while the SAM 2 takes `x1,y1,x2,y2` format as bbox input.

### Input is Video File

```
python scripts/demo.py --video_path <your_video.mp4> --txt_path <path_to_first_frame_bbox.txt>
```

### Input is Frame Folder
```
# Only JPG images are supported
python scripts/demo.py --video_path <your_frame_directory> --txt_path <path_to_first_frame_bbox.txt>
```

## FAQs
**Question 1:** Does SAMURAI need training? [issue 34](https://github.com/yangchris11/samurai/issues/34)

**Answer 1:** Unlike real-life samurai, the proposed samurai do not require additional training. It is a zero-shot method, we directly use the weights from SAM 2.1 to conduct VOT experiments. The Kalman filter is used to estimate the current and future state (bounding box location and scale in our case) of a moving object based on measurements over time, it is a common approach that had been adopted in the field of tracking for a long time, which does not require any training. Please refer to code for more detail.

**Question 2:** Does SAMURAI support streaming input (e.g. webcam)?

**Answer 2:** Not yet. The existing code doesn't support live/streaming video as we inherit most of the codebase from the amazing SAM 2. Some discussion that you might be interested in: facebookresearch/sam2#90, facebookresearch/sam2#388 (comment).

**Question 3:** How to use SAMURAI in longer video?

**Answer 3:** See the discussion from sam2 https://github.com/facebookresearch/sam2/issues/264.

**Question 4:** How do you run the evaluation on the VOT benchmarks?

**Answer 4:** For LaSOT, LaSOT-ext, OTB, NFS please refer to the [issue 74](https://github.com/yangchris11/samurai/issues/74) for more details. For GOT-10k-test and TrackingNet, please refer to the official portal for submission.

## Acknowledgment

SAMURAI is built on top of [SAM 2](https://github.com/facebookresearch/sam2?tab=readme-ov-file) by Meta FAIR.

The VOT evaluation code is modifed from [VOT Toolkit](https://github.com/votchallenge/toolkit) by Luka Čehovin Zajc.

## Citation

Please consider citing our paper and the wonderful `SAM 2` if you found our work interesting and useful.
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}

@misc{yang2024samurai,
  title={SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory}, 
  author={Cheng-Yen Yang and Hsiang-Wei Huang and Wenhao Chai and Zhongyu Jiang and Jenq-Neng Hwang},
  year={2024},
  eprint={2411.11922},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2411.11922}, 
}
```
