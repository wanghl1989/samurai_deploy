# SAMURAI Deployment

<div align="center">

**SAMURAI 行人跟踪模型部署方案**

基于 SAMURAI（Segment Anything Model for Visual Tracking with Motion-Aware Memory）的多后端部署实现，支持 ONNX Runtime、TensorRT 和 Docker 容器化部署。

</div>

---

## 项目简介

本项目提供了 SAMURAI 视觉跟踪模型的多种高性能部署方案，适用于不同的硬件平台和应用场景。SAMURAI 是一种零样本视觉跟踪方法，直接使用 SAM 2.1 的预训练权重.

### 核心特性

- **多后端支持**：提供 ONNX Runtime、TensorRT 两种推理后端
- **多语言实现**：包含 Python 和 C++ 两种实现
- **高性能优化**：支持 FP16、INT8 量化，优化推理速度
- **容器化部署**：提供完整的 Docker 导出和测试

### 项目结构

```
samurai_deploy/
├── samurai-onnx/           # ONNX 模型导出
│   ├── sam2/               # SAM 2 核心代码
│   ├── lib/                # 训练和评估代码
│   └── scripts/            # 推理脚本
├── samurai-onnxruntime/    # ONNX Runtime 推理实现
│   ├── python/             # Python 推理代码
│   └── cpp/                # C++ 推理代码
├── samurai-tensorrt/       # TensorRT 推理实现
│   ├── python/             # Python 推理代码
│   └── cpp/                # C++ 推理代码
├── docker/                 # Docker 部署配置
│   ├── Dockerfile
│   └── build_and_run.sh
└── checkpoints/            # 模型权重文件
```

---

### 快速开始

**onnx模型导出**

```
cd samura
pip install -r requirement.txt
```

**onnx模型导出**

```
python scripts/export_onnx.py --model_path <sam2.1 pytorch checkpoint path>
```


转换后的一个pt文件会保存4个onnx模型在一个文件夹中

**ONNX推理**

```
cd python
python onnx_inference.py  --onnx_model_path <onnx model directory path>
```

**TensorRT转换和推理**

```
cd python
python tensorrt_inference_cupy.py  --onnx_model_path <onnx model directory path>

```

**docker 运行**

```
cd docker
chmod +x build_and_run.sh
./build_and_run.sh
```


