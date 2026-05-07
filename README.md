# MNIST-V2.0

一个用 NumPy 从零实现的 MNIST 手写数字识别项目，包含 MLP、简易 CNN、训练记录可视化，以及浏览器手写预测页面。

## 项目结构

```text
data/       MNIST CSV 数据集，本地放置，不提交
models/     训练得到的 .npz 模型，本地生成
results/    训练历史，本地生成
predicts/   预测输出，本地生成
scripts/    训练、预测、可视化和 Web 服务脚本
web/        手写数字预测前端
```

## 环境

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate mnist-v2
```

也可以用 pip：

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

GPU 训练脚本需要另外安装与本机 CUDA 匹配的 CuPy，例如 `cupy-cuda11x` 或 `cupy-cuda12x`。

## 数据

把 CSV 格式的 MNIST 数据放到 `data/`：

```text
data/mnist_train.csv
data/mnist_test.csv
```

CSV 第一列为标签 `0-9`，后面 784 列为 `28 x 28` 像素值。

## 常用命令

训练 MLP：

```bash
cd scripts
python train.py --model-name mlp_demo --epochs 50 --batch-size 32
```

训练 CNN：

```bash
python train_cnn.py --model-name cnn_demo --epochs 3 --batch-size 64
```

GPU 训练 CNN：

```bash
pip install cupy-cuda12x
python train_cnn_gpu.py --model-name cnn_gpu_demo --gpu 0 --epochs 10 --batch-size 256
```

如果本机是 CUDA 11，把 `cupy-cuda12x` 换成 `cupy-cuda11x`。常用参数包括 `--gpu`、`--epochs`、`--batch-size`、`--learning-rate`、`--conv1-filters`、`--conv2-filters`。

预测：

```bash
python predict.py --data ../data/mnist_test.csv --model ../models/mlp_demo.npz
```

可视化训练记录：

```bash
python visualize.py --hist ../results/mlp_demo_training_history.npz
```

CNN 训练结束会自动在 `visualize/` 下生成训练曲线，也可以单独可视化 CNN 训练记录：

```bash
python visualize_cnn.py --hist ../results/cnn_demo_cnn_training_history.npz
```

启动手写预测页面：

```bash
cd ..
python scripts/web_app.py --port 8000
```

然后打开 `http://127.0.0.1:8000`。

## 许可证

本项目使用 MIT License，详见 `LICENSE`。
