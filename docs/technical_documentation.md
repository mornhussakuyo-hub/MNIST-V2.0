# MNIST-V2.0 技术文档

## 1. 项目概览

MNIST-V2.0 是一个手写数字识别项目，核心特点是从底层实现神经网络训练与推理，而不是直接依赖 PyTorch / TensorFlow。项目包含三类模型与一套 Web 体验：

- MLP：`scripts/model.py`，两层全连接神经网络。
- CPU CNN：`scripts/cnn_model.py`，NumPy 实现的两层卷积网络。
- GPU CNN：`scripts/cnn_model_gpu.py`，CuPy 实现的 CUDA 版本卷积网络。
- Web 推理：`scripts/web_app.py` + `web/`，提供网页手写输入和后端预测 API。

整体数据流如下：

```text
data/*.csv
  -> scripts/utils.py 读取、归一化、划分、one-hot
  -> scripts/train*.py 训练模型
  -> models/*.npz 保存权重
  -> results/*.npz 保存训练曲线数据
  -> scripts/visualize.py 输出曲线图片
  -> scripts/web_app.py 加载最新模型
  -> web/app.js 发送 784 维像素
  -> 后端返回预测类别、置信度、10 类概率
```

## 2. 目录说明

| 路径 | 作用 |
| --- | --- |
| `data/` | 存放 MNIST CSV 数据，例如 `mnist_train.csv`、`mnist_test.csv`。 |
| `models/` | 存放训练得到的 `.npz` 模型权重。 |
| `results/` | 存放训练历史，例如 loss、accuracy 曲线数据。 |
| `predicts/` | 存放命令行预测输出。 |
| `scripts/` | 核心 Python 脚本，包括模型、训练、预测、增强、可视化和 Web 服务。 |
| `visualize/` | 存放可视化生成的训练曲线图片。 |
| `web/` | 前端页面、样式和交互逻辑。 |
| `requirements.txt` | pip 环境依赖。 |
| `environment.yml` | Conda 环境依赖。 |

## 3. 运行环境

基础依赖：

- `numpy`：矩阵计算，CPU 模型核心。
- `pandas`：读取 CSV 数据。
- `matplotlib`：绘制训练曲线。
- `pillow`：数据增强时缩放图像。

GPU 训练额外依赖：

- `cupy-cuda11x` 或 `cupy-cuda12x`：与本机 CUDA 版本匹配。

## 4. 数据格式

项目默认使用 CSV 格式 MNIST：

```text
label,pixel1,pixel2,...,pixel784
5,0,0,...,255,...
```

第一列是标签，范围是 `0-9`。后面 784 列是 `28 x 28` 图像展开后的像素值，范围通常是 `0-255`。

训练前会做两件关键预处理：

1. 像素归一化：`pixel / 255.0`，把灰度值变成 `0-1`。
2. 标签 one-hot：例如数字 `3` 会变成 `[0,0,0,1,0,0,0,0,0,0]`。

## 5. 程序总览

| 程序 | 主要职责 |
| --- | --- |
| `scripts/common.py` | 文件路径校验与安全路径归一化。 |
| `scripts/utils.py` | 加载数据、归一化、one-hot 编码、训练/验证集划分。 |
| `scripts/model.py` | NumPy 实现两层 MLP。 |
| `scripts/train.py` | 训练 MLP，保存模型和训练历史。 |
| `scripts/cnn_model.py` | NumPy 实现 CNN，包括卷积、池化、ReLU、softmax、反向传播。 |
| `scripts/train_cnn.py` | 训练 CPU CNN。 |
| `scripts/cnn_model_gpu.py` | CuPy 实现 GPU CNN。 |
| `scripts/train_cnn_gpu.py` | 训练 CUDA CNN。 |
| `scripts/predict.py` | 使用保存的 MLP 模型做命令行批量预测。 |
| `scripts/visualize.py` | 根据训练历史生成 loss / accuracy 图片。 |
| `scripts/expand_data.py` | 对 MNIST 图像做放大、缩小、平移数据增强。 |
| `scripts/web_app.py` | 启动 Web 服务，加载最新模型，提供预测 API。 |
| `web/index.html` | Web 页面结构。 |
| `web/styles.css` | Web 页面样式。 |
| `web/app.js` | 画板交互、请求后端、展示预测概率。 |

## 6. `scripts/common.py`

这个文件很小，主要负责路径安全和文件存在性检查。

### 代码段说明

`import os`

导入系统路径工具，用于判断文件是否存在、展开用户目录、获取绝对路径。

`validate_file_path(path)`

检查传入路径是否是一个真实存在的文件。如果文件不存在，抛出 `FileNotFoundError`。训练、预测、可视化脚本都会用它提前发现路径错误。

`safe_path_resolution(path)`

对路径做三步处理：

1. `os.path.expanduser(path)`：支持 `~` 用户目录。
2. `os.path.abspath(expanded)`：转换为绝对路径。
3. `os.path.normpath(absolute)`：规范化路径，去掉多余的 `..`、`.` 等。

它的作用是让后续文件操作面对统一、明确的路径。

## 7. `scripts/utils.py`

这个文件负责通用数据处理，是训练脚本的基础工具。

### `load_data(train_path, test_path)`

作用：读取训练集和测试集 CSV。

代码流程：

1. 使用 `pd.read_csv(train_path)` 读取训练 CSV。
2. `train_df.iloc[:, 0].values` 取第一列作为标签。
3. `train_df.iloc[:, 1:].values` 取剩余 784 列作为图像像素。
4. 用同样方式读取测试集。
5. 打印图像和标签的形状，方便检查数据是否正确。
6. 返回 `train_images, train_labels, test_images, test_labels`。

### `normalize_data(images, data_name)`

作用：把像素从 `0-255` 缩放到 `0-1`。

核心代码是：

```python
return images / 255.0
```

神经网络训练时，如果输入数值过大，梯度和激活值会不稳定。归一化可以让训练更平稳。

### `one_hot_encode(labels, num_classes=10)`

作用：把数字标签转换成 one-hot 矩阵。

代码流程：

1. 获取样本数 `num_samples`。
2. 检查标签数据类型，如果不是整数则转成 `np.int32`。
3. 创建形状为 `(num_samples, 10)` 的零矩阵。
4. 使用高级索引：`one_hot[np.arange(num_samples), labels] = 1`。
5. 返回 one-hot 标签。

### `split_train_val(images, labels, val_radio=0.2, usage_radio=1)`

作用：从训练数据中随机抽取一部分使用，并切分训练集和验证集。

代码流程：

1. 计算总样本数。
2. 根据 `usage_radio` 计算实际使用的样本数。
3. 用 `np.random.choice` 随机抽取样本。
4. 用 `np.random.permutation` 打乱样本。
5. 根据 `val_radio` 计算训练/验证分割点。
6. 分别返回训练图像、训练标签、验证图像、验证标签。

注意：参数名里写的是 `radio`，语义上其实是 `ratio`。

## 8. `scripts/model.py`

这个文件实现一个两层全连接神经网络 `TwoLayerNet`。

### 模型结构

```text
输入层：784
隐藏层：hidden_size，默认 128
激活函数：ReLU
输出层：10
输出函数：softmax
损失函数：交叉熵
优化方式：小批量梯度下降
```

### `__init__`

作用：初始化网络结构和参数。

代码段说明：

- `input_size, hidden_size, output_size` 保存网络尺寸。
- `weight_init` 支持 `he`、`xavier`、`normal` 三种初始化方式。
- He 初始化：适合 ReLU，公式大意是权重标准差按 `sqrt(2 / fan_in)` 缩放。
- Xavier 初始化：常用于 sigmoid / tanh 一类激活。
- normal 初始化：用较小随机数初始化。
- `b1`、`b2` 初始化为零。
- `self.cache = {}` 用来保存前向传播中间变量，反向传播时会用到。

### `get_model_name`

返回模型名称，用于保存、预测输出和 Web 展示。

### `relu`

ReLU 激活函数：

```python
return np.maximum(0, x)
```

小于 0 的值变成 0，大于 0 的值保持不变。

### `d_relu`

ReLU 的导数：

```python
return (x > 0).astype(np.float32)
```

反向传播时，只有前向值大于 0 的神经元继续传递梯度。

### `softmax`

作用：把输出层分数转换成概率。

代码里先减去每行最大值：

```python
exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
```

这是为了避免指数爆炸，提高数值稳定性。最后除以每行指数和，得到 10 类概率。

### `forward`

作用：完成一次前向传播。

流程：

```text
X
  -> z1 = X @ W1.T + b1
  -> a1 = ReLU(z1)
  -> z2 = a1 @ W2.T + b2
  -> y_hat = softmax(z2)
```

中间结果 `X`、`z1`、`a1`、`z2`、`y_hat` 会放入 `cache`，供反向传播使用。

### `compute_loss`

作用：计算交叉熵损失。

代码要点：

- `np.clip(y_hat, epsilon, 1. - epsilon)` 防止 `log(0)`。
- `-np.sum(y * np.log(y_hat)) / num_samples` 是多分类交叉熵。

### `backward`

作用：根据损失计算所有参数梯度。

流程：

1. 输出层误差：`dz2 = y_hat - y`。
2. 输出层权重梯度：`dW2 = dz2.T @ a1 / num_samples + reg_lambda * W2`。
3. 输出层偏置梯度：`db2 = sum(dz2) / num_samples`。
4. 反传到隐藏层：`da1 = dz2 @ W2`。
5. 乘以 ReLU 导数：`dz1 = da1 * d_relu(z1)`。
6. 隐藏层权重梯度：`dW1 = dz1.T @ X / num_samples + reg_lambda * W1`。
7. 隐藏层偏置梯度：`db1 = sum(dz1) / num_samples`。
8. 把梯度打包成字典返回。

`reg_lambda` 是 L2 正则项，用来抑制权重过大，降低过拟合风险。

### `update_parameters`

作用：执行梯度下降。

每个参数都按下面形式更新：

```python
parameter -= learning_rate * gradient
```

### `predict`

作用：输出预测类别。

先调用 `forward` 得到概率，然后 `np.argmax(y_hat, axis=1)` 取概率最大的类别。

### `accuracy`

作用：计算准确率。

将预测类别和 one-hot 标签中的真实类别比较，求平均值。

### `save`

作用：把权重和模型名保存到 `.npz`。

保存内容包括：

- `W1`
- `b1`
- `W2`
- `b2`
- `model_name`

### `load`

作用：从 `.npz` 文件恢复 MLP 权重。

读取保存的矩阵后覆盖当前模型参数。

## 9. `scripts/train.py`

这个文件负责训练 MLP。

### 顶部线程配置

`THREAD_ENV_VARS` 定义常见 BLAS / NumPy 线程环境变量：

- `OMP_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `MKL_NUM_THREADS`
- `VECLIB_MAXIMUM_THREADS`
- `NUMEXPR_NUM_THREADS`

`configure_numpy_threads()` 会读取命令行参数 `--num-threads`。如果用户指定线程数，就写入这些环境变量；如果环境已经设置，则尊重环境；否则默认使用 CPU 核心数。

这段代码必须在 `import numpy as np` 前执行，因为很多 BLAS 后端会在 NumPy 导入时读取线程设置。

### 导入项目模块

`sys.path.append(...)` 把 `scripts` 目录加入模块搜索路径。

随后导入：

- `TwoLayerNet`：MLP 模型。
- `load_data`、`normalize_data`、`one_hot_encode`、`split_train_val`：数据工具。
- `validate_file_path`、`safe_path_resolution`：路径工具。

### `parse_args`

定义命令行参数。重要参数包括：

- `--traindt`：训练 CSV 路径。
- `--testdt`：测试 CSV 路径。
- `--model-name`：模型名，必填。
- `--epochs`：训练轮数。
- `--batch-size`：小批量大小。
- `--learning-rate`：学习率。
- `--hidden-size`：隐藏层大小。
- `--val-ratio`：验证集比例。
- `--usage-ratio`：实际使用训练数据比例。
- `--reg-rate`：L2 正则强度。
- `--LR-dc`、`--LR-dc-num`：学习率衰减开关和间隔。
- `--no-save`：只训练不保存。

### `main`

训练主流程。

第一段：读取参数并打印配置。

这里把命令行参数转成局部变量，如 `learning_rate`、`num_epochs`、`batch_size`，并打印网络结构和超参数。

第二段：检查数据路径。

调用 `safe_path_resolution` 规范化路径，再用 `validate_file_path` 检查文件是否存在。如果找不到文件，程序退出。

第三段：加载和预处理数据。

流程是：

```text
load_data
  -> normalize_data
  -> split_train_val
  -> one_hot_encode
```

第四段：初始化模型。

创建：

```python
model = TwoLayerNet(input_size, hidden_size, output_size, "he", args.model_name)
```

其中 `input_size=784`，`output_size=10`。

第五段：训练循环。

每个 epoch 内：

1. `np.random.permutation(num_train)` 打乱训练样本。
2. 按 `batch_size` 切成多个 mini-batch。
3. 对每个 batch 执行：
   - `model.forward`
   - `model.compute_loss`
   - `model.backward`
   - `model.update_parameters`
4. 统计平均训练 loss。
5. 计算训练准确率、验证准确率、验证 loss。
6. 记录历史曲线数据。
7. 如果开启学习率衰减，则每隔固定 epoch 把学习率减半。

第六段：测试集评估。

训练结束后调用 `model.accuracy(test_images, test_labels)` 得到最终测试准确率。

第七段：保存结果。

如果没有开启 `--no-save`：

- 在 `results/` 保存训练历史 `.npz`。
- 在 `models/` 保存模型 `.npz`。

## 10. `scripts/cnn_model.py`

这个文件是 CPU CNN 的核心。它使用 NumPy 从零实现卷积、池化、ReLU、softmax、反向传播和模型保存。

### `im2col`

作用：把输入图像中的局部卷积窗口展开成二维矩阵。

输入形状：

```text
(n, c, h, w)
```

分别表示 batch 数、通道数、高度、宽度。

代码流程：

1. 计算输出特征图大小 `out_h`、`out_w`。
2. 使用 `np.pad` 对图像补零。
3. 创建 `col` 数组，形状包含 batch、通道、卷积核高宽、输出高宽。
4. 双层循环遍历卷积核内部位置，把所有滑动窗口对应位置复制到 `col`。
5. 通过 `transpose` 和 `reshape` 转换成二维矩阵。

最终形状：

```text
(n * out_h * out_w, c * filter_h * filter_w)
```

这样卷积可以写成矩阵乘法，提高效率。

### `col2im`

作用：`im2col` 的反向过程，用于卷积和池化的反向传播。

代码流程：

1. 把二维 `col` 还原成包含 batch、通道、卷积核高宽、输出高宽的结构。
2. 创建补零后的 `img` 梯度数组。
3. 通过循环把窗口梯度累加回原图对应位置。
4. 去掉 padding 区域，返回输入图像形状的梯度。

注意这里用的是 `+=`，因为同一个输入像素可能参与多个卷积窗口，需要累加多个梯度贡献。

### `softmax`

作用：把分类分数转成概率。

先减去最大值防止指数溢出，再计算指数归一化。

### `Conv2D`

二维卷积层。

#### `__init__`

作用：初始化卷积核、偏置和缓存。

- `W` 形状是 `(out_channels, in_channels, kernel_size, kernel_size)`。
- 使用 He 初始化，适合 ReLU。
- `b` 是每个输出通道一个偏置。
- `stride` 和 `pad` 控制卷积步长与补零。
- `self.x`、`self.col` 用于反向传播。

#### `forward`

作用：卷积前向传播。

流程：

1. 根据输入和卷积核计算输出高宽。
2. 保存输入 `self.x`。
3. 调用 `im2col` 展开输入图像。
4. 把卷积核 reshape 成二维矩阵。
5. 执行矩阵乘法：`self.col @ W.T + b`。
6. reshape 回 `(n, out_channels, out_h, out_w)`。

#### `backward`

作用：卷积反向传播。

流程：

1. 把上游梯度 `dout` 转成二维矩阵。
2. `db = sum(dout)` 得到偏置梯度。
3. `dW = dout.T @ self.col` 得到卷积核梯度。
4. `dcol = dout @ W` 得到展开输入的梯度。
5. 调用 `col2im` 把梯度还原成输入图像形状。

### `MaxPool2D`

最大池化层。

#### `__init__`

保存池化窗口大小、步长、输入缓存和最大值索引缓存。

#### `forward`

作用：执行最大池化。

流程：

1. 计算池化输出尺寸。
2. 调用 `im2col` 把每个池化窗口展开。
3. reshape 成 `(窗口数量, 通道数, pool_h * pool_w)`。
4. `np.argmax` 记录每个窗口最大值位置。
5. `np.max` 得到池化结果。
6. reshape 回 `(n, c, out_h, out_w)`。

#### `backward`

作用：最大池化反向传播。

最大池化只有前向传播中取得最大值的位置会接收梯度，其他位置梯度为 0。

流程：

1. 转换 `dout` 维度。
2. 创建全零 `dmax`。
3. 根据 `self.arg_max` 把梯度放回最大值位置。
4. reshape 成 `dcol`。
5. 调用 `col2im` 还原为输入形状。

### `ReLU`

ReLU 激活层。

#### `forward`

保存 `x <= 0` 的布尔掩码，并把这些位置置 0。

#### `backward`

反向传播时，把前向阶段小于等于 0 的位置梯度置 0。

### `TwoConvNet`

完整 CNN 模型。

#### `__init__`

模型结构：

```text
Conv2D(1 -> conv1_filters, 3x3, pad=1)
ReLU
MaxPool2D(2x2)
Conv2D(conv1_filters -> conv2_filters, 3x3, pad=1)
ReLU
MaxPool2D(2x2)
Flatten
FC(conv2_filters * 7 * 7 -> 10)
Softmax
```

为什么是 `7 * 7`：

```text
28 x 28 -> 第一次 2x2 pool -> 14 x 14
14 x 14 -> 第二次 2x2 pool -> 7 x 7
```

#### `forward`

作用：完整 CNN 前向传播。

流程：

1. 卷积 1。
2. ReLU 1。
3. 池化 1。
4. 卷积 2。
5. ReLU 2。
6. 池化 2。
7. 保存 flatten 前形状。
8. 展平成二维矩阵。
9. 全连接计算 10 类分数。
10. softmax 输出概率。

#### `compute_loss`

作用：计算交叉熵损失，并可加入 L2 正则。

正则项包括 `conv1.W`、`conv2.W`、`W3` 三组权重。

#### `backward`

作用：完整 CNN 反向传播。

流程：

1. softmax + 交叉熵梯度：`dout = (probs - y) / batch_size`。
2. 计算全连接层 `W3`、`b3` 梯度。
3. 反传回 flatten 输入。
4. reshape 回第二个池化层输出形状。
5. 依次反传：pool2 -> relu2 -> conv2 -> pool1 -> relu1 -> conv1。
6. 把卷积层和全连接层梯度保存到 `self.grads`。

#### `update_parameters`

按学习率更新所有可训练参数：

- `conv1.W`
- `conv1.b`
- `conv2.W`
- `conv2.b`
- `W3`
- `b3`

#### `predict_proba`

返回 softmax 概率。

#### `predict`

返回概率最大的类别。

#### `accuracy`

按 batch 分批预测，避免一次性推理占用过多内存。最后与真实标签比较并求平均准确率。

#### `save`

保存 CNN 模型到 `.npz`，额外写入 `model_type="cnn"`。Web 服务会根据这个字段判断如何加载模型。

#### `load`

从 `.npz` 读取卷积层和全连接层参数，并按保存的 filter 数重建模型。

## 11. `scripts/train_cnn.py`

这个文件训练 CPU CNN。

### 顶部线程配置

和 `train.py` 类似，先设置 NumPy / BLAS 线程数，再导入 NumPy。

### `PROJECT_ROOT`

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

用于定位项目根目录，使默认数据路径、模型路径和结果路径不依赖当前命令行所在目录。

### `parse_args`

重要参数：

- `--conv1-filters`：第一层卷积核数量，默认 8。
- `--conv2-filters`：第二层卷积核数量，默认 16。
- `--epochs`：默认 3。
- `--batch-size`：默认 64。
- `--learning-rate`：默认 0.01。
- `--reg-rate`：L2 正则。
- `--no-save`：训练后不保存。

### `prepare_images`

作用：把扁平图像转为 CNN 需要的四维格式：

```python
images.astype(np.float32).reshape(-1, 1, 28, 28)
```

其中 `1` 表示 MNIST 是单通道灰度图。

### `main`

训练流程：

1. 解析参数并打印网络结构。
2. 检查训练集和测试集路径。
3. 读取 CSV。
4. 归一化像素。
5. 切分训练集和验证集。
6. 把图像 reshape 成 `(N, 1, 28, 28)`。
7. 把标签 one-hot 并转成 `float32`。
8. 创建 `TwoConvNet`。
9. 每个 epoch 随机打乱样本，按 batch 训练。
10. 每个 batch 执行 forward、loss、backward、update。
11. 每个 epoch 后计算训练准确率、验证 loss、验证准确率。
12. 最后计算测试准确率。
13. 保存训练历史和模型。

## 12. `scripts/cnn_model_gpu.py`

这个文件是 GPU CNN，整体结构与 `cnn_model.py` 几乎一致，但使用 `cupy as cp` 替代 `numpy as np`。

### 与 CPU 版本的主要区别

- 数组创建使用 `cp.zeros`、`cp.random.randn`。
- 矩阵运算在 CUDA GPU 上执行。
- 保存模型时使用 `cp.asnumpy` 把 GPU 数组转回 NumPy。
- 加载模型时使用 `cp.asarray` 把 NumPy 数组放回 GPU。
- `accuracy` 返回前用 `cp.asnumpy` 转成 Python float。

### 代码段说明

`im2col`、`col2im`、`softmax`、`Conv2D`、`MaxPool2D`、`ReLU` 和 `TwoConvNetGPU` 的逻辑与 CPU 版本对应。区别是所有中间张量都尽量保留在 GPU 显存中，避免频繁 CPU/GPU 传输。

演示时可以说：

> GPU 版本证明我们的实现不是只停留在理论层面，而是可以切换计算后端。由于 NumPy 和 CuPy API 很接近，我们在保留模型结构的同时，把大量矩阵计算迁移到了 CUDA。

## 13. `scripts/train_cnn_gpu.py`

这个文件训练 CUDA CNN。

### 导入部分

- `cupy as cp`：GPU 数组和 CUDA 调用。
- `numpy as np`：读取数据后的 CPU 侧处理，以及保存历史。
- `TwoConvNetGPU`：GPU CNN 模型。

### `parse_args`

GPU 训练额外参数：

- `--gpu`：选择 CUDA 设备编号。
- `--eval-batch-size`：评估准确率时的 batch 大小。
- `--metric-samples`：每个 epoch 只抽样一部分训练样本计算训练准确率，节省时间。设置为 0 表示使用全部样本。

### `prepare_images`

把输入变成 `(N, 1, 28, 28)` 的 `float32`，再用 `cp.asarray` 放到 GPU。

### `to_gpu_labels`

先调用 `one_hot_encode`，再转成 `float32`，最后放入 GPU。

### `scalar`

把 CuPy 标量转成 Python `float`，方便记录 loss。

### `synchronize`

调用 `cp.cuda.Stream.null.synchronize()`，等待 GPU 当前流计算完成。因为 GPU 计算常常是异步的，计时和日志输出前同步可以让统计更准确。

### `main`

主要流程：

1. `cp.cuda.Device(args.gpu).use()` 选择 GPU。
2. 打印 GPU 名称和训练配置。
3. 检查路径并读取数据。
4. 归一化、划分训练/验证、转成 GPU 张量。
5. 创建 `TwoConvNetGPU`。
6. 使用 `cp.random.permutation` 在 GPU 上打乱索引。
7. 对每个 batch 执行 GPU forward、loss、backward、update。
8. 每个 epoch 后根据 `metric_samples` 选择训练准确率评估样本。
9. 计算验证 loss 和验证准确率。
10. 同步 GPU，记录耗时。
11. 最后计算测试准确率。
12. 保存训练历史和模型。

## 14. `scripts/predict.py`

这个文件用于命令行批量预测，目前主要支持 MLP `.npz` 模型。

### `parse_args`

参数：

- `--data`：要预测的 CSV。
- `--model`：训练好的 `.npz` 模型。
- `--outdir`：预测结果输出目录。

### `load_model`

作用：从 MLP `.npz` 中恢复模型。

流程：

1. 检查模型路径。
2. 读取 `W1`、`b1`、`W2`、`b2`。
3. 根据权重形状推断输入层、隐藏层、输出层大小。
4. 创建 `TwoLayerNet`。
5. 用保存的权重覆盖随机初始化权重。
6. 返回模型对象。

### `main`

流程：

1. 加载模型。
2. 读取待预测 CSV。
3. 如果 CSV 有 785 列，则第一列视为真实标签；否则认为没有标签。
4. 归一化图像。
5. 调用 `model.predict` 得到预测结果。
6. 如果有真实标签，计算准确率。
7. 把每个样本预测结果写入 `predicts/`。

## 15. `scripts/visualize.py`

这个文件把训练历史 `.npz` 转成图片。

### `parse_args`

只需要一个参数：

- `--hist`：训练历史 `.npz` 文件路径。

### `main`

流程：

1. 检查历史文件是否存在。
2. 读取：
   - `train_loss`
   - `val_loss`
   - `train_acc`
   - `val_acc`
3. 根据历史文件名推断模型名。
4. 创建输出目录 `visualize/{model_name}_visualize/`。
5. 生成 2 x 2 子图：
   - 训练 loss
   - 验证 loss
   - 训练 accuracy
   - 验证 accuracy
6. 保存为 `{model_name}_training_history.png`。
7. 再生成一张合并图：
   - 上半部分对比 train / val loss
   - 下半部分对比 train / val accuracy
8. 保存为 `{model_name}_combined_history.png`。

## 16. `scripts/expand_data.py`

这个文件负责数据增强，目的是让模型更适应真实网页手写输入的尺度和位置变化。

### `PROJECT_ROOT`、默认输入输出

默认读取：

```text
data/mnist_train.csv
```

默认输出：

```text
data/mnist_train_expand.csv
```

### `parse_args`

重要参数：

- `--chunk-size`：分块处理行数，避免一次性加载太多数据。
- `--seed`：随机种子，保证平移增强可复现。
- `--enlarge-scale`：放大比例，默认 1.18。
- `--shrink-scale`：缩小比例，默认 0.82。
- `--shift-scale`：平移前缩放比例，默认 0.78。
- `--max-shift`：最大平移像素，默认 4。

### `digit_bbox`

作用：找到数字笔迹的边界框。

流程：

1. 用 `np.where(image > threshold)` 找到大于阈值的像素。
2. 如果没有有效像素，返回整张图范围。
3. 否则返回最小/最大行列坐标。

### `resize_digit`

作用：裁剪数字、缩放数字、再放回 28 x 28 画布。

流程：

1. 调用 `digit_bbox` 找到数字区域。
2. 裁剪出数字。
3. 根据 `scale` 计算新高度和宽度。
4. 用 Pillow 双线性插值缩放。
5. 计算新图像放回画布的位置。
6. 根据 `shift_y`、`shift_x` 加入平移。
7. 限制位置，保证不越界。
8. 创建黑色画布，把缩放后的数字贴回去。

### `make_shift`

作用：生成一个非零随机平移。

如果 `shift_y` 和 `shift_x` 都是 0，就重新采样，保证增强样本确实发生平移。

### `augment_row`

作用：把一行 MNIST 数据扩展成多行。

对每个原始样本生成 4 个版本：

1. 原图。
2. 放大图。
3. 缩小图。
4. 缩小后随机平移图。

最后返回形状为 `(4, 785)` 的数组。

### `main`

流程：

1. 解析参数。
2. 创建输出目录。
3. 如果输出文件已存在，先删除。
4. 创建随机数生成器。
5. 用 `pd.read_csv(..., chunksize=...)` 分块读取。
6. 对每个 chunk 的每一行调用 `augment_row`。
7. 用 `np.vstack` 合并增强结果。
8. 写入输出 CSV，第一块写 header，后续追加。
9. 打印输入输出样本数。

## 17. `scripts/web_app.py`

这个文件启动 Web 服务，并提供模型推理 API。

### 顶部常量

- `PROJECT_ROOT`：项目根目录。
- `WEB_ROOT`：前端静态文件目录。
- `DEFAULT_MODELS_DIR`：默认模型目录。

### `ModelStore`

`ModelStore` 是后端模型管理类，负责自动发现、加载和复用最新模型。

#### `__init__`

保存模型目录和模型缓存状态：

- `model_path`：当前加载的模型路径。
- `model_mtime`：模型修改时间。
- `model_type`：`mlp` 或 `cnn`。
- `cnn_model`：CNN 模型对象。
- `weights`：MLP 权重字典。
- `model_name`：模型名称。

#### `latest_model_path`

作用：找到 `models/` 中最新修改的 `.npz` 模型文件。

它会过滤掉训练历史文件，只保留真正模型。然后按修改时间选最新文件。

#### `load_latest`

作用：如果模型更新，就重新加载；如果没有变化，就复用缓存。

流程：

1. 调用 `latest_model_path` 找到最新模型。
2. 比较路径和修改时间，如果没变化直接返回。
3. 用 `np.load` 读取模型文件。
4. 判断 `model_type`，如果是 CNN 或文件中有 `conv1_W`，就调用 `TwoConvNet.load`。
5. 否则按 MLP 读取 `W1`、`b1`、`W2`、`b2`。
6. 更新缓存状态。

#### `info`

作用：返回前端可展示的模型信息。

包括：

- 模型名。
- 模型路径。
- 模型类型。
- MLP 隐藏层大小。
- 网络结构字符串。

#### `predict`

作用：接收 784 个像素，返回预测结果。

流程：

1. 调用 `load_latest`，确保使用最新模型。
2. 把前端传来的 `pixels` 转成 NumPy 数组。
3. reshape 成 `(1, 784)`。
4. 用 `np.clip` 限制在 `0-1`。
5. 如果是 CNN，reshape 成 `(1, 1, 28, 28)` 后调用 `cnn_model.predict_proba`。
6. 如果是 MLP，直接手写一遍 MLP 前向传播。
7. `np.argmax` 得到预测类别。
8. 返回预测类别、置信度、10 类概率和模型信息。

#### `architecture`

作用：生成网络结构描述字符串。

CNN 返回类似：

```text
Conv(8) -> Pool -> Conv(16) -> Pool -> FC(10)
```

MLP 返回类似：

```text
784 -> 128 -> 10
```

### `softmax`

Web 服务内部给 MLP 推理使用的 softmax，与模型中逻辑一致。

### `build_handler(model_store)`

作用：动态创建 HTTP 请求处理类。

内部类 `MNISTHandler` 继承 `SimpleHTTPRequestHandler`。

#### `__init__`

设置静态文件目录为 `WEB_ROOT`，所以浏览器访问 `/` 时会得到 `web/index.html`。

#### `do_GET`

如果请求路径是 `/api/model`，返回模型信息 JSON。否则交给父类处理静态文件。

#### `do_POST`

只处理 `/api/predict`。

流程：

1. 检查路径，不是 `/api/predict` 就返回 404。
2. 读取请求体长度。
3. 解析 JSON。
4. 取出 `pixels`。
5. 检查它必须是长度 784 的列表。
6. 调用 `model_store.predict(pixels)`。
7. 返回预测 JSON。
8. 出错时返回 400 和错误信息。

#### `write_json`

把 Python 字典转成 JSON 响应，并设置：

- HTTP 状态码。
- `Content-Type: application/json; charset=utf-8`。
- `Content-Length`。

### `parse_args`

Web 服务参数：

- `--host`：绑定地址，默认 `127.0.0.1`。
- `--port`：端口，默认 `8000`。
- `--models-dir`：模型目录。

### `main`

流程：

1. 检查 `web/` 目录是否存在。
2. 创建 `ModelStore`。
3. 预加载模型信息。
4. 创建 HTTP handler。
5. 启动 `ThreadingHTTPServer`。
6. 打印访问地址。
7. `serve_forever()` 持续服务。
8. Ctrl+C 时关闭服务器。

最后：

```python
os.chdir(PROJECT_ROOT)
main()
```

确保无论从哪里启动脚本，相对路径都以项目根目录为基准。

## 18. `web/index.html`

这个文件定义网页结构。

### 页面头部

设置 HTML5 文档类型、中文语言、UTF-8 编码、移动端 viewport、页面标题，并引入 `styles.css`。

### 主体结构

`main.shell` 是页面容器。

`section.hero` 包含标题：

```text
画一个数字，让神经网络来猜
```

以及隐藏的 `modelInfo` 容器。

`section.workspace` 是主要工作区，分为左右两块：

- 左侧 `draw-panel`：绘图区、清空按钮、28 x 28 网格。
- 右侧 `result-panel`：预测按钮、最终结果、置信度、概率柱状图。

底部引入 `app.js`，负责动态创建网格和交互。

## 19. `web/styles.css`

这个文件定义页面视觉效果。

### 全局变量

`:root` 里定义颜色、阴影和字体变量，例如：

- `--ink`：主文字颜色。
- `--paper`：页面底色。
- `--accent`：强调色。
- `--accent-2`：辅助强调色。
- `--display-font`：标题字体。
- `--text-font`：正文字体。

### 基础样式

- `* { box-sizing: border-box; }` 让尺寸计算更稳定。
- `body` 设置全屏背景、字体和文字颜色。
- `button` 统一按钮基础样式。

### 页面布局

- `.shell` 控制页面最大宽度和居中。
- `.hero` 控制标题区域。
- `.workspace` 使用 CSS Grid，桌面端左右两栏。
- `.panel` 定义面板背景、边框、阴影和圆角。

### 绘图网格

`.digit-grid` 使用：

```css
grid-template-columns: repeat(28, 1fr);
aspect-ratio: 1;
```

创建一个正方形 28 x 28 网格。

`.cell` 是每个像素格，默认黑色。

### 结果区域

- `.winner` 展示最终预测数字。
- `#winnerDigit` 用大字体突出结果。
- `.bars` 定义 10 个数字概率柱状图布局。
- `.bar-fill` 通过 JS 动态改变高度，展示概率。

### 响应式布局

`@media (max-width: 860px)` 下，页面改成单列布局，适配手机或窄屏。

## 20. `web/app.js`

这个文件负责网页交互。

### 顶部常量和状态

- `GRID_SIZE = 28`：网格宽高。
- `CELL_COUNT = 784`：像素总数。
- `pixels`：长度 784 的数组，保存用户绘制的灰度值。
- `grid`、`clearButton`、`predictButton` 等变量缓存 DOM 节点。
- `isDrawing`：记录当前是否正在拖拽绘制。
- `cells`：保存所有像素格 DOM 节点。

### `createGrid`

作用：动态创建 784 个格子。

每个格子：

- `className = "cell"`。
- `dataset.index = index`，记录自己对应的像素索引。
- 追加到 `#grid`。
- 存入 `cells` 数组。

### `createBars`

作用：创建 10 个概率柱状图卡片，对应数字 0-9。

每个卡片包含：

- 柱状条轨道。
- 柱状条填充。
- 数字标签。
- 百分比标签。

### `paintCell`

作用：给指定像素上色，同时模拟一个简单画笔。

代码逻辑：

1. 检查索引是否越界。
2. 根据 index 计算行列。
3. 构造画笔数组：中心点较亮，上下左右较淡。
4. 遍历画笔覆盖位置。
5. 更新 `pixels[brushIndex]`，最大不超过 1。
6. 把灰度值转成 `rgb(lightness, lightness, lightness)` 更新页面颜色。

这让用户画出来的笔迹不是单像素硬边，而是略有粗细。

### `cellFromPointer`

作用：根据鼠标或触控位置找到当前格子。

使用 `document.elementFromPoint(event.clientX, event.clientY)` 获取指针下的 DOM 元素，如果它是 `.cell`，就返回对应索引。

### `clearGrid`

作用：清空画布和结果。

流程：

1. `pixels.fill(0)` 清空像素。
2. 把所有 cell 背景改回黑色。
3. 重置结果标题、最终数字、置信度。
4. 调用 `updateBars` 清空概率柱状图。

### `updateBars`

作用：根据后端返回的概率更新柱状图。

对每个数字：

1. 找到对应卡片。
2. 找到 `.bar-fill` 和 `.percent-label`。
3. 把概率限制在 `0-1`。
4. 设置柱状条高度。
5. 显示百分比文本。

参数 `prediction` 目前传入但没有用来高亮 winner，后续可以扩展。

### `loadModelInfo`

作用：请求 `/api/model` 获取模型信息。

当前代码请求后只做错误处理，没有把模型信息显示到页面，因为 `modelInfo` 是 hidden。可以作为后续展示增强点。

### `predictDigit`

作用：发送绘图像素到后端并展示结果。

流程：

1. 禁用预测按钮，防止重复点击。
2. 把标题改成“正在识别...”。
3. `fetch("/api/predict", ...)` 发送 POST 请求。
4. 请求体是 `{ pixels }`。
5. 解析 JSON。
6. 如果响应失败，抛出错误。
7. 调用 `updateBars` 更新概率柱状图。
8. 显示预测数字和置信度。
9. 出错时显示错误信息。
10. 最后恢复按钮可点击。

### 绘图事件监听

`pointerdown`：

- 标记开始绘制。
- 捕获指针。
- 立即绘制当前格子。

`pointermove`：

- 如果正在绘制，就持续绘制指针经过的格子。

`pointerup` 和 `pointerleave`：

- 停止绘制。

按钮事件：

- 清空按钮调用 `clearGrid`。
- 预测按钮调用 `predictDigit`。

脚本最后执行：

```javascript
createGrid();
createBars();
clearGrid();
loadModelInfo();
```

完成页面初始化。

## 21. 训练与部署命令

训练 MLP：

```bash
cd scripts
python train.py --model-name mlp_demo --epochs 50 --batch-size 32
```

训练 CPU CNN：

```bash
python train_cnn.py --model-name cnn_demo --epochs 3 --batch-size 64
```

训练 GPU CNN：

```bash
python train_cnn_gpu.py --model-name cnn_gpu_demo --gpu 0 --epochs 10 --batch-size 256
```

启动 Web：

```bash
cd ..
python scripts/web_app.py --host 0.0.0.0 --port 8000
```

服务器部署时，通常需要让防火墙或反向代理开放对应端口，浏览器访问服务器地址即可体验。

## 22. 内部架构总结

```text
浏览器
  用户绘制 28 x 28 像素
  POST /api/predict
        |
        v
Python HTTP Server
  校验 pixels 长度
  加载 models/ 最新 .npz
  判断 MLP / CNN
        |
        v
模型推理
  MLP: 矩阵乘法 + ReLU + softmax
  CNN: Conv + ReLU + Pool + FC + softmax
        |
        v
JSON 响应
  prediction
  confidence
  probabilities
        |
        v
前端展示最终数字和概率柱状图
```

## 23. 未来方向：二值化推理设计文档

### 研究动机

当前模型使用 `float32` 或 `float64` 进行计算。虽然准确率较好，但乘法、加法和权重存储成本较高。二值化推理希望把权重和激活压缩到 1 bit，从而提高推理速度、降低模型大小，并为边缘设备部署做准备。

### 二值化目标

可以分三个阶段：

1. 权重二值化：`W_float -> W_binary`。
2. 激活二值化：`activation_float -> activation_binary`。
3. 位运算推理：用 bit packing + bitcount 替代普通乘加。

### 普通卷积和二值卷积对比

普通卷积：

```text
sum(input_float * weight_float)
```

二值卷积：

```text
input_bit 与 weight_bit 做 XNOR
统计相同 bit 的数量
根据 bitcount 结果还原近似点积
```

如果使用 `-1 / +1` 表示二值：

```text
dot(x, w) = same_count - different_count
```

因为：

- 相同符号相乘为 `+1`。
- 不同符号相乘为 `-1`。

### 可以新增的模块

建议后续新增：

| 文件 | 作用 |
| --- | --- |
| `scripts/binary_ops.py` | 二值化函数、bit packing、bitcount 工具。 |
| `scripts/binary_cnn_model.py` | 二值 CNN 推理模型。 |
| `scripts/evaluate_binary.py` | 对比 float CNN 和 binary CNN 的准确率、速度、模型大小。 |
| `scripts/train_binary_cnn.py` | 如果进一步做二值化训练，可放训练逻辑。 |

### 技术挑战

1. 准确率下降：二值化会损失表达能力。
2. 训练困难：sign 函数不可导，需要 STE 近似。
3. 输入预处理：网页输入是灰度值，如何二值化阈值需要实验。
4. 卷积实现：Python 层循环可能抵消位运算收益，需要尽量向量化。
5. 指标设计：除了准确率，还要比较推理耗时、模型大小、内存占用。

### 实验指标

建议记录：

- Float MLP / Float CNN 准确率。
- Binary Weight CNN 准确率。
- Binary Activation CNN 准确率。
- 单样本平均推理耗时。
- batch 推理吞吐量。
- 模型文件大小。
- Web 端响应延迟。

### 研讨会表达方式

可以这样讲：

> 我们后续不是简单追求更复杂的模型，而是想研究更轻量的推理方式。MNIST 是一个很适合做二值化实验的平台，因为任务足够清晰、数据规模适中、实验反馈快。我们可以在这个项目已有的 CNN 结构上逐步替换权重表示和计算方式，观察速度和准确率之间的平衡。

