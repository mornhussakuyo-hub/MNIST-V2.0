# CNN Manim 动画渲染说明

动画脚本位置：

```text
animations/cnn_forward_manim.py
```

场景名称：

```text
CNNForwardPass
```

## 安装 Manim

建议单独安装动画依赖，不影响原来的训练环境：

```bash
pip install manim
```

如果是在 Windows 上渲染中文文字，建议安装或保留 `Microsoft YaHei` 字体。脚本顶部的 `FONT` 可以改成你机器上已有的中文字体。

## 快速预览

在项目根目录运行：

```bash
manim -pql animations/cnn_forward_manim.py CNNForwardPass
```

含义：

- `-p`：渲染后自动打开视频。
- `-q l`：低质量快速预览，适合检查画面节奏。

## 导出研讨会视频

如果要放进 PPT，建议用中等或高质量：

```bash
manim -pqm animations/cnn_forward_manim.py CNNForwardPass
```

或者：

```bash
manim -pqh animations/cnn_forward_manim.py CNNForwardPass
```

输出视频默认在：

```text
media/videos/cnn_forward_manim/
```

## 动画内容

这段动画按项目中的 `TwoConvNet` 结构演示：

```text
1 x 28 x 28 输入图像
  -> Conv1: 8 个 3 x 3 卷积核
  -> ReLU
  -> MaxPool: 28 x 28 -> 14 x 14
  -> Conv2 + ReLU: 16 个特征图
  -> MaxPool: 14 x 14 -> 7 x 7
  -> Flatten: 16 x 7 x 7
  -> FC(10)
  -> Softmax 概率输出
```

最后一页还加入了未来二值化推理方向：

```text
float32 乘加
  -> 0/1 权重与激活
  -> XNOR + bitcount
```

这可以自然衔接你们研讨会里“后续研究二值化推理 MNIST”的部分。

