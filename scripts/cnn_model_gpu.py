from pathlib import Path

import cupy as cp
import numpy as np


def im2col(x, filter_h, filter_w, stride=1, pad=0):
    n, c, h, w = x.shape
    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1
    img = cp.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

    col = cp.zeros((n, c, filter_h, filter_w, out_h, out_w), dtype=x.dtype)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x_pos in range(filter_w):
            x_max = x_pos + stride * out_w
            col[:, :, y, x_pos, :, :] = img[:, :, y:y_max:stride, x_pos:x_max:stride]

    return col.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    n, c, h, w = input_shape
    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1
    col = col.reshape(n, out_h, out_w, c, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = cp.zeros((n, c, h + 2 * pad + stride - 1, w + 2 * pad + stride - 1), dtype=col.dtype)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x_pos in range(filter_w):
            x_max = x_pos + stride * out_w
            img[:, :, y:y_max:stride, x_pos:x_max:stride] += col[:, :, y, x_pos, :, :]

    return img[:, :, pad:h + pad, pad:w + pad]


def softmax(x):
    shifted = x - cp.max(x, axis=1, keepdims=True)
    exp_x = cp.exp(shifted)
    return exp_x / cp.sum(exp_x, axis=1, keepdims=True)


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        scale = cp.sqrt(cp.float32(2.0 / (in_channels * kernel_size * kernel_size)))
        self.W = (cp.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale).astype(cp.float32)
        self.b = cp.zeros(out_channels, dtype=cp.float32)
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.dW = None
        self.db = None

    def forward(self, x):
        fn, _, fh, fw = self.W.shape
        n, _, h, w = x.shape
        out_h = (h + 2 * self.pad - fh) // self.stride + 1
        out_w = (w + 2 * self.pad - fw) // self.stride + 1

        self.x = x
        self.col = im2col(x, fh, fw, self.stride, self.pad)
        out = self.col @ self.W.reshape(fn, -1).T + self.b
        return out.reshape(n, out_h, out_w, fn).transpose(0, 3, 1, 2)

    def backward(self, dout):
        fn, _, fh, fw = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, fn)

        self.db = cp.sum(dout, axis=0)
        self.dW = (dout.T @ self.col).reshape(self.W.shape)
        dcol = dout @ self.W.reshape(fn, -1)
        return col2im(dcol, self.x.shape, fh, fw, self.stride, self.pad)


class MaxPool2D:
    def __init__(self, pool_h=2, pool_w=2, stride=2):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.x = None
        self.arg_max = None

    def forward(self, x):
        n, c, h, w = x.shape
        out_h = (h - self.pool_h) // self.stride + 1
        out_w = (w - self.pool_w) // self.stride + 1

        self.x = x
        col = im2col(x, self.pool_h, self.pool_w, self.stride, 0)
        col = col.reshape(-1, c, self.pool_h * self.pool_w)
        self.arg_max = cp.argmax(col, axis=2)
        out = cp.max(col, axis=2)
        return out.reshape(n, out_h, out_w, c).transpose(0, 3, 1, 2)

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dout_flat = dout.reshape(-1, dout.shape[3])
        rows, channels = dout_flat.shape
        dmax = cp.zeros((rows, channels, pool_size), dtype=dout.dtype)
        dmax[cp.arange(rows)[:, None], cp.arange(channels)[None, :], self.arg_max] = dout_flat
        dcol = dmax.reshape(dmax.shape[0], -1)
        return col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, 0)


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout = dout.copy()
        dout[self.mask] = 0
        return dout


class TwoConvNetGPU:
    def __init__(self, model_name="cnn_gpu_model", conv1_filters=8, conv2_filters=16):
        self.model_name = model_name
        self.conv1 = Conv2D(1, conv1_filters, kernel_size=3, stride=1, pad=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()
        self.conv2 = Conv2D(conv1_filters, conv2_filters, kernel_size=3, stride=1, pad=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()

        flattened_size = conv2_filters * 7 * 7
        self.W3 = (cp.random.randn(flattened_size, 10) * cp.sqrt(cp.float32(2.0 / flattened_size))).astype(cp.float32)
        self.b3 = cp.zeros(10, dtype=cp.float32)
        self.flatten_shape = None
        self.flat_input = None
        self.probs = None
        self.grads = {}

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        self.flatten_shape = out.shape
        self.flat_input = out.reshape(out.shape[0], -1)
        self.probs = softmax(self.flat_input @ self.W3 + self.b3)
        return self.probs

    def compute_loss(self, probs, y, reg_lambda=0.0):
        epsilon = 1e-8
        clipped = cp.clip(probs, epsilon, 1.0 - epsilon)
        data_loss = -cp.sum(y * cp.log(clipped)) / y.shape[0]
        reg_loss = 0.5 * reg_lambda * (
            cp.sum(self.conv1.W * self.conv1.W) +
            cp.sum(self.conv2.W * self.conv2.W) +
            cp.sum(self.W3 * self.W3)
        )
        return data_loss + reg_loss

    def backward(self, x, y, reg_lambda=0.0):
        batch_size = x.shape[0]
        dout = (self.probs - y) / batch_size

        self.grads["W3"] = self.flat_input.T @ dout + reg_lambda * self.W3
        self.grads["b3"] = cp.sum(dout, axis=0)

        dout = dout @ self.W3.T
        dout = dout.reshape(self.flatten_shape)
        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        self.conv1.backward(dout)

        self.grads["conv1_W"] = self.conv1.dW + reg_lambda * self.conv1.W
        self.grads["conv1_b"] = self.conv1.db
        self.grads["conv2_W"] = self.conv2.dW + reg_lambda * self.conv2.W
        self.grads["conv2_b"] = self.conv2.db
        return self.grads

    def update_parameters(self, learning_rate):
        self.conv1.W -= learning_rate * self.grads["conv1_W"]
        self.conv1.b -= learning_rate * self.grads["conv1_b"]
        self.conv2.W -= learning_rate * self.grads["conv2_W"]
        self.conv2.b -= learning_rate * self.grads["conv2_b"]
        self.W3 -= learning_rate * self.grads["W3"]
        self.b3 -= learning_rate * self.grads["b3"]

    def predict_proba(self, x):
        return self.forward(x)

    def predict(self, x):
        return cp.argmax(self.predict_proba(x), axis=1)

    def accuracy(self, x, y, batch_size=512):
        predictions = []
        for start in range(0, x.shape[0], batch_size):
            predictions.append(self.predict(x[start:start + batch_size]))
        predictions = cp.concatenate(predictions)
        true_labels = cp.argmax(y, axis=1)
        return float(cp.asnumpy(cp.mean(predictions == true_labels)))

    def save(self, filepath):
        np.savez(
            filepath,
            model_type="cnn",
            model_name=self.model_name,
            conv1_W=cp.asnumpy(self.conv1.W),
            conv1_b=cp.asnumpy(self.conv1.b),
            conv2_W=cp.asnumpy(self.conv2.W),
            conv2_b=cp.asnumpy(self.conv2.b),
            W3=cp.asnumpy(self.W3),
            b3=cp.asnumpy(self.b3),
        )

    @classmethod
    def load(cls, filepath):
        data = np.load(filepath)
        model_name = str(data["model_name"].item() if "model_name" in data else Path(filepath).stem)
        conv1_filters = data["conv1_W"].shape[0]
        conv2_filters = data["conv2_W"].shape[0]
        model = cls(model_name=model_name, conv1_filters=conv1_filters, conv2_filters=conv2_filters)
        model.conv1.W = cp.asarray(data["conv1_W"])
        model.conv1.b = cp.asarray(data["conv1_b"])
        model.conv2.W = cp.asarray(data["conv2_W"])
        model.conv2.b = cp.asarray(data["conv2_b"])
        model.W3 = cp.asarray(data["W3"])
        model.b3 = cp.asarray(data["b3"])
        return model
