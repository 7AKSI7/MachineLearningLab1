"""Тензорные операции в TensorFlow, PyTorch и Keras."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import torch
from keras import ops


def create_tensorflow_tensor(data: np.ndarray) -> tf.Tensor:
    return tf.constant(np.asarray(data, dtype=np.float32))


def create_pytorch_tensor(data: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(data, dtype=np.float32))


def tensorflow_axw_b(x_tf: tf.Tensor) -> dict:
    """Считает A @ X @ W + B в TensorFlow."""
    n, m = x_tf.shape
    k, p = 5, 4

    a = tf.random.uniform((k, n), minval=0, maxval=10, dtype=tf.int32)
    w = tf.random.normal((m, p))
    b = tf.random.uniform((k, p))

    ax = tf.matmul(tf.cast(a, tf.float32), x_tf)
    axw = tf.matmul(ax, w)
    result = axw + b
    return {"result": result, "A": a, "W": w, "B": b}


def pytorch_axw_b(x_pt: torch.Tensor) -> dict:
    """Считает A @ X @ W + B в PyTorch."""
    n, m = x_pt.shape
    k, p = 5, 4

    a = torch.randint(0, 10, (k, n), dtype=torch.float32)
    w = torch.randn(m, p)
    b = torch.rand(k, p)

    result = a @ x_pt @ w + b
    return {"result": result, "A": a, "W": w, "B": b}


def create_TPQ(shape=(3, 10)):
    """Создаёт три случайных массива T, P, Q."""
    t = np.random.rand(*shape)
    p = np.random.rand(*shape)
    q = np.random.rand(*shape)
    return t, p, q


def keras_operation(t: np.ndarray, p: np.ndarray, q: np.ndarray):
    """V = |sin(T) - exp(P) * sqrt(Q)| в Keras."""
    t_k = ops.convert_to_tensor(t, dtype="float32")
    p_k = ops.convert_to_tensor(p, dtype="float32")
    q_k = ops.convert_to_tensor(q, dtype="float32")
    return ops.abs(ops.sin(t_k) - ops.exp(p_k) * ops.sqrt(q_k))


def pytorch_operation(t: np.ndarray, p: np.ndarray, q: np.ndarray):
    """Та же операция в PyTorch."""
    t_t = torch.tensor(t, dtype=torch.float32)
    p_t = torch.tensor(p, dtype=torch.float32)
    q_t = torch.tensor(q, dtype=torch.float32)
    return torch.abs(torch.sin(t_t) - torch.exp(p_t) * torch.sqrt(q_t))