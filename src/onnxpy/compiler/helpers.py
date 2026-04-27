"""Runtime helper preamble emitted into every generated ONNXPY module.

The generated Python file is self-contained: it inlines the helper
functions below so that the only import surface a downstream user
sees is ``numpy`` and ``safetensors``.

This module exposes the helpers as a single ``HELPERS_SOURCE`` string
rather than as importable callables, because they are an emission
detail and deliberately have no in-process equivalent. The end-to-end
tests cover their behavior by compiling and executing tiny ONNX
graphs.
"""

from __future__ import annotations

HELPERS_SOURCE = '''\
_ONNX_DTYPE_NAMES = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    16: "bfloat16",
}


def _onnxpy_dtype_name(value):
    if value is None:
        return None
    if isinstance(value, str):
        return {"double": "float64", "float": "float32", "bool_": "bool"}.get(value, value)
    return _ONNX_DTYPE_NAMES[int(value)]


def _onnxpy_dtype(xp, value):
    name = _onnxpy_dtype_name(value)
    if name is None:
        return None
    if hasattr(xp, name):
        return getattr(xp, name)
    if name == "bool":
        return bool
    return getattr(np, name)


def _onnxpy_asarray(value, dtype, xp):
    resolved = _onnxpy_dtype(xp, dtype)
    if resolved is None:
        return xp.asarray(value)
    return xp.asarray(value, dtype=resolved)


def _onnxpy_to_python(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _onnxpy_to_tuple(value):
    py = _onnxpy_to_python(value)
    if isinstance(py, (list, tuple)):
        return tuple(int(item) for item in py)
    return (int(py),)


def _onnxpy_resolve_pads(
    input_spatial, kernel_shape, strides, dilations, auto_pad, pads
):
    rank = len(kernel_shape)
    mode = (auto_pad or "NOTSET").upper()
    if mode == "NOTSET":
        if pads is None:
            return tuple((0, 0) for _ in range(rank))
        flat = list(pads)
        return tuple((int(flat[i]), int(flat[i + rank])) for i in range(rank))
    if mode == "VALID":
        return tuple((0, 0) for _ in range(rank))
    pairs = []
    for i in range(rank):
        in_size = int(input_spatial[i])
        s = int(strides[i])
        d = int(dilations[i])
        k = int(kernel_shape[i])
        out_size = -(-in_size // s)
        eff_k = (k - 1) * d + 1
        total = max((out_size - 1) * s + eff_k - in_size, 0)
        if mode == "SAME_UPPER":
            pairs.append((total // 2, total - total // 2))
        else:
            pairs.append((total - total // 2, total // 2))
    return tuple(pairs)


def _onnxpy_sliding_windows(x_padded, kernel_shape, strides, dilations):
    from numpy.lib.stride_tricks import sliding_window_view

    rank = len(kernel_shape)
    dilated = tuple((int(k) - 1) * int(d) + 1 for k, d in zip(kernel_shape, dilations))
    windows = sliding_window_view(
        x_padded, dilated, axis=tuple(range(2, 2 + rank))
    )
    sl = (
        (slice(None), slice(None))
        + tuple(slice(None, None, int(s)) for s in strides)
        + tuple(slice(None, None, int(d)) for d in dilations)
    )
    return windows[sl]


def _onnxpy_conv(
    x, w, b=None, *, kernel_shape=None, strides=None, pads=None,
    dilations=None, group=1, auto_pad="NOTSET", xp,
):
    x_np = np.asarray(x)
    w_np = np.asarray(w)
    rank = x_np.ndim - 2
    kernel = tuple(int(k) for k in (kernel_shape or w_np.shape[2:]))
    strides_t = tuple(int(s) for s in (strides if strides is not None else [1] * rank))
    dilations_t = tuple(
        int(d) for d in (dilations if dilations is not None else [1] * rank)
    )
    group_i = int(group)
    pad_pairs = _onnxpy_resolve_pads(
        x_np.shape[2:], kernel, strides_t, dilations_t, auto_pad, pads
    )
    pad_full = ((0, 0), (0, 0)) + tuple(pad_pairs)
    x_padded = np.pad(x_np, pad_full)
    windows = _onnxpy_sliding_windows(x_padded, kernel, strides_t, dilations_t)
    n = windows.shape[0]
    c_in = windows.shape[1]
    out_spatial = windows.shape[2 : 2 + rank]
    perm = (
        (0,)
        + tuple(range(2, 2 + rank))
        + (1,)
        + tuple(range(2 + rank, 2 + 2 * rank))
    )
    windows = np.transpose(windows, perm)
    out_prod = 1
    for s in out_spatial:
        out_prod *= int(s)
    k_prod = 1
    for k in kernel:
        k_prod *= int(k)
    c_per_group = c_in // group_i
    windows = windows.reshape(n, out_prod, group_i, c_per_group * k_prod)
    m_out = w_np.shape[0]
    m_per_group = m_out // group_i
    w_flat = w_np.reshape(group_i, m_per_group, c_per_group * k_prod)
    out = np.einsum("nogk,gmk->nogm", windows, w_flat)
    out = out.reshape(n, out_prod, m_out)
    out = np.transpose(out, (0, 2, 1)).reshape(n, m_out, *out_spatial)
    if b is not None:
        b_np = np.asarray(b)
        out = out + b_np.reshape((1, m_out) + (1,) * rank)
    return xp.asarray(out)


def _onnxpy_max_pool(
    x, *, kernel_shape, strides=None, pads=None, dilations=None,
    ceil_mode=0, auto_pad="NOTSET", xp,
):
    x_np = np.asarray(x)
    rank = x_np.ndim - 2
    kernel = tuple(int(k) for k in kernel_shape)
    strides_t = tuple(int(s) for s in (strides if strides is not None else [1] * rank))
    dilations_t = tuple(
        int(d) for d in (dilations if dilations is not None else [1] * rank)
    )
    pad_pairs = _onnxpy_resolve_pads(
        x_np.shape[2:], kernel, strides_t, dilations_t, auto_pad, pads
    )
    pad_full = ((0, 0), (0, 0)) + tuple(pad_pairs)
    x_padded = np.pad(x_np, pad_full, constant_values=-np.inf)
    windows = _onnxpy_sliding_windows(x_padded, kernel, strides_t, dilations_t)
    out = windows.max(axis=tuple(range(-rank, 0)))
    return xp.asarray(out)


def _onnxpy_avg_pool(
    x, *, kernel_shape, strides=None, pads=None, ceil_mode=0,
    auto_pad="NOTSET", count_include_pad=0, xp,
):
    x_np = np.asarray(x)
    rank = x_np.ndim - 2
    kernel = tuple(int(k) for k in kernel_shape)
    strides_t = tuple(int(s) for s in (strides if strides is not None else [1] * rank))
    dilations_t = (1,) * rank
    pad_pairs = _onnxpy_resolve_pads(
        x_np.shape[2:], kernel, strides_t, dilations_t, auto_pad, pads
    )
    pad_full = ((0, 0), (0, 0)) + tuple(pad_pairs)
    x_padded = np.pad(x_np, pad_full)
    windows = _onnxpy_sliding_windows(x_padded, kernel, strides_t, dilations_t)
    reduce_axes = tuple(range(-rank, 0))
    if count_include_pad:
        out = windows.mean(axis=reduce_axes)
    else:
        sums = windows.sum(axis=reduce_axes)
        mask = np.ones_like(x_np)
        mask_padded = np.pad(mask, pad_full)
        mask_windows = _onnxpy_sliding_windows(
            mask_padded, kernel, strides_t, dilations_t
        )
        counts = mask_windows.sum(axis=reduce_axes)
        out = sums / np.maximum(counts, 1)
    return xp.asarray(out)


def _onnxpy_global_avg_pool(x, *, xp):
    return xp.mean(x, axis=tuple(range(2, x.ndim)), keepdims=True)


def _onnxpy_softmax(x, *, axis=-1, xp):
    x_np = np.asarray(x)
    a = int(axis)
    if a < 0:
        a += x_np.ndim
    shifted = x_np - np.max(x_np, axis=a, keepdims=True)
    e = np.exp(shifted)
    out = e / np.sum(e, axis=a, keepdims=True)
    return xp.asarray(out)


def _onnxpy_log_softmax(x, *, axis=-1, xp):
    x_np = np.asarray(x)
    a = int(axis)
    if a < 0:
        a += x_np.ndim
    shifted = x_np - np.max(x_np, axis=a, keepdims=True)
    out = shifted - np.log(np.sum(np.exp(shifted), axis=a, keepdims=True))
    return xp.asarray(out)


def _onnxpy_flatten(x, *, axis=1, xp):
    a = int(axis)
    if a < 0:
        a += x.ndim
    if a == 0:
        return xp.reshape(x, (1, -1))
    leading = 1
    for d in x.shape[:a]:
        leading *= int(d)
    return xp.reshape(x, (leading, -1))


def _onnxpy_reshape(data, shape, *, allow_zero=0, xp):
    shape_py = _onnxpy_to_python(shape)
    if not isinstance(shape_py, (list, tuple)):
        shape_py = [int(shape_py)]
    target = []
    in_shape = tuple(int(d) for d in data.shape)
    for i, dim in enumerate(shape_py):
        d = int(dim)
        if d == 0 and not int(allow_zero):
            target.append(in_shape[i])
        else:
            target.append(d)
    return xp.reshape(data, tuple(target))


def _onnxpy_transpose(x, *, perm=None, xp):
    if perm is None:
        perm = tuple(range(x.ndim - 1, -1, -1))
    return xp.permute_dims(x, tuple(int(p) for p in perm))


def _onnxpy_concat(values, *, axis=0, xp):
    return xp.concat(tuple(values), axis=int(axis))


def _onnxpy_gemm(a, b, c=None, *, alpha=1.0, beta=1.0, trans_a=0, trans_b=0, xp):
    a2 = xp.permute_dims(a, (1, 0)) if int(trans_a) else a
    b2 = xp.permute_dims(b, (1, 0)) if int(trans_b) else b
    out = float(alpha) * (a2 @ b2)
    if c is not None:
        out = out + float(beta) * c
    return out


def _onnxpy_cast(x, *, to, xp):
    dtype = _onnxpy_dtype(xp, to)
    if hasattr(xp, "astype"):
        try:
            return xp.astype(x, dtype, copy=False)
        except TypeError:
            return xp.astype(x, dtype)
    return xp.asarray(x, dtype=dtype)
'''

__all__ = ["HELPERS_SOURCE"]
