"""Build sharpen_x4.onnx from the published Real-ESRGAN weights, without PyTorch.

Only needed if the .onnx file is ever lost or replaced; the built file is
committed next to this script and the server never runs this.

    curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
    python make_sharpen_model.py realesr-general-x4v3.pth sharpen_x4.onnx

The network is SRVGGNetCompact (Real-ESRGAN, BSD-3-Clause, Xintao Wang et al.):
conv 3->64, PReLU, 32 x (conv 64->64, PReLU), conv 64->48, pixel shuffle x4,
plus a nearest-neighbour x4 copy of the input added back as a residual. The
checkpoint is a zip holding a pickle; it is read here with a small unpickler
so that converting it does not need a 2GB PyTorch install.
"""
from __future__ import annotations

import pickle
import sys
import zipfile

import numpy as np


def load_state(path: str) -> dict[str, np.ndarray]:
    zf = zipfile.ZipFile(path)
    root = zf.namelist()[0].split("/")[0]

    class Storage:
        def __init__(self, key, dtype):
            self.key, self.dtype = key, dtype

    dtypes = {"FloatStorage": np.float32, "HalfStorage": np.float16,
              "DoubleStorage": np.float64, "LongStorage": np.int64}

    def rebuild(storage, offset, size, stride, *_):
        raw = np.frombuffer(zf.read(f"{root}/data/{storage.key}"), dtype=storage.dtype)
        arr = np.lib.stride_tricks.as_strided(
            raw[offset:], shape=tuple(size), strides=[s * raw.itemsize for s in stride])
        return np.array(arr, dtype=np.float32)

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild
            if module == "torch" and name in dtypes:
                return name
            if module == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            raise pickle.UnpicklingError(f"refusing {module}.{name}")

        def persistent_load(self, pid):
            _, kind, key, _location, _numel = pid
            return Storage(key, dtypes[kind])

    with zf.open(f"{root}/data.pkl") as fh:
        obj = Unpickler(fh).load()
    return obj.get("params_ema") or obj.get("params") or obj


def build(state: dict[str, np.ndarray], out: str) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    idx = sorted({int(k.split(".")[1]) for k in state if k.startswith("body.")})
    nodes, inits = [], []
    x = "input"
    for i in idx:
        w = state[f"body.{i}.weight"]
        if w.ndim == 4:                                   # convolution
            b = state[f"body.{i}.bias"]
            inits += [numpy_helper.from_array(w, f"w{i}"), numpy_helper.from_array(b, f"b{i}")]
            nodes.append(helper.make_node("Conv", [x, f"w{i}", f"b{i}"], [f"c{i}"],
                                          kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
            x = f"c{i}"
        else:                                              # PReLU, one slope per channel
            inits.append(numpy_helper.from_array(w.reshape(-1, 1, 1), f"p{i}"))
            nodes.append(helper.make_node("PRelu", [x, f"p{i}"], [f"a{i}"]))
            x = f"a{i}"

    nodes.append(helper.make_node("DepthToSpace", [x], ["shuffled"], blocksize=4, mode="CRD"))
    inits.append(numpy_helper.from_array(np.array([1, 1, 4, 4], dtype=np.float32), "scales"))
    nodes.append(helper.make_node("Resize", ["input", "", "scales"], ["base"], mode="nearest"))
    nodes.append(helper.make_node("Add", ["shuffled", "base"], ["output"]))

    graph = helper.make_graph(
        nodes, "sharpen_x4",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H", "W"])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                              producer_name="kamal-forge")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    state = load_state(src)
    print(f"{len(state)} tensors")
    build(state, dst)
    print(f"wrote {dst}")
