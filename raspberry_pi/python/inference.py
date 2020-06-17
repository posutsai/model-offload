#!/usr/local/bin/python3
import sys
import numpy as np
from tvm.contrib import graph_runtime
# from tvm.contrib.debugger import debug_runtime as graph_runtime
import tvm
import time

def inference(imgs):
    ctx = tvm.cpu(0)
    loaded_lib = tvm.runtime.load_module("../deploy/rpi_resnet18_lib.tar")
    with open("../deploy/rpi_resnet18_graph.json") as f:
        loaded_graph = f.read()
    with open("../deploy/rpi_resnet18_param.params", "rb") as f:
        loaded_params = bytearray(f.read())
    module = graph_runtime.create(loaded_graph, loaded_lib, ctx)
    module.load_params(loaded_params)
    input_name = 'input0'
    module.set_input(input_name, imgs[:32].astype("float32"))
    start = time.time_ns()
    module.run()
    end = time.time_ns()
    print(f"duration in python is {end - start} ns");

    # If getting output
    # tvm_output = module.get_output(0)


if __name__ == "__main__":
    BATCH_SIZE = 32
    with open("../dataset/dataset.npy", "rb") as f:
        imgs = np.load(f)
    inference(imgs)
