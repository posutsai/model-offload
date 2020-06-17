#!/usr/local/bin/python3
import sys
import tvm
from tvm import relay
import numpy as np
from tvm.contrib.download import download_testdata
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import glob

def preprocess(dataset_path):
    pp_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    imgs = []
    for f in glob.glob(dataset_path)[:100]:
        img = Image.open(f).resize((224, 224))
        img = pp_transform(img)
        imgs.append(img.numpy())
    return np.array(imgs)

def compile_model(batch_size):
    # Prepare pretrain model
    model_name = 'resnet18'
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()
    input_shape = [batch_size, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    # Import the graph to Relay
    input_name = 'input0'
    shape_list = [(input_name, (BATCH_SIZE, 3, 224, 224))]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    target = 'llvm'
    # target_host = 'llvm --system-lib -target=aarch64-linux-gnu'
    target_host = 'llvm'
    ctx = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod,
                                         target=target,
                                         target_host=target_host,
                                         params=params)

    lib.export_library("deploy/rpi_resnet18_lib.tar")
    with open("deploy/rpi_resnet18_graph.json", "w") as fo:
        fo.write(graph)
    with open("deploy/rpi_resnet18_param.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))

if __name__ == "__main__":
    DATASET_PATH = "dataset/*.jpg"
    BATCH_SIZE = 32
    imgs = preprocess(DATASET_PATH)
    with open("dataset/dataset.npy", "wb") as f:
        np.save(f, imgs)
    compile_model(BATCH_SIZE)
