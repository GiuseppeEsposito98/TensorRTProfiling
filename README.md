# Pytorch_to_TensorRT
This repository, can take any pytorch-based model and convert it into a TensorRT compatible format. 
- It allows a kernel-level profiling of the operations graph by inspecting each NN layer implementation.
- It leverages Hardware Program Counters to profile the inference time
- It leverages on internal hardware sensors to profile the Power consumption.
