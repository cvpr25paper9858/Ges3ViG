#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>
#include <iostream>
torch::Tensor process_octant(torch::Tensor cuboid);
