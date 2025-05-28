#include<iostream>
#include<stdio.h>

#include "cuda_runtime.h"
#include "utils/cuda_assert.cuh"

__global__ void GPUFunction() {
   // std::cout << "Thread: " << threadIdx.x  << " running in block: " << blockIdx.x << std::endl;
   printf("Thread:%d running in block: %d\n", threadIdx.x, blockIdx.x);
}

void CPUFunction() {
   std::cout << "CPU Execution\n";
}

int main() {
   cudaDeviceProp gpu_props = getGPUDeviceSettings();
   cudaDeviceProp cpu_props = getCPUDeviceSettings();

   printDeviceInfo(gpu_props);
   printDeviceInfo(cpu_props);

   // Launching kernels based on warpSize for optimized utilization
   CPUFunction();
   cuda_assert(<2,gpu_props.warpSize>>>());
   cudaDeviceSynchronize();
}
