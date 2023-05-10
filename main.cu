#include<iostream>
#include<stdio.h>

__global__ void GPUFunction() {
   // std::cout << "Thread: " << threadIdx.x  << " running in block: " << blockIdx.x << std::endl;
   printf("Thread:%d running in block: %d\n", threadIdx.x, blockIdx.x);
}

void CPUFunction() {
   std::cout << "CPU Execution\n";
}

cudaDeviceProp getGPUDeviceSettings() {
   int deviceID;
   cudaGetDevice(&deviceID);
   cudaDeviceProp props;
   cudaGetDeviceProperties(&props, deviceID);

   std::cout << props.name << ": " << deviceID << std::endl;

   return props;
}

cudaDeviceProp getCPUDeviceSettings() {
   cudaDeviceProp props;
   cudaGetDeviceProperties(&props, cudaCpuDeviceId);

   std::cout << props.name << ": " << cudaCpuDeviceId << std::endl;
   return props;
}


void printDeviceProps(cudaDeviceProp& props) {
   std::cout << props.name << '\n';
   std::cout << "Settings: \n";

   std::cout << "Multitprocessor Count: " << props.multiProcessorCount << '\n'; 
   std::cout << "Warp size: " << props.warpSize << '\n';
   std::cout << "Major Compute Capability: " << props.major << '\n';
   std::cout << "Minor Compute Capability: " << props.minor << '\n';
   std::cout << "Clock Rate: " << props.clockRate << '\n';

   std::cout << std::endl; // clean the buffer.
}

int main() {
   cudaDeviceProp gpu_props = getGPUDeviceSettings();
   cudaDeviceProp cpu_props = getCPUDeviceSettings();

   printDeviceProps(gpu_props);
   printDeviceProps(cpu_props);

   CPUFunction();
   GPUFunction<<<2,gpu_props.warpSize>>>();
   std::cout << "This line is written in the vim editor \n";
   cudaDeviceSynchronize();
}
