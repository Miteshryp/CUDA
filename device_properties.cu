#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include"utils/cuda_utils.cuh"
#include<stdio.h>


int main() {
	
	// Getting the number of supported devices
	int deviceCount = 0;
	cuda_assert(cudaGetDeviceCount(&deviceCount));

	if (!deviceCount) {
		printf("No CUDA supported device found\n");
		return 0;
	}

	printf("CUDA Supported devices: %d", deviceCount);

	// getting active device 
	int activeDevice = 0;
	cuda_assert(cudaGetDevice(&activeDevice));

	// getting device properties
	cudaDeviceProp props;
	cuda_assert(cudaGetDeviceProperties(&props, activeDevice));

	printDeviceInfo(props);

	return 0;
}