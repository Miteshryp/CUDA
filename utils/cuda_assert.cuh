
#ifndef CUDA_UTILS
#define CUDA_UTILS

#define BYTES_PER_KB 1024
#define BYTES_PER_GB 1024.0 * 1024.0 * 1024.0
#define BYTES_PER_MB 1024 * 1024

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#define cuda_assert(ans) __checkCudaCode((ans), __FILE__, __LINE__);

inline void __checkCudaCode(cudaError_t code, const char *filename, int line_number)
{
	if (code != cudaSuccess)
	{
		printf("CUDA ERROR at %s:%d\n", filename, line_number);
	}
}

inline void printDeviceInfo(cudaDeviceProp props)
{
	double totalGlobalMempory = (double)(props.totalGlobalMem) / (double)BYTES_PER_GB;

	printf("\nDEVICE PROPERTIES - %s\n", props.name);

	printf("Clock Rate: %d\n", props.clockRate);
	printf("CUDA version: %d.%d\n", props.major, props.minor);
	printf("Compute mode: %d\n", props.computeMode);

	printf("Multiprocessor count: %d\n", props.multiProcessorCount);
	printf("Warp size: %d\n", props.warpSize);

	printf("Major Compute Capability: %d\n", props.major);
	printf("Minor Compute Capability: %d\n", props.minor);

	printf("Constant memory: %lf KB\n", (double)props.totalConstMem / BYTES_PER_KB);
	printf("Shared memory per block: %lf KB\n", (double)props.sharedMemPerBlock / BYTES_PER_KB);
	printf("Global memory: %f GB \n", (float)(props.totalGlobalMem / BYTES_PER_GB));
	printf("Block size (threads per block): %d\n", props.maxThreadsPerBlock);

	printf("\n");
}

void printDeviceProps(cudaDeviceProp &props)
{
	printf("%s\n", props.name);

	printf("Multiprocessor count: %d\n", props.multiProcessorCount);
	printf("Warp size: %d\n", props.warpSize);

	printf("Major Compute Capability: %d\n", props.major);
	printf("Minor Compute Capability: %d\n", props.minor);
}

cudaDeviceProp getGPUDeviceSettings()
{
	int deviceID;
	cuda_assert(cudaGetDevice(&deviceID));
	cudaDeviceProp props;
	cuda_assert(cudaGetDeviceProperties(&props, deviceID));

	printf("%s: %d\n", props.name, deviceID);

	return props;
}

cudaDeviceProp getCPUDeviceSettings()
{
	cudaDeviceProp props;
	cuda_assert(cudaGetDeviceProperties(&props, cudaCpuDeviceId));

	printf("%s: %d\n", props.name, cudaCpuDeviceId);

	return props;
}

#endif // !CUDA_UTILS