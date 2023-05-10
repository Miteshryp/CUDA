// CUDA runtime headers
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

// C standard headers
#include<stdio.h>
#include<time.h>
#include<stdlib.h>

__global__ void printArr(const int* arr, int size) {
	int xoffset = (blockIdx.x * gridDim.x);
	int yoffset = (blockIdx.y);
}

//int main() {
//
//	const int size = 64;
//	int* arr = new int[size];
//	
//	// random seed generator
//	time_t t;
//	srand((unsigned int) &t);
//
//	// assigning random values
//	for (int i = 0; i < size; i++)
//		arr[i] = (unsigned int)(rand() & 64);
//
//	// shifting data to device
//	int* cuda_arr;
//	cudaMalloc(&cuda_arr, sizeof(int) * size);
//	cudaMemcpy(cuda_arr, arr, sizeof(int) * size, cudaMemcpyHostToDevice);
//
//	// creating kernel configurations
//	dim3 blockDimensions(2,2,2);
//	dim3 thread_per_block(2,2,2);
//
//	// executing the kernel
//	printArr << <blockDimensions, thread_per_block >> > (cuda_arr, size);
//
//	// waiting for all the computation to finish
//	cudaDeviceSynchronize();
//	
//	// reseting the GPU allocated resources
//	cudaDeviceReset();
//}