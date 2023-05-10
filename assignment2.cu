#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include "utils/cuda_assert.cuh"

#include<stdio.h>
#include<stdlib.h>
#include<time.h>

// c++ stl timing library
#include<chrono>


__global__ void sumArrGPU(const int* arr1, const int* arr2, const int* arr3, int* result, int size) {
	int index = threadIdx.x + (blockDim.x * blockIdx.x);

	//printf("Hello\n");

	if (index < size)
		result[index] = arr1[index] + arr2[index] + arr3[index];
}

void sumArrCPU(const int* arr1, const int* arr2, const int* arr3, int* result, int size) {
	for (int i = 0; i < size; i++) {
		result[i] = arr1[i] + arr2[i] + arr3[i];
	}
 }

//int main() {
//	const int size = 10000;
//	
//	// CPU memory variables
//	int* arr1 = new int[size];
//	int* arr2 = new int[size];
//	int* arr3 = new int[size];
//	int* result = new int[size];
//	int* gans = new int[size];
//
//	// GPU memory variables
//	int *garr1 = nullptr, *garr2 = nullptr, *garr3 = nullptr, *gresults = nullptr;
//	
//
//	// random seed initialiser
//	time_t t;
//	srand((unsigned int)&t);
//	
//
//	// intialising values
//	for (int i = 0; i < size; i++) {
//		arr1[i] = rand() & 0xffff;
//		arr2[i] = rand() & 0xffff;
//		arr3[i] = rand() & 0xffff;
//	}
//	
//	// setting GPU device
//	//cuda_assert(cudaSetDevice(0));
//
//	// allocating cuda memory
//	cuda_assert(cudaMalloc(&garr1, sizeof(int) * size));
//	cuda_assert(cudaMalloc(&garr2, sizeof(int) * size));
//	cuda_assert(cudaMalloc(&garr3, sizeof(int) * size));
//	cuda_assert(cudaMalloc(&gresults, sizeof(int) * size));
//
//
//	// monitering host to device memory transfer time
//	auto shtod = std::chrono::high_resolution_clock::now();
//	cuda_assert(cudaMemcpy(garr1, arr1, sizeof(int) * size, cudaMemcpyHostToDevice));
//	cuda_assert(cudaMemcpy(garr2, arr2, sizeof(int) * size, cudaMemcpyHostToDevice));
//	cuda_assert(cudaMemcpy(garr3, arr3, sizeof(int) * size, cudaMemcpyHostToDevice));
//	auto ehtod = std::chrono::high_resolution_clock::now();
//
//	// kernel settings
//	int threads_per_block = 512;
//	int no_of_blocks = (int)(size / threads_per_block) + (size % threads_per_block ? 1 : 0);
//	
//	// monitering kernel execution time
//	auto sgpu = std::chrono::high_resolution_clock::now();
//	sumArrGPU << <no_of_blocks, threads_per_block>> > (garr1, garr2, garr3, gresults, size);
//	cuda_assert(cudaDeviceSynchronize());
//	auto egpu = std::chrono::high_resolution_clock::now();
//
//	// monitering CPU execution time
//	auto scpu = std::chrono::high_resolution_clock::now();
//	sumArrCPU(arr1, arr2, arr3, result, size);
//	auto ecpu = std::chrono::high_resolution_clock::now();
//
//	// monitering device to host memory transfer
//	auto sdtoh = std::chrono::high_resolution_clock::now();
//	cuda_assert(cudaMemcpy(gans, gresults, sizeof(int) * size, cudaMemcpyDeviceToHost));
//	auto edtoh = std::chrono::high_resolution_clock::now();
//
//	// getting results sychronously
//	
//	printf("CPU execution time: %ld\n", std::chrono::duration_cast<std::chrono::microseconds>(ecpu - scpu).count() );
//	printf("GPU execution time: %ld\n\n", std::chrono::duration_cast<std::chrono::microseconds>(egpu - sgpu).count() );
//									   
//	printf("Host to device copy: %ld\n", std::chrono::duration_cast<std::chrono::microseconds>(ehtod - shtod).count() );
//	printf("Device to host copy: %ld\n", std::chrono::duration_cast<std::chrono::microseconds>(edtoh - sdtoh).count() );
//
//
//	//cuda_assert(cudaDeviceReset());
//}