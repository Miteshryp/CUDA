#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include<iostream>

__global__ void arraySumGPU(const int* arr1, const int* arr2, int* sumArr, int size) {
	int index = threadIdx.x + (blockDim.x * blockIdx.x);

	// perform sum if index is valid
	if (index < size) {
		sumArr[index] = arr1[index] + arr2[index];
	}
	//printf("index: %d\n", index);
}

void arraySumCPU(const int* arr1, const int* arr2, int* sumArr, int size) {
	for (int i = 0; i < size; i++) {
		sumArr[i] = arr1[i] + arr2[i];
	}
}

int main() {
	
	int size = 10000;
	cudaSetDevice(0);

	// allocating cpu arrays
	int* arr1 = new int[size];
	int* arr2 = new int[size];

	// initialising array to random values
	time_t t;
	srand((unsigned int)&t);
	for (int i = 0; i < size; i++)
		arr1[i] = rand() & 0xffff;
	for (int i = 0; i < size; i++)
		arr2[i] = rand() & 0xffff;

	int *result = new int[size];

	int *gpuArr1, *gpuArr2, *gpuResult;

	// allocating gpu memory
	cudaMalloc(&gpuArr1, sizeof(int) * size);
	cudaMalloc(&gpuArr2, sizeof(int) * size);
	cudaMalloc(&gpuResult, sizeof(int) * size);

	// copying values
	cudaMemcpy(gpuArr1, arr1, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuArr2, arr2, sizeof(int) * size, cudaMemcpyHostToDevice);

	// kernel settings
	int threads_per_block = 32;
	int block_count = (int)(size / threads_per_block) + (size % threads_per_block ? 1 : 0);
	

	// executing the kernel
	arraySumGPU << <block_count, threads_per_block >> > (gpuArr1, gpuArr2, gpuResult, size);

	// acquiring validation sum
	arraySumCPU(arr1, arr2, result, size);
	
	// getting GPU answer to CPU
	int* gpuAnswer = new int[size];
	cudaMemcpy(gpuAnswer, gpuResult, sizeof(int) * size, cudaMemcpyDeviceToHost);

	// checking the results
	bool correct = true;
	for (int i = 0; i < size; i++) {
		if (gpuAnswer[i] != result[i]) {
			correct = false; 
			break;
		}
	}

	// display data
	if (correct) std::cout << "Answer is correct\n";
	else std::cout << "Incorrect answer\n";

	// synchronize gpu
	cudaDeviceSynchronize();
	cudaDeviceReset(); // free gpu resources
}