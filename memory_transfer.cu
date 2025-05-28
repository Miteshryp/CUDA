#include <stdio.h>

#include "utils/cuda_utils.cuh"
#include<stdio.h>

__global__ void readFromArray(const int* arr, int size) {
	int index = threadIdx.x + (blockDim.x * blockIdx.x);

	if (index >= size) return;

	printf("%d\n", arr[index]);
}

int main() {
    // Allocating host buffer
    int n = 40;
    int *array = (int*)malloc(sizeof(*array) * n);

    // Initialising the host array
    for(int i = 0; i < n; i++) array[i] = i+1;

    // Allocating device buffer
    int *d_array;
    cuda_assert(cudaMalloc((void**)&d_array, sizeof(*array) * n));

    // Host to device data transfer
    cuda_assert(cudaMemcpy((void*)d_array, (void*)array, sizeof(*array) * n, cudaMemcpyHostToDevice));

    // Getting device props to determine warp size
    cudaDeviceProp props = getGPUDeviceSettings();

    unsigned int no_of_blocks = ceil((float)n / props.warpSize);
    printf("No of blocks: %u\n", no_of_blocks);
    readFromArray<<<no_of_blocks, props.warpSize>>>(d_array, n);

    cuda_assert(cudaDeviceSynchronize());
    return 0;

	// We should create threads in the multiples of 32 for boosted performance on the cuda SM's
	// But our array size might not be in the multiple of 32, and sometimes may be more than a multiple
	
	// To benefit from the performance boost of thread count in multiple of 32, we should ensure that the 
	// extra threads do not access the overflowed memory.
	// To prevent this, we can pass in the size of the array into the function, and we can put a check
	// in the function to see if the index is in valid bounds.
}
