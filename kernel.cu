#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>

__global__ void hello_cuda() {
    printf("(Tx, Ty, Tz) - (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("(Bx, By, Bz) - (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("(BDx, BDy, BDz) - (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("Gx, Gy, Gz - (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

    // gridDim - Dimension of the no of thread in a block
    // blockDim - Dimension of total blocks allocated on the Streaming multiprocessor (SM)
}

__global__ void exersiceKernel() {
    printf("tx, ty, tz - (%d, %d, %d) \n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("bx, by, bz - (%d, %d, %d) \n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gx, gy, gz - (%d, %d, %d) \n", gridDim.x, gridDim.y, gridDim.z);
}

__global__ void print2Darray(const int* array) {
    int xOffset = blockIdx.x * blockDim.x;
    int yOffset = blockIdx.y * (gridDim.x * blockDim.x); // Y index * (no of elements in a row = no of blocks in row * element in each block);

    int index = threadIdx.x + xOffset + yOffset;

    printf("element %d  - (index=%d, xOffset=%d, yOffset=%d\n", array[index], index, xOffset, yOffset);
}

//int main()
//{
//    int* arr = new int[16];
//
//    for (int i = 0; i < 4; i++)
//        for (int j = 0; j < 4; j++)
//            arr[(4*i) + j] = (4 * i) + j;
//
//    dim3 grid_blocks(2, 2, 1);
//    dim3 threads_per_block(4);
//
//    int* cuda_arr;
//    cudaMalloc(&cuda_arr, sizeof(int) * 16);
//    cudaMemcpy(cuda_arr, arr, sizeof(int) * 16, cudaMemcpyHostToDevice);
//    
//    print2Darray << < grid_blocks, threads_per_block>> > (cuda_arr);
//    cudaDeviceSynchronize();
//
//    cudaFree(cuda_arr);
//    cudaDeviceReset(); // free's all the resources occupied by the device
//
//    delete[] arr;
//    return 0;
//}