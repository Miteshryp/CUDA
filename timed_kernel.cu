// Timing the kernel and CPU runtime
#include "utils/cuda_utils.cuh"

__global__ void kernel(int* A, int *B, int* res, int n) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < n) {
        res[index] = A[index] + B[index];
    }
}

void CPU_kernel(int* A, int* B, int* res, int n) {
    for(int i = 0; i < n; i++) {
        res[i] = A[i] + B[i];
    }
}

bool check_results(int* A, int* B, int n) {
    for(int i = 0; i < n; i++) {
        if(A[i] != B[i]) return false;
    }
    return true;
}


int main() {
    int n = 100000000;

    // Creating host buffers
    clock_t cpu_allocation_start = clock();
    int *h_array_A = (int*)malloc(sizeof(*h_array_A) * n);
    int *h_array_B = (int*)malloc(sizeof(*h_array_B) * n);
    int *h_array_res = (int*)calloc(n, sizeof(*h_array_B));
    clock_t cpu_allocation_end = clock();

    for(int i = 0; i < n; i++){
        h_array_A[i] = i+1;
        h_array_B[i] = i+5;
    }

    // Allocating device buffers
    int *d_array_A, *d_array_B, *d_array_res;

    clock_t gpu_allocation_start = clock();
    cuda_assert(cudaMalloc((void**)&d_array_A, sizeof(*h_array_A) * n));
    cuda_assert(cudaMalloc((void**)&d_array_B, sizeof(*h_array_B) * n));
    cuda_assert(cudaMalloc((void**)&d_array_res, sizeof(*h_array_res) * n));
    clock_t gpu_allocation_end = clock();
    
    // Transferring data from host to device
    clock_t htd_transfer_start = clock();
    cuda_assert(cudaMemcpy(d_array_A, h_array_A, sizeof(*h_array_A) * n, cudaMemcpyHostToDevice));
    cuda_assert(cudaMemcpy(d_array_B, h_array_B, sizeof(*h_array_B) * n, cudaMemcpyHostToDevice));
    clock_t htd_transfer_end = clock();

    // Fetching GPU parameters
    cudaDeviceProp gpu = getGPUDeviceSettings();
    
    // Launching kernel
    clock_t gpu_kernel_start = clock();

    int threads_per_block = gpu.warpSize * 2;
    int no_of_blocks = ceil(n / threads_per_block);
    kernel<<<no_of_blocks, threads_per_block>>>(d_array_A, d_array_B, d_array_res, n);

    cuda_assert(cudaDeviceSynchronize());

    clock_t gpu_kernel_end = clock();

    // Transferring results from device to host
    clock_t dth_transfer_start = clock();
    cuda_assert(cudaMemcpy(h_array_res, d_array_res, sizeof(int) * n, cudaMemcpyDeviceToHost));
    clock_t dth_transfer_end = clock();
        
    // Launching CPU kernel
    int* cpu_results = (int*)calloc(n, sizeof(*cpu_results));
    clock_t cpu_kernel_start = clock();
    CPU_kernel(h_array_A, h_array_B, cpu_results, n);
    clock_t cpu_kernel_end = clock();

    // Check correctness
    if(check_results(h_array_res, cpu_results, n) == false) printf("Wrong results\n");
    else printf("Results are correct\n");

    // Printing time results
    double gpu_allocation_time = (double)((double)(gpu_allocation_end - gpu_allocation_start) / CLOCKS_PER_SEC);
    double cpu_allocation_time = (double)((double)(cpu_allocation_end - cpu_allocation_start) / CLOCKS_PER_SEC);
    double htd_transfer_time = (double)((double)(htd_transfer_end - htd_transfer_start) / CLOCKS_PER_SEC);
    double gpu_kernel_time = (double)((double)(gpu_kernel_end - gpu_kernel_start) / CLOCKS_PER_SEC);
    double dth_transfer_time = (double)((double)(dth_transfer_end - dth_transfer_start) / CLOCKS_PER_SEC);
    double cpu_kernel_time = (double)((double)(cpu_kernel_end - cpu_kernel_start) / CLOCKS_PER_SEC);

    printf("GPU Allocation time: %4.6f\n", gpu_allocation_time);
    printf("Host to Device transfer time: %4.6f\n", htd_transfer_time);
    printf("GPU kernel execution time: %4.6f\n", gpu_kernel_time);
    printf("Device to Host transfer time: %4.6f\n", dth_transfer_time);

    printf("CPU Allocation time: %4.6f\n", cpu_allocation_time);
    printf("CPU kernel execution time: %4.6f\n", cpu_kernel_time);

    printf("\n\n");

    double total_gpu_time = gpu_allocation_time + htd_transfer_time + gpu_kernel_time + dth_transfer_time;
    printf("Total GPU Time (memory + kernel latency): %4.6f\n", total_gpu_time);
    printf("Memory overhead (percentage): %3.2f", 100.f*((htd_transfer_time + dth_transfer_time + gpu_allocation_time) / total_gpu_time));
}
