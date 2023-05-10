#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include<stdio.h>

//__global__ void readFromArray(const int* arr, int size) {
//	int index = threadIdx.x + (blockDim.x * blockIdx.x);
//
//	if (index >= size) return;
//
//	printf("%d\n", arr[index]);
//}

//int main() {
//	// We should create threads in the multiples of 32 for boosted performance on the cuda SM's
//	// But our array size might not be in the multiple of 32, and sometimes may be more than a multiple
//	
//	// To benefit from the performance boost of thread count in multiple of 32, we should ensure that the 
//	// extra threads do not access the overflowed memory.
//	// To prevent this, we can pass in the size of the array into the function, and we can put a check
//	// in the function to see if the index is in valid bounds.
//
//
//}