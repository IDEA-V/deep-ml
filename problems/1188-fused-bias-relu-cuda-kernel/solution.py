#include <cuda_runtime.h>

__global__ void kernel(const float* input, const float* bias, float* output, int rows, int cols) {
    // TODO
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    if (row < rows && col < cols) {
        const float sum = input[row*cols+col] + bias[col];

        if (sum > 0) {
            output[row*cols+col] = sum;
        }else{
            output[row*cols+col] = 0.0;
        }
    }

}

void solve(const float* input, const float* bias, float* output, int rows, int cols) {
    // TODO: allocate device memory, copy in, launch kernel, copy out, free
    size_t size_A = rows * cols * sizeof(float);
    size_t size_b = cols * sizeof(float);

    float *d_A, *d_b, *d_C;

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_b, size_b);
    cudaMalloc((void**)&d_C, size_A);

    cudaMemcpy(d_A, input, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, bias, size_b, cudaMemcpyHostToDevice);

    dim3 grid(
        (cols + 31) / 32,
        (rows + 31) / 32
    );
    dim3 block(32, 32);

    kernel<<<grid, block>>>(d_A, d_b, d_C, rows, cols);

    cudaMemcpy(output, d_C, size_A, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_C);

}