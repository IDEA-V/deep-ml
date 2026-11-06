#include <cuda_runtime.h>

__global__ void kernel(const float* in, float* out, int N) {
    // TODO: block-wise reduction using shared memory
    __shared__ float sdata[256];
    int tx = threadIdx.x;
    int start = blockIdx.x*256;
    int alive = 128;

    if (start+tx < N) {
        sdata[tx] = in[start+tx];
    }else{
        sdata[tx] = 0.0f;
    }
    __syncthreads();

    for (int i=0;i<8;i++) {
        if (tx < alive) {
            sdata[tx] = sdata[tx] + sdata[tx+alive];
        }
        __syncthreads();
        alive = alive/2;
    }

    if (tx == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

void solve(const float* input, float* output, int N) {
    // TODO: allocate device memory, copy in, launch kernel, copy out, free
    size_t size_N = N * sizeof(float);

    float *d_A, *d_B;

    cudaMalloc((void**) &d_A, size_N);
    cudaMalloc((void**) &d_B, size_N);
    
    cudaMemcpy(d_A, input, size_N, cudaMemcpyHostToDevice);

    float* src = d_A;
    float* dst = d_B;
    int currentN = N;
    
    while (currentN > 1) {
        int next_N = (currentN+255)/256;

        kernel<<<next_N,256>>>(src,dst,currentN);
        currentN = next_N;

        float* tmp = src;
        src = dst;
        dst = tmp;
    }
    cudaMemcpy(output, src, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(src);
    cudaFree(dst);
}