#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>


__global__ void vec_conv(const float* in_vec1, const float* in_vec2, 
    size_t vec_len, float* out_vec ) 
{
    /*
    for (int a = blockIdx.x * vec_len; a < blockIdx.x * vec_len + vec_len; a++) {
        out_vec[blockIdx.x] = __fmaf_ieee_rn(in_vec1[a], in_vec2[a], out_vec[blockIdx.x]);
    }
    */
    for (int a = threadIdx.x * vec_len; a < threadIdx.x * vec_len + vec_len; a++) {
        out_vec[threadIdx.x] += in_vec1[a] * in_vec2[a];
    }
}


template<typename T>
static void print_vec(
    const std::vector<T> arr)
{
    for (int _cln = 0; _cln < arr.size(); _cln++)
    {
        std::cout << arr[_cln] << " ";
    }
    std::cout << "\n";
}

int main() {

    const size_t size = 10'000;
    const size_t thread_count = 200;

    std::vector<std::vector<float>> h_A(thread_count, std::vector<float>(size));
    std::vector<std::vector<float>> h_B(thread_count, std::vector<float>(size));
    std::vector<float> h_C(thread_count, 0);
    for (int j = 0; j < thread_count; j++) {
        for (int i = 0; i < size; i++)
        {
            h_A[j][i] = static_cast<float>(j) / 10000.0;
            h_B[j][i] = 1.0;
        }
        
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, h_A.size() * h_A[0].size() * sizeof(float));
    cudaMalloc((void**)&d_B, h_B.size() * h_B[0].size() * sizeof(float));
    cudaMalloc((void**)&d_C, h_C.size() * sizeof(float));

    int mem_region = h_A[0].size() * sizeof(float);
    for (int i = 0; i < h_A.size(); i++) 
    {
        cudaMemcpy(d_A + i * h_A[i].size(), h_A[i].data(), mem_region, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B + i * h_A[i].size(), h_B[i].data(), mem_region, cudaMemcpyHostToDevice);
    }

    auto begin = std::chrono::steady_clock::now();

    vec_conv << <1, thread_count >> > (d_A, d_B, size, d_C);

    auto end = std::chrono::steady_clock::now();

    std::cout << "Default Time:\t\t"
        << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";

    cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    print_vec(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}