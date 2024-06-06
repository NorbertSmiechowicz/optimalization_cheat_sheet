#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

template<typename T>
class cuda_uptr
{
public:

    cuda_uptr() noexcept : cuda_uptr{ nullptr } {}
    explicit cuda_uptr(size_t size) noexcept : 
        dev_size{ size }
    {
        cudaError_t state = cudaMalloc((void**)&dev_pointer, dev_size * sizeof(T));
        if (state == cudaErrorMemoryAllocation) 
        {
            dev_pointer = nullptr;
        }
    }

    // copy constructors
    cuda_uptr(const cuda_uptr&) = delete;
    cuda_uptr& operator=(const cuda_uptr&) = delete;

    // move constructors
    cuda_uptr(cuda_uptr&& other) noexcept : dev_pointer{ other.release() } {}
    cuda_uptr& operator=(cuda_uptr&& other) noexcept 
    {
        if (this != &other) 
        {
            reset(other.release());
        }
        return *this;
    }

    ~cuda_uptr() noexcept 
    {
        cudaFree(dev_pointer);
    }

    T* release() noexcept { return std::exchange(dev_pointer, nullptr); }
    cudaError_t reset(T* ptr_to_assign = nullptr) noexcept
    {
        cudaError_t errcode = cudaFree(dev_pointer);
        if(errcode != ) dev_pointer = ptr_to_assign;
        return errcode;
    }
    size_t size() { return dev_size; }
    T* data() { return dev_pointer; }


    inline cudaError_t load_from_host(const T *src, const size_t items)
    {
        size_t memory_region = items > dev_size ? dev_size : items;
        memory_region *= sizeof(T);
        return cudaMemcpy(dev_pointer, src, memory_region, cudaMemcpyHostToDevice);
    }

    inline cudaError_t load_from_device(const T* src, const size_t items)
    {
        size_t memory_region = items > dev_size ? dev_size : items;
        memory_region *= sizeof(T);
        return cudaMemcpy(dev_pointer, src, memory_region, cudaMemcpyDeviceToDevice);
    }

    inline cudaError_t load_into_host(T *dst, const size_t items)
    {
        size_t memory_region = items > dev_size ? dev_size : items;
        memory_region *= sizeof(T);
        return cudaMemcpy(dst, dev_pointer, memory_region, cudaMemcpyDeviceToHost);
    }

private:
    T* dev_pointer;
    size_t dev_size;
};


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

    /*
    {
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

        float* d_A, * d_B, * d_C;
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

        for (int i = 0; i < 1000; i++)
        {
            vec_conv << <1, thread_count >> > (d_A, d_B, size, d_C);
        }

        auto end = std::chrono::steady_clock::now();

        std::cout << "Default CUDA Time:\t\t\t"
            << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() << "[us]\n";

        cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    */

    {
        std::cout << "\ncreating two cuda pointers\n";
        cuda_uptr<float> a(64);
        cuda_uptr<float> b(64);
        std::cout << a.data() << "\n" << b.data();
    }

    {
        std::cout << "\nnew scope to test for freeing memory\n";
        cuda_uptr<float> a(64);
        cuda_uptr<float> b(64);
        std::cout << a.data() << "\n" << b.data() << "\n\n";
        

        std::cout << "\ntesting for data transfers\n";

        std::vector<float> v(10, 1.0);
        print_vec(v);

        a.load_from_host(v.data(), v.size());

        std::vector<float> w(10, 0.0);
        print_vec(w);

        a.load_into_host(w.data(), 5);
        print_vec(w);
    }

    {
        std::cout << "\ntesting memory content after freeing\n";
        cuda_uptr<float> a(64);
        std::cout << a.data() << "\n\n";

        std::vector<float> v(48);
        print_vec(v);

        a.load_into_host(v.data(), v.size());
        print_vec(v);

        // won't compile as it should
        // 
        // std::vector<int> i(32);
        // a.load_from_host(i.data(), 32);
        //
        // auto b = a;
    }
    return 0;
}