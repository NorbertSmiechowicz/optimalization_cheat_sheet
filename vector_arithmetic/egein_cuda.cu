#include<iostream>
#include<cuda_runtime.h>
#include<Eigen/Dense>

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

    cuda_uptr(const cuda_uptr&) = delete;
    cuda_uptr& operator=(const cuda_uptr&) = delete;

    T* release() noexcept { return std::exchange(dev_pointer, nullptr); }
    cudaError_t reset(T* ptr_to_assign = nullptr) noexcept
    {
        cudaError_t errcode = cudaFree(dev_pointer);
        if (errcode == cudaSuccess) dev_pointer = ptr_to_assign;
        return errcode;
    }

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
        cudaError_t state = cudaFree(dev_pointer);
    }

    size_t size() { return dev_size; }
    T* data() { return dev_pointer; }

private:
    T* dev_pointer;
    size_t dev_size;
};

template<typename T, size_t rows, size_t common, size_t columns>
__global__ void mult(
    const Eigen::Matrix<T, rows, common>* A,
    const Eigen::Matrix<T, common, columns>* B,
    Eigen::Matrix<T, rows, columns>* C)
{
    *C = (*A) * (*B);
    return;
}

int main()
{
    Eigen::Matrix<float, 3, 3> A {
        {0, 1, 2 },
        {3, 4, 5 },
        {6, 7, 8 }
    };
    Eigen::Matrix<float, 3, 3> B {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };

    Eigen::Matrix<float, 3, 3> C;

    //mult(&A, &B, &C);
    //std::cout << C;

    cuda_uptr<Eigen::Matrix<float, 3, 3>> cuda_A(sizeof(A)), cuda_B(sizeof(B)), cuda_C(sizeof(C));
    cudaMemcpy(cuda_A.data(), &A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B.data(), &B, sizeof(B), cudaMemcpyHostToDevice);

    mult << <1, 1 >> > (cuda_A.data(), cuda_B.data(), cuda_C.data());


    cudaMemcpy(&C, cuda_C.data(), sizeof(C), cudaMemcpyDeviceToHost);

    std::cout << C;
}