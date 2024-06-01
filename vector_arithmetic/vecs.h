#include <immintrin.h>
#include <array>

template<size_t rows, size_t columns, typename T>
static void print_arr(
    const std::array<std::array<T, columns>, rows> arr){};

template<size_t rows, size_t common, size_t columns>
std::array<std::array<float, rows>, columns> mat_mul(   
    const std::array<std::array<float, rows>, common> A, 
    const std::array<std::array<float, common>, columns> B){};