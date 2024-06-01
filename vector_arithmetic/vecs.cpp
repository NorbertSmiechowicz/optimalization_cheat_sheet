#include <immintrin.h>
#include <array>
#include <iostream>

template<size_t rows, size_t columns, typename T>
static void print_arr(const std::array<std::array<T, columns>, rows> arr){
    for(int _row = 0; _row < rows; _row++){
        for(int _cln = 0; _cln < columns; _cln++)
        {
            std::cout<<arr[_row][_cln]<<" ";
        }
        std::cout<<"\n";
    }
}

template<size_t rows, size_t common, size_t columns>
static std::array<std::array<float, columns>, rows> mat_mul(   
        const std::array<std::array<float, common>, rows> A, 
        const std::array<std::array<float, columns>, common> B) 
{
    std::array<std::array<float, columns>, rows> C = {};

    for(size_t _row = 0; _row < rows; _row++){
        for(size_t _cln = 0; _cln < columns; _cln++){
            for(size_t _cmn = 0; _cmn < common; _cmn++)
            {  
                C[_row][_cln] += A[_row][_cmn] * B[_cmn][_row];
            }
        }
    }
    return C;
}

int main(){

    std::array<std::array<float, 10>, 3> a = {};
    std::array<std::array<float, 5>, 10> b = {};
    print_arr(mat_mul(a,b));
    return 0;
}