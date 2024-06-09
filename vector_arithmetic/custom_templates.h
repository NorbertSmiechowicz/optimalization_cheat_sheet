#include <iostream>
#include <vector>
#include <emmintrin.h> 
#include <immintrin.h>


namespace nla{

template<typename T>
static void print(
    const std::vector<T> printable)
{
    for(const auto& element : printable)
    {
        std::cout<<element<<" ";
    }
    std::cout<<"\n";
}

template<typename T>
static void print(
    const std::vector<std::vector<T>> printable)
{
    for(const auto& row : printable){
        for(const auto& column : row)
        {
            std::cout<<column<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";
}

class vec8f32
{
public:
    __m256 simd_reg;

    // default constructors
    vec8f32() = default;
    ~vec8f32() = default;

    // array initialization
    vec8f32(const float* float_array) : simd_reg{_mm256_load_ps(float_array)} {};
    vec8f32& operator=(const float* float_array) { return *this = vec8f32(float_array); }

    // simd initialization
    vec8f32(const __m256 simd_register) { simd_reg = simd_register; }
    vec8f32& operator=(const __m256 simd_register) { return *this = vec8f32(simd_register); }

    // memory operations
    inline void clear() { simd_reg = _mm256_setzero_ps(); }
    inline void set(float value) {
        float temp[8] = {value, value, value, value, value, value, value, value};
        simd_reg = _mm256_load_ps(temp); 
    }
    inline void store(float* destination) { _mm256_storeu_ps(destination, simd_reg); };
    inline float sum() {
        float temp[8];
        _mm256_storeu_ps(temp, _mm256_hadd_ps(simd_reg, simd_reg));
        return (temp[0] + temp[2]) + (temp[4] + temp[6]);
    };

    // arithmetic operations
    inline vec8f32 operator + (const vec8f32 other) { return _mm256_add_ps(simd_reg, other.simd_reg); }
    inline vec8f32 operator + (const __m256 other) { return _mm256_add_ps(simd_reg, other); }
    inline vec8f32 operator + (const float* other) { return _mm256_add_ps(simd_reg, _mm256_load_ps(other)); }

    inline vec8f32 operator - (const vec8f32 other) { return _mm256_sub_ps(simd_reg, other.simd_reg); }
    inline vec8f32 operator - (const __m256 other) { return _mm256_sub_ps(simd_reg, other); }
    inline vec8f32 operator - (const float* other) { return _mm256_sub_ps(simd_reg, _mm256_load_ps(other)); }

    inline vec8f32 operator * (const vec8f32 other) { return _mm256_mul_ps(simd_reg, other.simd_reg); }
    inline vec8f32 operator * (const __m256 other) { return _mm256_mul_ps(simd_reg, other); }
    inline vec8f32 operator * (const float* other) { return _mm256_mul_ps(simd_reg, _mm256_load_ps(other)); }

    inline vec8f32 operator / (const vec8f32 other) { return _mm256_div_ps(simd_reg, other.simd_reg); }
    inline vec8f32 operator / (const __m256 other) { return _mm256_div_ps(simd_reg, other); }
    inline vec8f32 operator / (const float* other) { return _mm256_div_ps(simd_reg, _mm256_load_ps(other)); }

    inline void addmult(const vec8f32 vec1, const vec8f32 vec2) { 
        simd_reg = _mm256_fmadd_ps(vec1.simd_reg, vec2.simd_reg, simd_reg);
    }
    inline void addmult(const __m256 vec1, const __m256 vec2) { 
        simd_reg = _mm256_fmadd_ps(vec1, vec2, simd_reg); 
    }
    inline void addmult(const float* vec1, const float* vec2) { 
        simd_reg = _mm256_fmadd_ps(_mm256_load_ps(vec1), _mm256_load_ps(vec2), simd_reg); 
    }
};

}