#include <immintrin.h>
#include <array>
#include <vector>
#include <iostream>
// only used for testing in main
#include <chrono>

template<size_t length, typename T>
static void print_arr(
    const std::array<T, length> arr)
{
    for(int _len = 0; _len < length; _len++)
    {
        std::cout<<arr[_len]<<" ";
    }
    std::cout<<"\n";
}

template<size_t rows, size_t columns, typename T>
static void print_arr(
    const std::array<std::array<T, columns>, rows> arr)
{
    for(int _row = 0; _row < rows; _row++){
        for(int _cln = 0; _cln < columns; _cln++)
        {
            std::cout<<arr[_row][_cln]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";
}

template<typename T>
static void print_arr(
    const std::vector<std::vector<T>> arr)
{
    for(int _row = 0; _row < arr.size(); _row++){
        for(int _cln = 0; _cln < arr[0].size(); _cln++)
        {
            std::cout<<arr[_row][_cln]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";
}


template<size_t rows, size_t columns, typename T>
inline static void print_arr_dims(
    const std::array<std::array<T, columns>, rows> arr)
{
    std::cout<<"{ " << arr.size()<<" x "<<arr[0].size()<<" }";
}

template<typename T>
inline static void print_arr_dims(
    const std::vector<std::vector<T>> arr)
{
    std::cout<<"{ " << arr.size()<<" x "<<arr[0].size()<<" }";
}

template<size_t length>
static std::array<float, length> vec_mul(   
    const std::array<float, length> v,
    const std::array<float, length> w)
{
    std::array<float, length> u;
    auto up = u.data();
    auto vp = v.data();
    auto wp = w.data();

    //size_t len_n32 = length / 32;
    size_t len_n8 = length - (length % 8);

    for(size_t coef = 0; coef < len_n8; coef+=8)
    {
        _mm256_store_ps(up + coef, _mm256_mul_ps(_mm256_load_ps(vp + coef), _mm256_load_ps(wp + coef)));
    }

    for(size_t coef = len_n8; coef < length; coef++)
    {
        u[coef] = v[coef] * w[coef];
    }
    return u;
}

template<size_t rows, size_t common, size_t columns, typename T>
static std::array<std::array<T, columns>, rows> mat_mul(   
        const std::array<std::array<T, common>, rows> A, 
        const std::array<std::array<T, columns>, common> B) 
{
    std::array<std::array<T, columns>, rows> C = {};

    for(size_t _row = 0; _row < rows; _row++){
        for(size_t _cln = 0; _cln < columns; _cln++){
            for(size_t _cmn = 0; _cmn < common; _cmn++)
            {  
                C[_row][_cln] += A[_row][_cmn] * B[_cmn][_cln];
            }
        }
    }
    return C;
}

template<typename T>
static std::vector<std::vector<T>> mat_mul(   
        const std::vector<std::vector<T>> A, 
        const std::vector<std::vector<T>> B) 
{
    std::vector<std::vector<T>> C(A.size(), std::vector<T>(B[0].size()));

    for(size_t _row = 0; _row < A.size(); _row++){
        for(size_t _cln = 0; _cln < B[0].size(); _cln++){
            for(size_t _cmn = 0; _cmn < B.size(); _cmn++)
            {  
                C[_row][_cln] += A[_row][_cmn] * B[_cmn][_cln];
            }
        }
    }
    return C;
}

template<size_t rows, size_t columns, typename T>
static std::array<std::array<T, rows>, columns> mat_transpose(   
        const std::array<std::array<T, columns>, rows> A)
{
    std::array<std::array<T, rows>, columns> B;

    for(size_t _row = 0; _row < rows; _row++){
        for(size_t _cln = 0; _cln < columns; _cln++){
            B[_cln][_row] = A[_row][_cln];
        }
    }

    return B;
} 

template<size_t rows, size_t common, size_t columns>
static std::array<std::array<float, columns>, rows> mat_mul_1(   
        const std::array<std::array<float, common>, rows> A, 
        const std::array<std::array<float, common>, columns> B) 
{
    std::array<std::array<float, columns>, rows> C;

    for(size_t _row = 0; _row < rows; _row++){
        for(size_t _cln = 0; _cln < columns; _cln++){
            C[_row][_cln] = vec_conv_1(A[_row],B[_cln]);
        }
    }

    return C;
}

template<size_t length>
inline static float vec_conv_1(   
    const std::array<float, length> v,
    const std::array<float, length> w)
{
    auto vp = v.data();
    auto wp = w.data();
    __m256 u{};
    float uarr[8];

    //size_t len_n32 = length / 32;
    size_t len_n8 = length - (length % 8);

    for(size_t coef = 0; coef < len_n8; coef+=8)
    {
        u = _mm256_fmadd_ps(_mm256_load_ps(vp + coef), _mm256_load_ps(wp + coef), u);
    }
    _mm256_store_ps(uarr, u);

    for(uint8_t regs = 1; regs < 8; regs++){
        uarr[0] += uarr[regs];
    }

    for(size_t coef = len_n8; coef < length; coef++)
    {
        uarr[0] += v[coef] * w[coef];
    }
    return uarr[0];
}

template<size_t rows, size_t common, size_t columns>
static std::array<std::array<float, columns>, rows> mat_mul_2(   
        const std::array<std::array<float, common>, rows> A, 
        const std::array<std::array<float, common>, columns> B) 
{
    std::array<std::array<float, columns>, rows> C;

    for(size_t _row = 0; _row < rows; _row++){
        for(size_t _cln = 0; _cln < columns; _cln++){
            C[_row][_cln] = vec_conv_2(A[_row],B[_cln]);
        }
    }

    return C;
}

template<size_t length>
static float vec_conv_2(   
    const std::array<float, length> v,
    const std::array<float, length> w)
{
    auto vp = v.data();
    auto wp = w.data();
    __m256 u0{}, u1{}, u2{};
    float uarr[8];

    size_t len_n24 = length - (length % 24);
    size_t len_n8 = length - (length % 8);

    for(size_t coef = 0; coef < len_n24; coef+=24)
    {
        u0 = _mm256_fmadd_ps(_mm256_load_ps(vp + coef), _mm256_load_ps(wp + coef), u0);
        u1 = _mm256_fmadd_ps(_mm256_load_ps(vp + coef + 8), _mm256_load_ps(wp + coef + 8), u1);
        u2 = _mm256_fmadd_ps(_mm256_load_ps(vp + coef + 16), _mm256_load_ps(wp + coef + 16), u2);
    }

    for(size_t coef = len_n24; coef < len_n8; coef+=8)
    {
        u0 = _mm256_fmadd_ps(_mm256_load_ps(vp + coef), _mm256_load_ps(wp + coef), u0);
    }

    u0 = _mm256_add_ps(u0, u1);
    u0 = _mm256_add_ps(u0, u2);
    _mm256_store_ps(uarr, u0);

    for(uint8_t regs = 1; regs < 8; regs++){
        uarr[0] += uarr[regs];
    }

    for(size_t coef = len_n8; coef < length; coef++)
    {
        uarr[0] += v[coef] * w[coef];
    }
    return uarr[0];
}

template<size_t rows, size_t common, size_t columns>
static std::array<std::array<float, columns>, rows> mat_mul_3(   
        const std::array<std::array<float, common>, rows> A, 
        const std::array<std::array<float, common>, columns> B) 
{
    std::array<std::array<float, columns>, rows> C;

    for(size_t _row = 0; _row < rows; _row++){
        for(size_t _cln = 0; _cln < columns; _cln++)
        {
            C[_row][_cln] = vec_conv_3(A[_row].data(),B[_cln].data(),A[_row].size());
        }
    }

    return C;
}

static float vec_conv_3(   
    const float *v, const float *w, size_t length)
{
    __m256 u0{}, u1{}, u2{};
    float uarr[8];

    size_t len_n24 = length - (length % 24);
    size_t len_n8 = length - (length % 8);

    for(size_t coef = 0; coef < len_n24; coef+=24)
    {
        u0 = _mm256_fmadd_ps(_mm256_load_ps(v + coef), _mm256_load_ps(w + coef), u0);
        u1 = _mm256_fmadd_ps(_mm256_load_ps(v + coef + 8), _mm256_load_ps(w + coef + 8), u1);
        u2 = _mm256_fmadd_ps(_mm256_load_ps(v + coef + 16), _mm256_load_ps(w + coef + 16), u2);
    }

    for(size_t coef = len_n24; coef < len_n8; coef+=8)
    {
        u0 = _mm256_fmadd_ps(_mm256_load_ps(v + coef), _mm256_load_ps(w + coef), u0);
    }

    u0 = _mm256_add_ps(u0, u1);
    u0 = _mm256_add_ps(u0, u2);
    _mm256_store_ps(uarr, u0);

    for(uint8_t regs = 1; regs < 8; regs++){
        uarr[0] += uarr[regs];
    }

    for(size_t coef = len_n8; coef < length; coef++)
    {
        uarr[0] += v[coef] * w[coef];
    }
    return uarr[0];
}

static std::vector<std::vector<float>> mat_mul_3(   
        const std::vector<std::vector<float>> A, 
        const std::vector<std::vector<float>> B) 
{
    std::vector<std::vector<float>> C(A.size(), std::vector<float>(B.size()));

    for(size_t row = 0; row < A.size(); row++){
        for(size_t column = 0; column < B.size(); column++)
        {
            C[row][column] = vec_conv_3(A[row].data(),B[column].data(),A[row].size());
        }
    }

    return C;
}

template<typename T>
static std::vector<std::vector<T>> mat_transpose(   
        const std::vector<std::vector<T>> A)
{
    std::vector<std::vector<T>> B(A[0].size(), std::vector<float>(A.size()));

    for(size_t _row = 0; _row < A.size(); _row++){
        for(size_t _cln = 0; _cln < A[0].size(); _cln++){
            B[_cln][_row] = A[_row][_cln];
        }
    }

    return B;
} 

int main(){

    std::chrono::steady_clock::time_point begin, end;

    /*{
        auto a = [](){
            const size_t rows = 3, columns = 4;
            std::array<std::array<float, columns>, rows> _a;
            for(int i = 0; i < rows; i++){
                for(int j = 0; j<columns; j++)
                {
                    _a[i][j] = i;
                }
            }
            return _a;
        }();

        auto b = [](){
            const size_t rows = 4, columns = 2;
            std::array<std::array<float, columns>, rows> _a;
            for(int i = 0; i < rows; i++){
                for(int j = 0; j<columns; j++)
                {
                    _a[i][j] = j;
                }
            }
            return _a;
        }();

        print_arr(a);
        print_arr(b);
        print_arr(mat_mul(a,b));
    }*/

    /*{
        auto v = [](){
            const size_t columns = 2000;
            std::array<float, columns> _a;
                for(int j = 0; j<columns; j++)
                {
                    _a[j] = j;
                }
            return _a;
        }();

        auto w = [](){
            const size_t columns = 2000;
            std::array<float, columns> _a;
                for(int j = 0; j<columns; j++)
                {
                    _a[j] = 1999 - j;
                }
            return _a;
        }();

        print_arr(vec_mul(v,w));
    }*/

    /*{
        auto v = [](){
            const size_t columns = 2000;
            std::array<float, columns> _a;
                for(int j = 0; j<columns; j++)
                {
                    _a[j] = 1;
                }
            return _a;
        }();

        auto w = [](){
            const size_t columns = 2000;
            std::array<float, columns> _a;
                for(int j = 0; j<columns; j++)
                {
                    _a[j] = 2;
                }
            return _a;
        }();

        std::cout<<vec_conv(v,w);
    }*/

    {
        auto a = [](){
            const size_t rows = 100, columns = 200;
            std::array<std::array<float, columns>, rows> _a;
            for(int i = 0; i < rows; i++){
                for(int j = 0; j<columns; j++)
                {
                    _a[i][j] = static_cast<float>(i);
                }
            }
            return _a;
        }();

        auto b = [](){
            const size_t rows = 200, columns = 100;
            std::array<std::array<float, columns>, rows> _a;
            for(int i = 0; i < rows; i++){
                for(int j = 0; j<columns; j++)
                {
                    _a[i][j] = static_cast<float>(j);
                }
            }
            return _a;
        }();

        const size_t repeats = 1'000;

        std::cout<<"------------------------------------------------------\n";
        std::cout<<"# of calcs " << repeats << ", A : ";
        print_arr_dims(a);
        std::cout<<"  B : ";
        print_arr_dims(b);
        std::cout<<"\n------------------------------------------------------\n";

        begin = std::chrono::steady_clock::now();

        auto c = mat_mul(a,b);
        for(size_t i = 0; i < repeats; i++){
            c = mat_mul(a,b);
        }

        end = std::chrono::steady_clock::now();
        std::cout << "Default Time:\t\t\t\t" 
            << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() << "[us]\n";
        

        begin = std::chrono::steady_clock::now();

        auto bt = mat_transpose(b);
        c = mat_mul_1(a,bt);
        for(size_t i = 0; i < repeats; i++){
            bt = mat_transpose(b);
            c = mat_mul_1(a,bt);
        }

        end = std::chrono::steady_clock::now();
        std::cout << "1st Optim Time:\t\t\t\t" 
            << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() << "[us]\n";

        begin = std::chrono::steady_clock::now();

        bt = mat_transpose(b);
        c = mat_mul_2(a,bt);
        for(size_t i = 0; i < repeats; i++){
            bt = mat_transpose(b);
            c = mat_mul_2(a,bt);
        }

        end = std::chrono::steady_clock::now();
        std::cout << "2st Optim Time:\t\t\t\t" 
            << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() << "[us]\n";

        begin = std::chrono::steady_clock::now();

        c = mat_mul_3(a,mat_transpose(b));
        for(size_t i = 0; i < repeats; i++){
            c = mat_mul_3(a,mat_transpose(b));
        }

        end = std::chrono::steady_clock::now();
        std::cout << "3st Optim Time:\t\t\t\t" 
            << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() << "[us]\n";
    }

    /*{   // testing for correctness of outputs
        
        auto a = [](){
            const size_t rows = 10, columns = 10;
            std::vector<std::vector<float>> _a(rows, std::vector<float>(columns));
            for(int i = 0; i < rows; i++){
                for(int j = 0; j<columns; j++)
                {
                    _a[i][j] = i;
                }
            }
            return _a;
        }();

        auto b = [](){
            const size_t rows = 10, columns = 10;
            std::vector<std::vector<float>> _a(rows, std::vector<float>(columns));
            for(int i = 0; i < rows; i++){
                for(int j = 0; j<columns; j++)
                {
                    _a[i][j] = j;
                }
            }
            return _a;
        }();
        print_arr(a);
        print_arr(b);
        print_arr(mat_mul_3(a,mat_transpose(b)));
    }*/

    {
        auto a = [](){
            const size_t rows = 100, columns = 200;
            std::vector<std::vector<float>> _a(rows, std::vector<float>(columns));
            for(int i = 0; i < rows; i++){
                for(int j = 0; j<columns; j++)
                {
                    _a[i][j] = static_cast<float>(i);
                }
            }
            return _a;
        }();

        auto b = [](){
            const size_t rows = 200, columns = 100;
            std::vector<std::vector<float>> _a(rows, std::vector<float>(columns));
            for(int i = 0; i < rows; i++){
                for(int j = 0; j<columns; j++)
                {
                    _a[i][j] = static_cast<float>(j);
                }
            }
            return _a;
        }();

        const size_t repeats = 1'000;

        std::cout<<"------------------------------------------------------\n";
        std::cout<<"# of calcs " << repeats << ", A : ";
        print_arr_dims(a);
        std::cout<<"  B : ";
        print_arr_dims(b);
        std::cout<<"\n------------------------------------------------------\n";

        begin = std::chrono::steady_clock::now();

        auto bt = mat_transpose(b);
        auto c = mat_mul_3(a,bt);
        for(size_t i = 0; i < repeats; i++){
            bt = mat_transpose(b);
            c = mat_mul(a,bt);
        }

        end = std::chrono::steady_clock::now();
        std::cout << "Default Array -> Vector Time:\t\t" 
            << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() << "[us]\n";


        begin = std::chrono::steady_clock::now();

        c = mat_mul_3(a,mat_transpose(b));
        for(size_t i = 0; i < repeats; i++){
            c = mat_mul_3(a,mat_transpose(b));
        }

        end = std::chrono::steady_clock::now();
        std::cout << "Optim3 Array -> Vector Time:\t\t" 
            << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() << "[us]\n";

    }

    return 0;
}