#include <iostream>
#include <emmintrin.h> 
#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <thread>
#include <vector>
#include <future>


static float best_single_thread(const float* a, const float* b, const int no_iters) {

    __m256 s_0, s_1, s_2, s_3;
    float _s[8];

    for (int j = 0; j < no_iters; j++)
        {
            s_0 = _mm256_setzero_ps();
            s_1 = _mm256_setzero_ps();
            s_2 = _mm256_setzero_ps();
            s_3 = _mm256_setzero_ps();
            for (int i = 0; i < 2048; i += 32)
            {
                s_0 = _mm256_fmadd_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i), s_0);
                s_1 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8), s_1);
                s_2 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 16), _mm256_load_ps(b + i + 16), s_2);
                s_3 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 24), _mm256_load_ps(b + i + 24), s_3);
            }
            s_0 = _mm256_add_ps(s_0, s_1);
            s_0 = _mm256_add_ps(s_0, s_2);
            s_0 = _mm256_add_ps(s_0, s_3);

            _mm256_storeu_ps(_s, s_0);
            for (int i = 1; i < 8; i++)
            {
                _s[0] += _s[i];
            }
        }

    return *_s;
}

static float best_single_thread512(const float* a, const float* b, const int no_iters) {

    __m512 s_0, s_1, s_2, s_3;
    float _s[16];

    for (int j = 0; j < no_iters; j++)
        {
            s_0 = _mm512_setzero_ps();
            s_1 = _mm512_setzero_ps();
            s_2 = _mm512_setzero_ps();
            s_3 = _mm512_setzero_ps();
            for (int i = 0; i < 2048; i += 64)
            {
                s_0 = _mm512_fmadd_ps(_mm512_load_ps(a + i), _mm512_load_ps(b + i), s_0);
                s_1 = _mm512_fmadd_ps(_mm512_load_ps(a + i + 16), _mm512_load_ps(b + i + 16), s_1);
                s_2 = _mm512_fmadd_ps(_mm512_load_ps(a + i + 32), _mm512_load_ps(b + i + 32), s_2);
                s_3 = _mm512_fmadd_ps(_mm512_load_ps(a + i + 48), _mm512_load_ps(b + i + 48), s_3);
            }
            s_0 = _mm512_add_ps(s_0, s_1);
            s_0 = _mm512_add_ps(s_0, s_2);
            s_0 = _mm512_add_ps(s_0, s_3);

            _mm512_storeu_ps(_s, s_0);
            for (int i = 1; i < 16; i++)
            {
                _s[0] += _s[i];
            }
        }

    return *_s;
}

/*
static float best_single_thread_c(const float* cache, const int no_iters) {

    __m256 s_0, s_1, s_2, s_3;
    float _s[8];

    for (int j = 0; j < no_iters; j++)
    {
        s_0 = _mm256_setzero_ps();
        s_1 = _mm256_setzero_ps();
        s_2 = _mm256_setzero_ps();
        s_3 = _mm256_setzero_ps();
        for (int i = 0; i < 4096; i += 64)
        {
            s_0 = _mm256_fmadd_ps(_mm256_load_ps(cache + i), _mm256_load_ps(cache + 8 + i), s_0);
            s_1 = _mm256_fmadd_ps(_mm256_load_ps(cache + i + 16), _mm256_load_ps(cache + i + 24), s_1);
            s_2 = _mm256_fmadd_ps(_mm256_load_ps(cache + i + 32), _mm256_load_ps(cache + i + 40), s_2);
            s_3 = _mm256_fmadd_ps(_mm256_load_ps(cache + i + 48), _mm256_load_ps(cache + i + 56), s_3);
        }
        s_0 = _mm256_add_ps(s_0, s_1);
        s_0 = _mm256_add_ps(s_0, s_2);
        s_0 = _mm256_add_ps(s_0, s_3);

        _mm256_storeu_ps(_s, s_0);
        for (int i = 1; i < 8; i++)
        {
            _s[0] += _s[i];
        }
    }

    return *_s;
}
*/
int main(int argc, char* argv[])
{
    float _a[2048], _b[2048];
    for (int i = 0; i < 2048; i++)
    {
        _a[i] = i;
        _b[i] = 2047.0 - i;
    }

    std::chrono::steady_clock::time_point begin, end;
    
    {   
        const float *a = _a, *b = _b;
        float s;
        begin = std::chrono::steady_clock::now();
        for (int j = 0; j < 200000; j++)
        {
            s = 0;
            for (int i = 0; i < 2048; i++)
            {
                s += a[i] * b[i];
            }
        }
        end = std::chrono::steady_clock::now();
        std::cout << "RAW Implementation value:\t\t" << std::setprecision(10) << s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }

    {
        const float *a = _a, *b = _b;
        __m128 s;
        float _s[4];
        begin = std::chrono::steady_clock::now();
        for (int j = 0; j < 200000; j++)
        {
            s = _mm_setzero_ps();
            for (int i = 0; i < 2048; i += 4)
            {
                s = _mm_add_ps(s, _mm_mul_ps(_mm_load_ps(a + i), _mm_load_ps(b + i)));
            }
            _mm_storeu_ps(_s, s);
            for (int i = 1; i < 4; i++)
            {
                _s[0] += _s[i];
            }
        }
        end = std::chrono::steady_clock::now();
        std::cout << "SSE Instruction Set value:\t\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }

    {
        const float *a = _a, *b = _b;
        __m256 s;
        float _s[8];
        begin = std::chrono::steady_clock::now();
        for (int j = 0; j < 200000; j++)
        {
            s = _mm256_setzero_ps();
            for (int i = 0; i < 2048; i += 8)
            {
                s = _mm256_add_ps(s, _mm256_mul_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
            }
            _mm256_storeu_ps(_s, s);
            for (int i = 1; i < 8; i++)
            {
                _s[0] += _s[i];
            }
        }
        end = std::chrono::steady_clock::now();
        std::cout << "AVX Instruction Set value:\t\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }

    /*{
        const float *a = _a, *b = _b;
        __m512 s;
        float _s[16];
        begin = std::chrono::steady_clock::now();
        for (int j = 0; j < 200000; j++)
        {
            s = _mm512_setzero_ps();
            for (int i = 0; i < 2048; i += 16)
            {
                s = _mm512_add_ps(s, _mm512_mul_ps(_mm512_load_ps(a + i), _mm512_load_ps(b + i)));
            }
            _mm512_storeu_ps(_s, s);
            for (int i = 1; i < 16; i++)
            {
                _s[0] += _s[i];
            }
        }
        end = std::chrono::steady_clock::now();
        std::cout << "512 value:\t\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }*/

    std::cout << "-------------------------\n";

    /* {
        const float *a = _a, *b = _b;
        float s_0, s_1;
        begin = std::chrono::steady_clock::now();
        for (int j = 0; j < 200000; j++)
        {
            s_0 = 0;
            s_1 = 0;
            for (int i = 0; i < 2048; i += 2)
            {
                s_0 += a[i] * b[i];
                s_1 += a[i + 1] * b[i + 1];
            }
            s_0 += s_1;
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Default Chain Independet value:\t\t" << std::setprecision(10) << s_0
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }*/

    {
        const float *a = _a, *b = _b;
        __m256 s_0, s_1;
        float _s[8];
        begin = std::chrono::steady_clock::now();
        for (int j = 0; j < 200000; j++)
        {
            s_0 = _mm256_setzero_ps();
            s_1 = _mm256_setzero_ps();
            for (int i = 0; i < 2048; i += 16)
            {
                s_0 = _mm256_add_ps(s_0, _mm256_mul_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
                s_1 = _mm256_add_ps(s_1, _mm256_mul_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8)));
            }
            s_0 = _mm256_add_ps(s_0, s_1);
            _mm256_storeu_ps(_s, s_0);
            for (int i = 1; i < 8; i++)
            {
                _s[0] += _s[i];
            }
        }
        end = std::chrono::steady_clock::now();
        std::cout << "AVX Chain Independent x2 value:\t\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }

    {
        const float *a = _a, *b = _b;
        __m256 s_0, s_1, s_2;
        float _s[8];
        begin = std::chrono::steady_clock::now();
        for (int j = 0; j < 200000; j++)
        {
            s_0 = _mm256_setzero_ps();
            s_1 = _mm256_setzero_ps();
            s_2 = _mm256_setzero_ps();
            for (int i = 0; i < 2040; i += 24)
            {
                s_0 = _mm256_add_ps(s_0, _mm256_mul_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
                s_1 = _mm256_add_ps(s_1, _mm256_mul_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8)));
                s_2 = _mm256_add_ps(s_2, _mm256_mul_ps(_mm256_load_ps(a + i + 16), _mm256_load_ps(b + i + 16)));
            }
            s_0 = _mm256_add_ps(s_0, _mm256_mul_ps(_mm256_load_ps(a + 2040), _mm256_load_ps(b + 2040)));

            s_0 = _mm256_add_ps(s_0, s_1);
            s_0 = _mm256_add_ps(s_0, s_2);
            _mm256_storeu_ps(_s, s_0);
            for (int i = 1; i < 8; i++)
            {
                _s[0] += _s[i];
            }
        }
        end = std::chrono::steady_clock::now();
        std::cout << "AVX Chain Independent x3 value:\t\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }

    {
        const float *a = _a, *b = _b;
        __m256 s_0, s_1, s_2, s_3;
        float _s[8];
        begin = std::chrono::steady_clock::now();
        for (int j = 0; j < 200000; j++)
        {
            s_0 = _mm256_setzero_ps();
            s_1 = _mm256_setzero_ps();
            s_2 = _mm256_setzero_ps();
            s_3 = _mm256_setzero_ps();
            for (int i = 0; i < 2048; i += 32)
            {
                s_0 = _mm256_add_ps(s_0, _mm256_mul_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
                s_1 = _mm256_add_ps(s_1, _mm256_mul_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8)));
                s_2 = _mm256_add_ps(s_2, _mm256_mul_ps(_mm256_load_ps(a + i + 16), _mm256_load_ps(b + i + 16)));
                s_3 = _mm256_add_ps(s_3, _mm256_mul_ps(_mm256_load_ps(a + i + 24), _mm256_load_ps(b + i + 24)));
            }
            s_0 = _mm256_add_ps(s_0, s_1);
            s_0 = _mm256_add_ps(s_0, s_2);
            s_0 = _mm256_add_ps(s_0, s_3);

            _mm256_storeu_ps(_s, s_0);
            for (int i = 1; i < 8; i++)
            {
                _s[0] += _s[i];
            }
        }
        end = std::chrono::steady_clock::now();
        std::cout << "AVX Chain Independent x4 value:\t\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }

    std::cout << "-------------------------\n";

    {
        const float *a = _a, *b = _b;
        __m256 s;
        float _s[8];
        begin = std::chrono::steady_clock::now();
        for (int j = 0; j < 200000; j++)
        {
            s = _mm256_setzero_ps();
            for (int i = 0; i < 2048; i += 8)
            {
                s = _mm256_fmadd_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i), s);
            }
            _mm256_storeu_ps(_s, s);
            for (int i = 1; i < 8; i++)
            {
                _s[0] += _s[i];
            }
        }
        end = std::chrono::steady_clock::now();
        std::cout << "FMA3 mulladd value:\t\t\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }

    {
        const float *a = _a, *b = _b;
        __m256 s_0, s_1, s_2, s_3;
        float _s[8];

        begin = std::chrono::steady_clock::now();
        for (int j = 0; j < 200000; j++)
        {
            s_0 = _mm256_setzero_ps();
            s_1 = _mm256_setzero_ps();
            s_2 = _mm256_setzero_ps();
            s_3 = _mm256_setzero_ps();
            for (int i = 0; i < 2048; i += 32)
            {
                s_0 = _mm256_fmadd_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i), s_0);
                s_1 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8), s_1);
                s_2 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 16), _mm256_load_ps(b + i + 16), s_2);
                s_3 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 24), _mm256_load_ps(b + i + 24), s_3);
            }
            s_0 = _mm256_add_ps(s_0, s_1);
            s_0 = _mm256_add_ps(s_0, s_2);
            s_0 = _mm256_add_ps(s_0, s_3);

            _mm256_storeu_ps(_s, s_0);
            for (int i = 1; i < 8; i++)
            {
                _s[0] += _s[i];
            }
        }
        end = std::chrono::steady_clock::now();
        std::cout << "ChIx4 FMA3 mulladd value:\t\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }

    /*{
        const float *a = _a, *b = _b;
        __m256 s_0, s_1, s_2, s_3;
        float _s[8];
        float cache[4096];
        for (int i = 0; i < 2048; i+=8)
        {
            for (int j = 0; j < 8; j++)
            {
                cache[2*i + j] = a[i + j];
                cache[2 * i + j + 8] = b[i + j];
            }
        }

        begin = std::chrono::steady_clock::now();
        for (int j = 0; j < 200000; j++)
        {
            s_0 = _mm256_setzero_ps();
            s_1 = _mm256_setzero_ps();
            s_2 = _mm256_setzero_ps();
            s_3 = _mm256_setzero_ps();
            for (int i = 0; i < 4096; i += 64)
            {
                s_0 = _mm256_fmadd_ps(_mm256_load_ps(cache + i), _mm256_load_ps(cache + 8 + i), s_0);
                s_1 = _mm256_fmadd_ps(_mm256_load_ps(cache+ i + 16), _mm256_load_ps(cache + i + 24), s_1);
                s_2 = _mm256_fmadd_ps(_mm256_load_ps(cache+ i + 32), _mm256_load_ps(cache + i + 40), s_2);
                s_3 = _mm256_fmadd_ps(_mm256_load_ps(cache+ i + 48), _mm256_load_ps(cache + i + 56), s_3);
            }
            s_0 = _mm256_add_ps(s_0, s_1);
            s_0 = _mm256_add_ps(s_0, s_2);
            s_0 = _mm256_add_ps(s_0, s_3);

            _mm256_storeu_ps(_s, s_0);
            for (int i = 1; i < 8; i++)
            {
                _s[0] += _s[i];
            }
        }
        end = std::chrono::steady_clock::now();
        std::cout << "with data reformatting value:\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }*/

    std::cout << "-------------------------\n";
   
    /*{
        const float *a = _a, *b = _b;
        float _s[8];

        begin = std::chrono::steady_clock::now();

        *_s = best_single_thread(a, b, 200000);

        end = std::chrono::steady_clock::now();
        std::cout << "Stack overhead for Single Thread value:\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }*/

    {
        const float *a = _a, *b = _b;
        float _s[8];

        std::vector<std::future<float>> threads;

        begin = std::chrono::steady_clock::now();

        for (int i = 0; i < 4; i++)
        {
            threads.push_back(std::async(best_single_thread, a, b, 200000 / 4));
        }

        for (int i = 0; i < 4; i++) {
            _s[i] = threads[i].get();
        }

        end = std::chrono::steady_clock::now();
        std::cout << "4 Threads value:\t\t\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }

    {
        const float *a = _a, *b = _b;
        float _s[8];

        std::vector<std::future<float>> threads;

        begin = std::chrono::steady_clock::now();

        for (int i = 0; i < 8; i++)
        {
            threads.push_back(std::async(best_single_thread, a, b, 200000 / 8));
        }

        for (int i = 0; i < 8; i++) {
            _s[i] = threads[i].get();
        }


        end = std::chrono::steady_clock::now();
        std::cout << "8 Threads value:\t\t\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }

    /*{
        const float *a = _a, *b = _b;
        float _s[8];

        std::vector<std::future<float>> threads;
        
        begin = std::chrono::steady_clock::now();

        float cach[4096];
        for (int i = 0; i < 2048; i += 8)
        {
            for (int j = 0; j < 8; j++)
            {
                cach[2 * i + j] = a[i + j];
                cach[2 * i + j + 8] = b[i + j];
            }
        }
        const float* cache = cach;

        for (int i = 0; i < 8; i++)
        {
            threads.push_back(std::async(best_single_thread_c, cache, 200000 / 8));
        }

        for (int i = 0; i < 8; i++) {
            _s[i] = threads[i].get();
        }

        end = std::chrono::steady_clock::now();
        std::cout << "8 Threads value:\t\t\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }*/

    {
        const float *a = _a, *b = _b;
        float _s[8];

        std::vector<std::future<float>> threads;

        begin = std::chrono::steady_clock::now();

        for (int i = 0; i < 12; i++)
        {
            threads.push_back(std::async(best_single_thread512, a, b, 200000 / 12));
        }

        for (int i = 0; i < 12; i++) {
            _s[i] = threads[i].get();
        }


        end = std::chrono::steady_clock::now();
        std::cout << "8 Threads value:\t\t\t" << std::setprecision(10) << *_s
            << "\ttime:\t" << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]\n";
    }
    
    return 0;
}
