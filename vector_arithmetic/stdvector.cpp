#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <vector>
#include <emmintrin.h> 
#include <immintrin.h>
#include"custom_templates.h"

int main()
{
    /*
    {
        std::cout<<sizeof(std::vector<float>)<<"\t"<<sizeof(std::vector<std::vector<float>>)<<"\n\n";
        auto a = []()
        {
            size_t rows = 16, columns = 10;
        std::vector<std::vector<float>> temp(rows, std::vector<float>(columns));
            for(int _row = 0; _row < rows; _row++){
                for(int _cln = 0; _cln < columns; _cln++)
                {
                    temp[_row][_cln] = static_cast<float>(_row*10+_cln);
                }
            }
            return temp;
        }();

        nla::print(a);

        for(float* i = a[0].data(); i < a[0].data() + a.size()*a[0].size(); i++){
            std::cout<<*i<<" ";
        }
    }
    */
    /*
    {
        //std::cout<<sizeof(std::array<std::array<float,10>, 100>);

        auto gen = [](const size_t objects, const size_t elements)
        {
            constexpr float multer = 1.0 / static_cast<float>(1<<31 - 1);
            std::vector<std::vector<float>> temp(objects, std::vector<float>(elements));
            for(auto& obj : temp){
                for(auto i = 0; i < obj.size(); i++)
                {
                    obj[i] = multer * static_cast<float>(std::rand());
                }
            }
            return temp;
        };

        auto a = gen(10'000, 1000);
        auto b = gen(10'000, 1000);
        std::vector<float> c(10'000);

        for(size_t obj = 0; obj < a.size(); obj++){
            c[obj] = std::inner_product(a[obj].begin(), a[obj].end(), b[obj].begin(), (float)0.0);
        }

        nla::print(c);

    }
    */
    {
        auto gen = [](const size_t objects, const size_t elements)
        {
            constexpr float multer = 1.0 / static_cast<float>(1<<31 - 1);
            std::vector<std::vector<float>> temp(objects, std::vector<float>(elements));
            for(auto& obj : temp){
                for(auto i = 0; i < obj.size(); i++)
                {
                    obj[i] = multer * static_cast<float>(std::rand());
                }
            }
            return temp;
        };

        auto a = gen(10'000, 8);
        auto b = gen(10'000, 8);
        
        nla::print(a[0]);
        nla::print(b[0]);
        nla::vec8f32 c(a[0].data());
        nla::vec8f32 d(b[0].data());

        c = c+d;

        _mm256_store_ps(b[0].data(), c.simd_reg);
        nla::print(b[0]);

        std::cout<<"---------\n";        

        nla::print(a[1]);
        nla::print(b[1]);
        c = a[1].data();
        d = b[1].data();

        c = c+d;

        _mm256_store_ps(b[0].data(), c.simd_reg);
        nla::print(b[0]);

    }

    return 0;
}   