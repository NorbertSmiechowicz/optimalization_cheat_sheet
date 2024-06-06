#include <iostream>
#include <vector>
#include"custom_templates.h"

int main()
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

    print_vec(a);

    for(float* i = a[0].data(); i < a[0].data() + a.size()*a[0].size(); i++){
        std::cout<<*i<<" ";
    }

    return 0;
}