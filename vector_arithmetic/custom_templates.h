#include <iostream>
#include <vector>

template<typename T>
static void print_vec(
    const std::vector<T> printable)
{
    for(const auto& element : printable)
    {
        std::cout<<element<<" ";
    }
    std::cout<<"\n";
}

template<typename T>
static void print_vec(
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

