/*
 * Radix sort based on least significant bit
 */

#include <iostream>
#include <algorithm>
#include <cassert>


template <typename T>
void compare(const T* const test, const T* const ref, const int size)
{
    for (std::size_t i = 0; i < size; i++) {
        if (test[i] != ref[i]) {
            std::cerr << "Mismatch at index " << i << std::endl;
            std::cerr << "Test value = " << test[i] << std::endl;
            std::cerr << "Ref value  = " << ref[i] << std::endl;

            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Pass" << std::endl;
}

template <typename T>
void print(const T* const data, const int size)
{
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << ", ";
    }

    std::cout << std::endl;
}

const unsigned int bit_at(const unsigned int& value, const int& bitIdx)
{
    return (value >> bitIdx) & 1;
}


void histogram(unsigned int* histo, const unsigned int* const data, const int bitIdx, const int size)
{
    for (int i = 0; i < size; i++) {
        unsigned int bit = bit_at(data[i], bitIdx);
        histo[bit]++;
    }
}


/* exclusive scan
 */
void histo_to_cdf(unsigned int* cdf, const unsigned int* const histo)
{
    cdf[0] = 0;

    // bit has 2 value: 0 or 1
    cdf[1] = histo[0];
}


void move(unsigned int* output, const unsigned int* input, const int bitIdx, unsigned int* cdf, const int size)
{
    for (int i = 0; i < size; i++) {
        unsigned int bit = bit_at(input[i], bitIdx);

        int index = cdf[bit];
        assert(index >= 0);
        output[index] = input[i];

        cdf[bit]++;
    }
}


void count_sort(unsigned int* data, const int bitIdx, const int size)
{
    const int numBins = 2;

    // historgram
    unsigned int* histo = new unsigned int[numBins];
    memset(histo, 0, sizeof(unsigned int) * numBins);

    histogram(histo, data, bitIdx, size);

    // cdf
    unsigned int* cdf = new unsigned int[numBins];
    memset(cdf, 0, sizeof(unsigned int) * numBins);

    histo_to_cdf(cdf, histo);

    // move data
    unsigned int* tmp = new unsigned int[size];
    move(tmp, data, bitIdx, cdf, size);

    memcpy(data, tmp, sizeof(unsigned int) * size);

    if (histo) free(histo);
    if (cdf) free(cdf);
    if (tmp) free(tmp);
}


void radix_sort_lsb(const unsigned int* input, unsigned int* output, const int size)
{
    unsigned int totalBits = size * sizeof(unsigned int);

    for (int i = 0; i < totalBits; i++) {
        count_sort(output, i, size);
    }
}


int main()
{
    const int size = 7;
    const unsigned int input[] = { 1 << 8, 1 << 2, 1 << 3, 1 << 2, 1 << 10, 1 << 5, 1 << 12 };
    unsigned int output[size];
    unsigned int output_std[size];


    //unsigned int 
    std::cout << "Origianl data         = ";
    print(input, size);


    // sort using radix sort based on lsb
    std::copy(input, input + size, output);
    radix_sort_lsb(input, output, size);

    std::cout << "Sorted data using radix = ";

    print(output, size);

    // sort using std
    std::copy(input, input + size, output_std);

    std::sort(output_std, output_std + size);

    std::cout << "Sorted data using std = ";
    print(output_std, size);


    compare<unsigned int>(output, output_std, size);


	return 0;
}
