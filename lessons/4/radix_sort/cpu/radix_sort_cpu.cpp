#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>

template <typename T>
void compare (const std::vector<T>& values, const std::vector<T>& ref_values)
{
	if (values.size() != ref_values.size()) {
		std::cerr << "Mismatch size of values and ref_values on line " << __func__ << std::endl;
		exit(EXIT_FAILURE);
	}

	for (std::size_t i = 0; i < values.size(); ++i) {
		if (values[i] != ref_values[i]) {
			std::cerr << "Mismatch value at index " << i << std::endl;
			std::cerr << "value= " << values[i] << ", ref_value= " << ref_values[i] << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	std::cout << "Pass" << std::endl;
}


int getMax(std::vector<int>& array)
{
	// get max value in array
	int max_value = static_cast<int>(-2 ^ 20);

	for (auto value : array) {
		max_value = std::max(max_value, value);
	}

	return max_value;
}


constexpr unsigned int getDigit(const int value, const unsigned int exponent)
{
	int digit = (value / exponent) % 10;

	return digit;
}


void compute_histogram(std::vector<unsigned int>& histogram, const std::vector<int>& array, const unsigned int exponent)
{
	for (const auto& value : array) {
		unsigned int digit = getDigit(value, exponent);
		histogram[digit]++;
	}
}


void compute_cdf(std::vector<unsigned int>& cdf, const std::vector<unsigned int>& histogram)
{
	cdf[0] = histogram[0]; // inclusive scan	

	for (std::size_t i = 1; i < histogram.size(); ++i) {
		cdf[i] = cdf[i - 1] + histogram[i];
	}
}


void countSort(std::vector<int>& array, unsigned int exponent)
{
	std::vector<unsigned int> histogram(10, 0);		// 10 digits with value=0.
	std::vector<unsigned int> cdf(10, 0);			// 10 digits with value=0.
	std::vector<int> output(array.size());

	// compute histogram of a single digit
	 compute_histogram(histogram, array, exponent);

	// compute cdf from histogram by using inclusive scan
	compute_cdf(cdf, histogram);

	// move data to output upon cdf
	for (int i = array.size() - 1; i >= 0; i--) {
		unsigned int digit = getDigit(array[i], exponent);	// d is in the range of [0..9].
		int index = cdf[digit] - 1;

		assert(index >= 0);

		output[index] = array[i];

		cdf[digit]--;
	}

	array = output;
}

void radix_sort_cpu(std::vector<int>& array)
{
	std::vector<int> result;

	int max_val = getMax(array);

	for (std::size_t exponent = 1; exponent <= max_val; exponent *= 10) {
		countSort(array, exponent);
	}
}


int main()
{
	std::vector<int> array = { 170, 45, 75, 90, 802, 24, 2, 66 };
	std::vector<int> result(array);		// copy array to result.

	radix_sort_cpu(result);

	std::vector<int> ref_result(array);
	std::sort(ref_result.begin(), ref_result.end());

	compare<int>(result, ref_result);

	return 0;
}
