#include <iostream>
#include <vector>
#include <chrono>
#include "photo.hpp"

int main() {
    std::vector<float> sobel_x = {
        -1,  0,  1,
        -2,  0,  2,
        -1,  0,  1
    };

    std::vector<float> sobel_y = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };

    int width, height, channels;
    std::vector<unsigned char> input_image;

    input_image = load_png("input.png", width, height, channels);
    if (input_image.empty()) {
        std::cerr << "Error loading PNG image" << std::endl;
        return 1;
    }

    std::vector<unsigned char> sequential_output(width * height * channels);
    std::vector<unsigned char> parallel_output(width * height * channels);

    auto start_seq = std::chrono::high_resolution_clock::now();
    sequential_filter(input_image, sequential_output, width, height, channels, sobel_x, sobel_y);
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> seq_time = end_seq - start_seq;

    auto start_par = std::chrono::high_resolution_clock::now();
    parallel_filter(input_image, parallel_output, width, height, channels, sobel_x, sobel_y);
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> par_time = end_par - start_par;

    save_png("sequential_output.png", sequential_output, width, height, channels);
    save_png("parallel_output.png", parallel_output, width, height, channels);

    std::cout << "Sequential processing time: " << seq_time.count() << " seconds" << std::endl;
    std::cout << "Parallel processing time: " << par_time.count() << " seconds" << std::endl;

    return 0;
}
