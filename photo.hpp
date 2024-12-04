#ifndef PHOTO_HPP
#define PHOTO_HPP

#include <vector>
#include <string>

struct ThreadData {
    const std::vector<unsigned char>* input;
    std::vector<unsigned char>* output;
    int width;
    int height;
    const std::vector<float>* kernel_x;
    const std::vector<float>* kernel_y;
    int kernel_channels;
    int channel;
};

std::vector<unsigned char> load_png(const std::string& filename, int& width, int& height, int& channels);
void save_png(const std::string& filename, const std::vector<unsigned char>& image, int width, int height, int channels);

void apply_convolution(const std::vector<unsigned char>& input, std::vector<unsigned char>& output, int width, int height,
                       const std::vector<float>& kernel_x, const std::vector<float>& kernel_y, int channel, int channels);
void* thread_convolution(void* arg);
void sequential_filter(const std::vector<unsigned char>& input, std::vector<unsigned char>& output, int width, int height,
                       int channels, const std::vector<float>& kernel_x, const std::vector<float>& kernel_y);
void parallel_filter(const std::vector<unsigned char>& input, std::vector<unsigned char>& output, int width, int height,
                     int channels, const std::vector<float>& kernel_x, const std::vector<float>& kernel_y);

#endif // PHOTO_HPP
