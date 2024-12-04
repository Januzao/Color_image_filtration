#include "photo.hpp"
#include <iostream>
#include <cmath>
#include <png.h>
#include <pthread.h>
#include <algorithm>

std::vector<unsigned char> load_png(const std::string& filename, int& width, int& height, int& channels) {
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        // perror("Error opening PNG file");
        return {};
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        fclose(file);
        // std::cerr << "Failed to initialize libpng\n";
        return {};
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        fclose(file);
        // std::cerr << "Failed to initialize PNG info structure\n";
        return {};
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        fclose(file);
        // std::cerr << "Error reading PNG file\n";
        return {};
    }

    png_init_io(png_ptr, file);
    png_read_info(png_ptr, info_ptr);

    width    = png_get_image_width(png_ptr, info_ptr);
    height   = png_get_image_height(png_ptr, info_ptr);
    channels = png_get_channels(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth  = png_get_bit_depth(png_ptr, info_ptr);

    // Adjust PNG settings
    if (bit_depth == 16) png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png_ptr);

    png_read_update_info(png_ptr, info_ptr);

    std::vector<unsigned char> image(png_get_rowbytes(png_ptr, info_ptr) * height);
    std::vector<png_bytep> row_pointers(height);

    for (int y = 0; y < height; y++) {
        row_pointers[y] = image.data() + y * png_get_rowbytes(png_ptr, info_ptr);
    }

    png_read_image(png_ptr, row_pointers.data());

    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(file);
    return image;
}

void save_png(const std::string& filename, const std::vector<unsigned char>& image, int width, int height, int channels) {
    FILE* file = fopen(filename.c_str(), "wb");
    if (!file) {
        // perror("Error saving PNG file");
        return;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        fclose(file);
        // std::cerr << "Failed to initialize libpng for writing\n";
        return;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        fclose(file);
        // std::cerr << "Failed to initialize PNG info structure for writing\n";
        return;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(file);
        // std::cerr << "Error writing PNG file\n";
        return;
    }

    png_init_io(png_ptr, file);

    png_set_IHDR(
        png_ptr, info_ptr, width, height, 8,
        channels == 4 ? PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
    );

    png_write_info(png_ptr, info_ptr);

    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = const_cast<png_bytep>(image.data() + y * width * channels);
    }

    png_write_image(png_ptr, row_pointers.data());
    png_write_end(png_ptr, nullptr);

    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(file);
}

// Convolution function
void apply_convolution(const std::vector<unsigned char>& input, std::vector<unsigned char>& output, int width, int height,
                       const std::vector<float>& kernel_x, const std::vector<float>& kernel_y, int channel, int channels) {
    int kernel_size = 3;
    int half = kernel_size / 2;

    for (int y = half; y < height - half; y++) {
        for (int x = half; x < width - half; x++) {
            float gx = 0.0f;
            float gy = 0.0f;

            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int pixel_index = ((y + ky) * width + (x + kx)) * channels + channel;
                    int kernel_index = (ky + half) * kernel_size + (kx + half);

                    gx += input[pixel_index] * kernel_x[kernel_index];
                    gy += input[pixel_index] * kernel_y[kernel_index];
                }
            }

            float magnitude = std::sqrt(gx * gx + gy * gy);
            output[((y * width) + x) * channels + channel] = static_cast<unsigned char>(std::min(std::max(magnitude, 0.0f), 255.0f));
        }
    }
}

void* thread_convolution(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    apply_convolution(*data->input, *data->output, data->width, data->height,
                      *data->kernel_x, *data->kernel_y, data->channel, data->kernel_channels);
    return nullptr;
}

void sequential_filter(const std::vector<unsigned char>& input, std::vector<unsigned char>& output, int width, int height,
                       int channels, const std::vector<float>& kernel_x, const std::vector<float>& kernel_y) {
    for (int c = 0; c < channels; c++) {
        apply_convolution(input, output, width, height, kernel_x, kernel_y, c, channels);
    }
}

void parallel_filter(const std::vector<unsigned char>& input, std::vector<unsigned char>& output, int width, int height,
                     int channels, const std::vector<float>& kernel_x, const std::vector<float>& kernel_y) {
    std::vector<pthread_t> threads(channels);
    std::vector<ThreadData> thread_data(channels);

    for (int c = 0; c < channels; c++) {
        thread_data[c].input = &input;
        thread_data[c].output = &output;
        thread_data[c].width = width;
        thread_data[c].height = height;
        thread_data[c].kernel_x = &kernel_x;
        thread_data[c].kernel_y = &kernel_y;
        thread_data[c].kernel_channels = channels;
        thread_data[c].channel = c;

        pthread_create(&threads[c], nullptr, thread_convolution, &thread_data[c]);
    }

    for (int c = 0; c < channels; c++) {
        pthread_join(threads[c], nullptr);
    }
}
