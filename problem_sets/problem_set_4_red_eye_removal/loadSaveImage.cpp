#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cuda_runtime.h> //for uchar4


void loadImageRGBA(const std::string& filename,
    uchar4** imagePtr,
    size_t* numRows, size_t* numCols)
{
    cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
        std::cerr << "Couldn't open file: " << filename << std::endl;
        exit(1);
    }

    if (image.channels() != 3) {
        std::cerr << "Image must be color!" << std::endl;
        exit(1);
    }

    if (!image.isContinuous()) {
        std::cerr << "Image isn't continuous!" << std::endl;
        exit(1);
    }

    cv::Mat imageRGBA;
    cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

    *imagePtr = new uchar4[image.rows * image.cols];

    unsigned char* cvPtr = imageRGBA.ptr<unsigned char>(0);
    for (size_t i = 0; i < image.rows * image.cols; ++i) {
        (*imagePtr)[i].x = cvPtr[4 * i + 0];
        (*imagePtr)[i].y = cvPtr[4 * i + 1];
        (*imagePtr)[i].z = cvPtr[4 * i + 2];
        (*imagePtr)[i].w = cvPtr[4 * i + 3];
    }

    *numRows = image.rows;
    *numCols = image.cols;
}

void saveImageRGBA(const uchar4* const image,
    const size_t numRows, const size_t numCols,
    const std::string& output_file)
{
    int sizes[2];
    sizes[0] = numRows;
    sizes[1] = numCols;
    cv::Mat imageRGBA(2, sizes, CV_8UC4, (void*)image);
    cv::Mat imageOutputBGR;
    cv::cvtColor(imageRGBA, imageOutputBGR, CV_RGBA2BGR);
    //output the image
    cv::imwrite(output_file.c_str(), imageOutputBGR);
}