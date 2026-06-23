#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <Eigen/Dense>  // Eigen 头文件必须在 OpenCV 的 eigen.hpp 之前
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include "yunet.h"


class ImageProcessor {
public:
    // 构造函数
    ImageProcessor(int k_num = 6);

    // 处理单张图片的ROI区域
    void processSingleImage(const cv::Mat& image, 
                          const FaceBox* face_box,
                          std::vector<Eigen::MatrixXd>& block_means,
                          int frame_idx);

    // 计算ROI区域的像素平均值
    std::vector<Eigen::MatrixXd> calculateROIPixelMeans(const std::vector<cv::Mat>& images, 
                                                       const FaceBox* face_box,
                                                       int k_num);

    // 处理单帧图像
    Eigen::MatrixXd processSingleFrame(const cv::Mat& image,
                                     const FaceBox* face_box,
                                     int k_num);

    // 获取K_NUM值
    int getKNum() const;

private:
    const int K_NUM;  // 区域划分数量
};

#endif // IMAGE_PROCESSOR_H 