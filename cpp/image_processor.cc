#include "image_processor.h"

ImageProcessor::ImageProcessor(int k_num) : K_NUM(k_num) {}

void ImageProcessor::processSingleImage(const cv::Mat& image, 
                                     const FaceBox* face_box,
                                     std::vector<Eigen::MatrixXd>& block_means,
                                     int frame_idx) {
    // 裁剪人脸区域
    cv::Mat cropped = image(cv::Rect(face_box->left, face_box->top, 
                                   face_box->right - face_box->left, 
                                   face_box->bottom - face_box->top));

    // 分离通道并计算积分图
    std::vector<cv::Mat> bgr_channels, integral_channels(3);
    cv::split(cropped, bgr_channels);
    
    for (int c = 0; c < 3; ++c) {
        cv::integral(bgr_channels[c], integral_channels[c], CV_64F);
    }
    
    // 计算区域尺寸
    int roi_w = cropped.cols / K_NUM;
    int roi_h = cropped.rows / K_NUM;
    
    // 计算每个子区域的均值
    int block_idx = 0;
    for (int row = 0; row < K_NUM; ++row) {
        for (int col = 0; col < K_NUM; ++col) {
            int x0 = col * roi_w;
            int y0 = row * roi_h;
            int x1 = (col == K_NUM-1) ? cropped.cols : (x0 + roi_w);
            int y1 = (row == K_NUM-1) ? cropped.rows : (y0 + roi_h);
            
            // 计算当前block的三个通道的均值
            for (int c = 0; c < 3; ++c) {
                double sum = integral_channels[c].at<double>(y1, x1) - 
                           integral_channels[c].at<double>(y0, x1) - 
                           integral_channels[c].at<double>(y1, x0) + 
                           integral_channels[c].at<double>(y0, x0);
                
                // 存储到对应block的矩阵中
                block_means[block_idx](frame_idx, c) = sum / ((x1 - x0) * (y1 - y0));
            }
            block_idx++;
        }
    }
}

// std::vector<Eigen::MatrixXd> ImageProcessor::calculateROIPixelMeans(const std::vector<cv::Mat>& images, 
//                                                                   const FaceBox* face_box) {
//     // 使用Eigen存储所有帧的通道均值 [block][frame][channel]
//     std::vector<Eigen::MatrixXd> block_means(K_NUM * K_NUM, 
//         Eigen::MatrixXd(images.size(), 3));

//     for(int i = 0; i < images.size(); i++) {
//         processSingleImage(images[i], face_box, block_means, i);
//     }

//     return block_means;
// }


// 处理单帧图像ROI，输出一帧的block RGB均值矩阵 [block x 3]
Eigen::MatrixXd ImageProcessor::processSingleFrame(
    const cv::Mat& image,
    const FaceBox* face_box,
    int k_num)
{
    // 裁剪ROI
    cv::Mat cropped = image(cv::Rect(face_box->left, face_box->top, 
                                    face_box->right - face_box->left, 
                                    face_box->bottom - face_box->top));
    std::vector<cv::Mat> bgr_channels(3), integral_channels(3);
    cv::split(cropped, bgr_channels);
    for (int c = 0; c < 3; ++c)
        cv::integral(bgr_channels[c], integral_channels[c], CV_64F);

    int roi_w = cropped.cols / k_num;
    int roi_h = cropped.rows / k_num;
    int block_count = k_num * k_num;

    Eigen::MatrixXd block_means(block_count, 3);  // [block x channel]

    int block_idx = 0;
    for (int row = 0; row < k_num; ++row) {
        for (int col = 0; col < k_num; ++col) {
            int x0 = col * roi_w;
            int y0 = row * roi_h;
            int x1 = (col == k_num - 1) ? cropped.cols : (x0 + roi_w);
            int y1 = (row == k_num - 1) ? cropped.rows : (y0 + roi_h);

            for (int c = 0; c < 3; ++c) {
                Eigen::MatrixXd integral_mat;
                cv::cv2eigen(integral_channels[c], integral_mat);
                double sum_val = integral_mat(y1, x1) - integral_mat(y0, x1)
                               - integral_mat(y1, x0) + integral_mat(y0, x0);
                block_means(block_idx, c) = sum_val / ((x1 - x0) * (y1 - y0));
            }
            ++block_idx;
        }
    }
    return block_means;
}











// 计算ROI区域的像素平均值，返回frame_means，尺寸为[f][block][channel]
std::vector<Eigen::MatrixXd> ImageProcessor::calculateROIPixelMeans(const std::vector<cv::Mat>& images, 
                                                   const FaceBox* face_box,
                                                   int k_num) {
    int frame_count = images.size();
    int block_count = k_num * k_num;

    // 定义输出，每帧一个矩阵，尺寸为[block x 3]
    std::vector<Eigen::MatrixXd> frame_means(frame_count, Eigen::MatrixXd(block_count, 3));

    for (int i = 0; i < frame_count; ++i) {
        // 直接调用之前写好的单帧处理函数，得到当前帧的block均值矩阵
        frame_means[i] = processSingleFrame(images[i], face_box, k_num);
    }

    return frame_means;
}


int ImageProcessor::getKNum() const {
    return K_NUM;
} 
