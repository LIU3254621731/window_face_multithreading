#ifndef YUNET_H
#define YUNET_H
#include "opencv2/opencv.hpp"
#include <string>
#include <map>
#include <Eigen/Dense>
#include <vector>

// 定义人脸框结构体
struct FaceBox {
    int left;
    int top;
    int right;
    int bottom;
    float confidence;
};

class YuNet
{
public:
    /**
     * @brief 构造函数
     * @param model_path 模型文件路径
     * @param input_size 输入图像大小，默认320x320
     * @param conf_threshold 置信度阈值，默认0.9
     * @param nms_threshold NMS阈值，默认0.3
     * @param top_k 保留的最大检测框数量，默认5000
     * @param backend_id 后端ID，默认0
     * @param target_id 目标设备ID，默认0
     */
    YuNet(const std::string& model_path,
          const cv::Size& input_size,
          float conf_threshold = 0.9f,
          float nms_threshold = 0.3f,
          int top_k = 5000,
          int backend_id = 0,
          int target_id = 0);

    /**
     * @brief 设置输入图像大小
     * @param input_size 新的输入图像大小
     */
    void setInputSize(const cv::Size& input_size);

    /**
     * @brief 执行人脸检测
     * @param image 输入图像
     * @return 检测结果矩阵，每行包含一个人脸的检测信息
     */
    cv::Mat detect(const cv::Mat& image);

    /**
     * @brief 可视化检测结果
     * @param image 原始图像
     * @param faces 检测到的人脸信息
     * @param fps 帧率（可选）
     * @return 可视化后的图像
     */
    cv::Mat visualize(const cv::Mat& image, const cv::Mat& faces, float fps = -1);

    /**
     * @brief 获取后端类型映射表
     * @return 后端类型映射表
     */
    static const std::map<std::string, int>& getBackendMap();

    /**
     * @brief 获取目标设备类型映射表
     * @return 目标设备类型映射表
     */
    static const std::map<std::string, int>& getTargetMap();

private:
    cv::Ptr<cv::FaceDetectorYN> model;  // 人脸检测器实例
    std::string model_path_;    // 模型文件路径
    cv::Size input_size_;       // 输入图像大小
    float conf_threshold_;      // 置信度阈值
    float nms_threshold_;       // NMS阈值
    int top_k_;                 // 最大检测框数量
    int backend_id_;            // 后端ID
    int target_id_;             // 目标设备ID

    // 定义后端类型映射表
    static const std::map<std::string, int> str2backend;
    // 定义目标设备类型映射表
    static const std::map<std::string, int> str2target;
};

#endif // FACE_DETECTOR_H 