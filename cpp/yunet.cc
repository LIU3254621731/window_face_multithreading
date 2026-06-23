#include "yunet.h"

// 定义后端类型映射表
const std::map<std::string, int> YuNet::str2backend{
    {"opencv", cv::dnn::DNN_BACKEND_OPENCV}, {"cuda", cv::dnn::DNN_BACKEND_CUDA},
    {"timvx",  cv::dnn::DNN_BACKEND_TIMVX},  {"cann", cv::dnn::DNN_BACKEND_CANN}
};

// 定义目标设备类型映射表
const std::map<std::string, int> YuNet::str2target{
    {"cpu", cv::dnn::DNN_TARGET_CPU}, {"cuda", cv::dnn::DNN_TARGET_CUDA},
    {"npu", cv::dnn::DNN_TARGET_NPU}, {"cuda_fp16", cv::dnn::DNN_TARGET_CUDA_FP16}
};

YuNet::YuNet(const std::string& model_path,
             const cv::Size& input_size,
             float conf_threshold,
             float nms_threshold,
             int top_k,
             int backend_id,
             int target_id)
    : model_path_(model_path), input_size_(input_size),
      conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
      top_k_(top_k), backend_id_(backend_id), target_id_(target_id)
{
    // 创建人脸检测器实例
    model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
}

void YuNet::setInputSize(const cv::Size& input_size)
{
    input_size_ = input_size;
    model->setInputSize(input_size_);
}

cv::Mat YuNet::detect(const cv::Mat& image)
{
    cv::Mat res;
    model->detect(image, res);
    
    // 如果检测到多个人脸，只保留置信度最高的那个
    if (res.rows > 1)
    {
        float max_conf = 0.0f;
        int max_idx = 0;
        
        // 找到置信度最高的人脸
        for (int i = 0; i < res.rows; ++i)
        {
            float conf = res.at<float>(i, 14);
            if (conf > max_conf)
            {
                max_conf = conf;
                max_idx = i;
            }
        }
        
        // 只保留置信度最高的人脸
        cv::Mat single_face(1, res.cols, res.type());
        res.row(max_idx).copyTo(single_face.row(0));
        return single_face;
    }
    
    return res;
}

cv::Mat YuNet::visualize(const cv::Mat& image, const cv::Mat& faces, float fps)
{
    // 定义绘制颜色
    static cv::Scalar box_color{0, 255, 0};  // 边界框颜色（绿色）
    static std::vector<cv::Scalar> landmark_color{
        cv::Scalar(255,   0,   0), // 右眼（红色）
        cv::Scalar(  0,   0, 255), // 左眼（蓝色）
        cv::Scalar(  0, 255,   0), // 鼻尖（绿色）
        cv::Scalar(255,   0, 255), // 右嘴角（紫色）
        cv::Scalar(  0, 255, 255)  // 左嘴角（黄色）
    };
    static cv::Scalar text_color{0, 255, 0};  // 文本颜色（绿色）

    // 克隆原始图像用于绘制
    auto output_image = image.clone();

    // 如果提供了帧率，显示在图像上
    if (fps >= 0)
    {
        cv::putText(output_image, cv::format("FPS: %.2f", fps), cv::Point(0, 15), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
    }

    // 检查是否检测到人脸
    if (faces.rows > 0)
    {
        // 绘制边界框
        int x1 = static_cast<int>(faces.at<float>(0, 0));
        int y1 = static_cast<int>(faces.at<float>(0, 1));
        int w = static_cast<int>(faces.at<float>(0, 2));
        int h = static_cast<int>(faces.at<float>(0, 3));
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

        // 显示置信度
        float conf = faces.at<float>(0, 14);
        cv::putText(output_image, cv::format("Best Face - Confidence: %.4f", conf), cv::Point(x1, y1+12), 
                    cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);

        // 绘制关键点
        for (int j = 0; j < landmark_color.size(); ++j)
        {
            int x = static_cast<int>(faces.at<float>(0, 2*j+4));
            int y = static_cast<int>(faces.at<float>(0, 2*j+5));
            cv::circle(output_image, cv::Point(x, y), 2, landmark_color[j], 2);
        }
    }
    else
    {
        // 如果没有检测到人脸，显示提示信息
        cv::putText(output_image, "No face detected", cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }
    return output_image;
}

const std::map<std::string, int>& YuNet::getBackendMap()
{
    return str2backend;
}

const std::map<std::string, int>& YuNet::getTargetMap()
{
    return str2target;
} 