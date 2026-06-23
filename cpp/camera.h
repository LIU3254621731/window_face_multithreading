#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>

class Camera {
public:
    Camera();                      // 默认构造
    ~Camera();

    bool open();                   // 打开默认摄像头（索引0）
    void close();                  // 关闭摄像头
    bool read(cv::Mat& frame);     // 读取图像帧

    int get_width() const;
    int get_height() const;
    int get_fps() const;

private:
    cv::VideoCapture cap;
    int width = 0;
    int height = 0;
    int fps = 0;
    int device_index = 0;          // Windows 下默认使用 index 0
};

#endif // CAMERA_H
