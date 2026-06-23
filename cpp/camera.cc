#include "camera.h"
#include <iostream>

Camera::Camera() {
    device_index = 0; // Windows 下默认使用摄像头索引 0
}

Camera::~Camera() {
    close();
}

bool Camera::open() {
    if (!cap.open(device_index)) {
        std::cerr << "Failed to open camera at index " << device_index << std::endl;
        return false;
    }

    // 可尝试设置默认分辨率和帧率
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    // 获取实际值
    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    std::cout << "Camera opened: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;
    return true;
}

void Camera::close() {
    if (cap.isOpened()) {
        cap.release();
        std::cout << "Camera closed." << std::endl;
    }
}

bool Camera::read(cv::Mat& frame) {
    if (!cap.isOpened()) return false;
    return cap.read(frame);
}

int Camera::get_width() const {
    return width;
}

int Camera::get_height() const {
    return height;
}

int Camera::get_fps() const {
    return fps;
}
