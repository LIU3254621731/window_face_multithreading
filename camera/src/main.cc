#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <chrono>

void configure_camera(int width, int height, int fps) {
    // 构造控制命令
    std::string cmds[] = {
        "v4l2-ctl -d /dev/v4l-subdev2 --set-ctrl exp_time=20000",
        "v4l2-ctl -d /dev/v4l-subdev2 --set-ctrl roi_x=584",
        "v4l2-ctl -d /dev/v4l-subdev2 --set-ctrl roi_y=476"
    };

    for (const auto& cmd : cmds) {
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "Command failed: " << cmd << std::endl;
        }
    }
}

void read_cam(int width, int height, int fps) {
    configure_camera(width, height, fps);

    // 构造 GStreamer 管道字符串
    std::string pipeline =
        "v4l2src device=/dev/video0 ! video/x-raw,format=GRAY8,width=" +
        std::to_string(width) + ",height=" + std::to_string(height) +
        ",framerate=" + std::to_string(fps) + "/1 ! appsink";

    // 打开摄像头
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Camera failed to open." << std::endl;
        return;
    }

    cv::Mat raw, bgr;
    int frame_count = 0;  // 用于自动命名保存的图像

    while (true) {
        cap >> raw;
        if (raw.empty()) continue;

        // 将 Bayer 图像转换为 BGR
        cv::cvtColor(raw, bgr, cv::COLOR_BayerGBRG2BGR);  // Python中是 BAYER_GBRG2BGR

        cv::imshow("demo", bgr);
        
        int key = cv::waitKey(1);
        if (key == 27) break;  // ESC 键退出
        else if (key == 'a' || key == 'A') {  // 按下 'a' 或 'A' 键保存图像
            std::string filename = "frame_" + std::to_string(frame_count++) + ".jpg";
            cv::imwrite(filename, bgr);
            std::cout << "Saved frame as: " << filename << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}

int main() {
    read_cam(768, 512, 30);
    return 0;
}
