#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <deque>
#include <fstream>
#include <iostream>
#include <chrono>
#include <mutex>
#include <thread>
#include <atomic>
#include <cmath>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "filter.h"
#include "pos_algorithm.h"
#include "yunet.h"
#include "camera.h"
#include "image_processor.h"
#include "SignalProcessor.h"
#include <map>
#include <condition_variable>
#include <future>
#include <algorithm>
#include <fstream>

#define K_NUM 6                   // 子区域数量
#define THREAD_POOL_SIZE 6        // 线程池大小
#define PI 3.14159265358979323846 // 圆周率

// 定义POS任务结构体，包含ID和帧序列
struct POSJob {
    int id;
    std::deque<Eigen::MatrixXd> frames;
};

// 任务队列和同步相关变量
std::mutex task_mutex;
std::deque<POSJob> task_queue;
std::condition_variable task_cv;
std::atomic<bool> running(true);

// 结果缓存和显示同步
std::mutex result_mutex;
std::map<int, Eigen::MatrixXd> result_buffer;
std::atomic<int> next_display_id(0);

// 全局信号缓冲区
std::deque<double> signal_buffer;    // 实时波形显示缓存
std::deque<double> fft_buffer;       // 用于FFT分析的10秒缓存
std::mutex fft_buffer_mutex;
std::atomic<double> current_bpm(0.0); // 当前心率值
std::chrono::steady_clock::time_point last_bpm_update_time; // 上次心率更新的时间戳
const double fft_window_seconds = 30.0;  // // FFT窗口长度


// 使用OpenCV的FFT函数来计算心率（bpm）
double calculateBPM(const std::deque<double>& signal, double fps) {
    if (signal.size() < fft_window_seconds * fps) return 0.0; // 不足10秒数据则跳过

    // 转换信号为OpenCV Mat
    cv::Mat signal_mat(1, static_cast<int>(signal.size()), CV_64F);
    for (size_t i = 0; i < signal.size(); ++i) {
        signal_mat.at<double>(0, i) = signal[i];
    }

    // 汉宁窗减少泄漏
    cv::Mat window = cv::Mat::zeros(1, static_cast<int>(signal.size()), CV_64F);
    for (int i = 0; i < window.cols; ++i) {
        window.at<double>(0, i) = 0.5 * (1 - cos(2 * PI * i / (window.cols - 1)));
    }
    signal_mat = signal_mat.mul(window);

    // 填充到最佳FFT长度
    int m = cv::getOptimalDFTSize(signal_mat.cols);
    cv::Mat padded;
    cv::copyMakeBorder(signal_mat, padded, 0, 0, 0, m - signal_mat.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // 合并实部虚部准备FFT
    cv::Mat planes[] = { padded, cv::Mat::zeros(padded.size(), CV_64F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    // 执行DFT
    cv::dft(complexI, complexI);

    // 幅度谱
    cv::split(complexI, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag = planes[0];

    // 计算频率分辨率
    double df = fps / m;
    int min_idx = static_cast<int>(0.75 / df);  // 心率下限 45 bpm
    int max_idx = static_cast<int>(3.0 / df);   // 心率上限 180 bpm

    if (max_idx >= mag.cols / 2) {
        max_idx = mag.cols / 2 - 1; // 限制最大索引
    }

    // 在感兴趣范围内寻找峰值
    cv::Point max_loc;
    double max_val;
    cv::minMaxLoc(mag(cv::Rect(min_idx, 0, max_idx - min_idx + 1, 1)), nullptr, &max_val, nullptr, &max_loc);

    if (max_val <= 0) return 0.0; // 无有效峰值

    double peak_freq = (min_idx + max_loc.x) * df;
    return peak_freq * 60.0; // Hz转bpm
}

// 绘制实时POS信号波形
void drawSignalWaveform(const std::deque<double>& buffer, const std::string& window_name = "Filtered POS Signal") {
    static const int plot_width = 800;
    static const int plot_height = 400;

    if (buffer.empty()) return;

    cv::Mat plot(plot_height, plot_width, CV_8UC3, cv::Scalar(0, 0, 0));

    int mid_y = plot_height / 2;
    int y_range = plot_height / 4;

    // 画中心线
    cv::line(plot, cv::Point(0, mid_y), cv::Point(plot_width, mid_y), cv::Scalar(50, 50, 50), 1);

    // 绘制信号曲线
    for (size_t i = 1; i < buffer.size(); ++i) {
        int x1 = static_cast<int>(i - 1);
        int x2 = static_cast<int>(i);

        double val1 = buffer[i - 1];
        double val2 = buffer[i];

        int y1 = mid_y - static_cast<int>(val1 * y_range);
        int y2 = mid_y - static_cast<int>(val2 * y_range);

        cv::line(plot, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 1);
    }

    // 左上角显示当前心率
    std::string bpm_text = "HR: " + std::to_string(static_cast<int>(current_bpm.load())) + " bpm";
    cv::putText(plot, bpm_text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

    cv::imshow(window_name, plot);
}

// 在POS工作线程中，新增一个CSV文件流
std::ofstream pos_result_file("pos_result.csv", std::ios::out | std::ios::app); // 以追加方式打开文件

// 确保在第一次写入时写入表头
bool is_first_write = true;

// POS工作线程：从任务队列中取出任务执行
void pos_worker_thread(int fps) {
    ButterworthBandPassFilter filter(fps, 0.65, 2.5); // 带通滤波器

    while (running) {
        POSJob job;
        {
            std::unique_lock<std::mutex> lock(task_mutex);
            task_cv.wait(lock, [] { return !task_queue.empty() || !running; });
            if (!running && task_queue.empty()) break; // 停止条件
            job = std::move(task_queue.front());
            task_queue.pop_front();
        }

        // 计算POS信号
        Eigen::MatrixXd H = cpu_POS_from_frame_means(job.frames, fps);
        Eigen::RowVectorXd H_avg = H.colwise().mean();

        // === 2) 去均值 ===
        double mean_val = H_avg.mean();
        H_avg.array() -= mean_val;

        // 滤波
        Eigen::RowVectorXd filtered_H_avg(H_avg.size());
        for (int i = 0; i < H_avg.size(); ++i) {
            filtered_H_avg(i) = filter.process(H_avg(i));
        }

        // 可选归一化
        bool do_normalize = true;
        if (do_normalize) {
            double max_abs = filtered_H_avg.cwiseAbs().maxCoeff();
            if (max_abs > 1e-6) {
                filtered_H_avg.array() /= max_abs;
            }
        }

        // 第一块用全部，其它只取最后一个值
        Eigen::RowVectorXd chunk_to_use;
        if (job.id == 0) {
            chunk_to_use = filtered_H_avg;
        }
        else {
            int step_size = 1;
            chunk_to_use = filtered_H_avg.tail(step_size);
        }

        // 现在，锁定并同步结果写入文件和缓存
        {
            std::lock_guard<std::mutex> lock(result_mutex);

            // 如果是第一次写入，先写入表头
            if (is_first_write) {
                pos_result_file << "POS_Result" << std::endl; // 表头
                is_first_write = false;
            }

            // 将每个结果写入 CSV 文件，每个数值一行
            for (int i = 0; i < chunk_to_use.size(); ++i) {
                pos_result_file << chunk_to_use(i) << std::endl;  // 每个结果写一行
            }

            // 放入结果缓存
            result_buffer[job.id] = chunk_to_use;
        }
    }
}

int main(int argc, char** argv) {
    const char* model_path = "C:\\Users\\32546\\Desktop\\window_face_multithreading\\model\\face_detection_yunet_2023mar.onnx";
    int fps = 30, width = 640, height = 480;

    // 初始化摄像头
    Camera camera;
    if (!camera.open()) return -1;

    // 加载人脸检测器
    YuNet detector(model_path, cv::Size(width, height), 0.9f, 0.3f, 500);
    ImageProcessor processor(K_NUM);
    SignalProcessor Sig_processor(30.0, 0.45, 3.0);

    std::deque<Eigen::MatrixXd> frame_means;
    const int max_frames = 1.6 * fps; // 每块1.6秒
    int frame_counter = 0;
    auto fps_start = std::chrono::steady_clock::now();
    int job_id_counter = 0;

    // 启动POS线程池
    std::vector<std::thread> thread_pool;
    for (int i = 0; i < THREAD_POOL_SIZE; ++i) {
        thread_pool.emplace_back(pos_worker_thread, fps);
    }

    cv::Mat frame;
    bool face_detected = false;
    FaceBox* face_box = nullptr;
    int no_face_count = 0;

    last_bpm_update_time = std::chrono::steady_clock::now();

    while (running) {
        auto loop_start = std::chrono::steady_clock::now();
        if (!camera.read(frame)) continue;

        if (!face_detected) {
            if (no_face_count == 0) std::cout << "Searching for face..." << std::endl;
            cv::Mat faces = detector.detect(frame);
            if (faces.empty()) {
                no_face_count++;
                if (no_face_count >= 30) {
                    std::cout << "No face detected." << std::endl;
                    running = false;
                    break;
                }
                continue;
            }
            else {
                face_box = new FaceBox{
                    static_cast<int>(faces.at<float>(0, 0)),
                    static_cast<int>(faces.at<float>(0, 1)),
                    static_cast<int>(faces.at<float>(0, 0) + faces.at<float>(0, 2)),
                    static_cast<int>(faces.at<float>(0, 1) + faces.at<float>(0, 3)),
                    faces.at<float>(0, 14)
                };
                face_detected = true;
                std::cout << "Face detected!" << std::endl;
            }
        }

        if (face_detected && face_box) {
            Eigen::MatrixXd current_frame = processor.processSingleFrame(frame, face_box, K_NUM);
            frame_means.push_back(current_frame);
            if (frame_means.size() > max_frames) frame_means.pop_front();

            if (frame_means.size() == max_frames) {
                POSJob job{ job_id_counter++, frame_means };
                {
                    std::lock_guard<std::mutex> lock(task_mutex);
                    task_queue.push_back(std::move(job));
                }
                task_cv.notify_one();
            }
        }

        // 检查是否有新结果
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            while (result_buffer.count(next_display_id)) {
                const auto& chunk = result_buffer[next_display_id];

                // 加到实时波形缓冲
                for (int i = 0; i < chunk.size(); ++i) {
                    signal_buffer.push_back(chunk(i));
                }
                while (signal_buffer.size() > 800) {
                    signal_buffer.pop_front();
                }

                // 加到FFT缓冲
                {
                    std::lock_guard<std::mutex> fft_lock(fft_buffer_mutex);
                    for (int i = 0; i < chunk.size(); ++i) {
                        fft_buffer.push_back(chunk(i));
                    }
                    while (fft_buffer.size() > fft_window_seconds * fps) {
                        fft_buffer.pop_front();
                    }
                }

                // 每隔1秒计算一次心率
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_bpm_update_time).count() >= 1) {
                    std::lock_guard<std::mutex> fft_lock(fft_buffer_mutex);
                    if (fft_buffer.size() >= fft_window_seconds * fps) {
                        double bpm = calculateBPM(fft_buffer, fps);
                        current_bpm = bpm;
                        std::cout << "Calculated BPM: " << bpm << std::endl;
                    }
                    last_bpm_update_time = now;
                }

                // 绘制波形
                drawSignalWaveform(signal_buffer, "Filtered POS Signal");
                cv::waitKey(1);

                result_buffer.erase(next_display_id++);
            }
        }

        auto loop_end = std::chrono::steady_clock::now();
        int frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
        std::cout << "Frame Time: " << frame_time << " ms" << std::endl;

        frame_counter++;
        auto fps_now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(fps_now - fps_start).count() >= 1) {
            std::cout << "Real-Time FPS: " << frame_counter << std::endl;
            frame_counter = 0;
            fps_start = fps_now;
        }
    }

    // 结束线程池
    running = false;
    task_cv.notify_all();
    for (auto& t : thread_pool) t.join();

    camera.close();
    if (face_box) delete face_box;
    return 0;
}