#ifndef FILTER_H
#define FILTER_H

#include "Iir.h"

class ButterworthBandPassFilter {
public:
    /**
     * @brief 构造函数，初始化带通滤波器
     * @param sampleRate 采样率(Hz)
     * @param lowCutoff 低频截止频率(Hz)
     * @param highCutoff 高频截止频率(Hz)
     */
    ButterworthBandPassFilter(double sampleRate, double lowCutoff, double highCutoff);

    /**
     * @brief 处理单个采样点
     * @param input 输入样本
     * @return 滤波后的输出样本
     */
    double process(double input);

    /**
     * @brief 重置滤波器状态
     */
    void reset();

private:
    static const int ORDER = 4; // 4阶滤波器
    Iir::Butterworth::BandPass<ORDER> filter; // 巴特沃斯带通滤波器
    double sampleRate; // 采样率
    double lowCutoff;  // 低频截止
    double highCutoff; // 高频截止
};

#endif // FILTER_H