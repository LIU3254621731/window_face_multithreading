#include "filter.h"
#include <stdexcept>

ButterworthBandPassFilter::ButterworthBandPassFilter(double sampleRate,
    double lowCutoff,
    double highCutoff)
    : sampleRate(sampleRate), lowCutoff(lowCutoff), highCutoff(highCutoff) {

    // 验证参数有效性
    if (sampleRate <= 0) {
        throw std::invalid_argument("采样率必须大于0");
    }
    if (lowCutoff <= 0 || highCutoff <= 0) {
        throw std::invalid_argument("截止频率必须大于0");
    }
    if (lowCutoff >= highCutoff) {
        throw std::invalid_argument("低频截止必须小于高频截止");
    }
    if (highCutoff >= sampleRate / 2) {
        throw std::invalid_argument("高频截止必须小于奈奎斯特频率(采样率/2)");
    }

    // 设置滤波器参数
    filter.setup(sampleRate, lowCutoff, highCutoff);
}

double ButterworthBandPassFilter::process(double input) {
    return filter.filter(input);
}

void ButterworthBandPassFilter::reset() {
    // 重新设置滤波器参数会重置内部状态
    filter.setup(sampleRate, lowCutoff, highCutoff);
}