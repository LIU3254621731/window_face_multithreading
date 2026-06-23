#include "SignalProcessor.h"

SignalProcessor::SignalProcessor(double fps, double lowcut, double highcut)
    : fps(fps), lowcut(lowcut), highcut(highcut)
{
    design_filter();
}

Eigen::MatrixXd SignalProcessor::process_signal(const Eigen::MatrixXd& signal_data) {
    int rows = signal_data.rows();
    int cols = signal_data.cols();
    Eigen::MatrixXd output(rows, cols);

    for (int c = 0; c < cols; ++c) {
        std::vector<double> column(rows);
        for (int r = 0; r < rows; ++r) {
            column[r] = signal_data(r, c);
        }

        std::vector<double> filtered = apply_filter(column);

        for (int r = 0; r < rows; ++r) {
            output(r, c) = filtered[r];
        }
    }

    return output;
}

// 该参数需要使用
void SignalProcessor::design_filter() {
    // Python中预计算的系数 (order=4, fs=30Hz, band=[0.45Hz, 3.0Hz])
    b = { 0.00192564, 0.00000000, -0.00770255, 0.00000000, 0.01155382, 0.00000000, -0.00770255, 0.00000000, 0.00192564 };
    a = { 1.00000000, -6.43219075, 18.38777059, -30.54002861, 32.25124411, -22.18130268, 9.70384353, -2.46919077, 0.27989453 };
}

std::vector<double> SignalProcessor::apply_filter(const std::vector<double>& data) {
    int N = b.size();
    int data_len = data.size();
    std::vector<double> output(data_len, 0.0);
    std::vector<double> x_hist(N, 0.0);
    std::vector<double> y_hist(N, 0.0);

    for (int n = 0; n < data_len; ++n) {
        for (int i = N - 1; i > 0; --i) {
            x_hist[i] = x_hist[i - 1];
            y_hist[i] = y_hist[i - 1];
        }
        x_hist[0] = data[n];

        double y = 0.0;
        for (int i = 0; i < N; ++i) y += b[i] * x_hist[i];
        for (int i = 1; i < N; ++i) y -= a[i] * y_hist[i];
        y /= a[0];

        y_hist[0] = y;
        output[n] = y;
    }

    return output;
}
