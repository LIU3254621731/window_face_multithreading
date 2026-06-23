#ifndef SIGNAL_PROCESSOR_HPP
#define SIGNAL_PROCESSOR_HPP

#include <Eigen/Dense>
#include <vector>

class SignalProcessor {
public:
    SignalProcessor(double fps = 30.0, double lowcut = 0.45, double highcut = 3.0);

    Eigen::MatrixXd process_signal(const Eigen::MatrixXd& signal_data);

private:
    double fps;
    double lowcut;
    double highcut;
    std::vector<double> b;
    std::vector<double> a;

    void design_filter();
    std::vector<double> apply_filter(const std::vector<double>& data);
};

#endif // SIGNAL_PROCESSOR_HPP
