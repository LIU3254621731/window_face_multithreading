#ifndef POS_ALGORITHM_H
#define POS_ALGORITHM_H

#include <Eigen/Dense>
#include <vector>
#include <deque>
// 输入类型：每个 block 是 [f x 3] 的矩阵
using RGBSignal = std::deque<Eigen::MatrixXd>;  // signal[e] = MatrixXd(f, 3)

Eigen::MatrixXd cpu_POS_from_frame_means(const RGBSignal& frame_means, float fps);
#endif // POS_ALGORITHM_H 