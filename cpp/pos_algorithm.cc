#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>
#include <deque>

/**
 * @brief POS算法滑动窗口实现，输入为每帧的block RGB均值矩阵
 * 
 * 输入参数：
 *   frame_means - std::vector<Eigen::MatrixXd>，长度为帧数f，每个元素是一个大小为 [block x 3] 的矩阵，
 *                 表示该帧中所有block的3个通道（B,G,R）均值
 *   fps         - 帧率，用于确定滑动窗口大小（默认窗口长度为1.6秒）
 * 
 * 返回值：
 *   Eigen::MatrixXd - [block x f] 大小的矩阵，表示每个block随时间的rPPG信号
 */
Eigen::MatrixXd cpu_POS_from_frame_means(
    const std::deque<Eigen::MatrixXd>& frame_means, float fps)
{
    const double eps = 1e-9;
    int f = frame_means.size();       // 总帧数
    if (f == 0) return Eigen::MatrixXd();
    int e = frame_means[0].rows();    // block 数量
    int c = frame_means[0].cols();    // 通道数，必须是3（B,G,R）

    if (c != 3) {
        std::cerr << "Error: each frame matrix should have 3 columns (B,G,R channels).\n";
        exit(1);
    }

    int w = static_cast<int>(1.6 * fps);  // 窗口大小，单位：帧数

    if (f < w) {
        std::cerr << "Error: frame length must be >= 1.6s = " << w << " frames.\n";
        exit(1);
    }

    // 投影矩阵 P，尺寸为 2 x 3
    Eigen::Matrix<double, 2, 3> P;
    P << 0, 1, -1,
        -2, 1, 1;

    // 结果矩阵，尺寸为 [block x frame]
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(e, f);

    // 滑动窗口遍历
    for (int n = w - 1; n < f; ++n) {
        int m = n - w + 1;  // 窗口起始帧索引

        for (int i = 0; i < e; ++i) {  // 遍历每个block
            // 收集当前窗口内该block的3通道信号，尺寸为 [3 x w]
            Eigen::MatrixXd Cn(3, w);
            for (int k = 0; k < w; ++k) {
                // frame_means[m + k]: 第m+k帧，[block x 3]
                // 取第i个block的RGB，转置后放到Cn第k列
                Cn.col(k) = frame_means[m + k].row(i).transpose();
            }

            // 对每个通道按时间归一化（除以均值）
            Eigen::VectorXd mean = Cn.rowwise().mean();
            for (int ch = 0; ch < 3; ++ch) {
                Cn.row(ch) /= (mean(ch) + eps);
            }

            // 计算投影信号，尺寸为 [2 x w]
            Eigen::MatrixXd S = P * Cn;
            Eigen::VectorXd S1 = S.row(0);
            Eigen::VectorXd S2 = S.row(1);

            // 计算调整系数 alpha
            double stdS1 = std::sqrt((S1.array() - S1.mean()).square().sum() / w);
            double stdS2 = std::sqrt((S2.array() - S2.mean()).square().sum() / w);
            double alpha = stdS1 / (stdS2 + eps);

            // 合并信号并去均值
            Eigen::VectorXd Hn = S1 + alpha * S2;
            Hn.array() -= Hn.mean();

            // 叠加到输出结果中（重叠相加）
            H.row(i).segment(m, w) += Hn.transpose();
        }
    }

    return H;  // 返回尺寸 [block x frame]
}
