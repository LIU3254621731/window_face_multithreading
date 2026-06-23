import os
import sys
import urllib
import urllib.request
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from face_detector import FaceDetector



def read_raw_to_mat_frames(raw_path: str, width: int, height: int, frame_count: int):
    frame_size = width * height * 3  # RGB888: 每像素3字节

    file_size = os.path.getsize(raw_path)

    print("\nImage dimensions:")
    print(f"  Width: {width} pixels")
    print(f"  Height: {height} pixels")
    print("  Channels: 3 (BGR)")
    print(f"  Total pixels per frame: {width * height}")
    print(f"  Total frames: {frame_count}")

    if file_size < frame_size * frame_count:
        raise RuntimeError("File size is smaller than expected for given frame count and resolution.")

    frames = []

    with open(raw_path, 'rb') as f:
        for i in range(frame_count):
            raw_bytes = f.read(frame_size)
            if len(raw_bytes) != frame_size:
                raise RuntimeError(f"Failed to read full frame at index {i}")

            # 将原始字节转为 NumPy 数组并 reshape 为 H×W×3
            frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 3))

            # OpenCV 默认是 BGR 格式，如果是 RGB，可以转换：
            # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frames.append(frame)

    return frames


def process_all_frames_fixed_roi(frames, face_box, k_num=6):
    """
    从固定ROI区域中分块并计算每帧每块RGB均值。

    参数:
        frames (List[np.ndarray]): 所有帧，形状为 [H, W, 3]，RGB格式。
        face_box (tuple): ROI坐标 (x0, y0, x1, y1)。
        k_num (int): 每行/列划分的块数，总共 k_num*k_num 个块。
    
    返回:
        block_means (List[np.ndarray]): 每个 block 对应一个矩阵 (num_frames, 3)
    """
    x0, y0, x1, y1 = face_box
    num_frames = len(frames)
    block_count = k_num * k_num
    block_means = [np.zeros((num_frames, 3), dtype=np.float64) for _ in range(block_count)]

    roi_width = x1 - x0
    roi_height = y1 - y0
    block_w = roi_width // k_num
    block_h = roi_height // k_num

    for frame_idx, frame in enumerate(frames):
        roi = frame[y0:y1, x0:x1]  # shape: (roi_h, roi_w, 3)
        bgr = cv2.split(roi)  # OpenCV默认是BGR，这里假设是RGB就不翻转

        # 为三个通道计算积分图
        integral = [cv2.integral(ch, sdepth=cv2.CV_64F) for ch in bgr]

        block_idx = 0
        for row in range(k_num):
            for col in range(k_num):
                xs = col * block_w
                ys = row * block_h
                xe = roi.shape[1] if col == k_num - 1 else xs + block_w
                ye = roi.shape[0] if row == k_num - 1 else ys + block_h
                area = (xe - xs) * (ye - ys)

                for c in range(3):
                    I = integral[c]
                    sum_val = I[ye, xe] - I[ys, xe] - I[ye, xs] + I[ys, xs]
                    block_means[block_idx][frame_idx, c] = sum_val / area
                block_idx += 1

    return block_means

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    巴特沃斯带通滤波器
    Args:
        data: 输入信号
        lowcut: 低频截止频率
        highcut: 高频截止频率
        fs: 采样频率
        order: 滤波器阶数
    Returns:
        滤波后的信号
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def cpu_POS(signal, **kargs):
    """
    POS method on CPU using Numpy.

    The dictionary parameters are: {'fps':float}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
    """
    # Run the pos algorithm on the RGB color signal c with sliding window length wlen
    # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
    eps = 10**-9
    X = signal
    e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * kargs['fps'])   # window length

    # stack e times fixed mat P
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        # Start index of sliding window (4)
        m = n - w + 1
        # Temporal normalization (5)
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2)+eps)
        M = np.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = np.multiply(M, Cn)

        # Projection (6)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)    # remove 3-th dim

        # Tuning (7)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        # Overlap-adding (8)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H

if __name__ == '__main__':
    # 参数配置
    model_path = '/home/bianjc/rknn_model_zoo-v2.3.0-2024-11-08/examples/RetinaFace/model/RetinaFace.rknn'
    onnx_path = '../model/RetinaFace_mobile320.onnx'
    dataset_path = '../model/dataset.txt'
    raw_path = '/home/bianjc/Document/2025-6-6_0-46-35-300pic_bgr.raw'  # raw文件路径
    width = 768  # 图像宽度
    height = 512  # 图像高度
    frame_count = 300  # 帧数

    # 初始化人脸检测器
    detector = FaceDetector(model_path, onnx_path, dataset_path)
    
    # 读取raw文件
    print(f"Reading raw file: {raw_path}")
    frames = read_raw_to_mat_frames(raw_path, width, height, frame_count)
    if frames is None:
        print("Failed to read raw file")
        sys.exit(1)
    
    print(f"Successfully read {len(frames)} frames")
    print(f"Video shape: {frames[0].shape}")
    
    # 使用第一帧进行人脸检测
    print("\nDetecting face in first frame...")
    first_frame = frames[0]
    dets = detector.detect(first_frame)
    
    if len(dets) > 0:
        # 打印检测结果
        print("\nFace detection result:")
        det = dets[0]  # 只取第一个检测到的人脸
        print(det)
        print(f"  Box: ({int(det[0])}, {int(det[1])}, {int(det[2])}, {int(det[3])})")
        print(f"  Score: {det[4]:.4f}")
        print(f"  Landmarks:")
        for j in range(5):
            print(f"    Point {j+1}: ({int(det[5+j*2])}, {int(det[6+j*2])})")
        

        x0, y0, x1, y1 = det[:4].astype(int)
        roi_box = (x0, y0, x1, y1)
        print(roi_box)

        block_means = process_all_frames_fixed_roi(frames, roi_box, k_num=6)

        print(len(block_means), block_means[0].shape)

        # 对每个区域使用POS算法处理
        H_list = []
        # 将block_means转换为numpy数组并调整维度
        block_means_array = np.array(block_means)  # shape: (block_count, num_frames, 3)
        print(f"Original block means array shape: {block_means_array.shape}")
        
        # 调整维度为 (block_count, 3, num_frames)
        block_means_array = np.transpose(block_means_array, (0, 2, 1))
        print(f"Transposed block means array shape: {block_means_array.shape}")
        
        H = cpu_POS(block_means_array, fps=30)
        
        # 将多区域结果平均, 取平均值
        H_avg = np.mean(H, axis=0)
        H_avg_1d = H_avg.flatten()

        # 应用巴特沃斯带通滤波
        fs = 30  # 采样频率 (fps)
        lowcut = 0.45  # 低频截止频率 (Hz)
        highcut = 3.0  # 高频截止频率 (Hz)
        H_avg_filtered = butter_bandpass_filter(H_avg_1d, lowcut, highcut, fs)

        print("POS algorithm results:")
        print(f"Output shape: {H_avg.shape}")
        print("Signal values:")
        # print(H_avg)

        # 读取H_avg.txt中的信号
        try:
            H_avg_txt = np.loadtxt('H_avg.txt')
            print("\nLoaded signal from H_avg.txt:")
            print(f"Shape: {H_avg_txt.shape}")
            
            # 对H_avg.txt中的信号也进行滤波
            H_avg_txt_filtered = butter_bandpass_filter(H_avg_txt, lowcut, highcut, fs)
            
            # 计算两个滤波后信号的相似度
            correlation = np.corrcoef(H_avg_filtered, H_avg_txt_filtered)[0, 1]
            print(f"\nFiltered signal correlation: {correlation:.4f}")
            
            # 信号可视化
            plt.figure(figsize=(15, 12))
            
            # 绘制原始RGB信号
            plt.subplot(4, 1, 1)
            time_points = np.arange(block_means_array.shape[2])
            for i in range(3):
                plt.plot(time_points, block_means_array[0, i, :], label=['R', 'G', 'B'][i])
            plt.title('Original RGB Signals')
            plt.xlabel('Frame')
            plt.ylabel('Intensity')
            plt.legend()
            plt.grid(True)

            # 绘制原始POS信号
            plt.subplot(4, 1, 2)
            plt.plot(time_points, H_avg_1d, 'r-', label='Current POS Signal')
            plt.plot(time_points, H_avg_txt, 'b-', label='H_avg.txt Signal')
            plt.title('Original POS Signals')
            plt.xlabel('Frame')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)

            # 绘制滤波后的POS信号
            plt.subplot(4, 1, 3)
            plt.plot(time_points, H_avg_filtered, 'g-', label='Current Filtered Signal')
            plt.plot(time_points, H_avg_txt_filtered, 'm-', label='H_avg.txt Filtered Signal')
            plt.title(f'Filtered POS Signals ({lowcut}-{highcut} Hz)')
            plt.xlabel('Frame')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)

            # 绘制信号差异
            plt.subplot(4, 1, 4)
            signal_diff = H_avg_filtered - H_avg_txt_filtered
            plt.plot(time_points, signal_diff, 'k-', label='Signal Difference')
            plt.title('Difference Between Filtered Signals')
            plt.xlabel('Frame')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig('signal_comparison.png')
            print("\nSaved signal comparison to 'signal_comparison.png'")
            plt.close()

        except Exception as e:
            print(f"\nError loading H_avg.txt: {str(e)}")
            # 如果无法加载H_avg.txt，只显示当前信号
            plt.figure(figsize=(15, 7))
            
            # 绘制原始RGB信号
            plt.subplot(3, 1, 1)
            time_points = np.arange(block_means_array.shape[2])
            for i in range(3):
                plt.plot(time_points, block_means_array[0, i, :], label=['R', 'G', 'B'][i])
            plt.title('Original RGB Signals')
            plt.xlabel('Frame')
            plt.ylabel('Intensity')
            plt.legend()
            plt.grid(True)

            # 绘制原始POS信号
            plt.subplot(3, 1, 2)
            plt.plot(time_points, H_avg_1d, 'r-', label='Original POS Signal')
            plt.title('Original POS Signal')
            plt.xlabel('Frame')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)

            # 绘制滤波后的POS信号
            plt.subplot(3, 1, 3)
            plt.plot(time_points, H_avg_filtered, 'g-', label='Filtered POS Signal')
            plt.title(f'Filtered POS Signal ({lowcut}-{highcut} Hz)')
            plt.xlabel('Frame')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig('signal_visualization.png')
            print("\nSaved signal visualization to 'signal_visualization.png'")
            plt.close()

        # 在图像上绘制检测结果
        result_img = first_frame.copy()
        # 绘制人脸框
        x1, y1, x2, y2 = map(int, det[:4])
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制关键点
        for j in range(5):
            x, y = int(det[5+j*2]), int(det[6+j*2])
            cv2.circle(result_img, (x, y), 2, (0, 0, 255), -1)
        
        # 保存检测结果图像
        output_path = "test.jpg"
        cv2.imwrite(output_path, result_img)
        print(f"\nSaved detection result to {output_path}")
    else:
        print("No face detected in the first frame")
