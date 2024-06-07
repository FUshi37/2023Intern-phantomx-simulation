import numpy as np

def kalman_filter(measurements, initial_state_mean, initial_state_covariance, process_variance, measurement_variance):
    # 初始化滤波器参数
    current_state_mean = initial_state_mean
    current_state_covariance = initial_state_covariance

    # 用于存储滤波后的状态值
    filtered_states = []

    for measurement in measurements:
        # 预测步骤
        predicted_state_mean = current_state_mean
        predicted_state_covariance = current_state_covariance + process_variance

        # 更新步骤
        kalman_gain = predicted_state_covariance / (predicted_state_covariance + measurement_variance)
        current_state_mean = predicted_state_mean + kalman_gain * (measurement - predicted_state_mean)
        current_state_covariance = (1 - kalman_gain) * predicted_state_covariance

        # 存储滤波后的状态值
        filtered_states.append(current_state_mean)

    return filtered_states

# 测试数据
# measurements = [1, 2, 3, 4, 5]
measurements = [1]
initial_state_mean = 0
initial_state_covariance = 1
process_variance = 0.1
measurement_variance = 0.1

# 执行卡尔曼滤波
filtered_states = kalman_filter(measurements, initial_state_mean, initial_state_covariance, process_variance, measurement_variance)

print("Filtered States:", filtered_states)
