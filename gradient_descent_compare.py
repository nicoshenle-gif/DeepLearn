import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成数据
def generate_data(n=100, noise_std=1.0):
    """生成线性回归数据 y = 2x + 3 + noise"""
    np.random.seed()
    X = np.random.rand(n, 1) * 10  #生成n行1列的随机数数组，每个数的范围是0到10，是x的值
    y = 2 * X + 3 + np.random.randn(n, 1) * noise_std #y的值是2x+3+噪声，噪声的标准差是noise_std
    return X, y

# 批量梯度下降
def batch_gradient_descent(X, y, lr=0.01, epochs=100):
    """批量梯度下降算法"""
    m = len(y)   #数据量大小
    w = np.random.randn(1)    #随机初始化w
    b = np.random.randn(1)    #随机初始化b
    
    for _ in range(epochs):
        y_pred = X * w + b
        dw = (2/m) * np.sum((y_pred - y) * X)
        db = (2/m) * np.sum(y_pred - y)
        w -= lr * dw
        b -= lr * db
    
    return w[0], b[0]

# 随机梯度下降
def stochastic_gradient_descent(X, y, lr=0.01, epochs=100):
    """随机梯度下降算法"""
    m = len(y)
    w = np.random.randn(1)
    b = np.random.randn(1)
    
    for _ in range(epochs):
        for i in range(m):
            xi = X[i]
            yi = y[i]
            y_pred = xi * w + b
            dw = 2 * (y_pred - yi) * xi
            db = 2 * (y_pred - yi)
            w -= lr * dw
            b -= lr * db
    
    return w[0], b[0]

# 小批量梯度下降
def minibatch_gradient_descent(X, y, lr=0.01, epochs=100, batch_size=16):
    """小批量梯度下降算法"""
    m = len(y)
    w = np.random.randn(1)
    b = np.random.randn(1)
    
    for _ in range(epochs):
        idx = np.random.permutation(m)
        X_shuffled = X[idx]
        y_shuffled = y[idx]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            y_pred = X_batch * w + b
            dw = (2/len(y_batch)) * np.sum((y_pred - y_batch) * X_batch)
            db = (2/len(y_batch)) * np.sum(y_pred - y_batch)
            w -= lr * dw
            b -= lr * db
    
    return w[0], b[0]

# 多组实验
def run_experiments():
    """运行多组实验"""
    true_w, true_b = 2, 3  # 真实参数
    n_experiments = 30     # 实验次数
    results = {'BGD': [], 'SGD': [], 'MiniBatch': []}
    
    print("开始进行多组实验...")
    
    for i in range(n_experiments):
        if (i + 1) % 10 == 0:
            print(f"完成 {i + 1}/{n_experiments} 组实验")
        
        X, y = generate_data()
        
        # 三种方法
        w_bgd, b_bgd = batch_gradient_descent(X, y)
        w_sgd, b_sgd = stochastic_gradient_descent(X, y)
        w_mbgd, b_mbgd = minibatch_gradient_descent(X, y)
        
        results['BGD'].append([w_bgd, b_bgd])
        results['SGD'].append([w_sgd, b_sgd])
        results['MiniBatch'].append([w_mbgd, b_mbgd])
    
    return results, true_w, true_b

# 计算偏差和方差
def analyze_bias_variance(results, true_w, true_b):
    """分析偏差和方差"""
    print("\n=== 偏差和方差分析 ===")
    print(f"真实参数: w = {true_w}, b = {true_b}")
    print("-" * 50)
    
    analysis_results = {}
    
    for method in results:
        arr = np.array(results[method])
        mean_params = np.mean(arr, axis=0)
        
        # 计算偏差 (bias)
        bias_w = abs(mean_params[0] - true_w)
        bias_b = abs(mean_params[1] - true_b)
        bias_total = np.sqrt(bias_w**2 + bias_b**2)
        
        # 计算方差 (variance)
        var_w = np.var(arr[:, 0])
        var_b = np.var(arr[:, 1])
        var_total = var_w + var_b
        
        analysis_results[method] = {
            'mean_w': mean_params[0],
            'mean_b': mean_params[1],
            'bias_w': bias_w,
            'bias_b': bias_b,
            'bias_total': bias_total,
            'var_w': var_w,
            'var_b': var_b,
            'var_total': var_total
        }
        
        print(f"{method}:")
        print(f"  均值参数: w = {mean_params[0]:.4f}, b = {mean_params[1]:.4f}")
        print(f"  偏差: w偏差 = {bias_w:.4f}, b偏差 = {bias_b:.4f}, 总偏差 = {bias_total:.4f}")
        print(f"  方差: w方差 = {var_w:.4f}, b方差 = {var_b:.4f}, 总方差 = {var_total:.4f}")
        print()
    
    return analysis_results

# 可视化结果
def visualize_results(results, analysis_results):
    """可视化实验结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    methods = ['BGD', 'SGD', 'MiniBatch']
    colors = ['blue', 'red', 'green']
    
    # 参数分布图
    for i, method in enumerate(methods):
        arr = np.array(results[method])
        
        # w参数分布
        axes[0, i].hist(arr[:, 0], bins=10, alpha=0.7, color=colors[i])
        axes[0, i].axvline(x=2, color='black', linestyle='--', label='真实值')
        axes[0, i].axvline(x=analysis_results[method]['mean_w'], color='red', linestyle='-', label='均值')
        axes[0, i].set_title(f'{method} - w参数分布')
        axes[0, i].set_xlabel('w值')
        axes[0, i].set_ylabel('频次')
        axes[0, i].legend()
        
        # b参数分布
        axes[1, i].hist(arr[:, 1], bins=10, alpha=0.7, color=colors[i])
        axes[1, i].axvline(x=3, color='black', linestyle='--', label='真实值')
        axes[1, i].axvline(x=analysis_results[method]['mean_b'], color='red', linestyle='-', label='均值')
        axes[1, i].set_title(f'{method} - b参数分布')
        axes[1, i].set_xlabel('b值')
        axes[1, i].set_ylabel('频次')
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig('gradient_descent_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 偏差方差对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods_cn = ['批量梯度下降', '随机梯度下降', '小批量梯度下降']
    bias_values = [analysis_results[method]['bias_total'] for method in methods]
    var_values = [analysis_results[method]['var_total'] for method in methods]
    
    ax1.bar(methods_cn, bias_values, color=colors)
    ax1.set_title('偏差对比')
    ax1.set_ylabel('偏差值')
    
    ax2.bar(methods_cn, var_values, color=colors)
    ax2.set_title('方差对比')
    ax2.set_ylabel('方差值')
    
    plt.tight_layout()
    plt.savefig('bias_variance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 运行实验
    results, true_w, true_b = run_experiments()
    
    # 分析结果
    analysis_results = analyze_bias_variance(results, true_w, true_b)
    
    # 可视化
    visualize_results(results, analysis_results)
    
    print("\n=== 总结 ===")
    print("1. 批量梯度下降(BGD): 使用全部数据，收敛稳定但速度慢")
    print("2. 随机梯度下降(SGD): 使用单个样本，收敛快但波动大")
    print("3. 小批量梯度下降(Mini-batch): 平衡了BGD和SGD的优缺点")
    print("\n实验完成！结果已保存为图片文件。")