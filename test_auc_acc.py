__author__ = "Lemon Wei"
__email__ = "Lemon2922436985@gmail.com"
__version__ = "1.1.0"

import numpy as np
from scipy.stats import norm, chi2, binom
from sklearn.metrics import roc_auc_score

## 1. 验证每一折DenseNet121-CBAM模型之间不存在统计学差异
## 2. 验证DenseNet121-CBAM与SimpleCNN、MOB-CBAM、DenseNet121模型之间存在统计学差异

def delong_test_bootstrap(y_true, y_score1, y_score2, verbose=True):
    """
    使用 DeLong's method 计算两个模型AUC的差异显著性（p值）
    
    参数:
    y_true: 真实标签 (np.array)
    y_score1: 模型1的预测概率 (np.array)
    y_score2: 模型2的预测概率 (np.array)

    返回:
    auc1: 模型1的AUC
    auc2: 模型2的AUC
    p_value: DeLong's test 的 p 值
    """
    y_true = np.array(y_true)
    y_score1 = np.array(y_score1)
    y_score2 = np.array(y_score2)

    # 获取正负样本索引
    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = np.sum(pos)
    n_neg = np.sum(neg)

    # 计算每个正样本与负样本之间的预测得分差异
    def compute_v10(scores_pos, scores_neg):
        # V10: 每个正样本对所有负样本的比较
        return np.array([[1.0 if s_pos > s_neg else 0.5 if s_pos == s_neg else 0.0
                          for s_neg in scores_neg] for s_pos in scores_pos])

    def compute_v1h(scores1, scores2, pos):
        # V1h: 每个样本对其他样本的比较矩阵
        n = len(scores1)
        V1 = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if pos[i] and pos[j]:
                    V1[i, j] = 0.5
                elif not pos[i] and not pos[j]:
                    V1[i, j] = 0.5
                else:
                    if pos[i]:
                        s_pos = scores1[i]
                        s_neg = scores1[j]
                        val1 = 1.0 if s_pos > s_neg else 0.5 if s_pos == s_neg else 0.0
                        s_pos = scores2[i]
                        s_neg = scores2[j]
                        val2 = 1.0 if s_pos > s_neg else 0.5 if s_pos == s_neg else 0.0
                        V1[i, j] = val1 - val2
                    else:
                        s_neg = scores1[i]
                        s_pos = scores1[j]
                        val1 = 1.0 if s_pos > s_neg else 0.5 if s_pos == s_neg else 0.0
                        s_neg = scores2[i]
                        s_pos = scores2[j]
                        val2 = 1.0 if s_pos > s_neg else 0.5 if s_pos == s_neg else 0.0
                        V1[i, j] = val1 - val2
        return V1

    # Step 1: 计算 AUC
    auc1 = roc_auc_score(y_true, y_score1)
    auc2 = roc_auc_score(y_true, y_score2)

    # Step 2: 计算方差协方差矩阵
    scores1_pos = y_score1[pos]
    scores1_neg = y_score1[neg]
    scores2_pos = y_score2[pos]
    scores2_neg = y_score2[neg]

    # V10 for AUC1 and AUC2
    V10_1 = compute_v10(scores1_pos, scores1_neg)
    V10_2 = compute_v10(scores2_pos, scores2_neg)

    # 计算 S10（平均比较值）
    S10_1 = np.mean(V10_1)
    S10_2 = np.mean(V10_2)

    # V1h: 协方差项
    V1h = compute_v1h(y_score1, y_score2, pos)
    S1h = np.mean(V1h, axis=1)
    var_S1h = np.cov(V1h.T)

    # Step 3: 计算方差和协方差
    var_AUC1 = (np.var(V10_1.mean(axis=1)) / n_pos) + (np.var(V10_1.mean(axis=0)) / n_neg)
    var_AUC2 = (np.var(V10_2.mean(axis=1)) / n_pos) + (np.var(V10_2.mean(axis=0)) / n_neg)
    cov_AUC1_AUC2 = var_S1h[0, 1]

    # Step 4: Z-score and p-value
    var_diff = var_AUC1 + var_AUC2 - 2 * cov_AUC1_AUC2
    var_diff = np.clip(var_diff, a_min=1e-10, a_max=None)  # 防止负值
    se = np.sqrt(var_diff)
    z = (auc1 - auc2) / se
    p_value = 2 * norm.sf(abs(z))  # 双侧检验 p 值

    if verbose:
        print(f"AUC1: {auc1:.4f}, AUC2: {auc2:.4f}")
        print(f"DeLong's Test p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("AUC差异具有统计学意义")
        else:
            print("AUC差异无统计学意义")

    return auc1, auc2, p_value

def n_delong_test_bootstrap(y_true, y_score1, y_score2, verbose=True):
    # 假设你有三个类别：0, 1, 2
    classes = np.unique(y_true)

    for cls in classes:
        # 二值化真实标签
        y_true_bin = (y_true == cls)
        
        # 提取模型1和模型2对该类别的预测概率
        y_score1_cls = y_score1[:, cls]
        y_score2_cls = y_score2[:, cls]
        
        print(f"\n=== 对类别 {cls} 进行 DeLong's Test ===")
        delong_test_bootstrap(y_true_bin, y_score1_cls, y_score2_cls, verbose=True)

def my_binom_test(k, n, p=0.5, alternative='two-sided'):
    """
    手动实现 binom_test 的功能
    参数:
    k: 成功次数
    n: 总试验次数
    p: 成功概率（默认 0.5）
    alternative: {'two-sided', 'less', 'greater'}
    返回:
    p_value: p 值
    """
    if alternative == 'two-sided':
        # 双侧检验：计算对称性 p 值
        d = binom.pmf(k, n, p)
        # 向两边扩展，找到所有比 d 小的概率点
        p_values = [binom.pmf(x, n, p) for x in range(n+1)]
        total = sum([pval for x, pval in enumerate(p_values) if pval <= d + 1e-12])
        return min(total, 1.0)  # 确保不超过 1

    elif alternative == 'less':
        # 左尾检验: P(X <= k)
        return binom.cdf(k, n, p)

    elif alternative == 'greater':
        # 右尾检验: P(X >= k)
        return binom.sf(k - 1, n, p)

    else:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    
def mcnemar_test(y_true, y_pred1, y_pred2, exact=True, verbose=True):
    """
    手动实现 McNemar's Test
    
    参数:
    y_true: 真实标签 (np.array)
    y_pred1: 模型1的预测标签 (np.array)
    y_pred2: 模型2的预测标签 (np.array)
    exact: 是否使用精确二项检验（当b + c < 25时推荐使用）

    返回:
    p_value: p 值
    table: 2x2 配对列联表
    """
    # 检查输入长度一致
    assert len(y_true) == len(y_pred1) == len(y_pred2)

    # 初始化四个格子的计数
    a = b = c = d = 0
    for gt, p1, p2 in zip(y_true, y_pred1, y_pred2):
        if p1 == p2:
            if p1 == gt:
                a += 1  # 两个模型都预测正确
            else:
                d += 1  # 两个模型都预测错误
        else:
            if p1 == gt:
                c += 1  # 模型2错，模型1对
            else:
                b += 1  # 模型1错，模型2对

    table = np.array([[a, b], [c, d]])

    # 使用精确二项检验（b + c < 25 时推荐）
    if exact:
        n = b + c
        if n == 0:
            p_value = 1.0
        else:
            k = min(b, c)
            # 手动计算双侧 p 值
            p_value = my_binom_test(k, n=n, p=0.5, alternative='two-sided')
    else:
        # 使用卡方检验（当 b + c >= 25 时）
        stat = (abs(b - c) ** 2) / (b + c) if (b + c) > 0 else 0
        p_value = chi2.sf(stat, df=1)

    if verbose:
        print("配对四格表：")
        print(table[0])
        print(table[1])
        print(f"McNemar's Test p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("准确率差异具有统计学意义")
        else:
            print("准确率差异无统计学意义")

    return p_value, table
