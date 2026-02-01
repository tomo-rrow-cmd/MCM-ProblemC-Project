import pandas as pd
import numpy as np
from pathlib import Path

# 配置路径
DATA_DIR = Path("./data_prepare")
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """读取所有核心数据表"""
    files = {
        "latent": "q2_latent_estimates_filled.csv",
        "mcm": "q3_mcm_modeling_dataset.csv",
        "bias": "q3_compare_judge_vs_fan.csv",
        "rule_comp": "q2_dwts_rule_comparison.csv"
    }
    data = {}
    for key, name in files.items():
        path = DATA_DIR / name
        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            print(f"⚠️ 警告: 找不到文件 {path}")
    return data


def build_risk_engine(data):
    """
    ARHM 核心特征工程：
    1. 计算评委共识度 (Judge Consensus)
    2. 提取行业先验偏差 (Industry Prior Bias)
    3. 计算动态阻尼初值 k
    """
    df_latent = data['latent']
    df_mcm = data['mcm']
    df_bias = data['bias']

    # 1. 提取行业维度的“粉丝溢价”先验数据
    # 我们只关心 industry 类型的 bias
    industry_bias = df_bias[df_bias['factor_type'] == 'industry'][['factor_level', 'diff_fan_minus_judge']]
    industry_bias.columns = ['industry_category', 'prior_fan_bias']

    # 2. 合并基础表
    # 链接 latent 投票数据与 mcm 行业背景数据
    df = pd.merge(
        df_latent,
        df_mcm[['season', 'week', 'celebrity_name', 'industry_category', 'judge_variance']],
        on=['season', 'week', 'celebrity_name'],
        how='left'
    )

    # 3. 注入行业先验偏差
    df = pd.merge(df, industry_bias, on='industry_category', how='left').fillna({'prior_fan_bias': 0})

    # 4. 计算自适应风险算子 (Adaptive Risk Factor, ARF)
    # 逻辑：评委分标准差越大，说明技术断层越明显，ARF 越高（需要更强的阻尼）
    # 同时，如果选手属于“粉丝高溢价行业”，ARF 进一步修正
    df['sigma_j'] = df.groupby(['season', 'week'])['judge_raw'].transform(np.std)

    # 核心公式：计算动态阻尼系数 k_prime
    # 基准 k 设为 1.0 (无缩减)，当 sigma_j 高于均值时，k 减小 (对数压缩变强)
    avg_sigma = df['sigma_j'].mean()
    df['adaptive_k'] = 1.0 / (1.0 + np.maximum(0, df['sigma_j'] - avg_sigma) + np.abs(df['prior_fan_bias']))

    return df


def main():
    print("🚀 启动 Q4 特征引擎...")
    data = load_data()

    if not data:
        return

    # 生成风险监测特征表
    q4_feature_set = build_risk_engine(data)

    # 保存结果
    output_path = OUTPUT_DIR / "q4_risk_monitor_features.csv"
    q4_feature_set.to_csv(output_path, index=False)

    print(f"✅ 预处理完成！共生成 {len(q4_feature_set)} 条风险监测样本。")
    print(f"📂 结果已存至: {output_path}")

    # 打印预览
    print("\n--- 风险算子预览 (Top 5) ---")
    print(q4_feature_set[['celebrity_name', 'industry_category', 'sigma_j', 'adaptive_k']].head())


if __name__ == "__main__":
    main()