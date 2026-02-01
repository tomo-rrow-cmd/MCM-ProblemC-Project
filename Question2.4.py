import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ==========================================
# 1. DPB-TPD 模型核心引擎 (增加量化指标方法)
# ==========================================
class DPB_TPD_Model:
    def __init__(self, kappa_base=1.0, gamma=0.85):
        self.kappa_base = kappa_base
        self.gamma = gamma

    def get_kappa(self, n):
        return self.kappa_base * (self.gamma ** -n)

    def calculate_elim_probability(self, j_a, j_b, n_a, n_b):
        """返回淘汰 a 的概率及当场 TDE 指标"""
        k_a = self.get_kappa(n_a)
        k_b = self.get_kappa(n_b)
        exp_a = np.exp(-k_a * j_a)
        exp_b = np.exp(-k_b * j_b)
        prob_a = exp_a / (exp_a + exp_b)
        # 计算 TDE: 概率对分差的敏感度
        tde = k_a * prob_a * (1 - prob_a) * abs(j_a - j_b)
        return prob_a, tde


# ==========================================
# 2. 自动化参数拟合模块 (基于 S28+ 历史事实)
# ==========================================
def calibrate_parameters(df):
    """通过极大似然估计反推评委的 κ 和 γ"""
    # 筛选有 Save 机制的现代赛季
    modern_df = df[df['season'] >= 28].copy()
    training_data = []
    tracker = {}

    for (s, w), group in modern_df.groupby(['season', 'week']):
        # 基于百分比制规则确定名义上的 Bottom 2
        b2 = group.nsmallest(2, 'P_total_score').to_dict('records')
        if len(b2) < 2: continue

        a, b = b2[0], b2[1]
        n_a = tracker.get(a['celebrity_name'], 0) + 1
        n_b = tracker.get(b['celebrity_name'], 0) + 1
        tracker[a['celebrity_name']] = n_a
        tracker[b['celebrity_name']] = n_b

        # 记录真实结果：谁被淘汰了？
        # 0 代表 a 淘汰，1 代表 b 淘汰
        actual_elim = 0 if a['is_eliminated_this_week'] else 1
        # 修正：将归一化的分数放大，增强优化器的梯度感应
        ja_scaled = a['judge_metric'] * 20
        jb_scaled = b['judge_metric'] * 20
        training_data.append((ja_scaled, jb_scaled, n_a, n_b, actual_elim))

    def nll(params):
        k, g = params
        # 稍微放宽 gamma 的上限到 1.1，观察是否有增长趋势
        if g <= 0.1 or g > 1.1: return 1e12
        loss = 0
        for ja, jb, na, nb, actual in training_data:
            # 使用 softmax 概率模型：淘汰概率与 e^(-k*score) 成正比
            ka, kb = k * (g ** -na), k * (g ** -nb)

            # 能量函数：技术分越高，被淘汰概率越低
            exp_a = np.exp(-ka * ja)
            exp_b = np.exp(-kb * jb)
            prob_elim_a = exp_a / (exp_a + exp_b)

            # actual == 0 代表真实结果是 a 被淘汰
            p_obs = prob_elim_a if actual == 0 else (1.0 - prob_elim_a)
            loss -= np.log(max(p_obs, 1e-10))
        # --- 新增正则化项 ---
        # 惩罚项 1: 鼓励较大的 k (评委应具有基本的辨析力)
        loss += 5.0 / (k + 0.1)
            # 惩罚项 2: 鼓励较小的 g (评委权力应随次数显著衰减)
        loss += 10.0 * (g ** 2)

        return loss

    # 优化求解
    # 将 bounds 稍微放开，x0 设置在远离边界的中心位置
    res = minimize(nll,
                   x0=[5.0, 0.85],
                   bounds=[(0.5, 30.0), (0.6, 0.95)],
                   method='L-BFGS-B')  # 或者使用 'TNC'
    return res.x[0], res.x[1]


# ==========================================
# 3. 增强版仿真与量化指标输出
# ==========================================
def run_final_analysis(df, kappa, gamma):
    model = DPB_TPD_Model(kappa, gamma)
    cases = ["Jerry Rice", "Billy Ray Cyrus", "Bristol Palin", "Bobby Bones"]

    sim_logs = []
    tracker = {}
    reversed_count = 0
    total_b2 = 0

    for (s, w), group in df.groupby(['season', 'week']):
        b2 = group.nsmallest(2, 'P_total_score')
        if len(b2) < 2: continue

        total_b2 += 1
        a, b = b2.iloc[0], b2.iloc[1]
        n_a = tracker.get(a['celebrity_name'], 0) + 1
        n_b = tracker.get(b['celebrity_name'], 0) + 1
        tracker[a['celebrity_name']], tracker[b['celebrity_name']] = n_a, n_b

        p_a, tde = model.calculate_elim_probability(a['judge_metric'], b['judge_metric'], n_a, n_b)

        # CCI 指标计算：判断是否纠正了“低分晋级”
        if (a['judge_metric'] < b['judge_metric'] and p_a > 0.5) or \
                (b['judge_metric'] < a['judge_metric'] and (1 - p_a) > 0.5):
            reversed_count += 1

        for name in cases:
            if a['celebrity_name'] == name or b['celebrity_name'] == name:
                target = a if a['celebrity_name'] == name else b
                opp = b if target['celebrity_name'] == a['celebrity_name'] else a
                prob, t = model.calculate_elim_probability(
                    target['judge_metric'], opp['judge_metric'],
                    tracker[name], tracker.get(opp['celebrity_name'], 1)
                )
                sim_logs.append({"Name": name, "Prob": prob, "N": tracker[name], "TDE": t})

    cci = reversed_count / total_b2
    if abs(gamma - 1.0) < 1e-5:
        tlh = float('inf')  # 或者设为一个代表“永不衰减”的数值如 99
    else:
        tlh = np.log(0.5) / np.log(gamma)
    return pd.DataFrame(sim_logs), cci, tlh


# ==========================================
# 主执行流
# ==========================================
if __name__ == "__main__":
    # 加载合并数据
    df_comp = pd.read_csv('./outputs/dwts_rule_comparison.csv')
    df_latent = pd.read_csv('./outputs/latent_estimates.csv')

    full_df = pd.merge(df_comp,
                       df_latent[['season', 'week', 'celebrity_name', 'judge_metric']],
                       on=['season', 'week', 'celebrity_name'])

    # 【新增校验】确保淘汰列和得分列有效
    print(f"DEBUG: 合并后样本总数: {len(full_df)}")
    print(f"DEBUG: 包含淘汰记录的样本数: {full_df['is_eliminated_this_week'].sum()}")
    if full_df['is_eliminated_this_week'].sum() == 0:
        print("❌ 警告：未发现淘汰记录！请检查 dwts_rule_comparison.csv 是否包含正确的 is_eliminated_this_week 列。")

    # 第一步：校准参数（核心修改）
    print("🚀 正在基于 S28+ 数据校准模型参数...")
    k_opt, g_opt = calibrate_parameters(full_df)
    print(f"✅ 校准完成: κ_base = {k_opt:.3f}, γ = {g_opt:.3f}")

    # 第二步：运行量化仿真
    results, cci, tlh = run_final_analysis(full_df, k_opt, g_opt)

    # 第三步：可视化量化面板
    print("\n" + "═" * 60)
    print(f"{'DPB-TPD 赛制评估量化报告':^60}")
    print("═" * 60)
    print(f" 1. 争议修正指数 (CCI): {cci:.4f} (机制纠偏能力)")
    print(f" 2. 忍耐度半衰期 (TLH): {tlh:.2f} 周 (评委权力衰减速度)")
    print("-" * 60)

    # --- 新增 TDE 汇总统计 ---
    overall_tde = results['TDE'].mean()
    print(f" 3. 核心技术决策弹性 (Avg TDE): {overall_tde:.4f}")
    print(f"    [解释] TDE 反映了评委将“技术分差”转化为“淘汰概率”的杠杆力度")
    print("-" * 60)

    for name in results['Name'].unique():
        personal_data = results[results['Name'] == name]
        latest = personal_data.iloc[-1]
        avg_personal_tde = personal_data['TDE'].mean()

        print(f"【{name}】仿真摘要:")
        print(f"   ➤ 最终淘汰概率: {latest['Prob']:.2%}")
        print(f"   ➤ 个人经历平均 TDE: {avg_personal_tde:.4f}")
        print(f"   ➤ 累计进入危险区次数: {latest['N']}")
        print(f"   🚩 结论: {'机制有效拦截' if latest['Prob'] > 0.5 else '粉丝护体效应强'}")
    print("═" * 60)