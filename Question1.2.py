import pandas as pd
import numpy as np
import cvxpy as cp


def solve_q1_with_metrics(df_path):
    # 1. 加载数据
    df = pd.read_csv(df_path)
    results_list = []

    # 全局一致性统计变量
    total_weeks = 0
    perfect_consistent_weeks = 0

    # 自动探测求解器 (优先使用 Clarabel)
    available_solvers = cp.installed_solvers()
    solver_to_use = cp.CLARABEL if 'CLARABEL' in available_solvers else cp.SCS

    # 2. 按周进行博弈逆向求解
    for (season, week), group in df.groupby(['season', 'week']):
        n = len(group)
        if n <= 1: continue
        total_weeks += 1

        # 按照最终排名(elim_priority)排序：第一名在前
        group = group.sort_values('elim_priority', ascending=True).reset_index(drop=True)

        # 决策变量
        F = cp.Variable(n)  # 估算的粉丝得票率
        slack = cp.Variable(n - 1)  # 松弛变量：用于量化规则冲突强度

        mechanism = group['mechanism'].iloc[0]
        J = group['judge_share'].values if mechanism == 'Percent' else group['judge_rank'].values

        # 3. 构建约束
        constraints = [cp.sum(F) == 1.0, F >= 0, slack >= 0]
        dual_constraints = []
        epsilon = 1e-4

        for i in range(n - 1):
            if mechanism == 'Percent':
                # 百分比法：Score = J% + F% (高者胜)
                con = (J[i] + F[i]) - (J[i + 1] + F[i + 1]) >= epsilon - slack[i]
            else:
                # 排名法：Rank = J_rank + F_rank (小者胜)
                con = (J[i + 1] + F[i + 1]) - (J[i] + F[i]) >= epsilon - slack[i]

            constraints.append(con)
            dual_constraints.append(con)

        # 4. 目标函数
        # 目标 A: 尽量让 slack 为 0 (维持逻辑一致性)
        # 目标 B: 让 F 尽量接近平均分布 (最小先验偏差)
        objective = cp.Minimize(cp.sum_squares(F - 1.0 / n) + 1000 * cp.sum(slack))

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=solver_to_use)

            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # 判定一致性：若所有松弛变量均极小，则认为完全一致
                is_consistent = np.all(slack.value < 1e-3)
                if is_consistent: perfect_consistent_weeks += 1

                group['estimated_fan_share'] = F.value
                group['consistency_slack'] = np.append(slack.value, 0)  # 最后一项补0

                # 5. 确定性量化 (Certainty)
                # 提取对偶变量 (Lagrange Multipliers)
                lambdas = [abs(c.dual_value) if c.dual_value is not None else 0.0 for c in dual_constraints]

                # 计算每位选手的确定性指标
                node_certainty = np.zeros(n)
                for i in range(n - 1):
                    node_certainty[i] += lambdas[i]
                    node_certainty[i + 1] += lambdas[i]

                # 使用对数转换：Certainty = ln(1 + lambda_sum)
                group['certainty_index'] = np.log1p(node_certainty)
                results_list.append(group)
        except Exception as e:
            print(f"S{season}W{week} 出错: {e}")

    # 6. 生成最终报表
    final_df = pd.concat(results_list)

    consistency_rate = (perfect_consistent_weeks / total_weeks) * 100

    print("\n" + "=" * 40)
    print("📈 Q1 模型量化验证报告")
    print("-" * 40)
    print(f"【一致性指标】")
    print(f"预测一致性 (Global Consistency): {consistency_rate:.2f}%")
    print(f"说明: 在该百分比的周次中，模型估算的粉丝票数能 100% 解释淘汰结果。")
    print(f"\n【确定性指标】")
    print(f"平均确定性 (Mean Certainty): {final_df['certainty_index'].mean():.4f}")
    print(f"确定性标准差 (Certainty Std): {final_df['certainty_index'].std():.4f}")
    print(f"结论: 确定性在各周/选手间 {'显著不一致' if final_df['certainty_index'].std() > 0.5 else '较为一致'}。")
    print("=" * 40 + "\n")

    final_df.to_csv('Q1_Model_Results.csv', index=False)
    return final_df

# 执行
solve_q1_with_metrics('Cleaned_For_question1.csv')