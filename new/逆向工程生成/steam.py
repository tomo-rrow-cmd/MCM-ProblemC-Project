import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import os


def draw_designer_flowchart():
    # 1. 画布设置：宽屏高分辨率
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    # ================= 2. 设计风格定义 (Design System) =================
    # 配色方案 (McKinsey / Nature Style)
    colors = {
        'bg_zone': '#F9FAFB',  # 极淡的灰背景
        'box_fill': '#FFFFFF',  # 纯白卡片
        'box_edge': '#E5E7EB',  # 极细灰边框
        'primary': '#1E3A8A',  # 深蓝 (强调色)
        'secondary': '#3B82F6',  # 亮蓝 (次要强调)
        'accent': '#F59E0B',  # 金色 (高亮关键数学)
        'text_main': '#1F2937',  # 深灰字
        'text_sub': '#6B7280'  # 浅灰字
    }

    # ================= 3. 绘图辅助函数 =================

    def draw_card(x, y, w, h, title, content, border_color=colors['primary'], header_color=None):
        """绘制带有圆角、阴影和标题栏的卡片"""
        # 1. 阴影层 (Shadow)
        shadow = patches.FancyBboxPatch((x + 0.5, y - 0.5), w, h, boxstyle="Round,pad=0.5,rounding_size=0.8",
                                        ec="none", fc='#D1D5DB', alpha=0.5, zorder=1)
        ax.add_patch(shadow)

        # 2. 主卡片背景
        card = patches.FancyBboxPatch((x, y), w, h, boxstyle="Round,pad=0.5,rounding_size=0.8",
                                      ec=colors['box_edge'], fc=colors['box_fill'], lw=1, zorder=2)
        ax.add_patch(card)

        # 3. 顶部标题栏 (如果有颜色)
        if header_color:
            # 绘制左侧装饰条
            ax.plot([x, x], [y, y + h], color=header_color, lw=4, solid_capstyle='round', zorder=4)

        # 4. 文本
        # 标题 (去除 letter_spacing)
        ax.text(x + 1.5, y + h - 2, title, fontsize=11, fontweight='bold', color=colors['text_main'],
                ha='left', va='center', zorder=5, fontname='Arial')
        # 内容
        if content:
            ax.text(x + w / 2, y + h / 2 - 1, content, fontsize=9, color=colors['text_sub'],
                    ha='center', va='center', zorder=5, fontname='Arial', linespacing=1.6)

        return {'top': (x + w / 2, y + h), 'bottom': (x + w / 2, y), 'left': (x, y + h / 2),
                'right': (x + w, y + h / 2)}

    def draw_arrow(p1, p2, color='#9CA3AF', style='simple'):
        """绘制平滑的贝塞尔曲线箭头"""
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                    connectionstyle="arc3,rad=0.0" if style == 'simple' else "arc3,rad=-0.2",
                                    shrinkA=5, shrinkB=5), zorder=1)

    # ================= 4. 绘制内容 (The Logic) =================

    # --- ZONE 1: INPUTS (LEFT) ---
    # 修复：移除 letter_spacing 参数
    ax.text(15, 55, "PHASE I: DATA INGESTION", ha='center', fontsize=10, fontweight='bold', color=colors['secondary'])

    node_raw = draw_card(5, 40, 20, 8, "Raw Competition Data",
                         "• Judge Scores (Normalized)\n• Weekly Eliminations\n• Contestant Metadata",
                         header_color=colors['secondary'])

    node_process = draw_card(5, 25, 20, 8, "Preprocessing",
                             r"Normalizing $J_{c,t} \in [0,1]$" "\n" r"Weighting $\alpha \in [0,1]$",
                             header_color=colors['secondary'])

    draw_arrow(node_raw['bottom'], node_process['top'])

    # --- ZONE 2: THE ENGINE (CENTER - HERO SECTION) ---
    # 修复：移除 letter_spacing 参数
    ax.text(50, 55, "PHASE II: INVERSE OPTIMIZATION ENGINE", ha='center', fontsize=12, fontweight='bold',
            color=colors['primary'])

    # 背景大框
    engine_bg = patches.FancyBboxPatch((32, 18), 36, 34, boxstyle="Round,pad=1",
                                       fc='#EFF6FF', ec=colors['primary'], lw=1.5, linestyle='--', zorder=0)
    ax.add_patch(engine_bg)
    ax.text(34, 49, "Core Optimization Problem (QP)", fontsize=9, fontweight='bold', color=colors['primary'], zorder=1)

    # 核心数学框 1: 约束
    math_content_1 = (r"$\bf{Hard\ Constraint:}$" "\n"
                      r"$S_{Safe} \geq S_{Elim} - \xi$")
    node_constr = draw_card(38, 38, 24, 8, "Constraint Logic", math_content_1, header_color=colors['accent'])

    # 核心数学框 2: 目标函数
    math_content_2 = (r"$\bf{Minimize\ Z:}$" "\n"
                      r"$\sum \xi^2 + \lambda \sum ||\Delta F_t||^2$")
    node_obj = draw_card(38, 22, 24, 10, "Objective Function", math_content_2, header_color=colors['accent'])

    # 连接 Input -> Engine
    draw_arrow(node_process['right'], node_constr['left'], color=colors['primary'])
    draw_arrow(node_constr['bottom'], node_obj['top'], color=colors['primary'])

    # --- ZONE 3: OUTPUT & VALIDATION (RIGHT) ---
    # 修复：移除 letter_spacing 参数
    ax.text(85, 55, "PHASE III: INFERENCE", ha='center', fontsize=10, fontweight='bold', color='#10B981')

    node_output = draw_card(75, 40, 20, 8, "Latent Trajectories",
                            r"Fan Vote $F_{c,t}$ Estimation" "\n(Time-Series Output)", header_color='#10B981')

    node_ci = draw_card(75, 25, 20, 8, "Uncertainty Quant.",
                        r"95% Confidence Intervals" "\n(Bayesian Credible Region)", header_color='#10B981')

    node_valid = draw_card(75, 5, 20, 8, "Model Validation",
                           "• Hit Rate Check\n• Shocking Elimination Detection", header_color='#EF4444')

    # 连接 Engine -> Output
    draw_arrow(node_obj['right'], node_output['left'], color=colors['primary'], style='curved')
    draw_arrow(node_obj['right'], node_ci['left'], color=colors['primary'], style='curved')

    # 连接 Output -> Validation
    draw_arrow(node_output['bottom'], node_valid['top'], color='#EF4444')

    # --- 装饰性元素 (Annotations) ---
    ax.text(50, 19, "Solved via CVXPY / Gurobi", ha='center', fontsize=8, style='italic', color=colors['primary'],
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    plt.tight_layout()
    # 确保保存路径存在
    save_dir = r"C:\Users\17616\Desktop\E\比赛\美赛\new\逆向工程生成\outputs\paper_figures_modern"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'Fig_Flowchart_Designer.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✨ Designer Level Flowchart generated: {save_path}")


if __name__ == "__main__":
    draw_designer_flowchart()