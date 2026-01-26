import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 6类、簇大小不一致（有的大有的小），并且整体更挤、更重叠一点
# =========================================================
np.random.seed(7)

num_classes = 6

# 每类点数（让“区域大小”不一致：有的大、有的小）
n_main = [900, 650, 520, 420, 380, 260]

# 簇中心（布局类似示例：左棕、上橙、上红、右蓝、右下绿、下紫）
centers = np.array([
    [-2.9,  0.2],   # brown
    [-1.2,  1.25],  # orange
    [ 0.25, 1.10],  # red
    [ 2.05, 0.70],  # blue
    [ 1.20,-1.05],  # green
    [-0.70,-1.25],  # purple
], dtype=float)

# 每类协方差不同 -> 让“块状范围”不一致（有的更大更散、有的更小更紧）
covs = [
    [[0.55,  0.08], [0.08, 0.32]],  # brown: 最大、较散
    [[0.40, -0.06], [-0.06, 0.30]], # orange: 中等偏大
    [[0.34,  0.10], [0.10, 0.34]],  # red: 中等
    [[0.26, -0.06], [-0.06, 0.22]], # blue: 小一些
    [[0.30,  0.07], [0.07, 0.24]],  # green: 中等偏小
    [[0.18, -0.03], [-0.03, 0.16]], # purple: 最小、最紧
]

# 向全局中心“挤压”的比例（越大越重叠）
mix_noise_ratio = 0.20
global_center = np.array([-0.15, 0.20])

X_list, y_list = [], []
for k in range(num_classes):
    pts = np.random.multivariate_normal(centers[k], covs[k], size=n_main[k])

    # 让一部分点向全局中心漂移，制造更“挤”和边界重叠感
    m = int(len(pts) * mix_noise_ratio)
    if m > 0:
        idx = np.random.choice(len(pts), m, replace=False)
        pts[idx] = (
            0.62 * pts[idx] +
            0.38 * global_center +
            np.random.normal(0, 0.08, size=(m, 2))
        )

    X_list.append(pts)
    y_list.append(np.full(len(pts), k))

X = np.vstack(X_list)
y = np.concatenate(y_list)

# ---------------------------------------------------------
# 绘图：点更大、颜色变一点
# ---------------------------------------------------------
colors = [
    "#8d6e63",  # 柔棕
    "#ff8f00",  # 橙
    "#e53935",  # 红
    "#1e88e5",  # 蓝
    "#43a047",  # 绿
    "#8e24aa",  # 紫
]

plt.figure(figsize=(7.6, 4.4), dpi=160)

for k in range(num_classes):
    pts = X[y == k]
    plt.scatter(
        pts[:, 0], pts[:, 1],
        s=14, alpha=0.92,
        c=colors[k],
        edgecolors="none"
    )

# 论文风格：去坐标轴
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.show()
