import matplotlib.pyplot as plt
import numpy as np
import re

# 读取文本文件并解析为字典
def read_patch_shifts(file_path):
    patch_shifts = {}
    with open(file_path, 'r') as file:
        current_patch = None
        for line in file:
            line = line.strip()
            # 识别 Patch 标题行
            match = re.match(r'#Patch (\d{3}) raw and fit shifts', line)
            if match:
                current_patch = f'patch_{match.group(1)}'
                patch_shifts[current_patch] = []
            elif line and current_patch is not None and not line.startswith('#'):
                # 解析数据行
                parts = line.split()
                if len(parts) >= 5:
                    shift_values = [float(parts[3]), float(parts[4])]
                    patch_shifts[current_patch].append(shift_values)
    return patch_shifts

# 示例：假设你的数据存储在 "patch_shifts.txt" 文件中
file_path = "/home/peiyuan/code/pp/motion_plot/0646log550-Patch-Patch.log"
patch_shifts = read_patch_shifts(file_path)

num_patches = len(patch_shifts)
num_cols = int(np.ceil(np.sqrt(num_patches)))
num_rows = int(np.ceil(num_patches / num_cols))

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
axes = axes.flatten()

# 统一设置所有子图的X轴和Y轴范围
x_min, x_max = float('inf'), float('-inf')
y_min, y_max = float('inf'), float('-inf')
for shifts in patch_shifts.values():
    shifts = np.array(shifts)
    x_min = min(x_min, shifts[:, 0].min())
    x_max = max(x_max, shifts[:, 0].max())
    y_min = min(y_min, shifts[:, 1].min())
    y_max = max(y_max, shifts[:, 1].max())

for idx, (patch_name, shifts) in enumerate(patch_shifts.items()):
    shifts = np.array(shifts)
    x_shifts = shifts[:, 0]
    y_shifts = shifts[:, 1]
    ax = axes[idx]
    ax.plot(x_shifts, y_shifts, marker='o')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # 仅在最左边一列中间的子图上显示 Y 轴标签
    if idx == (num_rows // 2) * num_cols:
        ax.set_ylabel('Y Shift', fontsize=20)
    elif idx % num_cols == 0:
        ax.set_ylabel('', fontsize=12)
    else:
        ax.set_yticklabels([])
    # 仅在最下面一行中间的子图上显示 X 轴标签
    if idx == (num_rows - 1) * num_cols + (num_cols // 2):
        ax.set_xlabel('X Shift', fontsize=20)
    elif idx >= (num_rows - 1) * num_cols:
        ax.set_xlabel('', fontsize=12)
    else:
        ax.set_xticklabels([])
    ax.grid(True)

# 隐藏多余的子图
for idx in range(len(patch_shifts), len(axes)):
    fig.delaxes(axes[idx])

plt.subplots_adjust(wspace=0, hspace=0)  # 将子图之间的间距设为 0
plt.tight_layout()
plt.savefig('/home/peiyuan/code/pp/motion_plot/shift_trajectories.png')
plt.show()
