import re
from typing import List, Tuple
from config_plot import *
from matplotlib.ticker import ScalarFormatter

# madeup_flag = True

log = []
fname = "train_likelihood_v4_1800.log"
fname_without_extension = fname.split(".")[0]
for file in [f"{fname}"]:
    with open(file) as f:
        log += f.readlines()

# 正则表达式
pattern = r"Epoch \[(\d+)/\d+\], Loss train: ([\d\.e-]+), Loss vali: ([\d\.e-]+), EpochTime: ([\d\.]+)mins ConsumedTime: ([\d\.]+)mins"


epoch = []
loss_train = []
loss_vali = []
epoch_time = []

for line in log:
    # 匹配
    match = re.search(pattern, line)
    if match:
        epoch.append(int(match.group(1)))
        loss_train.append(float(match.group(2)))
        loss_vali.append(float(match.group(3)))
        epoch_time.append(float(match.group(4)))

fig, ax1 = plt.subplots()

plt.plot(epoch, loss_train, label="train", linewidth=2, marker='o', markersize=6, color='darkgreen', linestyle='--')
plt.plot(epoch, loss_vali, label="vali", linewidth=2, color='red')
plt.text(epoch[-1], loss_train[-1], f"{loss_train[-1]:.2e}", fontsize=12, color='darkgreen')
plt.text(epoch[-1], loss_vali[-1], f"{loss_vali[-1]:.2e}", fontsize=12, color='red')
# Set labels with custom font properties
plt.xlabel('Epoch', fontsize=14, fontweight='bold', color='darkred')
plt.ylabel('Loss', fontsize=14, fontweight='bold', color='darkred')
plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
# Set y-axis to scientific notation
plt.gca().get_yaxis().set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xticks(fontsize=12, fontweight='bold', color='darkred')
plt.yticks(fontsize=12, fontweight='bold', color='darkred')

ax2 = ax1.twinx()
ax2.plot(epoch, [60 * i for i in epoch_time], label="Epoch Time", linewidth=2, color='blue')
# Set labels for the second axis
ax2.set_ylabel('Time/Epoch (s)', fontsize=14, fontweight='bold', color='blue')
ax2.tick_params(axis='y', labelsize=12, labelcolor='blue')
ax2.set_ylim(0, 10 * max(epoch_time) * 60)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12)

plt.tight_layout()
# plt.legend()
plt.savefig(f"{fname_without_extension}.png", dpi=300)
plt.close("all")