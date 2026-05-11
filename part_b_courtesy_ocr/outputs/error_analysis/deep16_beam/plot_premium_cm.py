
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
from pathlib import Path

# Load confusion matrix data from the long format CSV generated earlier
csv_path = Path("arabic_check_pipeline/part_b_courtesy_ocr/outputs/error_analysis/deep16_beam/digit_confusion_long.csv")

DIGITS = [str(i) for i in range(10)]
rows = DIGITS + ["Insertion"]
cols = DIGITS + ["Deletion"]

matrix = np.zeros((len(rows), len(cols)), dtype=int)
row_map = {r: i for i, r in enumerate(rows)}
col_map = {c: i for i, c in enumerate(cols)}

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        r = "Insertion" if row["reference"] == "<INS>" else row["reference"]
        c = "Deletion" if row["prediction"] == "<DEL>" else row["prediction"]
        if r in row_map and c in col_map:
            matrix[row_map[r], col_map[c]] = int(row["count"])

# Styling: Reverting to the Original Dark Premium Theme
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 10.5), facecolor='#0f172a')
ax.set_facecolor('#0f172a')

# Original colormap
shown_matrix = np.sqrt(matrix.astype(float))
im = ax.imshow(shown_matrix, cmap='magma', interpolation='nearest')

# Labels and Ticks
ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(rows)))
ax.set_xticklabels(cols, fontsize=12, color='#94a3b8')
ax.set_yticklabels(rows, fontsize=12, color='#94a3b8')
ax.set_xlabel("Predicted Label", fontsize=14, labelpad=20, color='#cbd5e1', fontweight='bold')
ax.set_ylabel("True Label", fontsize=14, labelpad=15, color='#cbd5e1', fontweight='bold')
ax.set_title("OCR Digit Confusion Analysis - Deep16 Beam", fontsize=22, pad=35, fontweight='bold', color='white')

# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Add text annotations
max_val = shown_matrix.max()
for i in range(len(rows)):
    for j in range(len(cols)):
        val = matrix[i, j]
        if val > 0:
            # Color logic for readability against colormap
            color = "black" if shown_matrix[i, j] > max_val * 0.7 else "white"
            fontweight = 'bold' if (i == j and i < 10) else 'normal'
            fontsize = 11 if (i == j and i < 10) else 9
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=fontsize, fontweight=fontweight)

# Highlight Insertion/Deletion regions with original subtle neon glow
ax.add_patch(patches.Rectangle((-0.5, 9.5), 11, 1, fill=False, edgecolor='#10b981', linewidth=2.5, alpha=0.8)) # Insertion Row
ax.add_patch(patches.Rectangle((9.5, -0.5), 1, 11, fill=False, edgecolor='#f43f5e', linewidth=2.5, alpha=0.8)) # Deletion Column

# Add a colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Confidence Scale (sqrt)', color='#94a3b8', fontsize=12)
cbar.ax.tick_params(labelsize=10, colors='#94a3b8')

# Add grid lines
ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
ax.grid(which="minor", color="#1e293b", linestyle='-', linewidth=1.2)
ax.tick_params(which="minor", bottom=False, left=False)

# Fixed Overlapping Text: Increased spacing and adjusted position
fig.text(0.5, 0.02, "Error analysis for Arabic Check OCR (Part B). Note the highlight on Insertions and Deletions.", 
         ha='center', fontsize=11, color='#64748b', style='italic')

# Adjust layout to accommodate the footer text
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

output_path = Path("arabic_check_pipeline/part_b_courtesy_ocr/outputs/error_analysis/deep16_beam/premium_confusion_matrix.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, facecolor='#0f172a')
plt.close()
print(f"Premium confusion matrix saved to {output_path}")
