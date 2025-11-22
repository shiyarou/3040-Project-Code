import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD ORIGINAL DATA
# ==========================================
df = pd.read_csv("german.data", header=None, sep=r"\s+")

CLASS_COL = 20  # last column: 1 = Good, 2 = Bad

attr_names = {
    0:  "Status of checking account",
    1:  "Duration (months)",
    2:  "Credit history",
    3:  "Purpose",
    4:  "Credit amount",
    5:  "Savings account/bonds",
    6:  "Present employment since",
    7:  "Installment rate (%)",
    8:  "Personal status & sex",
    9:  "Other debtors/guarantors",
    10: "Present residence since",
    11: "Property",
    12: "Age",
    13: "Other installment plans",
    14: "Housing",
    15: "Number of existing credits",
    16: "Job",
    17: "Number of dependents",
    18: "Telephone",
    19: "Foreign worker",
    20: "Class (1=Good,2=Bad)"
}

# These are numeric in the original dataset spec
numeric_cols = [1, 4, 7, 10, 12, 15, 17]

print("\n================================================")
print(" CLASS DISTRIBUTION ")
print("================================================")
class_counts = df[CLASS_COL].value_counts().sort_index()
total = len(df)
for cls, cnt in class_counts.items():
    pct = cnt / total * 100
    print(f"Class {cls} ({'Good' if cls==1 else 'Bad'}): {cnt} ({pct:.2f}%)")

print("\n================================================")
print(" STATISTICS AND PLOTS FOR ALL ATTRIBUTES ")
print("================================================")

# Make sure numeric columns are numeric
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

for col in range(20):  # 0–19 are attributes
    name = attr_names[col]
    print("\n----------------------------------------")
    print(f"ATTRIBUTE {col}: {name}")
    print("----------------------------------------")

    if col in numeric_cols:
        # ========= NUMERIC: print mean/median per class =========
        for cls in [1, 2]:
            group = df[df[CLASS_COL] == cls][col]
            mean_val = group.mean()
            median_val = group.median()
            print(f"Class {cls} ({'Good' if cls==1 else 'Bad'}):")
            print(f"  Mean:   {mean_val:8.3f}")
            print(f"  Median: {median_val:8.3f}")
    else:
                # ========= CATEGORICAL: double bar chart normalized proportions =========
        ct = pd.crosstab(df[col], df[CLASS_COL])  # counts per category per class

        # Ensure both class columns exist
        for cls in [1, 2]:
            if cls not in ct.columns:
                ct[cls] = 0

        ct = ct[[1, 2]]  # order columns Good=1, Bad=2

        # Normalize: convert counts → proportions for each class separately
        total_good = class_counts[1]
        total_bad = class_counts[2]

        prop_good = ct[1] / total_good
        prop_bad  = ct[2] / total_bad

        categories = ct.index.astype(str)
        x = np.arange(len(categories))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(x - width/2, prop_good, width, label='Good (proportion)')
        plt.bar(x + width/2, prop_bad,  width, label='Bad (proportion)')

        plt.title(f"{name} – Proportion comparison (normalized)")
        plt.xlabel("Category")
        plt.ylabel("Proportion within class")
        plt.xticks(x, categories, rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.show()
