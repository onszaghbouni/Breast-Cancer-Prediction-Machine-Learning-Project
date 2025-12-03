import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import pandas as pd

# =====================
# LOAD DATA
# =====================
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# =====================
# 1. CLASS DISTRIBUTION
# =====================
plt.figure(figsize=(6,4))
df["target"].value_counts().plot(kind="bar", color=["skyblue","pink"])
plt.title("Distribution des classes (0 = bénigne, 1 = maligne)")
plt.xlabel("Classe")
plt.ylabel("Nombre d'échantillons")
plt.savefig("plot_cancer_class_distribution.png")
plt.show()

# =====================
# 2. HEATMAP CORRELATION
# =====================
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Matrice de corrélation du dataset Breast Cancer")
plt.savefig("plot_cancer_heatmap.png")
plt.show()

# =====================
# 3. SCATTER COMPARATIF
# =====================
plt.figure(figsize=(6,4))
plt.scatter(df["mean radius"], df["mean texture"], alpha=0.5)
plt.xlabel("mean radius")
plt.ylabel("mean texture")
plt.title("Comparaison de deux caractéristiques")
plt.savefig("plot_cancer_scatter.png")
plt.show()
