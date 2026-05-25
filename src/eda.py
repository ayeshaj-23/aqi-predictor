import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("features/feature_store.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns)

# =========================
# AQI DISTRIBUTION
# =========================

plt.figure()
plt.hist(df["aqi"], bins=30)
plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.savefig("aqi_distribution.png")

# =========================
# CORRELATION HEATMAP (simple)
# =========================

corr = df.corr(numeric_only=True)

plt.figure()
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.title("Feature Correlation Heatmap")

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.tight_layout()
plt.savefig("correlation_heatmap.png")

print("EDA completed!")
print("Saved: aqi_distribution.png, correlation_heatmap.png")