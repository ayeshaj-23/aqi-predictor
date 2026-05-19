import json
import pandas as pd

# Load JSON dataset
with open("data/aqi_dataset.json", "r") as f:
    data = json.load(f)

# Convert to DataFrame (table)
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("data/aqi_dataset.csv", index=False)

print("CSV file created successfully!")
print(df.head())