import pandas as pd
import matplotlib.pyplot as plt

age_groups = ["5-20", "20-40", "40-60", "60-80", "80-90"]
scores = list(range(19))
data = [
    [0, 2, 3, 0, 2, 0, 0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0],
    [5, 22, 34, 22, 15, 10, 14, 9, 10, 16, 11, 9, 5, 10, 4, 4, 1, 2, 1],
    [25, 67, 115, 86, 99, 89, 110, 105, 102, 90, 112, 81, 91, 62, 52, 39, 24, 14, 5],
    [16, 54, 114, 104, 116, 125, 190, 172, 204, 193, 206, 202, 168, 148, 144, 98, 76, 36, 13],
    [5, 9, 21, 28, 30, 43, 46, 50, 62, 44, 51, 47, 40, 35, 21, 19, 13, 8, 4]
]

df = pd.DataFrame(data, index=age_groups, columns=scores)

plt.figure(figsize=(12, 6))
for age_group, values in df.iterrows():
    plt.bar(scores, values, alpha=0.6, label=age_group)

plt.xlabel("Score")
plt.ylabel("Number of Samples")
plt.title("Score Distribution by Age Group")
plt.legend(title="Age Group")
plt.xticks(scores)
plt.show()
