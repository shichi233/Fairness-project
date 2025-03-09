import matplotlib.pyplot as plt
import numpy as np

models = ["MLP", "CNN", "Transformer (ViT)"]
mae_values = [2.83, 1.97, 3.05]

colors = ["#FFB6C1", "#ADD8E6", "#FFFF99"]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, mae_values, color=colors)

for bar, mae in zip(bars, mae_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{mae:.2f}", ha='center', va='bottom', fontsize=12)

plt.xlabel("Model", fontsize=14)
plt.ylabel("MAE", fontsize=14)
plt.title("Mean Absolute Error (MAE) of Different Models", fontsize=16)

# Show plot
plt.ylim(0, max(mae_values) + 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
