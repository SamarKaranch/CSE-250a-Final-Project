import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("test.csv")

window = 20
df["smooth_reward"] = df["reward"].rolling(window=window).mean()

plt.figure(figsize=(20, 16))
plt.plot(df["episode"], df["smooth_reward"])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Curve")
plt.tight_layout()
plt.savefig("reward_plot.png")

corr = df["episode"].corr(df["reward"])
print(corr)
