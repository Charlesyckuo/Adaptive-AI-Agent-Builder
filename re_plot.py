# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("reranking top3and5 1216 results.csv")
df.drop(columns=["Embedding Time", "Reranking and Evaluation Time", "Total Time"], inplace=True)
print(df.head(5))

# Plotting function
def plot_data(df, k, emb_model, save_path):
    title = f"MRR and Hit Rate Comparison Across Reranking Models based on {emb_model} (Top k = {k})"

    # 篩選 Top_k 的資料
    df = df[df["Top_k"] == k]
    df = df[df["Embedding Model"] == emb_model]

    # 根據 Hit Rate 降序排序
    df = df.sort_values(by='Hit Rate', ascending=False)
    print(df.head(5))

    plt.figure(figsize=(14, 10))
    
    # Plotting Hit Rate and MRR bars
    bars_hit_rate = plt.barh(df['Reranking Model'], df['Hit Rate'], color='#FFDAB9', label='Hit Rate', alpha=0.8, height=0.4)
    bars_mrr = plt.barh(df['Reranking Model'], df['MRR'], color='#87CEEB', label='MRR', alpha=0.8, height=0.4)

    # Adding labels to each bar
    for bar in bars_hit_rate:
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.4f}', va='center')

    for bar in bars_mrr:
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.4f}', va='center')

    # Customizing plot appearance
    plt.xlim(0.8, 1.0)
    plt.xticks([i / 100 for i in range(80, 101, 1)], fontsize=10)
    plt.xlabel('Score', fontsize=12)
    plt.title(title, fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.gca().invert_yaxis()  # 讓最高的 Hit Rate 在最上面
    plt.tight_layout()

    # Saving the plot
    plt.savefig(save_path, format='png', dpi=300)
    print(f"圖表已保存到 {save_path}")
    plt.show()

# Generating the plot
plot_data(df, 3, "voyage-multilingual-2", "Voyage_Reranking_top_3.png")
plot_data(df, 5, "voyage-multilingual-2", "Voyage_Reranking_top_5.png")
plot_data(df, 3, "intfloat/multilingual-e5-large", "E5_large_Reranking_top_3.png")
plot_data(df, 5, "intfloat/multilingual-e5-large", "E5_large_Reranking_top_5.png")