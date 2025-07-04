import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def plot_weight_histograms(model, save_path):
    """
    Визуализирует распределение весов всех линейных слоёв модели.
    """
    weights = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            weights.append(module.weight.detach().cpu().numpy().flatten())
    all_weights = np.concatenate(weights)
    plt.figure(figsize=(6, 4))
    plt.hist(all_weights, bins=40, color='skyblue', edgecolor='black')
    plt.title("Weight Distribution")
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_comparison_table(results, save_path):
    """
    Строит и сохраняет таблицу сравнения техник регуляризации.
    """
    import pandas as pd
    df = pd.DataFrame({
        "Config": [r["config"] for r in results],
        "Final Test Accuracy": [r["final_acc"] for r in results],
        "Stability (std last 5)": [r["stability"] for r in results]
    })
    plt.figure(figsize=(8, len(results)))
    plt.axis('off')
    table = plt.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.title("Сравнение техник регуляризации", fontsize=14)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
