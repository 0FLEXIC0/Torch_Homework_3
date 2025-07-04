import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from utils.experiment_utils import set_seed, make_classification_data, ClassificationDataset
from utils.model_utils import MLP, train_and_eval, grid_search_widths
from utils.visualization_utils import plot_heatmap

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

RESULTS_DIR = "Homework_3/results/width_experiments"
PLOTS_DIR = "Homework_3/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

if __name__ == "__main__":
    width_grid = [16, 32, 64, 128, 256]
    input_dim = 2
    num_classes = 2

    # Запускаем grid search по всем схемам ширины
    results_expand, results_shrink, results_const, width_grid = grid_search_widths(
        width_grid, input_dim, num_classes, epochs=20
    )

    # Визуализация heatmap для каждой схемы
    plot_heatmap(results_expand, width_grid, "Расширение", os.path.join(PLOTS_DIR, "heatmap_expand.png"))
    plot_heatmap(results_shrink, width_grid, "Сужение", os.path.join(PLOTS_DIR, "heatmap_shrink.png"))
    plot_heatmap(results_const, width_grid, "Постоянная", os.path.join(PLOTS_DIR, "heatmap_const.png"))

    # Находим оптимальную архитектуру по максимальной test accuracy среди всех схем и комбинаций
    max_expand = np.max(results_expand)
    max_shrink = np.max(results_shrink)
    max_const = np.max(results_const)
    max_overall = max(max_expand, max_shrink, max_const)

    if max_overall == max_expand:
        scheme = "Расширение"
        idx = np.unravel_index(np.argmax(results_expand), results_expand.shape)
        best_widths = [min(width_grid[idx[0]], width_grid[idx[1]]), int((width_grid[idx[0]] + width_grid[idx[1]])/2), max(width_grid[idx[0]], width_grid[idx[1]])]
    elif max_overall == max_shrink:
        scheme = "Сужение"
        idx = np.unravel_index(np.argmax(results_shrink), results_shrink.shape)
        best_widths = [max(width_grid[idx[0]], width_grid[idx[1]]), int((width_grid[idx[0]] + width_grid[idx[1]])/2), min(width_grid[idx[0]], width_grid[idx[1]])]
    else:
        scheme = "Постоянная"
        idx = np.unravel_index(np.argmax(results_const), results_const.shape)
        best_widths = [width_grid[idx[0]], width_grid[idx[0]], width_grid[idx[0]]]

    logging.info(f"Оптимальная архитектура: схема={scheme}, ширины слоёв={best_widths}, test accuracy={max_overall:.4f}")