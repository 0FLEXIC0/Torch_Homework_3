import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from utils.experiment_utils import set_seed, make_classification_data, ClassificationDataset
from utils.model_utils import AdaptiveMLP, train_and_log_adaptive_regularization
from utils.visualization_utils import plot_weight_histograms, plot_comparison_table

RESULTS_DIR = "Homework_3/results/regularization_experiments"
PLOTS_DIR = "Homework_3/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_adaptive_regularization_experiments():
    """
    Запускает серию экспериментов с адаптивными техниками регуляризации для MLP:
    - Dropout с изменяющимся коэффициентом по эпохам
    - BatchNorm с разными momentum по слоям
    - Комбинированные варианты
    Сохраняет обученные модели и визуализации весов.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim, num_classes, width, epochs, batch_size = 2, 2, 128, 20, 64

    # Данные
    X, y = make_classification_data(n=1000, num_features=input_dim, num_classes=num_classes, seed=42)
    dataset = ClassificationDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Конфигурации для экспериментов
    configs = [
        {
            "name": "Adaptive Dropout (linear decay)",
            "dropout_schedule": "linear",
            "dropout_init": 0.5,
            "bn_momentums": [0.1, 0.1, 0.1],
            "combine": False
        },
        {
            "name": "Adaptive Dropout (exp decay)",
            "dropout_schedule": "exp",
            "dropout_init": 0.5,
            "bn_momentums": [0.1, 0.1, 0.1],
            "combine": False
        },
        {
            "name": "BatchNorm (momentum 0.05 or 0.1 or 0.2)",
            "dropout_schedule": None,
            "dropout_init": 0.0,
            "bn_momentums": [0.05, 0.1, 0.2],
            "combine": False
        },
        {
            "name": "Adaptive Dropout (linear) + BatchNorm (diff momentum)",
            "dropout_schedule": "linear",
            "dropout_init": 0.5,
            "bn_momentums": [0.05, 0.1, 0.2],
            "combine": True
        },
    ]

    results = []
    for cfg in configs:
        logging.info(f"Эксперимент: {cfg['name']}")
        model = AdaptiveMLP(
            input_dim, num_classes, [width, width, width],
            dropout_init=cfg["dropout_init"],
            dropout_schedule=cfg["dropout_schedule"],
            bn_momentums=cfg["bn_momentums"],
            use_batchnorm=cfg["bn_momentums"] is not None,
            combine=cfg.get("combine", False),
            total_epochs=epochs
        ).to(device)
        # Обучение и логирование истории
        history, final_acc = train_and_log_adaptive_regularization(
            model, train_loader, test_loader, device, epochs=epochs
        )
        # Визуализация распределения весов
        plot_weight_histograms(
            model, save_path=os.path.join(PLOTS_DIR, f"weights_{cfg['name'].replace(' ', '_')}.png")
        )
        # Сохранение модели
        model_path = os.path.join(RESULTS_DIR, f"model_{cfg['name'].replace(' ', '_')}.pt")
        torch.save(model.state_dict(), model_path)
        results.append({
            "config": cfg["name"],
            "final_acc": final_acc,
            "stability": np.std(history["test_acc"][-5:]),
            "history": history,
            "model_path": model_path
        })

    return results

def print_comparison_table(results):
    """
    Печатает таблицу сравнения техник регуляризации в консоль.
    """
    print("\nСравнение техник регуляризации:")
    print("{:<45} {:<20} {:<20}".format("Config", "Final Test Accuracy", "Stability (std last 5)"))
    print("-" * 85)
    for r in results:
        print("{:<45} {:<20.4f} {:<20.6f}".format(
            r["config"], r["final_acc"], r["stability"]
        ))

if __name__ == "__main__":
    results = run_adaptive_regularization_experiments()
    print_comparison_table(results)
