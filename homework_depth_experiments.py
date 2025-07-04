import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.experiment_utils import set_seed, make_classification_data, ClassificationDataset
from utils.visualization_utils import plot_learning_curves
from utils.model_utils import accuracy_score

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
RESULTS_DIR = "Homework_3/results/depth_experiments"
PLOTS_DIR = "Homework_3/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_device():
    """
    Определяет доступное устройство для вычислений: GPU, если доступен, иначе CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    """
    Многослойный перцептрон с опциональными BatchNorm и Dropout.
    """
    def __init__(self, in_features, out_features, hidden_sizes, use_bn=False, use_dropout=False, dropout_p=0.5):
        super().__init__()
        layers = []
        last_size = in_features
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden))  # BatchNorm для стабилизации обучения
            layers.append(nn.ReLU())  # Нелинейность
            if use_dropout:
                layers.append(nn.Dropout(dropout_p))  # Dropout для борьбы с переобучением
            last_size = hidden
        layers.append(nn.Linear(last_size, out_features))  # Выходной слой (логиты)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Прямой проход по сети.
        """
        return self.net(x)

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=30):
    """
    Обучает модель и собирает метрики по эпохам.
    """
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    epoch_times = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).view(-1).long()
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Оценка на тестовом наборе
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).view(-1).long()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                test_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                test_correct += (preds == y_batch).sum().item()
                test_total += X_batch.size(0)
        test_loss /= test_total
        test_acc = test_correct / test_total

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        logging.info(f"Epoch {epoch:02d}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, "
                     f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, Time={epoch_time:.2f}s")
    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "epoch_times": epoch_times
    }

def run_depth_experiment(hidden_layers_list, input_dim, num_classes, epochs=30, batch_size=64, seed=42,
                         use_bn=False, use_dropout=False, dropout_p=0.5):
    """
    Запускает серию экспериментов с разной глубиной сети и опциональными BatchNorm и Dropout.
    """
    set_seed(seed)
    device = get_device()
    logging.info(f"Используемое устройство: {device}")

    # Генерация данных
    X, y = make_classification_data(n=1000, source='random')
    dataset = ClassificationDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_histories = {}
    all_times = {}

    for hidden_sizes in hidden_layers_list:
        model_name = f"{len(hidden_sizes)+1}_layers"
        if use_bn:
            model_name += "_BN"
        if use_dropout:
            model_name += "_DO"
        logging.info(f"==== Обучение модели: {model_name} ====")
        model = MLP(input_dim, num_classes, hidden_sizes, use_bn=use_bn, use_dropout=use_dropout, dropout_p=dropout_p).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        history = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs)
        all_histories[model_name] = history
        all_times[model_name] = sum(history["epoch_times"])

        # Сохраняем историю обучения
        np.savez(os.path.join(RESULTS_DIR, f"{model_name}_history.npz"), **history)
        # Сохраняем графики кривых обучения
        plot_learning_curves(
            history["train_losses"], history["test_losses"],
            history["train_accs"], history["test_accs"],
            title=f"Learning Curves ({model_name})",
            save_path=os.path.join(PLOTS_DIR, f"{model_name}_curves.png")
        )

    return all_histories, all_times

def analyze_overfitting(histories):
    """
    Анализирует, когда начинается переобучение (разрыв между train и test accuracy).
    """
    for model_name, hist in histories.items():
        train_acc = np.array(hist['train_accs'])
        test_acc = np.array(hist['test_accs'])
        gap = train_acc - test_acc
        overfit_epoch = np.argmax(gap > 0.05) if np.any(gap > 0.05) else None
        print(f"{model_name}: максимальный разрыв train/test acc = {gap.max():.3f} на эпохе {gap.argmax()+1 if gap.size else '-'}")
        if overfit_epoch:
            print(f"  Переобучение начинается примерно с эпохи: {overfit_epoch+1}")
        else:
            print("  Явного переобучения не обнаружено.")

def get_model_depth_from_name(model_name):
    """
    Получает глубину модели (число слоёв) из имени модели вида '3_layers', '5_layers_BN_DO' и т.д.
    """
    return int(model_name.split('_')[0])

def find_optimal_depth(histories):
    """
    Определяет оптимальную глубину (по слоям) на основе максимальной test accuracy 
    при минимальном разрыве между train и test accuracy.
    """
    best_model = None
    best_score = -float('inf')
    for model_name, hist in histories.items():
        test_acc = hist["test_accs"][-1]
        train_acc = hist["train_accs"][-1]
        gap = abs(train_acc - test_acc)
        score = test_acc - gap  # можно добавить штраф за переобучение
        if score > best_score:
            best_score = score
            best_model = model_name
    optimal_depth = get_model_depth_from_name(best_model)
    print(f"\nОптимальная глубина: слоёв: {optimal_depth} (модель: {best_model}, test accuracy: {histories[best_model]['test_accs'][-1]:.4f})")
    return optimal_depth

if __name__ == "__main__":
    # Конфигурации глубины: от 1 до 7 слоев
    hidden_layers_list = [
        [],                  # 1 слой (линейный)
        [64],                # 2 слоя (1 скрытый)
        [64, 64],            # 3 слоя (2 скрытых)
        [64, 64, 64, 64],    # 5 слоев (4 скрытых)
        [64, 64, 64, 64, 64, 64]  # 7 слоев (6 скрытых)
    ]
    input_dim = 2
    num_classes = 2
    epochs = 30

    # Базовые модели без регуляризации
    print("=== Базовые модели ===")
    histories, times = run_depth_experiment(
        hidden_layers_list, input_dim, num_classes, epochs=epochs
    )
    analyze_overfitting(histories)

    # Модели с BatchNorm и Dropout
    print("\n=== Модели с BatchNorm и Dropout ===")
    histories_bn_do, times_bn_do = run_depth_experiment(
        hidden_layers_list, input_dim, num_classes, epochs=epochs,
        use_bn=True, use_dropout=True, dropout_p=0.5
    )
    analyze_overfitting(histories_bn_do)

    # Оптимальная глубина
    print("\n=== Определение оптимальной глубины (без BN/DO) ===")
    find_optimal_depth(histories)

    print("\n=== Определение оптимальной глубины (с BN и DO) ===")
    find_optimal_depth(histories_bn_do)

    # Итоговое сравнение
    print("\n=== Итоговое сравнение точности и времени для всех моделей ===")
    all_histories_combined = {**histories, **histories_bn_do}
    all_times_combined = {**times, **times_bn_do}
    for model_name in sorted(all_histories_combined.keys(), key=lambda x: (int(x.split('_')[0]), x)):
        train_acc = all_histories_combined[model_name]["train_accs"][-1]
        test_acc = all_histories_combined[model_name]["test_accs"][-1]
        total_time = all_times_combined[model_name]
        print(f"{model_name}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, Time={total_time:.2f}s")
