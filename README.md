# Коды файлов

experiment_utils.py
```python
import random
import numpy as np
import torch
from torch.utils.data import Dataset

def set_seed(seed=42):
    """
    Фиксирует seed для воспроизводимости экспериментов.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_classification_data(n=1000, num_features=2, num_classes=2, seed=42, source='random'):
    """
    Генерирует синтетические данные для классификации.
    """
    np.random.seed(seed)
    if source == 'random':
        X = np.random.randn(n, num_features)
        w = np.random.randn(num_features, num_classes)
        logits = X @ w + np.random.randn(n, num_classes) * 0.5
        y = np.argmax(logits, axis=1)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    else:
        raise ValueError("Unknown source")

class ClassificationDataset(Dataset):
    """
    PyTorch Dataset для задачи классификации.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

```

visualization_utils.py
```python
import matplotlib.pyplot as plt

def plot_learning_curves(train_losses, test_losses, train_accs, test_accs, title, save_path):
    """
    Строит и сохраняет графики потерь и точности.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, test_accs, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

```

model_utils.py
```python
import torch

def accuracy_score(y_true, y_pred):
    """
    Вычисляет точность классификации (accuracy).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    return (y_true == y_pred).mean()
```

# 1.1 Сравнение моделей разной глубины (15 баллов)
Создайте файл homework_depth_experiments.py:
Создайте и обучите модели с различным количеством слоев:
- 1 слой (линейный классификатор)
- 2 слоя (1 скрытый)
- 3 слоя (2 скрытых)
- 5 слоев (4 скрытых)
- 7 слоев (6 скрытых)

файл homework_depth_experiments.py
```python
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.visualization_utils import plot_learning_curves
from utils.model_utils import accuracy_score
from utils.experiment_utils import make_classification_data, ClassificationDataset, set_seed

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

RESULTS_DIR = "Homework_3/results/depth_experiments"
PLOTS_DIR = "Homework_3/plots"

def get_device():
    """Определяет доступное устройство."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    """
    Многослойный перцептрон с произвольным количеством скрытых слоёв.
    """
    def __init__(self, in_features, out_features, hidden_sizes):
        super().__init__()
        layers = []
        last_size = in_features
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden))
            layers.append(nn.ReLU())
            last_size = hidden
        layers.append(nn.Linear(last_size, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=30):
    """
    Обучает модель и возвращает историю потерь и точности.
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

        # Оценка на тесте
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

        logging.info(f"Epoch {epoch:02d}: "
                     f"Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, "
                     f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, "
                     f"Time={epoch_time:.2f}s")
    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "epoch_times": epoch_times
    }

def run_depth_experiment(hidden_layers_list, input_dim, num_classes, epochs=30, batch_size=64, seed=42):
    """
    Запускает серию экспериментов с разной глубиной сети.
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
        logging.info(f"==== Обучение модели: {model_name} ====")
        model = MLP(input_dim, num_classes, hidden_sizes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        history = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs)
        all_histories[model_name] = history
        all_times[model_name] = sum(history["epoch_times"])

        # Сохраняем результаты
        np.savez(os.path.join(RESULTS_DIR, f"{model_name}_history.npz"), **history)
        print(f"Сохранено: {os.path.join(RESULTS_DIR, f'{model_name}_history.npz')}")
        # Визуализация кривых обучения
        plot_learning_curves(
            history["train_losses"], history["test_losses"],
            history["train_accs"], history["test_accs"],
            title=f"Learning Curves ({model_name})",
            save_path=os.path.join(PLOTS_DIR, f"{model_name}_curves.png")
        )
        print(f"График сохранён: {os.path.join(PLOTS_DIR, f'{model_name}_curves.png')}")

    # Сравнение времени обучения
    logging.info("==== Время обучения (сек) для каждой модели ====")
    for model_name, t in all_times.items():
        logging.info(f"{model_name}: {t:.2f} сек")

    return all_histories, all_times

def test_mlp():
    """
    Тест: проверка корректности MLP на простом примере.
    """
    model = MLP(in_features=4, out_features=3, hidden_sizes=[5, 5])
    x = torch.randn(2, 4)
    out = model(x)
    assert out.shape == (2, 3), "Ошибка в размере выхода MLP"
    print("Unit-тест MLP: OK")

if __name__ == "__main__":
    # Тестирование MLP
    test_mlp()

    # Запуск эксперимента по глубине
    hidden_layers_list = [
        [],          # 1 слой (линейный)
        [64],        # 2 слоя (1 скрытый)
        [64, 64],    # 3 слоя (2 скрытых)
        [64, 64, 64, 64],  # 5 слоев (4 скрытых)
        [64, 64, 64, 64, 64, 64]  # 7 слоев (6 скрытых)
    ]
    input_dim = 2
    num_classes = 2

    histories, times = run_depth_experiment(
        hidden_layers_list,
        input_dim=input_dim,
        num_classes=num_classes,
        epochs=30,
        batch_size=64,
        seed=42
    )

    # Анализ результатов
    print("\n=== Итоговое сравнение точности и времени ===")
    for model_name in histories:
        train_acc = histories[model_name]["train_accs"][-1]
        test_acc = histories[model_name]["test_accs"][-1]
        total_time = times[model_name]
        print(f"{model_name}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, Time={total_time:.2f}s")
```

Для каждого варианта:
- Сравните точность на train и test
```
=== Итоговое сравнение точности и времени ===
1_layers: Train Acc=0.7775, Test Acc=0.7900, Time=0.97s
2_layers: Train Acc=0.7700, Test Acc=0.7850, Time=0.95s
3_layers: Train Acc=0.7850, Test Acc=0.7850, Time=0.93s
5_layers: Train Acc=0.7863, Test Acc=0.7850, Time=1.06s
7_layers: Train Acc=0.7788, Test Acc=0.7500, Time=1.24s
```

Для моделей с 1–5 слоями разница между train и test accuracy минимальна (менее 2%), что говорит об отсутствии явного переобучения. Это хороший признак: сеть не просто запоминает обучающие данные, но и хорошо обобщает на тесте.\
Для самой глубокой модели (7 слоев) разрыв между train и test accuracy увеличивается (2.88%), а тестовая точность снижается до 0.75 — это первый признак переобучения: сеть становится слишком сложной для объёма данных и начинает хуже обобщать.

- Визуализируйте кривые обучения

**1 слой:**
![1 слой](plots/1_layers_curves.png)

**2 слоя:**
![2 слоя](plots/2_layers_curves.png)

**3 слоя:**
![3 слоя](plots/3_layers_curves.png)

**5 слоев:**
![5 слоев](plots/5_layers_curves.png)

**7 слоев:**
![7 слоев](plots/7_layers_curves.png)

- Проанализируйте время обучения
```
=== Итоговое сравнение точности и времени ===
1_layers: Train Acc=0.7775, Test Acc=0.7900, Time=0.97s
2_layers: Train Acc=0.7700, Test Acc=0.7850, Time=0.95s
3_layers: Train Acc=0.7850, Test Acc=0.7850, Time=0.93s
5_layers: Train Acc=0.7863, Test Acc=0.7850, Time=1.06s
7_layers: Train Acc=0.7788, Test Acc=0.7500, Time=1.24s
```

Время обучения растёт с увеличением числа слоёв (особенно заметно для 5 и 7 слоёв), что ожидаемо: больше параметров — больше вычислений и времени на обратное распространение ошибки.\
Для малых и средних глубин (1–3 слоя) время почти одинаково, но начиная с 5 слоёв рост становится заметным.
