import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

class AdaptiveDropout(nn.Module):
    """
    Dropout с адаптивным коэффициентом (меняется по эпохам).
    """
    def __init__(self, initial_p=0.5, schedule='linear', total_epochs=20):
        super().__init__()
        self.initial_p = initial_p
        self.schedule = schedule
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_p(self):
        if self.schedule == 'linear':
            return self.initial_p * (1 - self.current_epoch / self.total_epochs)
        elif self.schedule == 'exp':
            return self.initial_p * (0.95 ** self.current_epoch)
        else:
            return self.initial_p

    def forward(self, x):
        p = self.get_p()
        if not self.training or p == 0.0:
            return x
        return torch.nn.functional.dropout(x, p=p, training=True)

class CustomBatchNorm1d(nn.BatchNorm1d):
    """
    BatchNorm с возможностью динамически менять momentum.
    """
    def __init__(self, num_features, momentum=0.1):
        super().__init__(num_features, momentum=momentum)
        self._external_momentum = momentum

    def set_momentum(self, momentum):
        self.momentum = momentum
        self._external_momentum = momentum

class AdaptiveMLP(nn.Module):
    """
    Многослойный перцептрон с поддержкой адаптивного Dropout и BatchNorm.
    """
    def __init__(self, in_features, out_features, hidden_sizes,
                 dropout_init=0.0, dropout_schedule=None,
                 bn_momentums=None, use_batchnorm=False, combine=False, total_epochs=20):
        super().__init__()
        layers = []
        last_size = in_features
        n_layers = len(hidden_sizes)
        for i, hidden in enumerate(hidden_sizes):
            layers.append(nn.Linear(last_size, hidden))
            # BatchNorm с индивидуальным momentum для каждого слоя
            if use_batchnorm and bn_momentums is not None:
                layers.append(CustomBatchNorm1d(hidden, momentum=bn_momentums[i]))
            if combine or (use_batchnorm and bn_momentums is not None):
                layers.append(nn.ReLU())
            # Dropout с адаптивным коэффициентом
            if dropout_schedule is not None and dropout_init > 0:
                layers.append(AdaptiveDropout(dropout_init, schedule=dropout_schedule, total_epochs=total_epochs))
            last_size = hidden
        layers.append(nn.Linear(last_size, out_features))
        self.net = nn.Sequential(*layers)

    def set_epoch(self, epoch):
        """
        Устанавливает текущую эпоху для всех адаптивных слоёв Dropout и BatchNorm.
        """
        for module in self.modules():
            if isinstance(module, AdaptiveDropout):
                module.set_epoch(epoch)

    def set_bn_momentums(self, momentums):
        """
        Устанавливает momentum для всех BatchNorm слоёв.
        """
        idx = 0
        for module in self.modules():
            if isinstance(module, CustomBatchNorm1d):
                module.set_momentum(momentums[idx])
                idx += 1

    def forward(self, x):
        return self.net(x)

def train_and_log_adaptive_regularization(model, train_loader, test_loader, device, epochs=20):
    """
    Обучает модель с адаптивной регуляризацией, обновляет параметры Dropout и BatchNorm по эпохам.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    # Для BatchNorm: пример расписания momentum (можно сделать сложнее)
    base_momentums = [0.05, 0.1, 0.2]
    for epoch in range(epochs):
        # Адаптивный Dropout: обновить epoch
        model.set_epoch(epoch)
        # Пример: динамически менять momentum у BatchNorm (здесь просто фиксированное, можно сделать по эпохам)
        model.set_bn_momentums(base_momentums)

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).view(-1).long()
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += X_batch.size(0)
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Тест
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

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        logging.info(f"Epoch {epoch+1}/{epochs} | Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}")

    return history, history["test_acc"][-1]
