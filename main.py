from utilities.create_objects import model_create
from utilities.load_split import load_and_split, dataload
from models.model import NeuralNetwork
from train_test.test import test_model
from train_test.train import run_model
from loss.focused_loss import focused_loss
from imports import epochs, output_model_dir, output_graphics_dir
from utilities.graphics import loss_graphics, metric_bars
import torch
import os

def main():
    train, test = load_and_split()

    (train_loader, test_loader) = dataload(train, test)

    class_weights = focused_loss()

    model, optimizer, loss, statistic = model_create(class_weights)

    # Проверка наличия файла с весами
    weights_path = "models/best.pt"
    if os.path.exists(weights_path):
        print(f"Загрузка весов из {weights_path}")
        model.load_state_dict(torch.load(weights_path))
    else:
        print(f"Файл с весами {weights_path} не найден. Веса не загружены.")

    if __name__ == "__main__":
        for i in range(epochs):
            print(f"Epoch #{i}")
            run_model(model, optimizer, train_loader, test_loader, loss)

        print("Завершение обучения после всех эпох")

    print("Начало тестирования модели")

    torch.save(model.state_dict(), output_model_dir + "over_epoch.pt")
    accuracy, precision, recall = test_model(model, test_loader, statistic)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

    loss_graphics(output_graphics_dir)
    metric_bars(output_graphics_dir, accuracy, precision, recall)

if __name__ == "__main__":
    main()
