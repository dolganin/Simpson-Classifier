# Нейронная сеть для классификации персонажей Симпсонов

Этот репозиторий содержит реализацию сверточной нейронной сети (CNN) для классификации персонажей из мультсериала "Симпсоны". Модель обучается на наборе данных, содержащем изображения персонажей, и может быть использована для предсказания классов новых изображений.

## Структура репозитория
```bash

├── configs
│   └── config.yaml
├── imports.py
├── inference
│   └── inference.ipynb
├── LICENSE
├── loss
│   └── focused_loss.py
├── main.py
├── models
│   ├── model.py
├── README.md
├── requirements.txt
├── scripts
│   └── train.sh
├── train_test
│   ├── test.py
│   └── train.py
└── utilities
    ├── create_objects.py
    ├── graphics.py
    ├── load_split.py
    └── yaml_reader.py
```

## Архитектура модели

Модель представляет собой стандартную сверточную нейронную сеть (CNN) с несколькими слоями свертки, пулинга и полносвязными слоями. Архитектура модели описана в файле `models/model.py`:

```python
from imports import *
#Model-class
class NeuralNetwork(nn.Module):
    """
    We use standard CNN for this classification, without any tricks from ResNet, MobileNet or Inception. Maybe (?) these model will be
    rewritted with only one convolution block in different parameters. Bias in this model increase converge. But it may be a little bit
    overfitting.
    """
    __channels_list__ = [3, 16, 32, 64, 128, 256, 512]  # List of in/out channels for convolutions. If you have a lot of memory on GPU you can expand it.

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        def __block__(in_channels, out_channels):
            """
            This architecture of Neural Networks is obvious, I guess. BTW, without BN model converge in slowly in 3-4 times.
            """
            return nn.Sequential(
             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=True),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.BatchNorm2d(out_channels)
             )

        self.__conv_list__ = [__block__(self.__channels_list__[i], self.__channels_list__[i+1]) for i in range(len(self.__channels_list__)-1)]

        self.conv_stack = nn.Sequential()

        for conv in self.__conv_list__:
            self.conv_stack.extend(conv)

        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 120),
            nn.Linear(120, 84),
            nn.Dropout(dropout_rate),
            nn.Linear(84, classnum),
        )

    def forward(self, x):
        for conv in self.conv_stack:
            x = conv(x)
        x = self.linear_stack(x)
        return x
```

## Запуск обучения

Для запуска обучения используется скрипт `train.sh`, расположенный в папке `scripts`. Перед запуском необходимо сделать скрипт исполняемым:
```bash
chmod +x scripts/train.sh
```

Затем можно запустить скрипт:
```bash
./scripts/train.sh
```

Обучение автоматически начнется на GPU устройстве, если оно доступно.

## Конфигурация

Параметры обучения и модели задаются в файле `configs/config.yaml`:
```yml
dataset_parameters:
    testdir: "../datasets/the_simpson_dataset/test"
    traindir: "../datasets/the_simpson_dataset/train"

model_parameters:
    weight_decay: 0.0001
    dropout_rate: 0.1

training_parameters:
  batch_size: 64
  learning_rate: 0.002
  num_epochs: 150

output_parameters:
    part_of_partitions: 5
    out_model_directory: "model/"
    out_graphics_directory: "graphics/"
    out_inference_directory: "inference/"
```

Количество разбиений (`part_of_partitions`) используется для улучшения читаемости графика лосса, разделяя его на равные части.

## Графики и метрики

### Потери

![](graphics/"0th part of learning and testing.png")
После завершения обучения будут сохранены графики лосса и метрик (точность, полнота, F1-мера) в папке `graphics/`. Заготовки для графиков уже включены в код.

## Лицензия

Этот проект распространяется под лицензией MIT. Подробности см. в файле `LICENSE`.
