#!/bin/bash

# Имя виртуального окружения
VENV_NAME="simpson_venv"

# Проверка существования виртуального окружения
if [ ! -d "$VENV_NAME" ]; then
    echo "Создание виртуального окружения $VENV_NAME..."
    python3 -m venv $VENV_NAME
    echo "Виртуальное окружение создано."
else
    echo "Виртуальное окружение $VENV_NAME уже существует."
fi

# Активация виртуального окружения
echo "Активация виртуального окружения..."
source $VENV_NAME/bin/activate

# Установка или обновление зависимостей
echo "Установка/обновление зависимостей..."
pip install -r requirements.txt

# Запуск main.py с переданными аргументами
echo "Запуск main.py с аргументами: $@"
python3 main.py "$@"

# Деактивация виртуального окружения (опционально)
deactivate
echo "Скрипт завершён."