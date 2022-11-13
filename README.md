### Feature extractor — [Audio Spectrogram Transformer](https://github.com/YuanGongND/ast)

- Топ-3 модель на Audio Classification on ESC-50 датасете (лучшая из доступных).

- Препроцессинг - вычисление банка фильтров, взят из оригинальной модели

- Последний слой с классификатором был заменен на собственный.

### Classificator - MLP

- Классификатор - однослойный персептрон с ReLU активацией

    ```python
    classifier = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Linear(256, 41)
    )
    ```

### Модификации

1. Data Augmentation
    - Выбор произвольного фрагмента wav длиной от 4-х секунд
    - Смещение старта на случайную величину до 4000 сэмплов

2. Feature extractor
    - Добавить в качестве фичей выход MLP оригинальной модели 


Использование модификаций не дало существенных улучшений

### F1 score

 - 0.85576