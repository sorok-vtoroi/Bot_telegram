from datasets import load_dataset, Features, Sequence, Value, Array2D, Array3D
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification, TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
from seqeval.metrics import classification_report, f1_score
import numpy as np
import os

# Загрузка датасета FUNSD
def load_funsd_dataset():
    dataset = load_dataset("nielsr/funsd-layoutlmv3")
    return dataset

# Подготовка данных
def prepare_datasets(dataset, processor, label_list):
    def prepare_examples(examples):
        images = examples["image"]
        words = examples["tokens"]
        boxes = examples["bboxes"]
        word_labels = examples["ner_tags"]

        encoding = processor(
            images,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length"
        )
        return encoding

    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(feature=Value(dtype='int64')),
    })

    column_names = dataset["train"].column_names
    train_dataset = dataset["train"].map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features,
    )
    eval_dataset = dataset["test"].map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features,
    )

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    return train_dataset, eval_dataset

# Добавление функции метрик
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Убираем токены заполнителей ([PAD])
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Вычисление метрик с помощью seqeval
    f1 = f1_score(true_labels, true_predictions)
    print(classification_report(true_labels, true_predictions))

    return {
        "f1": f1
    }

# Обучение модели
def train_model():
    # Загрузка датасета
    dataset = load_funsd_dataset()

    # Загрузка процессора
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    # Настройка меток
    label_list = dataset["train"].features["ner_tags"].feature.names
    global id2label, label2id
    id2label = {k: v for k, v in enumerate(label_list)}
    label2id = {v: k for k, v in enumerate(label_list)}
    num_labels = len(label_list)

    # Подготовка данных
    train_dataset, eval_dataset = prepare_datasets(dataset, processor, label_list)

    # Инициализация модели
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels,
    )

    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir="./results",
        max_steps=1000,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="f1",  # Используем F1 как метрику для лучшей модели
    )

    # Инициализация тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,  # Подключение метрики F1
    )

    # Запуск обучения
    trainer.train()

    # Сохранение модели
    model_dir = "./fine_tuned_model"
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)

    print(f"Модель сохранена в {model_dir}")

if __name__ == "__main__":
    train_model()
