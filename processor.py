from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
from PIL import Image, ImageDraw, ImageFont
import torch

# Инициализация процессора и модели
model = LayoutLMv3ForTokenClassification.from_pretrained("./fine_tuned_model")
processor = AutoProcessor.from_pretrained("./fine_tuned_model", apply_ocr=False)


def process_document(image_path):
    try:
        print("DEBUG: process_document started")

        # Загружаем изображение
        image = Image.open(image_path).convert("RGB")
        print(f"DEBUG: Image loaded successfully, size={image.size}")
        image_width, image_height = image.size

        # Прогоняем изображение через модель
        encoding = processor(image, return_tensors="pt", truncation=True)
        print(f"DEBUG: Encoding generated: {encoding.keys()}")

        outputs = model(**{k: v for k, v in encoding.items()})
        predictions = torch.argmax(outputs.logits, dim=2)

        # Извлекаем токены
        tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())
        print(f"DEBUG: Tokens extracted: {tokens[:10]}...")  # Первые 10 токенов

        token_boxes = encoding["bbox"].squeeze().tolist()
        print(f"DEBUG: Token boxes extracted: {len(token_boxes)} boxes")

        # Извлекаем метки
        labels = [model.config.id2label[pred] for pred in predictions.squeeze().tolist()]
        print(f"DEBUG: Labels extracted: {labels[:10]}...")  # Первые 10 меток

        # Визуализация
        draw = ImageDraw.Draw(image, "RGBA")
        font = ImageFont.load_default()
        for token, label, box in zip(tokens, labels, token_boxes):
            if token.strip():  # Игнорируем пустые токены
                unnormalized_box = [
                    box[0] / 1000 * image_width,
                    box[1] / 1000 * image_height,
                    box[2] / 1000 * image_width,
                    box[3] / 1000 * image_height
                ]
                draw.rectangle(unnormalized_box, outline="blue", width=2)
                draw.text((unnormalized_box[0] + 5, unnormalized_box[1] - 10), token, fill="black", font=font)

        # Сохраняем изображение с разметкой
        visualized_path = "processed_image.jpg"
        image.save(visualized_path)
        print(f"DEBUG: Visualization saved to {visualized_path}")

        # Очистка текста
        extracted_text = " ".join(
            token.replace("Ġ", "").strip() for token in tokens if token not in ["<s>", "</s>", "<pad>"]
        )
        print(f"DEBUG: Extracted text: {extracted_text[:100]}...")  # Показываем первые 100 символов

        return extracted_text, visualized_path

    except Exception as e:
        raise RuntimeError(f"Ошибка при обработке документа: {str(e)}")
