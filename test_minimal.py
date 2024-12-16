def process_document(image_path):
    try:
        print("DEBUG: Simulating process_document")
        # Возвращаем ровно два значения для теста
        return "Dummy extracted text", "Dummy visualized path"
    except Exception as e:
        raise RuntimeError(f"Ошибка при обработке документа: {str(e)}")

if __name__ == "__main__":
    test_image_path = "test_image.jpg"  # Укажите любой путь для теста
    try:
        print("DEBUG: Starting test for process_document")
        result = process_document(test_image_path)
        print("DEBUG: Function output:", result)
    except Exception as e:
        print(f"DEBUG: Error occurred during test: {str(e)}")
