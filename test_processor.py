from processor import process_document

if __name__ == "__main__":
    test_image_path = "test_image.jpg"
    try:
        print("DEBUG: Starting test for process_document")
        result = process_document(test_image_path)
        print(f"DEBUG: Function output type: {type(result)}")
        print(f"DEBUG: Function output: {result}")
        # Проверим распаковку
        extracted_text, visualization_path = result
        print(f"DEBUG: Extracted text: {extracted_text}")
        print(f"DEBUG: Visualization path: {visualization_path}")
    except Exception as e:
        print(f"DEBUG: Error occurred: {str(e)}")
