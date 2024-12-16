from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from processor import process_document
import os

# Обработчик команды /start
async def start(update: Update, context):
    await update.message.reply_text(
        "Привет! Отправьте мне изображение документа, и я извлеку из него текст."
    )

# Обработчик изображений
async def handle_image(update: Update, context):
    photo = update.message.photo[-1] if update.message.photo else update.message.document
    file = await photo.get_file()
    file_path = "temp_image.jpg"
    await file.download_to_drive(file_path)

    try:
        print(f"DEBUG: About to process file at path {file_path}")
        extracted_text, visualized_path = process_document(file_path)
        await update.message.reply_text(f"Извлеченный текст:\n{extracted_text}")
        await context.bot.send_photo(chat_id=update.message.chat_id, photo=open(visualized_path, "rb"))
    except Exception as e:
        print(f"DEBUG: Error occurred in bot: {str(e)}")
        await update.message.reply_text(f"Произошла ошибка: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        if 'visualized_path' in locals() and os.path.exists(visualized_path):
            os.remove(visualized_path)

# Основной запуск
if __name__ == "__main__":
    import platform
    import asyncio

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    application = ApplicationBuilder().token("7469991610:AAGRju-1gp8FLHU80v51vgrOkpuUm_iMxRg").build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))

    application.run_polling()
