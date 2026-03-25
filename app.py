import logging
import os
from dotenv import load_dotenv
from telegram.ext import ContextTypes, ApplicationBuilder, CommandHandler, MessageHandler, filters, ConversationHandler
from telegram import Update, ReplyKeyboardRemove
from bot.handlers import handle_ask, handle_help, handle_image, handle_start, handle_summarize, check_multiple_commands

load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

token = os.environ.get("TELEGRAM_TOKEN")

photo_state = 1


async def start_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if check_multiple_commands(update.message):
        await update.message.reply_text("""Usage: /image <optional_text>\n**Entered multiple commands**""")
        return

    context.user_data["instruction"] = " ".join(context.args).strip()
    await update.message.reply_text(
        "📸 Please upload the image you'd like me to summarize.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return photo_state


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Image summary cancelled.")
    return ConversationHandler.END


def main() -> None:

    if not token:
        raise RuntimeError("TELEGRAM_TOKEN is not set in environment / .env")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("help", handle_help))
    app.add_handler(CommandHandler("ask", handle_ask))
    app.add_handler(CommandHandler("summarize", handle_summarize))
    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("image", start_image_command)],
        states={
            photo_state: [MessageHandler(filters.PHOTO, handle_image)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    ))
    # app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(
        filters.COMMAND | filters.TEXT, handle_help))
    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
