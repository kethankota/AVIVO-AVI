import logging
import re
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes, ConversationHandler
from telegram import MessageEntity
from bot.router import route_text, route_image
from utils.history import get_history, add_to_history

logger = logging.getLogger(__name__)

help_text = """
*AVIVO AVI* — your AI assistant

*Commands:*
/ask <query> — answer from the knowledge base (RAG)
/image - generate caption with tags for the image
/summarize — summarise your last interaction 
/help — show this message
"""


def escape_markdown(text):
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)


def check_multiple_commands(message) -> bool:
    commands = message.parse_entities(types=[MessageEntity.BOT_COMMAND])
    return len(commands) > 1


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! I'm your GenAI assistant. Use /help to see what I can do."
    )


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_markdown(help_text)


async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text("Usage: /ask <query>")
        return

    if check_multiple_commands(update.message):
        await update.message.reply_markdown("""Usage: /ask <query>\n**Entered multiple commands**""")
        return

    user_id = update.effective_user.id
    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        result = await route_text(query, user_id)
        reply = result["answer"]
        if result.get("source"):
            reply += f"\n\n_Source: {result['source']}_"
        add_to_history(user_id, "text", query, reply)
        safe_answer = escape_markdown(reply)
        await update.message.reply_text(safe_answer, parse_mode="MarkdownV2")
    except Exception as e:
        logger.error("handle_ask error: %s", e)
        await update.message.reply_text("Sorry, something went wrong. Please try again.")


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    await update.message.chat.send_action(ChatAction.TYPING)

    if not update.message.photo:
        await update.message.reply_text(
            "Please attach a photo along with the /image command.\n"
            "Example: send a photo with caption `/image focus on the background`"
        )
        return
    user_hint = context.user_data.get("instruction", "")

    try:
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        result = await route_image(bytes(image_bytes), user_id, user_hint)
        caption = result["caption"]
        tags = result["tags"]
        reply = f"*Caption:* {caption}\n\n*Tags:* {', '.join(tags)}"
        add_to_history(user_id, "image", "[photo]", reply)
        await update.message.reply_markdown(reply)
    except Exception as e:
        logger.error("handle_image error: %s", e)
        await update.message.reply_text("Sorry, I couldn't process that image.")
    return ConversationHandler.END


async def handle_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    history = get_history(user_id)

    if not history:
        await update.message.reply_text("No history yet. Try /ask or send an image first.")
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        from bot.router import route_summarize
        summary = await route_summarize(history)
        await update.message.reply_markdown(f"*Summary of your last interactions:*\n\n{summary}")
    except Exception as e:
        logger.error("handle_summarize error: %s", e)
        await update.message.reply_text("Sorry, I couldn't summarise your history.")
