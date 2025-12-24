# ==========================================
# 📦 File: bot/telegram_bot.py
# ==========================================
import sys
import os
import tempfile
import logging
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.request import HTTPXRequest
from telegram.helpers import escape_markdown

# === Tambahkan path agar bisa import predict_pipeline ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.predict_pipeline import predict_pipeline

# === TOKEN BOT TELEGRAM ===
BOT_TOKEN = "8050295895:AAFJm7S9yCyuN4dA0_BQfilI-k5t2zahzLU"

# === Logging setup ===
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# === State per user ===
user_last_photo = {}   # user_id -> path gambar
user_state = {}        # user_id -> "foto" | "suhu" | "kelembapan"

# === Keyboard untuk suhu & kelembapan ===
suhu_keyboard = ReplyKeyboardMarkup(
    [["dingin", "normal", "panas"]], one_time_keyboard=True, resize_keyboard=True
)
kelembapan_keyboard = ReplyKeyboardMarkup(
    [["rendah", "sedang", "tinggi"]], one_time_keyboard=True, resize_keyboard=True
)

# ==========================================
# 🟢 /start handler
# ==========================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_state[user_id] = "foto"
    logger.info("[START] %s memulai bot", update.message.from_user.first_name)

    await update.message.reply_text(
        "🌶️ *Selamat datang di Bot Deteksi Hama Cabai!* 🌿\n\n"
        "📋 Langkah penggunaan:\n"
        "1️⃣ Kirim *foto daun cabai* yang ingin diperiksa.\n"
        "2️⃣ Pilih kondisi *suhu* (dingin / normal / panas).\n"
        "3️⃣ Pilih *kelembapan* (rendah / sedang / tinggi).\n\n"
        "📸 Silakan kirim foto daun terlebih dahulu.",
        parse_mode="Markdown",
    )

# ==========================================
# 🖼️ Handle Foto
# ==========================================
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    user_id = user.id
    user_state[user_id] = "suhu"

    logger.info("[PHOTO] Foto diterima dari %s (ID: %s)", user.first_name, user_id)
    photo = update.message.photo[-1]  # ambil resolusi tertinggi
    file = await photo.get_file()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            await file.download_to_drive(tmp.name)
            user_last_photo[user_id] = tmp.name
            logger.info("[PHOTO] Disimpan sementara di: %s", tmp.name)
    except Exception as e:
        logger.exception("[ERROR] Gagal mengunduh foto: %s", e)
        await update.message.reply_text("⚠️ Gagal mengunduh foto, coba kirim ulang ya!")
        return

    await update.message.reply_text(
        "✅ Gambar daun berhasil diterima!\n\n"
        "Sekarang pilih kondisi *suhu* lingkungan 🌡️\n"
        "• ❄️ *Dingin* (<25°C)\n"
        "• 🌤️ *Normal* (25–30°C)\n"
        "• 🔥 *Panas* (>30°C)\n\n"
        "Silakan pilih salah satu:",
        parse_mode="Markdown",
        reply_markup=suhu_keyboard,
    )

# ==========================================
# 🌡️ Handle Suhu & Kelembapan
# ==========================================
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    user_id = user.id
    text = update.message.text.strip().lower()
    logger.info("[TEXT] %s -> %s", user.first_name, text)

    # Validasi urutan langkah
    if user_id not in user_state or user_state[user_id] == "foto":
        await update.message.reply_text("⚠️ Kirim foto daun dulu sebelum memilih suhu ya 🌿")
        return

    # === Tahap pilih suhu ===
    if user_state[user_id] == "suhu":
        if text not in ["dingin", "normal", "panas"]:
            await update.message.reply_text(
                "⚠️ Pilihan tidak valid. Gunakan tombol di bawah:",
                reply_markup=suhu_keyboard,
            )
            return

        user_state[user_id] = "kelembapan"
        context.user_data["suhu"] = text

        await update.message.reply_text(
            f"🌡️ Suhu terpilih: *{text.capitalize()}*\n\n"
            "Sekarang pilih kondisi *kelembapan udara* 💧\n"
            "• 💧 *Rendah* (<60%)\n"
            "• 🌤️ *Sedang* (60–75%)\n"
            "• 💦 *Tinggi* (>75%)",
            parse_mode="Markdown",
            reply_markup=kelembapan_keyboard,
        )
        return

    # === Tahap pilih kelembapan ===
    if user_state[user_id] == "kelembapan":
        if text not in ["rendah", "sedang", "tinggi"]:
            await update.message.reply_text(
                "⚠️ Pilihan tidak valid. Gunakan tombol di bawah:",
                reply_markup=kelembapan_keyboard,
            )
            return

        suhu = context.user_data.get("suhu")
        kelembapan = text
        image_path = user_last_photo.get(user_id)

        await update.message.reply_text(
            "🔍 Sedang menganalisis daun... Mohon tunggu sebentar ⏳",
            reply_markup=ReplyKeyboardRemove(),
        )

        try:
            # === Jalankan prediksi ===
            result = predict_pipeline(image_path, suhu, kelembapan)

            # Escape semua teks agar aman dari error Markdown
            daun = escape_markdown(result["daun"], version=2)
            hama = escape_markdown(result["hama"], version=2)
            mitigasi = escape_markdown(result["mitigasi"], version=2)
            suhu_escaped = escape_markdown(suhu, version=2)
            kelembapan_escaped = escape_markdown(kelembapan, version=2)

            # === Format hasil analisis ===
            msg = (
                f"🌿 *HASIL DETEKSI HAMA CABAI*\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"📷 *Daun:* {daun}\n"
                f"🌡️ *Suhu:* {suhu_escaped}\n"
                f"💧 *Kelembapan:* {kelembapan_escaped}\n"
                f"🐛 *Hama:* {hama}\n\n"
                f"🧩 *Mitigasi Disarankan:*\n{mitigasi}\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"Kirim foto lain jika ingin analisis baru 🌶️"
            )

            # === Kirim hasil ke Telegram ===
            await update.message.reply_photo(
                photo=open(image_path, "rb"),
                caption=msg,
                parse_mode="MarkdownV2",
            )

            logger.info("[RESULT] %s → %s", user.first_name, result)

        except Exception as e:
            logger.exception("❌ Gagal saat prediksi: %s", e)
            await update.message.reply_text("❌ Terjadi kesalahan, coba kirim ulang fotonya!")

        finally:
            # Bersihkan file & state sementara
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
            user_last_photo.pop(user_id, None)
            user_state.pop(user_id, None)
            context.user_data.clear()

# ==========================================
# 📘 /help handler
# ==========================================
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📘 *Panduan Bot:*\n"
        "1️⃣ Kirim foto daun cabai.\n"
        "2️⃣ Pilih suhu (dingin/normal/panas).\n"
        "3️⃣ Pilih kelembapan (rendah/sedang/tinggi).\n"
        "🤖 Bot akan menampilkan hasil deteksi hama dan saran mitigasi.",
        parse_mode="Markdown",
    )

# ==========================================
# 🚀 MAIN
# ==========================================
def main():
    print("🚀 Inisialisasi Bot Telegram...")
    request = HTTPXRequest(read_timeout=60, connect_timeout=30)
    app = ApplicationBuilder().token(BOT_TOKEN).request(request).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("🤖 Bot Telegram aktif dan siap menerima pesan!\n")
    app.run_polling(poll_interval=3)

if __name__ == "__main__":
    main()
