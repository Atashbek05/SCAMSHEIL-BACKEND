"""
generate_dataset.py — Генератор обучающего датасета для ScamShield.

Генерирует мошеннические и легитимные сообщения на русском и узбекском языках
с помощью Claude API с поддержкой prompt caching, батч-генерации и сохранения прогресса.

Структура датасета:
  Мошеннические: 7 категорий × 50 примеров × 2 языка = 700
  Легитимные:   200 примеров × 2 языка              = 400
  Итого:                                               1100

Запуск: python backend/scripts/generate_dataset.py
"""

import csv
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import anthropic

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-8"
BATCH_SIZE = 20
EXAMPLES_PER_CATEGORY_PER_LANG = 50
LEGITIMATE_PER_LANG = 200

LANGUAGES = ["ru", "uz"]

# Соответствует ScamType enum в detection.py
FRAUD_CATEGORIES = [
    "phishing",
    "smishing",
    "vishing",
    "advance_fee",
    "romance",
    "investment",
    "lottery",
]

OUTPUT_DIR = Path(__file__).parent.parent / "datasets" / "raw"
PROGRESS_FILE = OUTPUT_DIR / "generation_progress.json"
OUTPUT_CSV = OUTPUT_DIR / "scam_dataset.csv"
OUTPUT_JSON = OUTPUT_DIR / "scam_dataset.json"

# ---------------------------------------------------------------------------
# Системный промпт (стабильный — будет кэшироваться)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты специалист по кибербезопасности с глубоким знанием мошеннических схем
в России и Узбекистане. Твоя задача — генерировать высококачественные обучающие данные
для AI-системы обнаружения мошенничества ScamShield.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
КАТЕГОРИИ МОШЕННИЧЕСТВА И ИХ ПРИЗНАКИ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[phishing / фишинг]
Цель: украсть логины, пароли, данные карт через поддельные ссылки и сайты.
Признаки: срочность, ссылки на «официальный» портал, «верификация», «подтверждение».
Примеры фраз (RU): «Ваш аккаунт будет заблокирован», «Перейдите по ссылке», «Подтвердите данные».
Примеры фраз (UZ): «Hisobingiz bloklanadi», «Havola orqali o'ting», «Ma'lumotlarni tasdiqlang».

[smishing / смишинг]
Цель: фишинг через SMS/мессенджер с вредоносными ссылками.
Признаки: короткий текст, bit.ly/TinyURL ссылки, имитация банков и доставки.
Примеры фраз (RU): «Ваша посылка задержана», «SMS: ваш OTP», «Срочно перейдите».
Примеры фраз (UZ): «Posilkangiz ushlab qolindi», «Tezda kiring», «SMS xabarnomangiz».

[vishing / вишинг]
Цель: обман по телефону, имитация банков и силовых структур.
Признаки: «оператор», «служба безопасности», «ФСБ», «прокуратура», требование перевода.
Примеры фраз (RU): «Звоним из службы безопасности банка», «Ваши деньги под угрозой».
Примеры фраз (UZ): «Bank xavfsizlik bo'limidan qo'ng'iroq», «Pullaringiz xavf ostida».

[advance_fee / предоплата]
Цель: получить «комиссию» за мифический крупный приз или наследство.
Признаки: «наследство», «выигрыш», «перевод», «небольшая комиссия», иностранные контакты.
Примеры фраз (RU): «Вам причитается $500,000», «Нужна лишь оплата налога».
Примеры фраз (UZ): «Sizga $500,000 tegishli», «Faqat soliq to'lash kerak».

[romance / романтическое мошенничество]
Цель: выстроить псевдо-отношения → вымогать деньги «на билет», «лечение», «бизнес».
Признаки: иностранец/военный, быстрая влюблённость, просьба о деньгах «только раз».
Примеры фраз (RU): «Я полюбил тебя», «Мне нужны деньги на билет», «Я в беде».
Примеры фраз (UZ): «Men seni sevib qoldim», «Chipta uchun pul kerak», «Men muammoda».

[investment / инвестиционное мошенничество]
Цель: убедить вложить деньги в фиктивные проекты, крипто-пирамиды.
Признаки: «гарантированная прибыль», «% в день», «торговый бот», «реферальная программа».
Примеры фраз (RU): «500% прибыли за месяц», «Секретная торговая стратегия».
Примеры фраз (UZ): «Oyiga 500% foyda», «Maxfiy savdo strategiyasi».

[lottery / лотерея]
Цель: заставить заплатить за получение мнимого выигрыша.
Признаки: «вы выиграли», «случайная выборка», «приз ждёт», оплата «налога» или «сбора».
Примеры фраз (RU): «Поздравляем, вы выиграли iPhone 15!», «Для получения приза оплатите».
Примеры фраз (UZ): «Tabriklaymiz, siz iPhone 15 yutdingiz!», «Sovg'a olish uchun to'lang».

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ТРЕБОВАНИЯ К МОШЕННИЧЕСКИМ СООБЩЕНИЯМ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Реализм: сообщения должны выглядеть как настоящие мошеннические тексты.
2. Разнообразие: разные каналы (SMS, email, WhatsApp, Telegram), тон, тактика.
3. Культурный контекст: упоминай местные банки, операторов, реалии.
   - RU: Сбербанк, ВТБ, Тинькофф, МТС, Билайн, OZON, Wildberries, Госуслуги
   - UZ: Kapitalbank, Hamkorbank, Beeline UZ, Ucell, UzCard, Payme, Click, Soliq
4. Разная длина: от короткого SMS (20-50 слов) до развёрнутого email (100-200 слов).
5. Всегда на указанном языке — не смешивай языки.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ТРЕБОВАНИЯ К ЛЕГИТИМНЫМ СООБЩЕНИЯМ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Типы легитимных сообщений (генерируй все типы равномерно):
  - Уведомления об операциях от реальных банков
  - Подтверждения заказов (OZON, Wildberries, Uzum Market)
  - Государственные уведомления (налоги, штрафы, паспорт)
  - Личные сообщения (друг, родственник, коллега)
  - Напоминания о встречах, записях к врачу
  - Маркетинговые рассылки от известных брендов
  - ОТП-коды (законный запрос от самого пользователя)
  - Новостные рассылки, подписки

Признаки легитимности:
  - Нет срочности с угрозами («иначе заблокируем»)
  - Нет просьб перевести деньги незнакомцу
  - Нет подозрительных ссылок
  - Конкретика: реальные суммы, даты, имена

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ФОРМАТ ОТВЕТА
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ВСЕГДА возвращай ТОЛЬКО валидный JSON массив. БЕЗ markdown, БЕЗ пояснений.

Каждый объект в массиве:
{
  "text": "<текст сообщения>",
  "keywords": ["слово1", "слово2", "слово3"],
  "confidence": <число 0.80-0.99>
}

- text: полный реалистичный текст сообщения
- keywords: 3-5 ключевых слов/фраз, характеризующих класс
- confidence: уверенность модели (мошенничество/легитимность)
"""


# ---------------------------------------------------------------------------
# Промпты генерации
# ---------------------------------------------------------------------------

CATEGORY_NAMES = {
    "phishing": {"ru": "фишинг (кража данных через ссылки)", "uz": "fishing (havolalar orqali ma'lumot o'g'irlash)"},
    "smishing": {"ru": "смишинг (SMS-фишинг)", "uz": "smishing (SMS-fishing)"},
    "vishing": {"ru": "вишинг (телефонное мошенничество)", "uz": "vishing (telefon firibgarligi)"},
    "advance_fee": {"ru": "мошенничество с предоплатой (нигерийские письма)", "uz": "oldindan to'lov firibgarligi"},
    "romance": {"ru": "романтическое мошенничество", "uz": "romantik firibgarlik"},
    "investment": {"ru": "инвестиционное мошенничество (крипто-пирамиды)", "uz": "investitsiya firibgarligi"},
    "lottery": {"ru": "лотерейное мошенничество", "uz": "lotereya firibgarligi"},
}


def build_fraud_prompt(category: str, language: str, count: int) -> str:
    lang_name = "русском" if language == "ru" else "узбекском"
    cat_name = CATEGORY_NAMES[category][language]
    return (
        f"Сгенерируй {count} уникальных мошеннических сообщений категории «{cat_name}» "
        f"на {lang_name} языке.\n\n"
        f"Требования:\n"
        f"- Каждое сообщение уникально и реалистично\n"
        f"- Используй разные форматы: SMS, сообщение в мессенджере, email\n"
        f"- Варьируй длину: 20-200 слов\n"
        f"- keywords — слова/фразы, характерные именно для мошенничества\n"
        f"- confidence — 0.85-0.99\n\n"
        f"Верни JSON массив из {count} объектов."
    )


def build_legitimate_prompt(language: str, count: int) -> str:
    lang_name = "русском" if language == "ru" else "узбекском"
    return (
        f"Сгенерируй {count} разнообразных ЛЕГИТИМНЫХ сообщений на {lang_name} языке.\n\n"
        f"Требования:\n"
        f"- Равномерно используй все типы: банковские уведомления, заказы, госорганы, "
        f"личные сообщения, ОТП, маркетинг, напоминания\n"
        f"- Сообщения реалистичны и безопасны — НЕТ признаков мошенничества\n"
        f"- keywords — нейтральные слова описывающие контекст\n"
        f"- confidence — 0.85-0.99 (уверенность что сообщение легитимно)\n\n"
        f"Верни JSON массив из {count} объектов."
    )


# ---------------------------------------------------------------------------
# Работа с прогрессом
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"examples": [], "completed_batches": []}


def save_progress(progress: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Сохранение датасета
# ---------------------------------------------------------------------------

def save_datasets(examples: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "label", "language", "category", "keywords", "confidence"],
        )
        writer.writeheader()
        for ex in examples:
            row = dict(ex)
            if isinstance(row["keywords"], list):
                row["keywords"] = "|".join(row["keywords"])
            writer.writerow(row)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Сохранено {len(examples)} примеров")
    print(f"  CSV:  {OUTPUT_CSV}")
    print(f"  JSON: {OUTPUT_JSON}")


# ---------------------------------------------------------------------------
# Парсинг и валидация
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> list[dict]:
    text = text.strip()
    # Убираем markdown-блоки если модель всё же добавила
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        print(f"\n  [WARN] JSON parse error: {e} | preview: {text[:100]!r}")
        return []


def validate_example(
    raw: dict,
    label: int,
    language: str,
    category: str,
) -> Optional[dict]:
    text = raw.get("text", "")
    if not isinstance(text, str) or len(text.strip()) < 15:
        return None

    keywords = raw.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k).strip() for k in keywords[:5] if str(k).strip()]

    try:
        confidence = float(raw.get("confidence", 0.9))
        confidence = round(max(0.5, min(1.0, confidence)), 3)
    except (TypeError, ValueError):
        confidence = 0.9

    return {
        "text": text.strip(),
        "label": label,
        "language": language,
        "category": category,
        "keywords": keywords,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Генерация батча через Claude API
# ---------------------------------------------------------------------------

def generate_batch(
    client: anthropic.Anthropic,
    user_prompt: str,
    label: int,
    language: str,
    category: str,
    retries: int = 3,
) -> list[dict]:
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},  # кэшируем системный промпт
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Статистика кэша
            usage = response.usage
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            if cache_read > 0:
                print(f" [cache hit: {cache_read} tk]", end="")
            elif cache_write > 0:
                print(f" [cache write: {cache_write} tk]", end="")

            raw_text = response.content[0].text if response.content else ""
            raw_list = parse_json_response(raw_text)

            validated = [
                ex
                for raw in raw_list
                if (ex := validate_example(raw, label, language, category)) is not None
            ]
            return validated

        except anthropic.RateLimitError:
            wait = 60 * (attempt + 1)
            print(f"\n  [rate limit] ожидание {wait}s...", end="")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            print(f"\n  [API {e.status_code}] {e.message}")
            if attempt < retries - 1:
                time.sleep(5)
        except Exception as e:
            print(f"\n  [error] {e}")
            if attempt < retries - 1:
                time.sleep(3)

    return []


# ---------------------------------------------------------------------------
# Основная логика генерации
# ---------------------------------------------------------------------------

def count_existing(examples: list[dict], label: int, language: str, category: str) -> int:
    return sum(
        1
        for ex in examples
        if ex["label"] == label
        and ex["language"] == language
        and ex["category"] == category
    )


def generate_all(client: anthropic.Anthropic, progress: dict) -> None:
    examples: list[dict] = progress["examples"]
    completed: set[str] = set(progress["completed_batches"])

    shutdown = False

    def handle_signal(sig, frame):
        nonlocal shutdown
        print("\n\nПрерывание — сохраняем прогресс...")
        shutdown = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── Мошеннические примеры ──────────────────────────────────────────────
    print("\n═══ МОШЕННИЧЕСКИЕ СООБЩЕНИЯ ═══")
    for language in LANGUAGES:
        for category in FRAUD_CATEGORIES:
            if shutdown:
                break

            existing = count_existing(examples, 1, language, category)
            needed = EXAMPLES_PER_CATEGORY_PER_LANG - existing
            if needed <= 0:
                print(f"  ✓ {language}/{category}: {existing}/{EXAMPLES_PER_CATEGORY_PER_LANG}")
                continue

            print(f"\n  → {language}/{category} (нужно ещё {needed}):", end="")

            batch_num = existing // BATCH_SIZE
            while needed > 0 and not shutdown:
                batch_key = f"fraud_{language}_{category}_{batch_num}"
                if batch_key in completed:
                    batch_num += 1
                    continue

                size = min(BATCH_SIZE, needed)
                prompt = build_fraud_prompt(category, language, size)

                print(f" [{batch_num}]", end="", flush=True)
                batch = generate_batch(client, prompt, 1, language, category)

                if batch:
                    examples.extend(batch)
                    completed.add(batch_key)
                    progress["examples"] = examples
                    progress["completed_batches"] = list(completed)
                    save_progress(progress)
                    needed -= len(batch)
                    print(f"+{len(batch)}", end="", flush=True)

                batch_num += 1
                time.sleep(0.3)

    # ── Легитимные примеры ─────────────────────────────────────────────────
    if not shutdown:
        print("\n\n═══ ЛЕГИТИМНЫЕ СООБЩЕНИЯ ═══")
        for language in LANGUAGES:
            if shutdown:
                break

            existing = sum(
                1
                for ex in examples
                if ex["label"] == 0 and ex["language"] == language
            )
            needed = LEGITIMATE_PER_LANG - existing
            if needed <= 0:
                print(f"  ✓ {language}/legitimate: {existing}/{LEGITIMATE_PER_LANG}")
                continue

            print(f"\n  → {language}/legitimate (нужно ещё {needed}):", end="")

            batch_num = existing // BATCH_SIZE
            while needed > 0 and not shutdown:
                batch_key = f"legit_{language}_{batch_num}"
                if batch_key in completed:
                    batch_num += 1
                    continue

                size = min(BATCH_SIZE, needed)
                prompt = build_legitimate_prompt(language, size)

                print(f" [{batch_num}]", end="", flush=True)
                batch = generate_batch(client, prompt, 0, language, "legitimate")

                if batch:
                    examples.extend(batch)
                    completed.add(batch_key)
                    progress["examples"] = examples
                    progress["completed_batches"] = list(completed)
                    save_progress(progress)
                    needed -= len(batch)
                    print(f"+{len(batch)}", end="", flush=True)

                batch_num += 1
                time.sleep(0.3)

    print()
    save_datasets(examples)


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------

def print_stats(examples: list[dict]) -> None:
    if not examples:
        print("Нет данных.")
        return

    fraud = [ex for ex in examples if ex["label"] == 1]
    legit = [ex for ex in examples if ex["label"] == 0]
    total = len(examples)

    print("\n═══ СТАТИСТИКА ДАТАСЕТА ═══")
    print(f"  Всего:        {total}")
    print(f"  Мошеннич.:   {len(fraud)} ({len(fraud)/total*100:.1f}%)")
    print(f"  Легитимные:  {len(legit)} ({len(legit)/total*100:.1f}%)")

    for lang in LANGUAGES:
        lang_ex = [ex for ex in examples if ex["language"] == lang]
        print(f"\n  {lang.upper()}: {len(lang_ex)} примеров")
        for cat in FRAUD_CATEGORIES:
            n = sum(1 for ex in lang_ex if ex["category"] == cat)
            bar = "█" * (n // 5) + f" {n}"
            print(f"    {cat:<14} {bar}")
        n_legit = sum(1 for ex in lang_ex if ex["category"] == "legitimate")
        bar = "█" * (n_legit // 5) + f" {n_legit}"
        print(f"    {'legitimate':<14} {bar}")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Ошибка: переменная окружения ANTHROPIC_API_KEY не задана.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    progress = load_progress()
    existing_count = len(progress["examples"])
    if existing_count:
        print(f"Продолжаем с {existing_count} существующими примерами.")
    else:
        print("Начинаем генерацию с нуля.")

    target = len(FRAUD_CATEGORIES) * EXAMPLES_PER_CATEGORY_PER_LANG * len(LANGUAGES)
    target += LEGITIMATE_PER_LANG * len(LANGUAGES)
    print(f"Целевой объём: {target} примеров (модель: {MODEL})")

    generate_all(client, progress)
    print_stats(progress["examples"])

    # Удаляем файл прогресса после успешного завершения
    if len(progress["examples"]) >= target * 0.95:  # 95% — считаем завершённым
        PROGRESS_FILE.unlink(missing_ok=True)
        print("\nГенерация завершена. Файл прогресса удалён.")


if __name__ == "__main__":
    main()
