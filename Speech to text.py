import whisper
import spacy
import json
import os

# ── spaCy Model ────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")


# ══════════════════════════════════════════════════
#  INVENTORY MODULE
# ══════════════════════════════════════════════════

def load_inventory_from_file(filepath: str) -> dict:
    inventory = {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                if len(parts) != 3:
                    print(f"[Line {line_num}] Invalid format, skipping: '{line}'")
                    continue

                name, quantity, price = parts
                name = name.strip()

                try:
                    quantity = int(quantity.strip())
                    price = float(price.strip())
                except ValueError:
                    print(f"[Line {line_num}] Invalid quantity/price, skipping: '{line}'")
                    continue

                inventory[name] = {
                    "quantity": quantity,
                    "price": price
                }

    except FileNotFoundError:
        print(f"File not found: {filepath}")

    return inventory


def display_inventory(inventory: dict):
    if not inventory:
        print("Inventory is empty.")
        return

    print(f"\n{'Name':<20} {'Quantity':>10} {'Price':>12}")
    print("-" * 44)
    for name, info in inventory.items():
        print(f"{name:<20} {info['quantity']:>10} {info['price']:>12,.0f}")


# ══════════════════════════════════════════════════
#  SPEECH-TO-TEXT MODULE
# ══════════════════════════════════════════════════

class FoodLibraryManager:
    def __init__(self, db_path='food_db.json'):
        self.db_path = db_path
        self.known_foods = self._load_db()
        self.units = {'gram', 'grams', 'g', 'kg', 'kilogram', 'kilograms', 'lbs', 'pound', 'pounds'}

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                try:
                    return set(json.load(f))
                except:
                    return set()
        return {'chicken', 'pork', 'beef', 'watermelon', 'flour'}

    def confirm_and_save(self, potential_food):
        response = input(f"❓ I found '{potential_food}'. Is this a food item? (y/n): ").lower()
        if response == 'y':
            self.known_foods.add(potential_food)
            with open(self.db_path, 'w') as f:
                json.dump(list(self.known_foods), f, indent=4)
            return True
        return False


def convert_to_grams(value, unit):
    unit = unit.lower()
    if unit in ['kg', 'kilogram', 'kilograms']:
        return int(value * 1000)
    if unit in ['lbs', 'pound', 'pounds']:
        return int(value * 453.59)
    return int(value)


def extract_smart(text, manager):
    doc = nlp(text.lower())
    found_data = {}
    food_context = nlp("food meat vegetable fruit ingredient")

    for i, token in enumerate(doc):
        if token.like_num:
            try:
                quantity = float(token.text)
            except ValueError:
                continue

            unit = ""
            food_item = ""

            start = max(0, i - 3)
            end = min(len(doc), i + 4)
            window = doc[start:end]

            for t in window:
                if t.text in manager.units:
                    unit = t.text
                elif t.pos_ in ["NOUN", "PROPN"] and t.text not in manager.units:
                    if t.similarity(food_context) > 0.25:
                        food_item = t.text

            if food_item:
                food_item = nlp(food_item)[0].lemma_

                if food_item not in manager.known_foods:
                    if manager.confirm_and_save(food_item):
                        found_data[food_item] = convert_to_grams(quantity, unit)
                else:
                    found_data[food_item] = convert_to_grams(quantity, unit)

    return found_data


def process_audio(mp3_file: str):
    """Transcribe an audio file and extract food items with quantities."""
    if not os.path.exists(mp3_file):
        print(f"❌ Error: {mp3_file} not found.")
        return {}

    manager = FoodLibraryManager()
    with open('library.json', 'w') as f:
        json.dump({}, f)

    print(f"--- Processing: {mp3_file} ---")
    model = whisper.load_model("base")
    result = model.transcribe(mp3_file, fp16=False)
    print(f"Transcript: {result['text']}")

    final_library = extract_smart(result['text'], manager)

    with open('library.json', 'w') as f:
        json.dump(final_library, f, indent=4)

    print("\n✅ Session Results:", final_library)
    return final_library


# ══════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== 1. INVENTORY FROM FILE ===")
    inventory = load_inventory_from_file("products.txt")
    display_inventory(inventory)

    print("\n=== 2. SPEECH-TO-TEXT FOOD EXTRACTION ===")
    process_audio(os.path.join(os.path.dirname(__file__), "Grocery.mp3"))
