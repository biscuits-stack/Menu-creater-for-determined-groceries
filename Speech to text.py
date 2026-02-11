import whisper
import spacy
import re
import json
import os

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

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

    # Iterate through tokens to find numbers
    for i, token in enumerate(doc):
        if token.like_num:
            try:
                quantity = float(token.text)
            except ValueError:
                continue
                
            unit = ""
            food_item = ""

            # BROAD SEARCH: Look at 3 words before and 3 words after the number
            # This is more reliable for unpunctuated transcripts than dependency trees.
            start = max(0, i - 3)
            end = min(len(doc), i + 4)
            window = doc[start:end]

            for t in window:
                # 1. Check for Unit
                if t.text in manager.units:
                    unit = t.text
                # 2. Check for Food (Noun that isn't a unit)
                elif t.pos_ in ["NOUN", "PROPN"] and t.text not in manager.units:
                    # Semantic similarity check
                    if t.similarity(food_context) > 0.25: # Slightly lowered threshold
                        food_item = t.text

            if food_item:
                # Lemma helps match 'apples' to 'apple'
                food_item = nlp(food_item)[0].lemma_ 
                
                if food_item not in manager.known_foods:
                    if manager.confirm_and_save(food_item):
                        found_data[food_item] = convert_to_grams(quantity, unit)
                else:
                    found_data[food_item] = convert_to_grams(quantity, unit)

    return found_data

def main(mp3_file):
    if not os.path.exists(mp3_file):
        print(f"❌ Error: {mp3_file} not found.")
        return

    # Initialize manager and reset library.json
    manager = FoodLibraryManager()
    with open('library.json', 'w') as f:
        json.dump({}, f)

    print(f"--- Processing: {mp3_file} ---")
    model = whisper.load_model("base")
    # fp16=False is essential for CPU-only systems to avoid errors
    result = model.transcribe(mp3_file, fp16=False)
    print(f"Transcript: {result['text']}")

    final_library = extract_smart(result['text'], manager)
    
    with open('library.json', 'w') as f:
        json.dump(final_library, f, indent=4)
    
    print("\n✅ Session Results:", final_library)

if __name__ == "__main__":
    main("D:\Python data\Test 3\Grocery.mp3")