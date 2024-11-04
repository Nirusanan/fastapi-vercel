from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import difflib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
import uvicorn

app = FastAPI()


class TextInput(BaseModel):
    text: str

tokenizer = None
model = None
id2label = None

def load_model():
    global tokenizer, model, id2label
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained("nirusanan/tinyBert-keyword")
        model = AutoModelForTokenClassification.from_pretrained("nirusanan/tinyBert-keyword")
        id2label = model.config.id2label


# tokenizer = AutoTokenizer.from_pretrained("nirusanan/tinyBert-keyword")
# model = AutoModelForTokenClassification.from_pretrained("nirusanan/tinyBert-keyword")
# id2label = model.config.id2label


def predict(text, model, tokenizer, device="cpu"):
    model.eval()
    tokenized = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_predictions = [id2label[pred.item()] for pred in predictions[0]]

    entities = []
    current_entity = None

    for idx, (token, pred) in enumerate(zip(tokens, token_predictions)):
        if pred.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"type": pred[2:], "start": idx, "text": token}
        elif pred.startswith("I-") and current_entity:
            current_entity["text"] += f" {token}"
        elif current_entity:
            entities.append(current_entity)
            current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities


def clean_keyword(keyword):
    return keyword.replace(" ##", "")


# find the closest match for each keyword
def find_closest_word(keyword, word_positions):
    keyword_cleaned = clean_keyword(keyword['text'])
    best_match = None
    best_score = float('inf')  # lower is better for Levenshtein distance

    for pos, word in word_positions.items():
        # Calculate similarity (Levenshtein distance)
        score = difflib.SequenceMatcher(None, keyword_cleaned, word).ratio()
        # Select words with high similarity (threshold set to 0.8 here)
        if score > 0.8 and (best_match is None or score > best_score):
            best_match = word
            best_score = score

    return best_match or keyword_cleaned


class finalKeywords:
    def generate_final_keywords(self, text: str):
        load_model()
        keywords = []
        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            predictions = predict(paragraph, model, tokenizer, device="cpu")
            for i in predictions:
                keywords.append(i)

        words = text.split()
        word_positions = {i: word.strip(".,") for i, word in enumerate(words)}

        cleaned_keywords = []
        for keyword in keywords:
            closest_word = find_closest_word(keyword, word_positions)
            cleaned_keywords.append({'type': keyword['type'], 'start': keyword['start'], 'text': closest_word})

        unique_keywords = {}
        for item in cleaned_keywords:
            text = item['text'].lower()
            if text not in unique_keywords:
                unique_keywords[text] = item

        cleaned_keywords_unique = list(unique_keywords.values())
        if len(cleaned_keywords_unique) > 5:
            final_keywords = cleaned_keywords_unique[:5]
        else:
            final_keywords = cleaned_keywords_unique

        text_values = [item['text'] for item in final_keywords]
        return text_values


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/")
async def generate_keyphrases(request: TextInput):
    final_keywords_instance = finalKeywords()
    keyphrases = final_keywords_instance.generate_final_keywords(request.text)
    return {"keyphrases": keyphrases}

