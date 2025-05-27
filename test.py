import os
import warnings
import time
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from pygments.styles.dracula import background
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from apscheduler.schedulers.background import BackgroundScheduler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

model_name = "blanchefort/rubert-base-cased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7'}

app = FastAPI()

def scrape_review():
    base_url = "https://otzyvmarketing.ru/"
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    divs = [
        div for div in soup.find_all('div', class_='text')
        if div.find("p")
    ]
    return divs
def send_to_n8n(text, label, score):
    payload = {
        "text":text,
        "label":label,
        "score":score
    }
    try:
        response = requests.post("http://localhost:5678/webhook/incoming-review", json=payload)
        print("sending to n8n... ",response.status_code)
    except Exception as e:
        print("error occured.:",e)
sent_reviews = set()
def review_delay(texts):
    for div in texts:
        cleaned = div.get_text(separator=" ", strip=True)
        if cleaned and cleaned not in sent_reviews:
            sentiment = classifier(cleaned)[0]
            label = sentiment["label"]
            score = float(sentiment["score"])
            send_to_n8n(cleaned, label, score)
            sent_reviews.add(cleaned)
            time.sleep(15)

@app.get("/analyze")
def analyze_sentiment(background_tasks: BackgroundTasks):
    texts = scrape_review()
    #results = []
    #sent_reviews = set()
    if not texts:
        return {"error":"No reviews found"}

    background_tasks.add_task(review_delay, texts)
    #return results
        #"message":"Started sending reviews",
        #"to_send": len(texts),

