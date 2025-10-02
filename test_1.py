import os
import time

import httpx
import requests
import asyncio

from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "blanchefort/rubert-base-cased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7'}

app = FastAPI()

scraping_targets = [
    {
        'url':"https://otzyvmarketing.ru/",
        'selector':"div.text"
    }
]
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD","OPTIONS","GET"]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount('https://',adapter)
session.mount('http://',adapter)

sent_reviews = set()

async def scrape_review():
    results = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for target in scraping_targets:
            try:
                response = await client.get(target["url"], headers=headers)
                response.raise_for_status()
            except httpx.RequestError as e:
                print(f"Failed to fetch {target["url"]}: {e}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            elements = soup.select(target["selector"])
            for element in elements:
                paragraph = [p.get_text(strip=True) for p in element.find_all('p')]
                review = " ".join(paragraph).strip()
                if review:
                    results.append(review)

        return results
async def send_to_n8n(text, label, score):
    payload = {
        "text":text,
        "label":label,
        "score":score
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post("https://n8n-production-af79.up.railway.app/webhook/collect_analyzed_reviews", json=payload)
            print("sending to n8n... ",response.status_code)
        except httpx.RequestError as e:
            print(f"error occured.:, {repr(e)}")

async def process_reviews(reviews):
    for review in reviews:
        if review and review not in sent_reviews:
            try:
                sentiment = classifier(review)[0]
                label = sentiment["label"]
                score = float(sentiment["score"])
                await send_to_n8n(review, label, score)
                sent_reviews.add(review)
            except Exception as e:
                print(f"Exception occured,{e}")
            await asyncio.sleep(1)
@app.get("/analyze")
async def analyze_sentiment():
    reviews = await scrape_review()
    if not reviews:
        return {"error":"No reviews found"}
    asyncio.create_task(process_reviews(reviews))

