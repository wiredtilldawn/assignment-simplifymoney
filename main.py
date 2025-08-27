import os
import json
import uuid
import aiohttp
import base64
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, ForeignKey, DateTime, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

import google.generativeai as genai

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("GENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in environment variables!")
if not API_KEY:
    raise ValueError("GENAI_API_KEY is not set in environment variables!")
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY is not set in environment variables!")

genai.configure(api_key=API_KEY)

app = FastAPI()

Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String)
    phone = Column(String)
    orders = relationship("Order", back_populates="user")


class GoldPrice(Base):
    __tablename__ = "gold_price"
    id = Column(Integer, primary_key=True, index=True)
    price_per_gram = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    grams = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="orders")


Base.metadata.create_all(bind=engine)


def ensure_initial_price():
    db = SessionLocal()
    try:
        count = db.query(GoldPrice).count()
        if count == 0:
            db.add(GoldPrice(price_per_gram=6000.0))
            db.commit()
    finally:
        db.close()


ensure_initial_price()

# ---------------- Deepgram Helpers ----------------
async def transcribe_audio(file: UploadFile):
    url = "https://api.deepgram.com/v1/listen"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=await file.read()) as resp:
            result = await resp.json()
            return result["results"]["channels"][0]["alternatives"][0]["transcript"]


async def synthesize_audio(text: str, filename: str = "output.wav"):
    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            audio_bytes = await resp.read()
            with open(filename, "wb") as f:
                f.write(audio_bytes)
    return filename

# ---------------- Core Logic ----------------

sessions = {}


def parse_genai_json_response(response):
    raw_text = ""
    try:
        if getattr(response, "candidates", None):
            raw_text = response.candidates[0].content.parts[0].text.strip()
    except Exception:
        raw_text = str(response)

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").replace("json", "", 1).strip()

    try:
        return json.loads(raw_text)
    except Exception:
        return {"answer": raw_text, "intent": "info"}


@app.post("/gold-assistant")
async def gold_assistant(query: Optional[str] = Form(None), audio: Optional[UploadFile] = File(None)):
    if audio:
        query = await transcribe_audio(audio)

    if not query:
        return {"error": "No query provided"}

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        f"""User query: {query}.
        Task: Decide intent: Is the user asking to BUY digital gold or just general info?
        Respond in STRICT JSON with two fields (no markdown, no extra text):
        {{
           "answer": "...",
           "intent": "buy_gold" OR "info"
        }}"""
    )

    parsed = parse_genai_json_response(response)
    reply_text = parsed.get("answer", "")

    if parsed.get("intent") == "buy_gold":
        reply_text = "It looks like you want to buy gold. Please continue via /buy-gold API."

    audio_file = await synthesize_audio(reply_text)

    return {
        "reply": reply_text,
        "intent": parsed.get("intent", "info"),
        "audio_file": audio_file
    }


@app.post("/buy-gold")
async def buy_gold(session_id: Optional[str] = Form(None), query: Optional[str] = Form(None), audio: Optional[UploadFile] = File(None)):
    if audio:
        query = await transcribe_audio(audio)

    if not query:
        return {"error": "No query provided"}

    if not session_id:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"stage": "name", "data": {}}
        reply = "Great! Let's get started. What's your full name?"
        audio_file = await synthesize_audio(reply)
        return {"session_id": session_id, "reply": reply, "audio_file": audio_file}

    if session_id in sessions:
        session = sessions[session_id]
        stage = session["stage"]

        if stage == "name":
            session["data"]["name"] = query
            session["stage"] = "email"
            reply = "Thanks! Please provide your email."

        elif stage == "email":
            session["data"]["email"] = query
            session["stage"] = "phone"
            reply = "Got it. What's your phone number?"

        elif stage == "phone":
            session["data"]["phone"] = query
            session["stage"] = "grams"
            reply = "Perfect. How many grams of gold would you like to buy?"

        elif stage == "grams":
            try:
                grams = float(query)
            except ValueError:
                reply = "Please enter a valid number for grams."
                audio_file = await synthesize_audio(reply)
                return {"session_id": session_id, "reply": reply, "audio_file": audio_file}

            session["data"]["grams"] = grams
            db = SessionLocal()
            try:
                latest_price = db.query(GoldPrice).order_by(GoldPrice.id.desc()).first()
                price_per_gram = latest_price.price_per_gram if latest_price else 6000.0
            finally:
                db.close()

            total_amount = grams * price_per_gram
            session["data"]["price_per_gram"] = price_per_gram
            session["data"]["total_amount"] = total_amount
            session["stage"] = "confirm"

            reply = f"Gold price is ₹{price_per_gram}/gram. For {grams} grams, total = ₹{total_amount}. Do you want to confirm purchase? (yes/no)"

        elif stage == "confirm":
            if query.lower() == "yes":
                data = session["data"]
                db = SessionLocal()
                try:
                    user = User(name=data["name"], email=data["email"], phone=data["phone"])
                    db.add(user)
                    db.commit()
                    db.refresh(user)

                    order = Order(user_id=user.id, grams=data["grams"], total_amount=data["total_amount"])
                    db.add(order)
                    db.commit()
                finally:
                    db.close()

                sessions.pop(session_id, None)
                reply = "✅ Order confirmed and placed successfully!"
            else:
                sessions.pop(session_id, None)
                reply = "❌ Order cancelled."
        else:
            reply = "Invalid stage. Restart process."

        audio_file = await synthesize_audio(reply)
        return {"session_id": session_id, "reply": reply, "audio_file": audio_file}

    return {"reply": "Invalid session. Please restart from /buy-gold."}
