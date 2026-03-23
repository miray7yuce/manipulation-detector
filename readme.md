# News Article Analyzer

A web application that analyzes news articles to detect **bias** and **manipulation techniques** using AI.

Users can either paste the **article text** or provide a **news URL**. The system extracts the article content and generates an analysis including detected bias, manipulation techniques, highlighted phrases, and an explanation.

## Tech Stack

* **Backend:** FastAPI (Python)
* **Frontend:** Blazor (.NET)
* **AI Providers:** Groq

## Features

* Bias detection in news articles
* Identification of common manipulation techniques
* Highlighting suspicious or emotionally loaded phrases
* Automatic language detection
* Source bias information for known news domains

# How it works

#Step 1 — Get a free Groq API key

Go to https://console.groq.com and create a free account
Navigate to API Keys from the left sidebar
Click Create API Key, give it a name, and copy the key

Groq's free tier supports thousands of requests per day using powerful models like LLaMA 3.3 70B. No payment info required.

#Step 2 — Configure the environment file
In the backend/ folder you'll find a file called .env.example.
Rename it to .env and paste your key:


GROQ_API_KEY=your_key_here
It should look like this when filled in:
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#Step 3 — Run the backend
Open a terminal in the backend/ folder:

pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000


#Step 4 — Run the frontend
Open a second terminal in the frontend/ folder (keep the backend running):

dotnet run
