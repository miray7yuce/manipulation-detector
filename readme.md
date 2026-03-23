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

## How it works

1. User enters article text or a URL.
2. Backend extracts and processes the article.
3. An AI model analyzes the text.
4. The results (bias, techniques, highlights, explanation) are displayed in the UI.
