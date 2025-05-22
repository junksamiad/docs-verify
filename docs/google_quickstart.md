Gemini API quickstart

This quickstart shows you how to install our libraries and make your first Gemini API request.

Note: All our code snippets use Google Gen AI SDK, a new set of libraries we have been rolling out since early 2025. You can find out more about this change at our Libraries page.
Before you begin
You need a Gemini API key. If you don't already have one, you can get it for free in Google AI Studio.

Install the Google GenAI SDK
Python
JavaScript
Go
Apps Script
Using Python 3.9+, install the google-genai package using the following pip command:


pip install -q -U google-genai
Make your first request
Use the generateContent method to send a request to the Gemini API.

Python
JavaScript
Go
Apps Script
REST

from google import genai

client = genai.Client(api_key="YOUR_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a few words"
)
print(response.text)
