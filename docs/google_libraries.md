Gemini API libraries

This page provides information on downloading and installing the latest libraries for the Gemini API. If you're new to the Gemini API, get started with the API quickstart.

Important note about our new libraries
Since late 2024, we have been progressively rolling out Google Gen AI SDK, a new set of libraries that provide a unified interface for accessing Google's generative AI models via both the Gemini API and Vertex AI, streamlining the transition between these platforms.

The Google Gen AI SDK is now the recommended pathway for accessing Gemini models and all code snippets in our documentation utilize these libraries. For further details on migrating between the Gemini Developer API and Vertex AI, please consult the Vertex AI page.

Key Library Updates

Language	Old library	New library (Recommended)
Python	google-generativeai	google-genai
JavaScript
and TypeScript	@google/generative-ai	@google/genai
Go	google.golang.org/generative-ai	google.golang.org/genai
We strongly encourage all users to migrate to the Google GenAI SDK, as it provides a well maintained path, alongside access to new features like multi-modal outputs (e.g., image and video generation) and Live API features, which are unavailable in our legacy libraries. We expect the feature gap to grow over time.

Install a library
Python
JavaScript
Go
You can install our Python library by running:


pip install google-genai
Previous libraries and SDKs
For platforms where a new library is available, we strongly recommend migrating to the new libraries. Previous libraries will not receive new features, and their specific support status is detailed below.

For platforms where a new official library is not yet available, you can call our APIs directly, use community-maintained libraries, or leverage trusted third-party options. We are actively working to expand our official SDK coverage and will list new libraries here as they are released.

The following are the previous libraries:

Python (previous): All support, including bug fixes, ends August 2025.
Node.js (previous): All support, including bug fixes, ends August 2025.
Go (previous): All support, including bug fixes, ends August 2025.
Dart and Flutter (Previous): Not actively maintained
Swift (Previous): Not actively maintained
Android (Previous): Not actively maintained


Upgrade to the Google Gen AI SDK

When we introduced the Gemini 2.0 model family, we also released a new set of Google Gen AI libraries for working with the Gemini API:

Python
TypeScript and JavaScript
Go
These updated libraries will be fully compatible with all Gemini API models and features, including recent additions like the Live API and Veo.

We recommend that you start migrating your projects from the old Gemini SDK to the new Gen AI SDK. This guide provides before-and-after examples of migrated code to help you get started. We'll continue to add examples here to help you get up and running with the new libraries.

Note: The Go examples omit imports and other boilerplate code to improve readability.
Install the SDK
Before

Python
JavaScript
Go

pip install -U -q "google-generativeai"
After

Python
JavaScript
Go

pip install -U -q "google-genai"
Authenticate
Authenticate using an API key. You can create your API key in Google AI Studio.

Before

Python
JavaScript
Go
The old SDK handled the API client object implicitly. In the new SDK you create the API client and use it to call the API. Remember, in either case the SDK will pick up your API key from the GOOGLE_API_KEY environment variable if you don't pass one to the client.


import google.generativeai as genai

genai.configure(api_key=...)
After

Python
JavaScript
Go

export GOOGLE_API_KEY="YOUR_API_KEY"

from google import genai

client = genai.Client() # Set the API key using the GOOGLE_API_KEY env var.
                        # Alternatively, you could set the API key explicitly:
                        # client = genai.Client(api_key="your_api_key")
Generate content
Before

Python
JavaScript
Go
The new SDK provides access to all the API methods through the Client object. Except for a few stateful special cases (chat and live-api sessions), these are all stateless functions. For utility and uniformity, objects returned are pydantic classes.


import google.generativeai as genai

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content(
    'Tell me a story in 300 words'
)
print(response.text)
After

Python
JavaScript
Go

from google import genai
client = genai.Client()

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='Tell me a story in 300 words.'
)
print(response.text)

print(response.model_dump_json(
    exclude_none=True, indent=4))
Before

Python
JavaScript
Go
Many of the same convenience features exist in the new SDK. For example, PIL.Image objects are automatically converted.


import google.generativeai as genai

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content([
    'Tell me a story based on this image',
    Image.open(image_path)
])
print(response.text)
After

Python
JavaScript
Go

from google import genai
from PIL import Image

client = genai.Client()

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=[
        'Tell me a story based on this image',
        Image.open(image_path)
    ]
)
print(response.text)
Streaming
Before

Python
JavaScript
Go

import google.generativeai as genai

response = model.generate_content(
    "Write a cute story about cats.",
    stream=True)
for chunk in response:
    print(chunk.text)
After

Python
JavaScript
Go

from google import genai

client = genai.Client()

for chunk in client.models.generate_content_stream(
  model='gemini-2.0-flash',
  contents='Tell me a story in 300 words.'
):
    print(chunk.text)
Configuration
Before

Python
JavaScript
Go
For all methods in the new SDK, the required arguments are provided as keyword arguments. All optional inputs are provided in the config argument. Config arguments can be specified as either Python dictionaries or Config classes in the google.genai.types namespace. For utility and uniformity, all definitions within the types module are pydantic classes.


import google.generativeai as genai

model = genai.GenerativeModel(
  'gemini-1.5-flash',
    system_instruction='you are a story teller for kids under 5 years old',
    generation_config=genai.GenerationConfig(
      max_output_tokens=400,
      top_k=2,
      top_p=0.5,
      temperature=0.5,
      response_mime_type='application/json',
      stop_sequences=['\n'],
    )
)
response = model.generate_content('tell me a story in 100 words')
After

Python
JavaScript
Go

from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
  model='gemini-2.0-flash',
  contents='Tell me a story in 100 words.',
  config=types.GenerateContentConfig(
      system_instruction='you are a story teller for kids under 5 years old',
      max_output_tokens= 400,
      top_k= 2,
      top_p= 0.5,
      temperature= 0.5,
      response_mime_type= 'application/json',
      stop_sequences= ['\n'],
      seed=42,
  ),
)
Safety settings
Generate a response with safety settings:

Before

Python
JavaScript

import google.generativeai as genai

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content(
    'say something bad',
    safety_settings={
        'HATE': 'BLOCK_ONLY_HIGH',
        'HARASSMENT': 'BLOCK_ONLY_HIGH',
  }
)
After

Python
JavaScript

from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
  model='gemini-2.0-flash',
  contents='say something bad',
  config=types.GenerateContentConfig(
      safety_settings= [
          types.SafetySetting(
              category='HARM_CATEGORY_HATE_SPEECH',
              threshold='BLOCK_ONLY_HIGH'
          ),
      ]
  ),
)
Async
Before

Python
To use the new SDK with asyncio, there is a separate async implementation of every method under client.aio.


import google.generativeai as genai

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content_async(
    'tell me a story in 100 words'
)
After

Python

from google import genai

client = genai.Client()

response = await client.aio.models.generate_content(
    model='gemini-2.0-flash',
    contents='Tell me a story in 300 words.'
)
Chat
Start a chat and send a message to the model:

Before

Python
JavaScript
Go

import google.generativeai as genai

model = genai.GenerativeModel('gemini-1.5-flash')
chat = model.start_chat()

response = chat.send_message(
    "Tell me a story in 100 words")
response = chat.send_message(
    "What happened after that?")
After

Python
JavaScript
Go

from google import genai

client = genai.Client()

chat = client.chats.create(model='gemini-2.0-flash')

response = chat.send_message(
    message='Tell me a story in 100 words')
response = chat.send_message(
    message='What happened after that?')
Function calling
Before

Python
In the new SDK, automatic function calling is the default. Here, you disable it.


import google.generativeai as genai
from enum import Enum

def get_current_weather(location: str) -> str:
    """Get the current whether in a given location.

    Args:
        location: required, The city and state, e.g. San Franciso, CA
        unit: celsius or fahrenheit
    """
    print(f'Called with: {location=}')
    return "23C"

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    tools=[get_current_weather]
)

response = model.generate_content("What is the weather in San Francisco?")
function_call = response.candidates[0].parts[0].function_call
After

Python

from google import genai
from google.genai import types

client = genai.Client()

def get_current_weather(location: str) -> str:
    """Get the current whether in a given location.

    Args:
        location: required, The city and state, e.g. San Franciso, CA
        unit: celsius or fahrenheit
    """
    print(f'Called with: {location=}')
    return "23C"

response = client.models.generate_content(
  model='gemini-2.0-flash',
  contents="What is the weather like in Boston?",
  config=types.GenerateContentConfig(
      tools=[get_current_weather],
      automatic_function_calling={'disable': True},
  ),
)

function_call = response.candidates[0].content.parts[0].function_call
Automatic function calling
Before

Python
The old SDK only supports automatic function calling in chat. In the new SDK this is the default behavior in generate_content.


import google.generativeai as genai

def get_current_weather(city: str) -> str:
    return "23C"

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    tools=[get_current_weather]
)

chat = model.start_chat(
    enable_automatic_function_calling=True)
result = chat.send_message("What is the weather in San Francisco?")
After

Python

from google import genai
from google.genai import types
client = genai.Client()

def get_current_weather(city: str) -> str:
    return "23C"

response = client.models.generate_content(
  model='gemini-2.0-flash',
  contents="What is the weather like in Boston?",
  config=types.GenerateContentConfig(
      tools=[get_current_weather]
  ),
)
Code execution
Code execution is a tool that allows the model to generate Python code, run it, and return the result.

Before

Python
JavaScript

import google.generativeai as genai

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    tools="code_execution"
)

result = model.generate_content(
  "What is the sum of the first 50 prime numbers? Generate and run code for "
  "the calculation, and make sure you get all 50.")
After

Python
JavaScript

from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='What is the sum of the first 50 prime numbers? Generate and run '
            'code for the calculation, and make sure you get all 50.',
    config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution)],
    ),
)
Search grounding
GoogleSearch (Gemini>=2.0) and GoogleSearchRetrieval (Gemini < 2.0) are tools that allow the model to retrieve public web data for grounding, powered by Google.

Before

Python

import google.generativeai as genai

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content(
    contents="what is the Google stock price?",
    tools='google_search_retrieval'
)
After

Python

from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='What is the Google stock price?',
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                google_search=types.GoogleSearch()
            )
        ]
    )
)
JSON response
Generate answers in JSON format.

Before

Python
JavaScript
By specifying a response_schema and setting response_mime_type="application/json" users can constrain the model to produce a JSON response following a given structure. The new SDK uses pydantic classes to provide the schema (although you can pass a genai.types.Schema, or equivalent dict). When possible, the SDK will parse the returned JSON, and return the result in response.parsed. If you provided a pydantic class as the schema the SDK will convert that JSON to an instance of the class.


import google.generativeai as genai
import typing_extensions as typing

class CountryInfo(typing.TypedDict):
    name: str
    population: int
    capital: str
    continent: str
    major_cities: list[str]
    gdp: int
    official_language: str
    total_area_sq_mi: int

model = genai.GenerativeModel(model_name="gemini-1.5-flash")
result = model.generate_content(
    "Give me information of the United States",
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema = CountryInfo
    ),
)
After

Python
JavaScript

from google import genai
from pydantic import BaseModel

client = genai.Client()

class CountryInfo(BaseModel):
    name: str
    population: int
    capital: str
    continent: str
    major_cities: list[str]
    gdp: int
    official_language: str
    total_area_sq_mi: int

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='Give me information of the United States.',
    config={
        'response_mime_type': 'application/json',
        'response_schema': CountryInfo,
    },
)

response.parsed
Files
Upload
Upload a file:

Before

Python

import requests
import pathlib
import google.generativeai as genai

# Download file
response = requests.get(
    'https://storage.googleapis.com/generativeai-downloads/data/a11.txt')
pathlib.Path('a11.txt').write_text(response.text)

file = genai.upload_file(path='a11.txt')

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content([
    'Can you summarize this file:',
    my_file
])
print(response.text)
After

Python

import requests
import pathlib
from google import genai

client = genai.Client()

# Download file
response = requests.get(
    'https://storage.googleapis.com/generativeai-downloads/data/a11.txt')
pathlib.Path('a11.txt').write_text(response.text)

my_file = client.files.upload(file='a11.txt')

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=[
        'Can you summarize this file:',
        my_file
    ]
)
print(response.text)
List and get
List uploaded files and get an uploaded file with a filename:

Before

Python

import google.generativeai as genai

for file in genai.list_files():
  print(file.name)

file = genai.get_file(name=file.name)
After

Python

from google import genai
client = genai.Client()

for file in client.files.list():
    print(file.name)

file = client.files.get(name=file.name)
Delete
Delete a file:

Before

Python

import pathlib
import google.generativeai as genai

pathlib.Path('dummy.txt').write_text(dummy)
dummy_file = genai.upload_file(path='dummy.txt')

file = genai.delete_file(name=dummy_file.name)
After

Python

import pathlib
from google import genai

client = genai.Client()

pathlib.Path('dummy.txt').write_text(dummy)
dummy_file = client.files.upload(file='dummy.txt')

response = client.files.delete(name=dummy_file.name)
Context caching
Context caching allows the user to pass the content to the model once, cache the input tokens, and then refer to the cached tokens in subsequent calls to lower the cost.

Before

Python
JavaScript

import requests
import pathlib
import google.generativeai as genai
from google.generativeai import caching

# Download file
response = requests.get(
    'https://storage.googleapis.com/generativeai-downloads/data/a11.txt')
pathlib.Path('a11.txt').write_text(response.text)

# Upload file
document = genai.upload_file(path="a11.txt")

# Create cache
apollo_cache = caching.CachedContent.create(
    model="gemini-1.5-flash-001",
    system_instruction="You are an expert at analyzing transcripts.",
    contents=[document],
)

# Generate response
apollo_model = genai.GenerativeModel.from_cached_content(
    cached_content=apollo_cache
)
response = apollo_model.generate_content("Find a lighthearted moment from this transcript")
After

Python
JavaScript

import requests
import pathlib
from google import genai
from google.genai import types

client = genai.Client()

# Check which models support caching.
for m in client.models.list():
  for action in m.supported_actions:
    if action == "createCachedContent":
      print(m.name)
      break

# Download file
response = requests.get(
    'https://storage.googleapis.com/generativeai-downloads/data/a11.txt')
pathlib.Path('a11.txt').write_text(response.text)

# Upload file
document = client.files.upload(file='a11.txt')

# Create cache
model='gemini-1.5-flash-001'
apollo_cache = client.caches.create(
      model=model,
      config={
          'contents': [document],
          'system_instruction': 'You are an expert at analyzing transcripts.',
      },
  )

# Generate response
response = client.models.generate_content(
    model=model,
    contents='Find a lighthearted moment from this transcript',
    config=types.GenerateContentConfig(
        cached_content=apollo_cache.name,
    )
)
Count tokens
Count the number of tokens in a request.

Before

Python
JavaScript

import google.generativeai as genai

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.count_tokens(
    'The quick brown fox jumps over the lazy dog.')
After

Python
JavaScript

from google import genai

client = genai.Client()

response = client.models.count_tokens(
    model='gemini-2.0-flash',
    contents='The quick brown fox jumps over the lazy dog.',
)
Generate images
Generate images:

Before

Python

#pip install https://github.com/google-gemini/generative-ai-python@imagen
import google.generativeai as genai

imagen = genai.ImageGenerationModel(
    "imagen-3.0-generate-001")
gen_images = imagen.generate_images(
    prompt="Robot holding a red skateboard",
    number_of_images=1,
    safety_filter_level="block_low_and_above",
    person_generation="allow_adult",
    aspect_ratio="3:4",
)
After

Python

from google import genai

client = genai.Client()

gen_images = client.models.generate_images(
    model='imagen-3.0-generate-001',
    prompt='Robot holding a red skateboard',
    config=types.GenerateImagesConfig(
        number_of_images= 1,
        safety_filter_level= "BLOCK_LOW_AND_ABOVE",
        person_generation= "ALLOW_ADULT",
        aspect_ratio= "3:4",
    )
)

for n, image in enumerate(gen_images.generated_images):
    pathlib.Path(f'{n}.png').write_bytes(
        image.image.image_bytes)
Embed content
Generate content embeddings.

Before

Python
JavaScript

import google.generativeai as genai

response = genai.embed_content(
  model='models/text-embedding-004',
  content='Hello world'
)
After

Python
JavaScript

from google import genai

client = genai.Client()

response = client.models.embed_content(
  model='text-embedding-004',
  contents='Hello world',
)
Tune a Model
Create and use a tuned model.

The new SDK simplifies tuning with client.tunings.tune, which launches the tuning job and polls until the job is complete.

Before

Python

import google.generativeai as genai
import random

# create tuning model
train_data = {}
for i in range(1, 6):
  key = f'input {i}'
  value = f'output {i}'
  train_data[key] = value

name = f'generate-num-{random.randint(0,10000)}'
operation = genai.create_tuned_model(
    source_model='models/gemini-1.5-flash-001-tuning',
    training_data=train_data,
    id = name,
    epoch_count = 5,
    batch_size=4,
    learning_rate=0.001,
)
# wait for tuning complete
tuningProgress = operation.result()

# generate content with the tuned model
model = genai.GenerativeModel(model_name=f'tunedModels/{name}')
response = model.generate_content('55')
After

Python

from google import genai
from google.genai import types

client = genai.Client()

# Check which models are available for tuning.
for m in client.models.list():
  for action in m.supported_actions:
    if action == "createTunedModel":
      print(m.name)
      break

# create tuning model
training_dataset=types.TuningDataset(
        examples=[
            types.TuningExample(
                text_input=f'input {i}',
                output=f'output {i}',
            )
            for i in range(5)
        ],
    )
tuning_job = client.tunings.tune(
    base_model='models/gemini-1.5-flash-001-tuning',
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        epoch_count= 5,
        batch_size=4,
        learning_rate=0.001,
        tuned_model_display_name="test tuned model"
    )
)

# generate content with the tuned model
response = client.models.generate_content(
    model=tuning_job.tuned_model.model,
    contents='55',
)
