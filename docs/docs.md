python3 -m venv backend/.venv
python3 -m venv frontend/.venv

pip install -r requirements.txt

source .venv/bin/activate

*** frontend ***

cd frontend

source .venv/bin/activate

python app.py


*** backend ***

***macOS***
cd into backend and activate venv

source .venv/bin/activate

then cd back into docs-verify and run:

uvicorn server:app --reload --port 8000

***windows***

backend\.venv\Scripts\activate.bat

