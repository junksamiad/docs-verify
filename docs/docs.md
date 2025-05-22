python3 -m venv backend/.venv
python3 -m venv frontend/.venv

pip install -r requirements.txt

source .venv/bin/activate

*** frontend ***

python app.py


*** backend ***

cd into backend and activate venv, then cd back into docs-verify and run:

uvicorn backend.server:app --reload --port 8000

