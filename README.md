# Emotion Detection & Self-Learning Support Engine

An AI-powered system that detects student emotion states from text using a **BiLSTM + BERT ensemble**, then delivers empathetic, personalized learning support via **Gemini 2.5 Flash**.

---

## Features

- 🔐 **User Authentication** — Sign Up & Sign In with hashed passwords
- 🧠 **Dual-model emotion analysis** — BiLSTM + BERT ensemble
- 💬 **Gemini AI responses** — personalized, field-aware tutoring
- 📊 **Session dashboard** — history, stats, and export
- 🌐 **Deployment-ready** — Streamlit Cloud, Railway, Heroku

---

## Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/bhawishyasharma45/Emotional-detection-and-Self-Learning-support-Engine.git
cd Emotional-detection-and-Self-Learning-support-Engine
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your-gemini-api-key-here
```

Or copy the Streamlit secrets template:
```bash
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
# Then edit .streamlit/secrets.toml with your key
```

### 5. Run the app
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## Default Test Accounts

On first run, two seed accounts are created in `users.json`:

| Username | Password |
|---|---|
| `student1` | `password123` |
| `naman_gaur` | `securepass` |

> `users.json` is gitignored — it stays on your local machine / server only.

---

## Deployment

### Streamlit Cloud (Recommended — Free)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo and set **Main file** to `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Deploy ✅

### Railway / Heroku
The `Procfile` and `runtime.txt` are already configured.
Set the `GEMINI_API_KEY` environment variable in your platform dashboard.

---

## Project Structure

```
├── app.py                          # Main Streamlit application
├── src/
│   ├── preprocessing.py            # Text cleaning + BiLSTM predictor
│   ├── bert_model.py               # BERT emotion classifier
│   ├── model.py                    # Model architecture
│   ├── predict.py                  # Inference helpers
│   └── train.py                    # Training script
├── models/                         # Trained model weights (gitignored)
├── notebooks/                      # Jupyter exploration notebooks
├── .streamlit/
│   ├── config.toml                 # Streamlit server + theme settings
│   └── secrets.toml.template       # Secret keys template
├── requirements.txt
├── Procfile                        # Heroku/Railway process definition
├── runtime.txt                     # Python version pin
└── README.md
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for AI responses |