# Deployment Guide for Heart Minder

This project is a Flask-based web application. Below are instructions for deploying it to various platforms.

## Prerequisites
- Python 3.8+
- The project structure has been updated to be a standard Flask app.

## 1. Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python app.py
   ```
3. Open `http://localhost:5000` in your browser.

## 2. Deploying to Render / Railway / Heroku
These platforms typically use the `Procfile` and `requirements.txt`.

### Steps:
1. **Create a new Web Service** on your chosen platform.
2. **Connect your GitHub Repository**.
3. **Environment Variables:**
   - `SECRET_KEY`: Set this to a random string for security.
   - `PORT`: Usually handled automatically by the platform (default 5000).
4. **Build Command:** `pip install -r requirements.txt`
5. **Start Command:** `gunicorn app:app` (This is already in the `Procfile`).

## 3. Database Note (SQLite)
The application uses SQLite (`heartminder.db`). 
- On platforms like **Render** or **Railway**, if you don't use a "Persistent Disk", the database will be reset every time the application restarts.
- For a final year project, this is usually acceptable. If you need permanent data, you would need to:
  1. Use a Persistent Disk (available on Render/Railway).
  2. Or, switch to a managed database like PostgreSQL (requires code changes to use `psycopg2`).

## 4. Key Fixes Applied:
- **Restructured:** Moved `app.py` and other files to the root directory so Flask can find the `templates` folder automatically.
- **Fixed Crash:** Removed a broken `joblib.load` call that caused the app to crash on startup.
- **Security:** Added support for `SECRET_KEY` environment variable.
- **Deployment Ready:** Added `Procfile` for production WSGI servers (gunicorn).
