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

## 3. Deploying to Vercel
Vercel is a serverless platform. This project has been configured with `vercel.json` to support it.

### Steps:
1. **Push your code** to a GitHub repository.
2. **Import the project** in Vercel.
3. **Framework Preset:** Vercel should auto-detect the configuration from `vercel.json`.
4. **Environment Variables:**
   - `SECRET_KEY`: Set this to a random string.
5. **Database Note:** 
   - Vercel uses a read-only filesystem. The application has been updated to use `/tmp/heartminder.db` for the database.
   - **Important:** Data in `/tmp` is temporary and will be cleared when the serverless function restarts. For permanent data, you should switch to a managed database like PostgreSQL (Supabase/Neon).

## 4. Database Note (General)
- **Restructured:** Moved `app.py` and other files to the root directory so Flask can find the `templates` folder automatically.
- **Fixed Crash:** Removed a broken `joblib.load` call that caused the app to crash on startup.
- **Security:** Added support for `SECRET_KEY` environment variable.
- **Deployment Ready:** Added `Procfile` for production WSGI servers (gunicorn).
