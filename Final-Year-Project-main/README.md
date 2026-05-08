# Heart Minder — Full Stack Flask App

## Quick Start
```bash
pip install -r requirements.txt
python app.py
# Open: http://localhost:5000
```

## Default Login
- Admin: `admin` / `admin123`
- Register new users at /register

## Features
- Beige/warm colour palette UI
- SQLite database (heartminder.db auto-created)
- Login / Register / Logout
- Home dashboard with stats
- Predict page (13 clinical inputs)
- Every prediction saved to DB
- Model retrains after each new prediction (DB data + Cleveland base)
- History page — users see own records, admin sees ALL records
- Admin: delete records, force retrain, manage users
- PDF report download

## Pages
| Route | Description |
|---|---|
| `/` | Redirects to login |
| `/login` | Login page |
| `/register` | Register new account |
| `/home` | Dashboard with stats + recent predictions |
| `/predict` | Run a prediction |
| `/history` | View DB records |
| `/admin/users` | Admin: manage users |
