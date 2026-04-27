from flask import (Flask, render_template, request, redirect,
                   url_for, session, send_file, flash, jsonify)
import sqlite3, io, hashlib, os, pickle, json
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER

app = Flask(__name__)
app.secret_key = 'heartminder_secret_2025'
DB_PATH = 'heartminder.db'

# ─── Cleveland base dataset ───────────────────────────────────────────────────
BASE_DATA = [
    [63,1,1,145,233,1,2,150,0,2.3,3,0,6,0],[67,1,4,160,286,0,2,108,1,1.5,2,3,3,1],
    [67,1,4,120,229,0,2,129,1,2.6,2,2,7,1],[37,1,3,130,250,0,0,187,0,3.5,3,0,3,0],
    [41,0,2,130,204,0,2,172,0,1.4,1,0,3,0],[56,1,2,120,236,0,0,178,0,0.8,1,0,3,0],
    [62,0,4,140,268,0,2,160,0,3.6,3,2,3,1],[57,0,4,120,354,0,0,163,1,0.6,1,0,3,0],
    [63,1,4,130,254,0,2,147,0,1.4,2,1,7,1],[53,1,4,140,203,1,2,155,1,3.1,3,0,7,1],
    [57,1,4,140,192,0,0,148,0,0.4,2,0,6,0],[56,0,2,140,294,0,2,153,0,1.3,2,0,3,0],
    [56,1,3,130,256,1,2,142,1,0.6,2,1,6,1],[44,1,2,120,263,0,0,173,0,0,1,0,7,0],
    [52,1,3,172,199,1,0,162,0,0.5,1,0,7,0],[57,1,3,150,168,0,0,174,0,1.6,1,0,3,0],
    [48,1,2,110,229,0,0,168,0,1,3,0,7,1],[54,1,4,140,239,0,0,160,0,1.2,1,0,3,0],
    [48,0,3,130,275,0,0,139,0,0.2,1,0,3,0],[49,1,2,130,266,0,0,171,0,0.6,1,0,3,0],
    [64,1,1,110,211,0,2,144,1,1.8,2,0,3,0],[58,0,1,150,283,1,2,162,0,1,1,0,3,0],
    [58,1,2,120,284,0,2,160,0,1.8,2,0,3,1],[58,1,3,132,224,0,2,173,0,3.2,1,2,7,1],
    [60,1,4,130,206,0,2,132,1,2.4,2,2,7,1],[50,0,3,120,219,0,0,158,0,1.6,2,0,3,0],
    [58,0,3,120,340,0,0,172,0,0,1,0,3,0],[66,0,3,150,226,0,0,114,0,2.6,3,0,3,0],
    [43,1,4,150,247,0,0,171,0,1.5,1,0,3,0],[40,1,4,110,167,0,2,114,1,2,2,0,7,1],
    [69,0,3,140,239,0,0,151,0,1.8,1,2,3,0],[60,1,4,117,230,1,0,160,1,1.4,1,2,7,1],
    [64,1,3,140,335,0,0,158,0,0,1,0,3,0],[59,1,4,135,234,0,0,161,0,0.5,2,0,7,0],
    [44,1,2,130,233,0,0,179,1,0.4,1,0,3,0],[42,1,4,140,226,0,0,178,0,0,1,0,3,0],
    [43,1,4,120,177,0,2,120,1,2.5,2,0,7,1],[57,1,4,150,276,0,2,112,1,0.6,2,1,6,1],
    [55,1,4,132,353,0,0,132,1,1.2,2,1,7,1],[61,1,3,150,243,1,0,137,1,1,2,0,3,0],
    [65,0,4,150,225,0,2,114,0,1,2,3,7,1],[40,1,1,140,199,0,0,178,1,1.4,1,0,7,0],
    [71,0,2,160,302,0,0,162,0,0.4,1,2,3,0],[59,1,3,150,212,1,0,157,0,1.6,1,0,3,0],
    [61,0,4,130,330,0,2,169,0,0,1,0,3,1],[58,1,3,112,230,0,2,165,0,2.5,2,1,7,1],
    [51,1,3,110,175,0,0,123,0,0.6,1,0,3,0],[50,1,4,150,243,0,2,128,0,2.6,2,0,7,1],
    [65,0,3,140,417,1,2,157,0,0.8,1,1,3,0],[53,1,2,130,197,1,2,152,0,1.2,3,0,3,0],
    [41,0,2,105,198,0,0,168,0,0,1,1,3,0],[65,1,4,120,177,0,0,140,0,0.4,1,0,7,0],
    [44,1,4,112,290,0,2,153,0,0,1,1,3,1],[44,1,2,130,219,0,2,188,0,0,1,0,3,0],
    [60,1,4,130,253,0,0,144,1,1.4,1,1,7,1],[54,1,4,124,266,0,2,109,1,2.2,2,1,7,1],
    [50,1,3,140,341,0,2,125,1,2.8,2,1,7,1],[46,1,3,140,311,0,0,120,1,1.8,2,2,7,1],
    [41,1,4,110,172,0,2,158,0,0,1,0,7,0],[54,1,2,125,273,0,2,152,0,0.5,3,1,3,0],
    [51,1,1,125,213,0,2,125,1,1.4,1,1,3,0],[51,0,3,130,256,0,2,149,0,0.5,1,0,3,0],
    [46,0,3,142,177,0,2,160,1,1.4,3,0,3,0],[58,1,4,128,216,0,2,131,1,2.2,2,3,7,1],
    [54,0,3,135,304,1,0,170,0,0,1,0,3,0],[54,1,4,120,188,0,0,113,0,1.4,2,1,7,1],
    [60,1,4,145,282,0,2,142,1,2.8,2,2,7,1],[60,0,3,102,318,0,0,160,0,0,1,1,3,0],
    [55,1,4,132,342,0,0,166,1,1.2,1,0,3,0],[61,1,4,150,243,1,0,137,1,1,2,0,3,0],
    [71,0,3,110,265,1,2,130,0,0,1,1,3,0],[70,1,2,156,245,0,2,143,0,0,2,0,3,0],
    [76,0,3,140,197,0,2,116,0,1.1,2,0,3,0],[67,0,3,152,277,0,0,172,0,0,1,1,3,0],
    [45,1,4,142,309,0,2,147,1,0,2,3,7,1],[68,1,4,180,274,1,2,150,1,1.6,2,0,7,1],
    [57,1,4,140,214,0,2,144,1,2,2,0,7,0],[57,0,4,128,303,0,2,159,0,0,1,1,3,0],
    [38,1,3,138,175,0,0,173,0,0,1,0,3,0],[44,1,2,120,220,0,0,170,0,0,1,0,3,0],
    [54,1,4,110,239,0,0,126,1,2.8,2,1,7,1],[48,0,2,130,275,0,0,139,0,0.2,1,0,3,0],
    [39,1,3,120,200,0,0,160,1,1,2,0,3,0],[45,0,3,130,234,0,2,175,0,0.6,2,0,3,0],
    [68,1,4,144,193,1,0,141,0,3.4,2,2,7,1],[52,1,4,128,255,0,0,161,1,0,1,1,7,1],
    [53,1,4,137,208,0,2,127,1,3.5,2,1,7,1],[43,1,4,132,247,1,2,143,1,0.1,2,3,7,1],
    [66,0,1,150,226,0,0,114,0,2.6,3,0,3,0],[65,1,4,140,306,1,2,87,1,1.5,2,1,7,1],
    [61,0,4,130,330,0,2,169,0,0,1,0,3,1],[62,1,4,120,267,0,0,99,1,1.8,2,2,7,1],
    [52,0,3,136,196,0,2,169,0,0.1,2,0,3,0],[59,1,4,126,218,1,0,134,0,2.2,2,1,7,1],
    [60,1,4,140,293,0,2,170,0,1.2,2,2,7,1],[52,1,2,134,201,0,0,158,0,0.8,1,1,3,0],
    [48,1,4,122,222,0,2,186,0,0,1,0,3,0],[45,1,4,115,260,0,2,185,0,0,1,0,3,0],
    [34,1,1,118,182,0,2,174,0,0,1,0,3,0],[57,0,4,128,303,0,2,159,0,0,1,1,3,0],
    [54,1,4,108,309,0,0,156,0,0,1,0,7,0],[52,1,3,118,186,0,2,190,0,0,2,0,6,0],
    [41,1,4,135,203,0,0,132,0,0,2,0,6,1],[58,1,4,140,211,1,2,165,0,0,1,0,3,0],
    [35,0,4,138,183,0,0,182,0,1.4,1,0,3,0],[51,1,3,100,222,0,0,143,1,1.2,2,0,3,0],
    [44,1,4,120,169,0,0,144,1,2.8,3,0,6,1],[62,0,3,130,263,0,0,97,0,1.2,2,1,7,1],
    [57,0,2,174,179,0,0,161,0,1,1,0,3,0],[48,1,4,124,255,1,0,175,0,0,1,2,3,1],
    [56,1,2,130,184,0,0,100,0,1.6,1,0,3,0],[59,1,1,140,177,0,0,162,1,0,1,1,3,0],
    [56,1,3,128,223,0,2,128,0,1,2,1,6,1],[63,0,4,124,197,0,0,136,1,0,2,0,3,1],
    [64,1,4,120,246,0,2,96,1,2.2,3,1,3,1],[74,0,2,120,269,0,2,121,1,0.2,1,1,3,0],
    [55,1,4,160,289,0,2,145,1,0.8,2,1,7,1],[46,1,2,140,275,0,0,165,1,0,2,0,3,0],
    [64,0,1,130,294,0,2,153,0,1.4,2,1,3,0],[59,1,4,110,239,0,2,142,1,1.2,2,1,7,1],
    [41,0,2,112,268,0,2,172,1,0,1,0,3,0],[54,0,2,108,267,0,2,167,0,0,1,0,3,0],
    [39,0,3,94,199,0,0,179,0,0,1,0,3,0],[53,1,4,123,282,0,0,95,1,2,2,2,7,1],
    [63,0,2,108,269,0,0,169,1,1.8,2,2,3,0],[34,0,2,118,210,0,0,192,0,0.7,1,0,3,0],
    [47,1,4,112,204,0,0,143,0,0.1,1,0,3,0],[54,1,3,125,273,0,2,152,0,0.5,3,1,3,0],
    [66,1,4,160,228,0,2,138,0,2.3,1,0,6,0],[52,1,3,160,331,0,0,166,1,0.5,1,1,3,0],
    [64,1,4,110,211,0,2,144,1,1.8,2,0,3,1],[59,1,4,164,176,1,2,90,0,1,2,2,6,1],
    [57,0,3,140,241,0,0,123,1,0.2,2,0,7,1],[45,1,1,110,264,0,0,132,0,1.2,2,0,7,1],
    [57,1,4,130,131,0,0,115,1,1.2,2,1,7,1],[57,0,2,130,236,0,2,174,0,0,2,1,3,1],
    [46,1,3,120,231,0,0,115,1,0,1,0,3,0],[43,0,3,122,213,0,0,165,0,0.2,2,0,3,0],
    [57,1,4,150,156,0,2,144,1,0.8,2,1,7,1],[48,1,2,122,222,0,2,186,0,0,1,0,3,0],
    [67,1,4,125,254,1,0,163,0,0.2,2,2,7,1],[65,1,4,110,248,0,2,158,0,0.6,1,2,6,1],
    [56,0,2,200,288,1,2,133,1,4,3,2,7,1],[77,1,4,125,304,0,2,162,1,0,1,3,3,1],
    [56,1,4,130,283,1,2,103,1,1.6,3,0,7,1],[35,1,4,120,198,0,0,130,1,1.6,2,0,7,1],
    [65,0,3,155,269,0,0,148,0,0.8,1,0,3,0],[46,0,2,142,177,0,2,160,1,1.4,3,0,3,0],
    [55,1,4,132,353,0,0,132,1,1.2,2,1,7,1],[54,0,2,160,201,0,0,163,0,0,1,1,3,0],
    [65,0,3,140,417,1,2,157,0,0.8,1,1,3,0],[60,1,4,120,246,0,2,135,0,0,2,0,7,0],
    [56,1,1,130,221,0,2,163,0,0,1,0,7,0],[66,0,1,178,228,1,0,165,1,1,2,2,7,1],
    [61,1,4,134,234,0,0,145,0,2.6,2,2,7,1],[57,1,4,110,335,0,0,143,1,3,2,1,7,1],
    [57,1,2,128,229,0,2,150,0,0.4,2,1,7,1],[42,0,3,120,209,0,0,173,0,0,2,0,3,0],
    [59,1,4,178,270,0,2,145,0,4.2,3,0,7,0],[41,1,4,110,172,0,2,158,0,0,1,0,7,0],
    [62,0,4,138,294,1,0,106,0,1.9,2,3,3,1],[46,1,4,120,249,0,2,144,0,0.8,1,0,7,1],
    [36,1,1,120,267,0,0,160,0,3,2,0,3,1],[62,1,3,130,231,0,0,146,0,1.8,2,3,7,0],
    [56,1,2,120,240,0,0,169,0,0,3,0,3,0],[58,1,2,146,218,0,0,105,0,2,2,1,7,0],
    [44,0,3,118,242,0,0,149,0,0.3,2,1,3,0],[60,1,4,150,258,0,2,157,0,2.6,2,2,7,1],
    [59,1,4,126,218,1,0,134,0,2.2,2,1,7,1],[44,1,4,132,290,0,2,145,0,2.8,2,2,7,1],
    [42,1,4,136,315,0,0,125,1,1.8,2,0,6,1],[57,1,4,150,255,0,0,92,1,3,2,1,7,1],
    [60,1,4,140,293,0,2,170,0,1.2,2,2,7,1],[54,1,3,150,195,0,0,122,0,0,1,0,3,0],
    [63,0,2,140,195,0,0,179,0,0,1,2,3,0],[46,1,4,150,231,0,0,147,0,3.6,2,0,3,1],
    [56,0,2,140,294,0,2,153,0,1.3,2,0,3,0],[54,1,4,120,171,0,2,137,0,2,2,0,3,0],
    [49,1,4,120,188,0,0,139,0,2,2,3,7,1],[39,1,3,118,219,0,0,140,1,1.2,2,0,7,0],
    [47,1,4,108,243,0,0,152,0,0,1,0,3,1],[50,0,3,120,244,0,0,162,0,1.1,1,0,3,0],
    [43,0,3,122,213,0,0,165,0,0.2,2,0,3,0],[51,0,4,130,305,0,0,142,1,1.2,2,0,7,1],
    [49,1,2,130,266,0,0,171,0,0.6,1,0,3,0],[56,1,4,130,283,1,2,103,1,1.6,3,0,7,1],
    [43,1,4,140,288,0,2,135,1,2,2,0,6,1],[42,1,3,120,240,1,0,194,0,0.8,3,0,7,0],
    [67,0,3,106,223,0,0,142,0,0.3,2,2,3,0],[45,1,2,104,208,0,2,148,1,3,2,0,3,0],
    [52,1,4,125,212,0,0,168,0,1,1,2,7,1],[57,1,2,150,126,1,0,173,0,0.2,1,1,7,0],
]

FEAT = ['age','trestbps','chol','thalach','oldpeak','sex','cp','fbs','restecg','exang','slope','ca','thal']
BASE_X = np.array([r[:13] for r in BASE_DATA])
BASE_Y = np.array([r[13] for r in BASE_DATA], dtype=int)

# ─── DB Init ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        username TEXT,
        age REAL, sex INTEGER, cp INTEGER, trestbps REAL, chol REAL,
        fbs INTEGER, restecg INTEGER, thalach REAL, exang INTEGER,
        oldpeak REAL, slope INTEGER, ca REAL, thal REAL,
        prediction INTEGER, probability REAL,
        lr_prob REAL, dt_prob REAL, rf_prob REAL, gb_prob REAL,
        stack_acc REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    # Default admin
    pw = hashlib.sha256('admin123'.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                  ('admin', pw, 'admin'))
    except: pass
    conn.commit(); conn.close()

init_db()

# ─── ML Training ──────────────────────────────────────────────────────────────
MODELS = {}
STACKED = None
ACCS = {}
AUCS = {}
STACK_ACC = 0.0
STACK_AUC = 0.0

def train_models():
    global MODELS, STACKED, ACCS, AUCS, STACK_ACC, STACK_AUC

    # Merge base data + DB data
    conn = get_db()
    rows = conn.execute(
        "SELECT age,trestbps,chol,thalach,oldpeak,sex,cp,fbs,restecg,exang,slope,ca,thal,prediction FROM predictions"
    ).fetchall()
    conn.close()

    if rows:
        db_X = np.array([[r[i] for i in range(13)] for r in rows])
        db_Y = np.array([r[13] for r in rows], dtype=int)
        X = np.vstack([BASE_X, db_X])
        y = np.concatenate([BASE_Y, db_Y])
    else:
        X, y = BASE_X.copy(), BASE_Y.copy()

    if len(np.unique(y)) < 2:
        X, y = BASE_X.copy(), BASE_Y.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    base_est = [
        ('lr', Pipeline([('sc', StandardScaler()),
                         ('clf', LogisticRegression(max_iter=1000, random_state=42))])),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]

    MODELS = {}
    ACCS = {}
    AUCS = {}
    for key, pipe in base_est:
        pipe.fit(X_train, y_train)
        yp = pipe.predict(X_test)
        ypr = pipe.predict_proba(X_test)[:,1]
        ACCS[key] = accuracy_score(y_test, yp)
        AUCS[key] = roc_auc_score(y_test, ypr)
        MODELS[key] = pipe

    stk = StackingClassifier(estimators=base_est,
                              final_estimator=LogisticRegression(max_iter=1000),
                              cv=5, n_jobs=-1)
    stk.fit(X_train, y_train)
    yps = stk.predict(X_test)
    yprs = stk.predict_proba(X_test)[:,1]
    STACK_ACC = accuracy_score(y_test, yps)
    STACK_AUC = roc_auc_score(y_test, yprs)
    STACKED = stk
    print(f"[Train] DB rows: {len(rows)} | Stack: {STACK_ACC*100:.1f}% AUC:{STACK_AUC:.3f}")

train_models()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session: return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'admin':
            flash('Admin access required.', 'error')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated

LABEL_MAP = {
    'age':'Age (years)','sex':'Sex','cp':'Chest Pain Type',
    'trestbps':'Resting BP (mmHg)','chol':'Cholesterol (mg/dL)',
    'fbs':'Fasting Blood Sugar > 120','restecg':'Resting ECG',
    'thalach':'Max Heart Rate','exang':'Exercise Angina',
    'oldpeak':'ST Depression','slope':'ST Slope',
    'ca':'Major Vessels (0-3)','thal':'Thalassemia'
}
def fmt_val(k,v):
    maps = {
        'sex':{0:'Female',1:'Male'},
        'cp':{0:'Typical Angina',1:'Atypical Angina',2:'Non-Anginal',3:'Asymptomatic'},
        'fbs':{0:'No',1:'Yes'},'restecg':{0:'Normal',1:'ST-T Abnormality',2:'LV Hypertrophy'},
        'exang':{0:'No',1:'Yes'},'slope':{0:'Upsloping',1:'Flat',2:'Downsloping'},
        'thal':{1.0:'Normal',2.0:'Fixed Defect',3.0:'Reversable Defect'}
    }
    if k in maps:
        try: return maps[k].get(int(float(v)), str(v))
        except: return str(v)
    if k == 'ca':
        try: return str(int(float(v)))
        except: return str(v)
    return str(v)

# ─── PDF ──────────────────────────────────────────────────────────────────────
def generate_pdf(patient_data, prediction, probability, username):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    BEIGE  = colors.HexColor('#f5f0e8')
    BROWN  = colors.HexColor('#6b4c2a')
    RED    = colors.HexColor('#c0392b')
    GREEN  = colors.HexColor('#27ae60')
    DARK   = colors.HexColor('#2c2c2c')

    ts = ParagraphStyle('T', parent=styles['Title'], fontSize=20,
                         textColor=BROWN, alignment=TA_CENTER, spaceAfter=4)
    ss = ParagraphStyle('S', parent=styles['Normal'], fontSize=9,
                         textColor=colors.HexColor('#888'), alignment=TA_CENTER)
    h2s = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12,
                          textColor=BROWN, spaceBefore=14, spaceAfter=6)
    ns = styles['Normal']

    story = []
    story.append(Paragraph("❤ Heart Minder", ts))
    story.append(Paragraph("See the Risk. Save the Life.", ss))
    story.append(Paragraph(f"Patient Report — {datetime.now().strftime('%d %b %Y, %I:%M %p')} | User: {username}", ss))
    story.append(HRFlowable(width="100%", thickness=2, color=BROWN, spaceAfter=12))

    # Result
    res_color = RED if prediction==1 else GREEN
    res_text  = "HEART DISEASE DETECTED" if prediction==1 else "NO HEART DISEASE DETECTED"
    res_bg    = colors.HexColor('#fdf2f2') if prediction==1 else colors.HexColor('#eafaf1')
    story.append(Paragraph("Prediction Result", h2s))
    rt = Table([[Paragraph(f'<b><font color="{res_color.hexval()}">{res_text}</font></b>',
                            ParagraphStyle('r',parent=ns,fontSize=14,alignment=TA_CENTER))],
                [Paragraph(f'Risk Probability: <b>{probability*100:.1f}%</b>',
                            ParagraphStyle('rp',parent=ns,fontSize=12,alignment=TA_CENTER))]],
               colWidths=[16*cm])
    rt.setStyle(TableStyle([
        ('BOX',(0,0),(-1,-1),1.5,res_color),
        ('BACKGROUND',(0,0),(-1,-1),res_bg),
        ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),
    ]))
    story.append(rt); story.append(Spacer(1,12))

    # Patient data table
    story.append(Paragraph("Patient Input Summary", h2s))
    rows = [['Parameter','Value']]
    for k in FEAT:
        rows.append([LABEL_MAP.get(k,k), fmt_val(k, patient_data.get(k,'–'))])
    pt = Table(rows, colWidths=[9*cm,7*cm])
    pt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),BROWN),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),10),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[BEIGE,colors.white]),
        ('BOX',(0,0),(-1,-1),0.5,colors.HexColor('#ccc')),
        ('INNERGRID',(0,0),(-1,-1),0.25,colors.HexColor('#ddd')),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LEFTPADDING',(0,0),(-1,-1),8),
    ]))
    story.append(pt); story.append(Spacer(1,12))

    # Model accuracy table
    story.append(Paragraph("Model Performance", h2s))
    name_map = {'lr':'Logistic Regression','dt':'Decision Tree',
                'rf':'Random Forest','gb':'XGBoost (GradientBoosting)'}
    ar = [['Model','Accuracy','AUC-ROC']]
    for k,full in name_map.items():
        ar.append([full, f"{ACCS.get(k,0)*100:.2f}%", f"{AUCS.get(k,0):.4f}"])
    ar.append(['Stacked Ensemble (Best)', f"{STACK_ACC*100:.2f}%", f"{STACK_AUC:.4f}"])
    at = Table(ar, colWidths=[9*cm,3.5*cm,3.5*cm])
    at.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),BROWN),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTNAME',(0,-1),(-1,-1),'Helvetica-Bold'),
        ('BACKGROUND',(0,-1),(-1,-1),colors.HexColor('#d5e8d4')),
        ('FONTSIZE',(0,0),(-1,-1),10),
        ('ROWBACKGROUNDS',(0,1),(-1,-2),[BEIGE,colors.white]),
        ('BOX',(0,0),(-1,-1),0.5,colors.HexColor('#ccc')),
        ('INNERGRID',(0,0),(-1,-1),0.25,colors.HexColor('#ddd')),
        ('ALIGN',(1,0),(-1,-1),'CENTER'),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LEFTPADDING',(0,0),(-1,-1),8),
    ]))
    story.append(at); story.append(Spacer(1,16))
    story.append(HRFlowable(width="100%",thickness=0.5,color=colors.grey,spaceAfter=6))
    story.append(Paragraph(
        "<i>Disclaimer: This is an AI-generated report for informational purposes only. "
        "Consult a qualified healthcare provider for medical decisions.</i>",
        ParagraphStyle('disc',parent=ns,fontSize=8,textColor=colors.grey)))
    doc.build(story)
    buf.seek(0)
    return buf

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def root():
    if 'user_id' in session: return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        uname = request.form.get('username','').strip()
        pw    = hash_pw(request.form.get('password',''))
        conn  = get_db()
        user  = conn.execute("SELECT * FROM users WHERE username=? AND password=?",
                              (uname, pw)).fetchone()
        conn.close()
        if user:
            session['user_id']  = user['id']
            session['username'] = user['username']
            session['role']     = user['role']
            return redirect(url_for('home'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        uname = request.form.get('username','').strip()
        pw    = request.form.get('password','')
        pw2   = request.form.get('password2','')
        if pw != pw2:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        conn = get_db()
        try:
            conn.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                          (uname, hash_pw(pw), 'user'))
            conn.commit()
            flash('Account created! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already taken.', 'error')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    disease = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction=1").fetchone()[0]
    no_disease = total - disease
    my_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?",
                             (session['user_id'],)).fetchone()[0]
    recent = conn.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 5").fetchall()
    conn.close()
    return render_template('home.html', total=total, disease=disease,
                           no_disease=no_disease, my_count=my_count,
                           recent=recent, stack_acc=STACK_ACC,
                           accs=ACCS, stack_auc=STACK_AUC)

@app.route('/predict', methods=['GET','POST'])
@login_required
def predict_page():
    if request.method == 'POST':
        try:
            fd = request.form
            vals = {k: float(fd[k]) for k in FEAT}
            X = np.array([[vals[k] for k in FEAT]])
            prediction = int(STACKED.predict(X)[0])
            probability = float(STACKED.predict_proba(X)[0][1])
            ind = {k: float(MODELS[k].predict_proba(X)[0][1]) for k in MODELS}

            # Save to DB
            conn = get_db()
            conn.execute('''INSERT INTO predictions
                (user_id,username,age,sex,cp,trestbps,chol,fbs,restecg,thalach,
                 exang,oldpeak,slope,ca,thal,prediction,probability,
                 lr_prob,dt_prob,rf_prob,gb_prob,stack_acc)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (session['user_id'], session['username'],
                 vals['age'], int(vals['sex']), int(vals['cp']),
                 vals['trestbps'], vals['chol'], int(vals['fbs']),
                 int(vals['restecg']), vals['thalach'], int(vals['exang']),
                 vals['oldpeak'], int(vals['slope']), vals['ca'], vals['thal'],
                 prediction, probability,
                 ind.get('lr',0), ind.get('dt',0),
                 ind.get('rf',0), ind.get('gb',0), STACK_ACC))
            conn.commit()
            conn.close()

            # Retrain with new data
            train_models()

            return jsonify({
                'prediction': prediction,
                'probability': round(probability*100,2),
                'individual': {k: round(v*100,2) for k,v in ind.items()},
                'model_accs': {k: round(v*100,2) for k,v in ACCS.items()},
                'stack_acc': round(STACK_ACC*100,2),
                'stack_auc': round(STACK_AUC,4),
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    return render_template('predict.html', accs=ACCS, stack_acc=STACK_ACC)

@app.route('/download_pdf', methods=['POST'])
@login_required
def download_pdf():
    try:
        fd = request.form
        vals = {k: fd.get(k,'0') for k in FEAT}
        X = np.array([[float(vals[k]) for k in FEAT]])
        prediction = int(STACKED.predict(X)[0])
        probability = float(STACKED.predict_proba(X)[0][1])
        buf = generate_pdf(vals, prediction, probability, session['username'])
        return send_file(buf, as_attachment=True,
                         download_name='heartminder_report.pdf',
                         mimetype='application/pdf')
    except Exception as e:
        return f"Error: {e}", 400

@app.route('/history')
@login_required
def history():
    conn = get_db()
    if session['role'] == 'admin':
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC",
            (session['user_id'],)).fetchall()
    conn.close()
    return render_template('history.html', rows=rows,
                           is_admin=(session['role']=='admin'))

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    conn = get_db()
    users = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return render_template('admin_users.html', users=users)

@app.route('/admin/delete_prediction/<int:pid>', methods=['POST'])
@login_required
@admin_required
def delete_prediction(pid):
    conn = get_db()
    conn.execute("DELETE FROM predictions WHERE id=?", (pid,))
    conn.commit(); conn.close()
    train_models()
    flash('Record deleted and model retrained.', 'success')
    return redirect(url_for('history'))

@app.route('/admin/retrain', methods=['POST'])
@login_required
@admin_required
def retrain():
    train_models()
    flash(f'Model retrained! Stack accuracy: {STACK_ACC*100:.1f}%', 'success')
    return redirect(url_for('home'))

@app.route('/api/stats')
@login_required
def api_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    disease = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction=1").fetchone()[0]
    conn.close()
    return jsonify({'total': total, 'disease': disease,
                    'no_disease': total-disease,
                    'stack_acc': round(STACK_ACC*100,2)})

if __name__ == '__main__':
    import os 
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
