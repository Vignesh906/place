from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3, hashlib, os
from placement_utils import load_model, generate_feedback, prepare_input, get_model_stats
from placement_graphs import generate_dynamic_graphs

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'placement_secret_key_2024')

DB = 'placement_database.db'

# ── DB init ────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT NOT NULL,
        role TEXT DEFAULT 'student',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        result TEXT,
        probability REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    # default admin
    admin_pw = hashlib.sha256('admin123'.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username,password,email,role) VALUES (?,?,?,?)",
                  ('admin', admin_pw, 'admin@placement.ai', 'admin'))
    except:
        pass
    conn.commit(); conn.close()

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def get_user(username):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    row = c.fetchone(); conn.close(); return row

def save_prediction(user_id, result, prob):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO predictions (user_id,result,probability) VALUES (?,?,?)",
              (user_id, result, prob))
    conn.commit(); conn.close()

def get_user_predictions(user_id):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT result,probability,created_at FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 10",
              (user_id,))
    rows = c.fetchall(); conn.close(); return rows

def get_all_stats():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE role='student'")
    students = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions")
    preds = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE result='Placed'")
    placed = c.fetchone()[0]
    conn.close()
    return students, preds, placed

def get_all_users():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        SELECT u.id, u.username, u.email, u.role, u.created_at,
               COUNT(p.id) as total_preds,
               SUM(CASE WHEN p.result='Placed' THEN 1 ELSE 0 END) as placed_count,
               MAX(p.created_at) as last_active
        FROM users u
        LEFT JOIN predictions p ON u.id = p.user_id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """)
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_predictions():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        SELECT u.username, p.result, p.probability, p.created_at
        FROM predictions p
        JOIN users u ON p.user_id = u.id
        ORDER BY p.created_at DESC
        LIMIT 100
    """)
    rows = c.fetchall()
    conn.close()
    return rows

def delete_user(user_id):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("DELETE FROM predictions WHERE user_id=?", (user_id,))
    c.execute("DELETE FROM users WHERE id=? AND role!='admin'", (user_id,))
    conn.commit()
    conn.close()

# ── load model ─────────────────────────────────────────────────────────────
model, scaler, features = load_model()

# ── startup init (runs for both gunicorn and direct python) ────────────────
os.makedirs('static/user_graphs', exist_ok=True)
os.makedirs('static/graphs', exist_ok=True)
init_db()

# ── routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','')
        user = get_user(username)
        if user and user[2] == hash_pw(password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[4]
            return redirect(url_for('dashboard'))
        flash('Invalid credentials. Please try again.', 'error')
    return render_template('placement_login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','')
        email    = request.form.get('email','').strip()
        confirm  = request.form.get('confirm_password','')

        if len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
        elif password != confirm:
            flash('Passwords do not match.', 'error')
        elif not email or '@' not in email:
            flash('Enter a valid email.', 'error')
        else:
            conn = sqlite3.connect(DB)
            c = conn.cursor()
            try:
                c.execute("INSERT INTO users (username,password,email) VALUES (?,?,?)",
                          (username, hash_pw(password), email))
                conn.commit()
                flash('Account created! Please login.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username already exists.', 'error')
            finally:
                conn.close()
    return render_template('placement_register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    stats = get_model_stats()
    history = get_user_predictions(session['user_id'])
    sys_stats = get_all_stats() if session.get('role') == 'admin' else None
    return render_template('placement_dashboard.html',
                           stats=stats, history=history, sys_stats=sys_stats)

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        form_data = request.form.to_dict()
        try:
            X_input = prepare_input(form_data, features)
            prob = model.predict_proba(X_input)[0][1] * 100
            result = 'Placed' if prob >= 50 else 'Not Placed'
            feedback = generate_feedback(form_data)
            graphs = generate_dynamic_graphs(form_data, round(prob, 2))
            save_prediction(session['user_id'], result, round(prob, 2))
            return render_template('placement_result.html',
                                   result=result, probability=round(prob, 2),
                                   feedback=feedback, form_data=form_data,
                                   graphs=graphs)
        except Exception as e:
            import traceback
            print("PREDICT ERROR:", traceback.format_exc())
            flash(f'Prediction error: {str(e)}', 'error')
            return redirect(url_for('predict'))

    return render_template('placement_predict.html')

@app.route('/analytics')
def analytics():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('placement_analytics.html')

@app.route('/admin')
def admin_panel():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') != 'admin':
        flash('Access denied. Admin only.', 'error')
        return redirect(url_for('dashboard'))
    users       = get_all_users()
    predictions = get_all_predictions()
    sys_stats   = get_all_stats()
    placed_pct  = round((sys_stats[2] / sys_stats[1] * 100) if sys_stats[1] > 0 else 0, 1)
    return render_template('placement_admin.html',
                           users=users, predictions=predictions,
                           sys_stats=sys_stats, placed_pct=placed_pct)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    delete_user(user_id)
    flash('User deleted successfully.', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    os.makedirs('static/user_graphs', exist_ok=True)
    os.makedirs('static/graphs', exist_ok=True)
    init_db()
    app.run(debug=True, port=5002)