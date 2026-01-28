import sqlite3
import bcrypt
import datetime

def init_db():
    conn = sqlite3.connect('ycry_users.db', check_same_thread=False)
    c = conn.cursor()
    
    # Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, name TEXT)''')
    
    # Profiles Table (Added profile_pic column)
    c.execute('''CREATE TABLE IF NOT EXISTS profiles
                 (username TEXT PRIMARY KEY, 
                  baby_name TEXT, 
                  dob TEXT, 
                  gender TEXT,
                  blood_group TEXT,
                  weight_birth REAL,
                  height_birth REAL,
                  head_circ REAL,
                  chest_circ REAL,
                  apgar_score INTEGER,
                  health_summary TEXT,
                  profile_pic TEXT, 
                  FOREIGN KEY(username) REFERENCES users(username))''')
                  
    # Growth & Vaccine Tables (Same as before)
    c.execute('''CREATE TABLE IF NOT EXISTS growth
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, 
                  date TEXT, weight REAL, height REAL,
                  FOREIGN KEY(username) REFERENCES users(username))''')

    c.execute('''CREATE TABLE IF NOT EXISTS vaccines
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, 
                  vaccine_name TEXT, date_given TEXT,
                  FOREIGN KEY(username) REFERENCES users(username))''')
    
    conn.commit()
    conn.close()

# --- USER FUNCTIONS ---
def create_user(username, password, name):
    conn = sqlite3.connect('ycry_users.db')
    c = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        c.execute('INSERT INTO users VALUES (?,?,?)', (username, hashed_pw, name))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('ycry_users.db')
    c = conn.cursor()
    c.execute('SELECT password, name FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    if data and bcrypt.checkpw(password.encode('utf-8'), data[0]):
        return data[1]
    return None

# --- PROFILE FUNCTIONS (UPDATED) ---
def save_full_profile(username, data):
    """
    Saves profile including picture.
    data = (name, dob, gender, blood, weight, height, head, chest, apgar, summary, pic_path)
    """
    conn = sqlite3.connect('ycry_users.db')
    c = conn.cursor()
    # Now inserting 12 values total (username + 11 fields)
    c.execute('''INSERT OR REPLACE INTO profiles VALUES 
                 (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
              (username, *data))
    conn.commit()
    conn.close()

def get_profile(username):
    conn = sqlite3.connect('ycry_users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM profiles WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    return data

# --- GROWTH & VACCINE FUNCTIONS ---
def add_growth_record(username, weight, height):
    conn = sqlite3.connect('ycry_users.db')
    c = conn.cursor()
    today = datetime.date.today()
    c.execute('INSERT INTO growth (username, date, weight, height) VALUES (?,?,?,?)', 
              (username, str(today), weight, height))
    conn.commit()
    conn.close()

def get_latest_growth(username):
    conn = sqlite3.connect('ycry_users.db')
    c = conn.cursor()
    c.execute('SELECT weight, height FROM growth WHERE username = ? ORDER BY id DESC LIMIT 1', (username,))
    data = c.fetchone()
    conn.close()
    return data

    conn.close()
    return data

def get_growth_history(username):
    conn = sqlite3.connect('ycry_users.db')
    c = conn.cursor()
    c.execute('SELECT date, weight, height FROM growth WHERE username = ? ORDER BY date ASC', (username,))
    data = c.fetchall()
    conn.close()
    return data

def mark_vaccine_done(username, vaccine_name):
    conn = sqlite3.connect('ycry_users.db')
    c = conn.cursor()
    today = datetime.date.today()
    c.execute('SELECT id FROM vaccines WHERE username=? AND vaccine_name=?', (username, vaccine_name))
    if not c.fetchone():
        c.execute('INSERT INTO vaccines (username, vaccine_name, date_given) VALUES (?,?,?)', 
                  (username, vaccine_name, str(today)))
    conn.commit()
    conn.close()

def get_completed_vaccines(username):
    conn = sqlite3.connect('ycry_users.db')
    c = conn.cursor()
    c.execute('SELECT vaccine_name FROM vaccines WHERE username=?', (username,))
    data = c.fetchall()
    conn.close()
    return [x[0] for x in data]