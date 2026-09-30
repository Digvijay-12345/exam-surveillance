import sqlite3, datetime as dt

DB = "events.db"

def _conn():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, student_id INTEGER, event_type TEXT,
        details TEXT, snapshot TEXT)""")
    return c

def log_event(student_id, event_type, details, snapshot):
    with _conn() as c:
        c.execute("INSERT INTO events(ts,student_id,event_type,details,snapshot) VALUES(?,?,?,?,?)",
                  (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), student_id, event_type, details, snapshot))

def fetch_events(limit=100):
    with _conn() as c:
        return c.execute("SELECT ts,student_id,event_type,details,snapshot FROM events ORDER BY id DESC LIMIT ?",
                         (limit,)).fetchall()

def clear_events():
    with _conn() as c:
        c.execute("DELETE FROM events")
