import sqlite3
import time

from glydi_bot.names_hint import KnownNames


def test_hint_lists_gallery_names_and_refreshes(tmp_path):
    db = tmp_path / "people.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE persons (person_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL, last_seen_at REAL)")
        c.execute("INSERT INTO persons VALUES ('1','Karyan',1.0,5.0)")
        c.execute("INSERT INTO persons VALUES ('2','Anvitha',2.0,NULL)")
    names = KnownNames(db)
    assert names.hint() == "Karyan, Anvitha."
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO persons VALUES ('3','Priyanka',3.0,9.0)")
    assert names.hint() == "Karyan, Anvitha."  # cached
    names._read_at = time.monotonic() - 100
    assert names.hint() == "Priyanka, Karyan, Anvitha."


def test_hint_is_none_without_a_gallery(tmp_path):
    assert KnownNames(tmp_path / "missing.db").hint() is None
