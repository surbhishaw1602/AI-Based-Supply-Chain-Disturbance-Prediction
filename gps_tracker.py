
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "shipments.db"
MAX_GPS_ACCURACY = 200.0


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.row_factory = sqlite3.Row
    return conn


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def initialize_gps_database():
    """Safely initialize GPS storage and migrate existing records."""
    conn = get_connection()
    try:
        cursor = conn.cursor()

        shipment_columns = {
            row["name"]
            for row in cursor.execute(
                "PRAGMA table_info(shipments)"
            ).fetchall()
        }

        if not shipment_columns:
            raise sqlite3.OperationalError(
                "The shipments table does not exist. "
                "Restore the existing shipment database first."
            )

        additions = {
            "source_latitude": "REAL",
            "source_longitude": "REAL",
            "destination_latitude": "REAL",
            "destination_longitude": "REAL",
        }

        for column, data_type in additions.items():
            if column not in shipment_columns:
                cursor.execute(
                    f"ALTER TABLE shipments ADD COLUMN {column} {data_type}"
                )

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracking_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                shipment_id INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                stopped_at TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                FOREIGN KEY (shipment_id)
                    REFERENCES shipments(shipment_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracking_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shipment_id INTEGER NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                accuracy REAL,
                recorded_at TEXT NOT NULL,
                session_id INTEGER,
                FOREIGN KEY (shipment_id)
                    REFERENCES shipments(shipment_id),
                FOREIGN KEY (session_id)
                    REFERENCES tracking_sessions(session_id)
            )
        """)

        update_columns = {
            row["name"]
            for row in cursor.execute(
                "PRAGMA table_info(tracking_updates)"
            ).fetchall()
        }

        if "session_id" not in update_columns:
            cursor.execute("""
                ALTER TABLE tracking_updates
                ADD COLUMN session_id INTEGER
            """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tracking_shipment_id
            ON tracking_updates(shipment_id, id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tracking_session_id
            ON tracking_updates(session_id, id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tracking_active_sessions
            ON tracking_sessions(shipment_id, status)
        """)

        conn.commit()
    finally:
        conn.close()


def start_tracking_session(shipment_id):
    """
    Start a shared database tracking session.
    If one is already active for this shipment, reuse it.
    """
    shipment = int(shipment_id)
    conn = get_connection()

    try:
        exists = conn.execute(
            "SELECT 1 FROM shipments WHERE shipment_id = ?",
            (shipment,)
        ).fetchone()

        if not exists:
            raise ValueError(f"Shipment {shipment} does not exist.")

        active = conn.execute("""
            SELECT session_id
            FROM tracking_sessions
            WHERE shipment_id = ? AND status = 'active'
            ORDER BY session_id DESC
            LIMIT 1
        """, (shipment,)).fetchone()

        if active:
            return int(active["session_id"])

        cursor = conn.execute("""
            INSERT INTO tracking_sessions (
                shipment_id, started_at, status
            )
            VALUES (?, ?, 'active')
        """, (shipment, _now()))

        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def stop_tracking_session(shipment_id):
    """Stop the active tracking session without deleting GPS history."""
    shipment = int(shipment_id)
    conn = get_connection()

    try:
        cursor = conn.execute("""
            UPDATE tracking_sessions
            SET status = 'stopped', stopped_at = ?
            WHERE shipment_id = ? AND status = 'active'
        """, (_now(), shipment))

        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_active_tracking_session(shipment_id):
    """Return the active session as a dictionary, or None."""
    conn = get_connection()

    try:
        row = conn.execute("""
            SELECT session_id, shipment_id, started_at, stopped_at, status
            FROM tracking_sessions
            WHERE shipment_id = ? AND status = 'active'
            ORDER BY session_id DESC
            LIMIT 1
        """, (int(shipment_id),)).fetchone()

        return dict(row) if row else None
    finally:
        conn.close()


def get_latest_session_location(shipment_id, session_id=None):
    """Return the latest GPS record from the active/current session only."""
    conn = get_connection()

    try:
        if session_id is None:
            session = conn.execute("""
                SELECT session_id
                FROM tracking_sessions
                WHERE shipment_id = ? AND status = 'active'
                ORDER BY session_id DESC
                LIMIT 1
            """, (int(shipment_id),)).fetchone()

            if not session:
                return None

            session_id = int(session["session_id"])

        row = conn.execute("""
            SELECT latitude, longitude, accuracy, recorded_at
            FROM tracking_updates
            WHERE shipment_id = ? AND session_id = ?
            ORDER BY id DESC
            LIMIT 1
        """, (int(shipment_id), int(session_id))).fetchone()

        return tuple(row) if row else None
    finally:
        conn.close()


def get_session_tracking_history(shipment_id, session_id=None):
    """Return chronological GPS history for the active/current session."""
    conn = get_connection()

    try:
        if session_id is None:
            session = conn.execute("""
                SELECT session_id
                FROM tracking_sessions
                WHERE shipment_id = ? AND status = 'active'
                ORDER BY session_id DESC
                LIMIT 1
            """, (int(shipment_id),)).fetchone()

            if not session:
                return []

            session_id = int(session["session_id"])

        rows = conn.execute("""
            SELECT latitude, longitude, accuracy, recorded_at
            FROM tracking_updates
            WHERE shipment_id = ? AND session_id = ?
            ORDER BY id ASC
        """, (int(shipment_id), int(session_id))).fetchall()

        return [tuple(row) for row in rows]
    finally:
        conn.close()


def update_shipment_coordinates(
    shipment_id,
    source_lat,
    source_lon,
    destination_lat,
    destination_lon,
):
    try:
        shipment = int(shipment_id)
        values = tuple(
            float(value)
            for value in (
                source_lat,
                source_lon,
                destination_lat,
                destination_lon,
            )
        )
    except (TypeError, ValueError, OverflowError):
        return False

    source_latitude, source_longitude, destination_latitude, destination_longitude = values

    if not (
        all(math.isfinite(value) for value in values)
        and -90 <= source_latitude <= 90
        and -180 <= source_longitude <= 180
        and -90 <= destination_latitude <= 90
        and -180 <= destination_longitude <= 180
    ):
        return False

    conn = get_connection()
    try:
        cursor = conn.execute("""
            UPDATE shipments
            SET source_latitude = ?,
                source_longitude = ?,
                destination_latitude = ?,
                destination_longitude = ?
            WHERE shipment_id = ?
        """, (*values, shipment))

        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def save_gps_location(
    shipment_id,
    latitude,
    longitude,
    accuracy=None,
    session_id=None
):
    """
    Save a valid GPS fix.

    When session_id is supplied, the session must still be active
    and must belong to the specified shipment.
    """
    try:
        shipment = int(shipment_id)
        lat = float(latitude)
        lon = float(longitude)
        acc = float(accuracy)
    except (TypeError, ValueError, OverflowError):
        return False

    if not (
        math.isfinite(lat)
        and math.isfinite(lon)
        and math.isfinite(acc)
        and -90 <= lat <= 90
        and -180 <= lon <= 180
        and 0 <= acc <= MAX_GPS_ACCURACY
    ):
        return False

    if session_id is None:
        return False

    conn = get_connection()
    try:
        active = conn.execute("""
            SELECT 1
            FROM tracking_sessions
            WHERE session_id = ?
              AND shipment_id = ?
              AND status = 'active'
        """, (int(session_id), shipment)).fetchone()

        if not active:
            return False

        conn.execute("""
            INSERT INTO tracking_updates (
                shipment_id,
                latitude,
                longitude,
                accuracy,
                recorded_at,
                session_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            shipment,
            lat,
            lon,
            acc,
            _now(),
            int(session_id)
        ))

        conn.commit()
        return True
    finally:
        conn.close()


def get_latest_location(shipment_id):
    """Return the latest historical GPS fix, regardless of session."""
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT latitude, longitude, accuracy, recorded_at
            FROM tracking_updates
            WHERE shipment_id = ?
            ORDER BY id DESC
            LIMIT 1
        """, (int(shipment_id),)).fetchone()

        return tuple(row) if row else None
    finally:
        conn.close()


def get_tracking_history(shipment_id, limit=None):
    """Return all historical GPS fixes, including older sessions."""
    conn = get_connection()
    try:
        if limit is None:
            rows = conn.execute("""
                SELECT latitude, longitude, accuracy, recorded_at
                FROM tracking_updates
                WHERE shipment_id = ?
                ORDER BY id ASC
            """, (int(shipment_id),)).fetchall()
        else:
            safe_limit = max(1, int(limit))
            rows = conn.execute("""
                SELECT latitude, longitude, accuracy, recorded_at
                FROM (
                    SELECT id, latitude, longitude, accuracy, recorded_at
                    FROM tracking_updates
                    WHERE shipment_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
            """, (int(shipment_id), safe_limit)).fetchall()

        return [tuple(row) for row in rows]
    finally:
        conn.close()


def get_tracking_update_count(shipment_id=None):
    conn = get_connection()
    try:
        if shipment_id is None:
            row = conn.execute(
                "SELECT COUNT(*) FROM tracking_updates"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM tracking_updates WHERE shipment_id = ?",
                (int(shipment_id),)
            ).fetchone()

        return int(row[0] or 0)
    finally:
        conn.close()
