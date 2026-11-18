# festival_service.py

import sqlite3
from pathlib import Path
from datetime import datetime


# Use the same SQLite database as app.py
DB_PATH = Path(__file__).resolve().parent / "shipments.db"


def get_festival_connection():
    """Create a connection to the shipment database."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_festival_database():
    """Create the festival_events table and insert starter events."""

    connection = get_festival_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS festival_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            event_type TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            state TEXT NOT NULL,
            affected_location TEXT NOT NULL,
            impact_level TEXT NOT NULL,
            description TEXT,
            source_url TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(name, start_date, state, affected_location)
        )
    """)

    # Starter events for West Bengal.
    # Dates are stored in YYYY-MM-DD format.
    starter_events = [
        (
            "Kali Puja",
            "Festival",
            "2026-11-08",
            "2026-11-08",
            "West Bengal",
            "Kolkata, Howrah, West Bengal",
            "High",
            "Kali Puja celebrations may cause local traffic congestion.",
            "https://sahaj.wb.gov.in/SiteController",
        ),
        (
            "Govardhan Puja",
            "Festival",
            "2026-11-09",
            "2026-11-09",
            "West Bengal",
            "West Bengal",
            "Medium",
            "Local religious celebrations may affect traffic in some areas.",
            "https://packolkata.imd.gov.in/panchang/index.php/en/festivals",
        ),
        (
            "Bhratridwitiya (Bhai Dooj)",
            "Festival",
            "2026-11-11",
            "2026-11-11",
            "West Bengal",
            "West Bengal",
            "Medium",
            "Festival-related local travel may increase road traffic.",
            "https://msme.wb.gov.in/cmsholiday",
        ),
        (
            "Chhath Puja",
            "Festival",
            "2026-11-15",
            "2026-11-16",
            "West Bengal",
            "Kolkata, Howrah and river ghats in West Bengal",
            "High",
            "Crowds and processions near river ghats may cause road restrictions.",
            "https://wbchse.wb.gov.in/annual-working-plan/list-of-holidays/",
        ),
        (
            "Jagaddhatri Puja",
            "Festival",
            "2026-11-16",
            "2026-11-19",
            "West Bengal",
            "Chandannagar, Hooghly, West Bengal",
            "High",
            "Puja pandals and processions may cause local road congestion.",
            "https://sahaj.wb.gov.in/SiteController",
        ),
        (
            "Birthday of Guru Nanak",
            "Public Holiday",
            "2026-11-24",
            "2026-11-24",
            "West Bengal",
            "West Bengal",
            "Low",
            "Public holiday. Check local traffic and business schedules.",
            "https://msme.wb.gov.in/cmsholiday",
        ),
        (
            "Christmas Day",
            "Public Holiday",
            "2026-12-25",
            "2026-12-25",
            "West Bengal",
            "Kolkata, Howrah and other major cities",
            "Medium",
            "Celebrations may increase traffic around churches and public areas.",
            "https://msme.wb.gov.in/cmsholiday",
        ),
        (
            "Saraswati Puja",
            "Festival",
            "2027-02-11",
            "2027-02-11",
            "West Bengal",
            "West Bengal",
            "Medium",
            "Educational institutions and local celebrations may affect traffic.",
            "https://packolkata.imd.gov.in/panchang/en/festivals",
        ),
    ]

    now = datetime.now().isoformat(timespec="seconds")

    for event in starter_events:
        cursor.execute("""
            INSERT OR IGNORE INTO festival_events (
                name,
                event_type,
                start_date,
                end_date,
                state,
                affected_location,
                impact_level,
                description,
                source_url,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (*event, now))

    connection.commit()
    connection.close()


def get_festival_events(start_date=None, end_date=None, state=None):
    """
    Retrieve festival events.

    Optional date filters include events that overlap the selected period.
    """

    connection = get_festival_connection()

    query = """
        SELECT *
        FROM festival_events
        WHERE 1 = 1
    """
    parameters = []

    if start_date:
        query += " AND end_date >= ?"
        parameters.append(str(start_date))

    if end_date:
        query += " AND start_date <= ?"
        parameters.append(str(end_date))

    if state:
        query += " AND LOWER(state) = LOWER(?)"
        parameters.append(state.strip())

    query += " ORDER BY start_date ASC, name ASC"

    rows = connection.execute(query, parameters).fetchall()
    connection.close()

    return [dict(row) for row in rows]


def add_festival_event(
    name,
    event_type,
    start_date,
    end_date,
    state,
    affected_location,
    impact_level,
    description="",
    source_url=""
):
    """Add a new festival or holiday."""

    name = name.strip()
    state = state.strip()
    affected_location = affected_location.strip()

    if not name or not state or not affected_location:
        raise ValueError(
            "Event name, state and affected location are required."
        )

    try:
        start = datetime.strptime(str(start_date), "%Y-%m-%d").date()
        end = datetime.strptime(str(end_date), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Dates must use YYYY-MM-DD format.")

    if end < start:
        raise ValueError("End date cannot be earlier than start date.")

    connection = get_festival_connection()

    try:
        connection.execute("""
            INSERT INTO festival_events (
                name,
                event_type,
                start_date,
                end_date,
                state,
                affected_location,
                impact_level,
                description,
                source_url,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            event_type,
            start.isoformat(),
            end.isoformat(),
            state,
            affected_location,
            impact_level,
            description.strip(),
            source_url.strip(),
            datetime.now().isoformat(timespec="seconds")
        ))

        connection.commit()
        return True

    finally:
        connection.close()


def update_festival_event(
    event_id,
    name,
    event_type,
    start_date,
    end_date,
    state,
    affected_location,
    impact_level,
    description="",
    source_url=""
):
    """Update an existing festival or holiday."""

    name = name.strip()
    state = state.strip()
    affected_location = affected_location.strip()

    if not name or not state or not affected_location:
        raise ValueError(
            "Event name, state and affected location are required."
        )

    try:
        start = datetime.strptime(str(start_date), "%Y-%m-%d").date()
        end = datetime.strptime(str(end_date), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Dates must use YYYY-MM-DD format.")

    if end < start:
        raise ValueError("End date cannot be earlier than start date.")

    connection = get_festival_connection()

    try:
        cursor = connection.execute("""
            UPDATE festival_events
            SET
                name = ?,
                event_type = ?,
                start_date = ?,
                end_date = ?,
                state = ?,
                affected_location = ?,
                impact_level = ?,
                description = ?,
                source_url = ?
            WHERE event_id = ?
        """, (
            name,
            event_type,
            start.isoformat(),
            end.isoformat(),
            state,
            affected_location,
            impact_level,
            description.strip(),
            source_url.strip(),
            int(event_id)
        ))

        connection.commit()
        return cursor.rowcount > 0

    finally:
        connection.close()


def delete_festival_event(event_id):
    """Delete a festival or holiday."""

    connection = get_festival_connection()

    try:
        cursor = connection.execute(
            "DELETE FROM festival_events WHERE event_id = ?",
            (int(event_id),)
        )

        connection.commit()
        return cursor.rowcount > 0

    finally:
        connection.close()