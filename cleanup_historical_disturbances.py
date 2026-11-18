

import sqlite3
from pathlib import Path
from datetime import datetime

from disturbance_service import is_historical_report


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DB_PATH = PROJECT_DIR / "shipments.db"

BACKUP_DIR = PROJECT_DIR / "database_backups"

TABLE_NAME = "disturbance_events"


# --------------------------------------------------
# CREATE A SAFE DATABASE BACKUP
# --------------------------------------------------

def create_database_backup():

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    BACKUP_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = (
        BACKUP_DIR /
        f"shipments_backup_{timestamp}.db"
    )

    # SQLite backup API creates a consistent backup.
    with sqlite3.connect(DB_PATH) as source_conn:
        with sqlite3.connect(backup_path) as backup_conn:
            source_conn.backup(backup_conn)

    print("\nBACKUP CREATED SUCCESSFULLY")
    print(f"Backup location: {backup_path}")

    return backup_path


# --------------------------------------------------
# FIND HISTORICAL RECORDS
# --------------------------------------------------

def find_historical_records():

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:

        cursor = conn.cursor()

        # Confirm that the expected table exists.
        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name = ?
            """,
            (TABLE_NAME,)
        )

        if cursor.fetchone() is None:
            raise RuntimeError(
                f"Table '{TABLE_NAME}' was not found."
            )

        cursor.execute(
            f"PRAGMA table_info({TABLE_NAME})"
        )

        columns = {
            row["name"]
            for row in cursor.fetchall()
        }

        if "title" not in columns:
            raise RuntimeError(
                "The disturbance_events table has no title column."
            )

        # Support common summary column names.
        summary_column = next(
            (
                name for name in (
                    "summary",
                    "description"
                )
                if name in columns
            ),
            None
        )

        cursor.execute(
            f"SELECT rowid AS cleanup_rowid, * FROM {TABLE_NAME}"
        )

        records = cursor.fetchall()

        historical_records = []

        for record in records:

            title = record["title"] or ""

            summary = (
                record[summary_column] or ""
                if summary_column
                else ""
            )

            # Use the historical detection already
            # implemented in disturbance_service.py.
            if is_historical_report(title, summary):

                historical_records.append({
                    "rowid": record["cleanup_rowid"],
                    "title": title,
                    "summary": summary
                })

        return historical_records

    finally:
        conn.close()


# --------------------------------------------------
# DELETE ONLY AFTER USER CONFIRMATION
# --------------------------------------------------

def delete_historical_records(records):

    if not records:
        print("\nNo historical records were identified.")
        return

    conn = sqlite3.connect(DB_PATH)

    try:

        cursor = conn.cursor()

        for record in records:

            cursor.execute(
                f"""
                DELETE FROM {TABLE_NAME}
                WHERE rowid = ?
                """,
                (record["rowid"],)
            )

        conn.commit()

        print(
            f"\nSuccessfully deleted {len(records)} "
            "historical records."
        )

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    print("=" * 55)
    print("HISTORICAL DISTURBANCE DATABASE CLEANUP")
    print("=" * 55)

    # Always create the backup first.
    create_database_backup()

    print("\nScanning saved disturbance announcements...")

    records = find_historical_records()

    print("\nHISTORICAL RECORDS FOUND:", len(records))

    if not records:
        print("Nothing needs to be deleted.")
        return

    print("\nRECORDS PROPOSED FOR DELETION")
    print("-" * 55)

    for number, record in enumerate(records, start=1):

        print(f"\n{number}. {record['title']}")

        if record["summary"]:
            print(
                "Summary:",
                record["summary"][:250]
            )

    print("\n" + "=" * 55)
    print("IMPORTANT: Review the records above.")
    print("Only historical reports identified by the")
    print("existing historical-news detection are listed.")
    print("=" * 55)

    confirmation = input(
        "\nType YES to permanently delete these records: "
    )

    if confirmation.strip() != "YES":

        print("\nCleanup cancelled. No records were deleted.")
        return

    delete_historical_records(records)


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nCLEANUP FAILED")
        print("Reason:", error)
        print("No successful deletion should be assumed.")
