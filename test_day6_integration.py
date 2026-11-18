"""Focused Day 6 checks for alert logic, project files and SQLite structure.

Run from the project directory:
    python test_day6_integration.py

This script does not make network requests or modify the database.
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import py_compile
import sqlite3
import sys
import traceback

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "shipments.db"

REQUIRED_MODULES = [
    "app.py",
    "gps_tracker.py",
    "map_utils.py",
    "routing_optimizer.py",
    "weather_service.py",
    "festival_service.py",
    "disturbance_service.py",
    "day6_alerts.py",
]

passed = 0
failed = 0
checks = []


def report(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        status = "PASS"
    else:
        failed += 1
        status = "FAIL"
    print(f"{status} - {name}" + (f": {detail}" if detail else ""))


print("\nDAY 6 INTEGRATION CHECK")
print("=" * 55)

# 1. Required files and syntax.
for filename in REQUIRED_MODULES:
    path = PROJECT_DIR / filename
    exists = path.is_file()
    report(f"File exists: {filename}", exists)
    if exists:
        try:
            py_compile.compile(str(path), doraise=True)
            report(f"Python syntax: {filename}", True)
        except py_compile.PyCompileError as exc:
            report(f"Python syntax: {filename}", False, str(exc))

# 2. Unit checks for the alert builder.
try:
    from day6_alerts import build_live_alerts

    fixed_now = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
    shipment = {
        "shipment_id": 501,
        "source": "Howrah, West Bengal",
        "destination": "New Town, Kolkata, West Bengal",
        "delivery_deadline": "2026-11-18T15:00:00+00:00",
        "delivery_status": "In Transit",
    }

    # Fresh GPS record: should not create a GPS freshness alert.
    fresh_gps = (22.57, 88.32, 10.0, "2026-11-18T11:55:00+00:00")
    alerts = build_live_alerts(shipment, latest_gps=fresh_gps, now=fixed_now)
    report(
        "Fresh GPS does not create stale/missing GPS alert",
        not any(a["category"] == "GPS" for a in alerts),
    )

    # Stale GPS record: should create Medium GPS alert.
    stale_gps = (22.57, 88.32, 10.0, "2026-11-18T11:00:00+00:00")
    alerts = build_live_alerts(shipment, latest_gps=stale_gps, now=fixed_now)
    report(
        "Stale GPS creates a Medium GPS alert",
        any(a["category"] == "GPS" and a["level"] == "Medium" for a in alerts),
    )

    # No GPS: should be visible as a Medium information/visibility alert.
    alerts = build_live_alerts(shipment, latest_gps=None, now=fixed_now)
    report(
        "Missing GPS creates a Medium GPS visibility alert",
        any(a["category"] == "GPS" and "No GPS" in a["message"] for a in alerts),
    )

    # Today's severe forecast: High alert.
    weather = {
        "Howrah": [{
            "date": "2026-11-18",
            "rainfall": 5,
            "wind_speed": 12,
            "severity": "Severe",
        }]
    }
    alerts = build_live_alerts(
        shipment, weather_forecasts=weather, latest_gps=fresh_gps, now=fixed_now
    )
    report(
        "Severe weather forecast creates a High alert",
        any(a["category"] == "Weather" and a["level"] == "High" for a in alerts),
    )

    # Matching festival date/location.
    festivals = [{
        "name": "Local Festival",
        "start_date": "2026-11-18",
        "end_date": "2026-11-19",
        "affected_location": "Howrah, West Bengal",
        "impact_level": "medium",
    }]
    alerts = build_live_alerts(
        shipment, festivals=festivals, latest_gps=fresh_gps, now=fixed_now
    )
    report(
        "Active matching festival creates a Festival alert",
        any(a["category"] == "Festival" for a in alerts),
    )

    # Matching RSS disturbance; resolved event must be ignored.
    disturbances = [{
        "title": "Road advisory",
        "location": "Howrah, West Bengal",
        "event_date": "2026-11-18",
        "severity": "high",
        "status": "Unverified",
    }, {
        "title": "Resolved item",
        "location": "Howrah, West Bengal",
        "event_date": "2026-11-18",
        "severity": "high",
        "status": "resolved",
    }]
    alerts = build_live_alerts(
        shipment, disturbances=disturbances,
        latest_gps=fresh_gps, now=fixed_now
    )
    rss_alerts = [a for a in alerts if a["category"] == "RSS Disturbance"]
    report(
        "Unresolved RSS item matches and resolved item is ignored",
        len(rss_alerts) == 1 and rss_alerts[0]["level"] == "High",
    )

    # High alerts should sort before Medium, then Low.
    levels = [a["level"] for a in alerts]
    order = {"High": 0, "Medium": 1, "Low": 2}
    report(
        "Alerts are sorted by priority",
        levels == sorted(levels, key=lambda level: order.get(level, 3)),
    )

except Exception as exc:
    report("Alert-builder unit tests", False, f"{type(exc).__name__}: {exc}")
    traceback.print_exc()

# 3. Read-only database diagnostics. Table names can vary by project version,
# so missing optional tables are reported as CHECK, not as a test failure.
if DB_PATH.is_file():
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True) as conn:
            tables = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            print("\nDATABASE DIAGNOSTICS")
            for table in (
                "shipments", "tracking_updates", "tracking_sessions",
                "festival_events", "disturbance_events"
            ):
                print(f"{'FOUND' if table in tables else 'CHECK'} - {table}")

            if "shipments" in tables:
                count = conn.execute("SELECT COUNT(*) FROM shipments").fetchone()[0]
                print(f"Registered shipments in database: {count}")
            if "tracking_updates" in tables:
                count = conn.execute(
                    "SELECT COUNT(*) FROM tracking_updates"
                ).fetchone()[0]
                print(f"GPS records in database: {count}")
    except sqlite3.Error as exc:
        report("Read-only database inspection", False, str(exc))
else:
    print("\nDATABASE DIAGNOSTICS")
    print("CHECK - shipments.db was not found beside this test script.")

print("\nSUMMARY")
print("=" * 55)
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(
    "\nNote: These checks test Python syntax and alert-builder behaviour. "
    "They do not prove that the Streamlit UI refreshes automatically, "
    "browser GPS permissions work, or external APIs return live data."
)
sys.exit(1 if failed else 0)
