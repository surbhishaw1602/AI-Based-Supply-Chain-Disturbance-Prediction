"""Day 6 live shipment alert generation.

Alerts are informational screening signals. Geographic/text matches and forecasts
must not be treated as confirmation of a road closure or actual delay.
"""
from datetime import datetime, date, timezone
import math


def _safe_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _parse_date(value):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _normalise_now(now=None):
    """Return an aware datetime; default to the machine's local timezone."""
    value = now or datetime.now().astimezone()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _location_matches(event_location, source, destination):
    """Conservative text match; this is not a geographic proximity test."""
    location = str(event_location or "").strip().lower()
    if not location or location in {
        "unknown", "location not identified", "india"
    }:
        return False

    places = (str(source or "").strip().lower(),
              str(destination or "").strip().lower())

    for place in places:
        if len(place) >= 4 and place in location:
            return True
        # Permit a complete comma-separated component match, e.g. "Kolkata"
        # in "Kolkata, West Bengal", without fuzzy substring matching.
        components = [part.strip() for part in place.split(",") if part.strip()]
        if any(len(part) >= 4 and part in location for part in components):
            return True
    return False


def _gps_timestamp(latest_gps):
    """Accept the legacy GPS tuple/list or a mapping with a timestamp field."""
    if latest_gps is None:
        return None
    if isinstance(latest_gps, dict):
        value = (
            latest_gps.get("recorded_at")
            or latest_gps.get("timestamp")
            or latest_gps.get("last_recorded")
        )
    else:
        try:
            value = latest_gps[3]
        except (TypeError, IndexError, KeyError):
            return None

    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_live_alerts(
    shipment,
    weather_forecasts=None,
    festivals=None,
    disturbances=None,
    latest_gps=None,
    now=None,
):
    """Build sorted alert dictionaries for one shipment.

    Expected GPS tuple format is the existing legacy shape:
    (latitude, longitude, accuracy, recorded_at).
    A mapping with recorded_at/timestamp/last_recorded is also accepted.
    """
    if not shipment:
        return []

    current_time = _normalise_now(now)
    today = current_time.date()
    shipment_id = shipment.get("shipment_id", "Unknown")
    source = shipment.get("source", "")
    destination = shipment.get("destination", "")
    alerts = []

    def add(level, category, message, event_date=None, **extra):
        item = {
            "shipment_id": shipment_id,
            "level": level,
            "category": category,
            "message": message,
            "date": str(event_date or "Unknown"),
        }
        item.update(extra)
        alerts.append(item)

    # 1. Weather: only today's forecast rows are used for live alerts.
    for location_name, forecast in (weather_forecasts or {}).items():
        for day in forecast or []:
            if str(day.get("date", ""))[:10] != today.isoformat():
                continue

            rainfall = _safe_float(day.get("rainfall"))
            wind = _safe_float(day.get("wind_speed"))
            severity = str(day.get("severity", "")).strip().lower()

            if severity == "severe":
                add("High", "Weather",
                    f"Severe weather forecast near {location_name}.",
                    today.isoformat())
            elif rainfall is not None and rainfall >= 50:
                add("High", "Weather",
                    f"Heavy rainfall forecast near {location_name}: {rainfall:.1f} mm.",
                    today.isoformat())
            elif rainfall is not None and rainfall >= 20:
                add("Medium", "Weather",
                    f"Moderate-to-heavy rainfall forecast near {location_name}: {rainfall:.1f} mm.",
                    today.isoformat())

            if wind is not None and wind >= 60:
                add("High", "Weather",
                    f"Strong wind forecast near {location_name}: {wind:.1f} km/h.",
                    today.isoformat())
            elif wind is not None and wind >= 40:
                add("Medium", "Weather",
                    f"Elevated wind forecast near {location_name}: {wind:.1f} km/h.",
                    today.isoformat())

    # 2. Festivals and holidays: active dates + conservative endpoint text match.
    for event in festivals or []:
        start = _parse_date(event.get("start_date"))
        end = _parse_date(event.get("end_date")) or start
        if not start or not end or not (start <= today <= end):
            continue

        location = event.get("affected_location", "")
        if not _location_matches(location, source, destination):
            continue

        impact = str(event.get("impact_level", "Unknown")).strip().lower()
        level = "High" if impact in {"high", "severe"} else "Medium"
        add(
            level,
            "Festival",
            f"{event.get('name', 'Festival / holiday')} may affect operations near {location}.",
            start.isoformat(),
            verification="Check local traffic and official advisories.",
        )

    # 3. RSS disturbances: skip resolved/expired items and old dated events.
    for event in disturbances or []:
        status = str(event.get("status", "Unverified")).strip().lower()
        if status in {"resolved", "expired", "dismissed"}:
            continue

        event_date = _parse_date(event.get("event_date"))
        if event_date and event_date < today:
            continue

        location = event.get("location", "")
        if not _location_matches(location, source, destination):
            continue

        severity = str(event.get("severity", "Low")).strip().lower()
        level = (
            "High" if severity == "high"
            else "Medium" if severity == "medium"
            else "Low"
        )
        add(
            level,
            "RSS Disturbance",
            f"{event.get('title', 'Public disturbance announcement')} mentions {location}. Verify with official sources.",
            event_date.isoformat() if event_date else "Date not confirmed",
            verification="Unverified announcement",
        )

    # 4. GPS freshness. A missing record is a Medium operational visibility alert.
    if latest_gps:
        try:
            recorded_at = _gps_timestamp(latest_gps)
            if recorded_at is None:
                raise ValueError("GPS timestamp is missing.")
            age_minutes = (
                current_time.astimezone(timezone.utc) - recorded_at
            ).total_seconds() / 60

            if age_minutes < 0:
                raise ValueError("GPS timestamp is in the future.")
            if age_minutes > 15:
                add(
                    "Medium", "GPS",
                    f"Latest GPS update is {age_minutes:.0f} minutes old.",
                    recorded_at.isoformat(),
                )
        except (ValueError, TypeError, OverflowError):
            add(
                "Medium", "GPS",
                "Latest GPS timestamp could not be verified.",
                "Unknown",
            )
    else:
        add(
            "Medium", "GPS",
            "No GPS tracking record is available yet.",
            "Unknown",
        )

    # 5. Delivery deadline.
    deadline_value = shipment.get("delivery_deadline")
    if deadline_value:
        try:
            deadline = datetime.fromisoformat(
                str(deadline_value).replace("Z", "+00:00")
            )
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)

            remaining_minutes = (
                deadline.astimezone(timezone.utc)
                - current_time.astimezone(timezone.utc)
            ).total_seconds() / 60

            delivery_status = str(
                shipment.get("delivery_status", "In Transit")
            ).strip().lower()

            if delivery_status not in {"delivered", "cancelled", "returned"}:
                if remaining_minutes < 0:
                    add("High", "Deadline", "Delivery deadline has passed.",
                        deadline.isoformat())
                elif remaining_minutes <= 30:
                    add("High", "Deadline",
                        "Delivery deadline is within 30 minutes.",
                        deadline.isoformat())
                elif remaining_minutes <= 120:
                    add("Medium", "Deadline",
                        "Delivery deadline is within 2 hours.",
                        deadline.isoformat())
        except (ValueError, TypeError, OverflowError):
            add("Low", "Deadline",
                "Delivery deadline could not be parsed.", "Unknown")

    priority = {"High": 0, "Medium": 1, "Low": 2}
    alerts.sort(key=lambda item: priority.get(item["level"], 3))
    return alerts
