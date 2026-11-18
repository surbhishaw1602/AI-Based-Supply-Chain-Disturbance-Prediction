import streamlit as st
import pandas as pd
import folium
import sqlite3
import requests
import math
import re
from html import escape

from pathlib import Path
from datetime import datetime, date, timedelta, timezone

from streamlit_folium import st_folium

# Automatic GPS component
from gps_component import automatic_gps_tracker

from gps_tracker import (
    initialize_gps_database,
    save_gps_location,
    get_latest_location,
    get_tracking_history,
    update_shipment_coordinates
)

# ROUTING AND GEOCODING
from map_utils import (
    generate_route,
    reverse_geocode,
    geocode_location,
    geocode_disturbance_locality,
    distance_from_location_to_route_km
)

# DAY 5 - INTELLIGENT ROUTING ENGINE
from routing_optimizer import (
    generate_route_candidates,
    integrate_route_conditions,
    rank_routes_intelligently,
    predict_additional_delay,
    calculate_revised_eta,
    calculate_route_risk
)

# DAY 3 - WEATHER MONITORING
from weather_service import get_weather_forecast

# DAY 3 - FESTIVAL AND HOLIDAY MONITORING
from festival_service import (
    initialize_festival_database,
    get_festival_events,
    add_festival_event,
    update_festival_event,
    delete_festival_event
)

# DAY 3 - RSS DISTURBANCE MONITORING
from disturbance_service import (
    initialize_disturbance_database,
    fetch_disturbance_feeds,
    get_disturbance_events
)


# ==========================================
# DAY 4 - MACHINE LEARNING
# ==========================================
from ml_service import (
    initialize_ml_database,
    add_training_record,
    get_training_data,
    get_dataset_summary,
    train_random_forest_model,
    get_random_forest_model_status,
    predict_disturbance_risk,
    predict_seven_day_risk,
    save_auto_observation,
    get_auto_observations,
    get_auto_dataset_summary,
    verify_auto_observation,
    generate_synthetic_dataset,
    get_synthetic_dataset_summary,
    get_synthetic_training_data,
    train_synthetic_random_forest_model
)

initialize_ml_database()



# ==================================================
# DAY 3 STEP 4 - DISTURBANCE / ROUTE MATCHING HELPERS
# ==================================================

DISTURBANCE_MATCH_RADIUS_KM = 15.0

def _is_historical_disturbance(event, days=30):
    """Exclude clearly old event dates; unknown dates remain reviewable."""
    value = event.get("event_date")
    if not value:
        return False
    try:
        event_day = date.fromisoformat(str(value)[:10])
        return (date.today() - event_day).days > days
    except (ValueError, TypeError):
        return False


def find_shipment_disturbance_matches(
    route_points,
    events,
    radius_km=15.0
):
    """
    Compare specific, geocoded RSS event locations with an OSRM route.

    These are potential geographic matches, not confirmed
    road closures or verified shipment disruptions.
    """

    if not route_points or not events:
        st.session_state["last_route_disturbance_diagnostics"] = {
            "events_checked": len(events or []),
            "events_with_location": 0,
            "events_rejected_by_confidence": 0,
            "geocoding_attempts": 0,
            "geocoding_successes": 0,
            "events_outside_radius": 0,
            "reason": "No route geometry or no saved RSS events",
        }
        return []

    matches = []
    diagnostics = {
        "events_checked": len(events),
        "events_with_location": 0,
        "events_rejected_by_confidence": 0,
        "geocoding_attempts": 0,
        "geocoding_successes": 0,
        "events_outside_radius": 0,
    }

    broad_locations = {
        "india",
        "west bengal",
        "odisha",
        "bihar",
        "jharkhand",
        "andhra pradesh",
        "karnataka",
        "punjab",
        "delhi",
        "maharashtra",
        "uttar pradesh",
        "assam",
        "rajasthan",
        "gujarat",
        "tamil nadu",
        "telangana",
        "kerala",
        "madhya pradesh"
    }

    for event in events:

        location_text = str(
            event.get("location") or ""
        ).strip()

        if not location_text:
            continue

        if location_text.lower() in {
            "location not identified",
            "unknown",
            "india"
        }:
            continue

        if _is_historical_disturbance(event):
            continue

        # Only high-confidence settlement names may create proximity alerts.
        # Missing metadata is treated conservatively for compatibility.
        if event.get("route_matchable") is not True:
            diagnostics["events_rejected_by_confidence"] += 1
            continue
        if str(event.get("location_confidence", "")).lower() != "high":
            diagnostics["events_rejected_by_confidence"] += 1
            continue

        diagnostics["events_with_location"] += 1
        parts = [
            part.strip()
            for part in re.split(
                r",|;|\||\band\b",
                location_text,
                flags=re.I
            )
            if part.strip()
        ]

        # Identify district context before geocoding localities.
        district = next(
            (
                part for part in parts
                if "district" in part.lower()
            ),
            None
        )

        state = None

        state_names = {
            "andhra pradesh",
            "west bengal",
            "odisha",
            "bihar",
            "jharkhand",
            "karnataka",
            "tamil nadu",
            "telangana",
            "kerala",
            "maharashtra",
            "punjab",
            "rajasthan",
            "gujarat",
            "assam"
        }

        for part in parts:
            if part.lower() in state_names:
                state = part
                break

        candidates = [
            part for part in parts
            if (
                len(part) >= 3
                and part.lower() not in broad_locations
                and part.lower() != (district or "").lower()
            )
        ]

        best = None

        for place in candidates[:3]:

            try:
                diagnostics["geocoding_attempts"] += 1
                geo = load_disturbance_locality(
                    place,
                    district=district,
                    state=state
                )
            except Exception as error:
                print(
                    f"Disturbance geocoding failed for "
                    f"{place}: {error}"
                )
                geo = None

            if not geo or geo.get("is_locality") is not True:
                continue

            diagnostics["geocoding_successes"] += 1
            distance = distance_from_location_to_route_km(
                geo,
                route_points
            )

            if distance is None:
                continue

            if best is None or distance < best["distance_km"]:
                best = {
                    "place": place,
                    "latitude": geo["latitude"],
                    "longitude": geo["longitude"],
                    "display_name": geo.get("display_name"),
                    "location_confidence": event.get("location_confidence", "High"),
                    "location_evidence": event.get("location_evidence", "article_text"),
                    "distance_km": float(distance)
                }

        if best is None:
            continue

        if best["distance_km"] > float(radius_km):
            diagnostics["events_outside_radius"] += 1
            continue

        best.update({
            "disturbance_id": event.get("disturbance_id"),
            "title": event.get("title") or "Untitled announcement",
            "event_type": event.get("event_type") or "Unknown",
            "severity": event.get("severity") or "Unknown",
            "status": event.get("status") or "Unverified",
            "event_date": event.get("event_date"),
            "published_at": event.get("published_at"),
            "source_name": event.get("source_name"),
            "source_url": event.get("source_url"),
            "match_label": (
                "Very close to route"
                if best["distance_km"] <= 5
                else "Potential nearby impact"
            )
        })

        matches.append(best)

    matches.sort(
        key=lambda item: item["distance_km"]
    )

    # Diagnostics distinguish missing RSS locations from geocoding failures
    # and events that are genuinely outside the route corridor.
    st.session_state["last_route_disturbance_diagnostics"] = diagnostics
    return matches


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Shipment Management System",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Shipment Management System")

st.markdown(
    "### Day 1 + Day 2 + Day 3: Shipment Registration, GPS Tracking, "
    "Weather and Festival Monitoring"
)

DB_PATH = Path(__file__).resolve().parent / "shipments.db"

MAX_GPS_ACCURACY = 200.0

# A phone reading must be reasonably close to the planned route.
MAX_GPS_DISTANCE_FROM_ROUTE_KM = 5.0

# Use a stricter limit when accepting a GPS position for ETA calculations.
MAX_GPS_DISTANCE_FOR_ETA_KM = 2.0

# Reject implausible movement between consecutive GPS records.
MAX_GPS_SPEED_KMH = 160.0
GPS_JUMP_CHECK_MAX_GAP_MINUTES = 30.0

# A GPS record older than 15 minutes is not treated as live.
MAX_GPS_STALENESS_MINUTES = 15

# A shipment whose revised ETA is within this many minutes of the
# deadline is classified as "At Risk".
DEADLINE_AT_RISK_BUFFER_MINUTES = 30


# ==================================================
# 2. DATABASE CONNECTION
# ==================================================

def get_connection():

    conn = sqlite3.connect(DB_PATH)

    conn.execute("PRAGMA foreign_keys = ON")

    return conn


# ==================================================
# 3. INITIALIZE DATABASES
# ==================================================

initialize_gps_database()
initialize_festival_database()
initialize_disturbance_database()


# ==================================================
# 4. DATABASE FUNCTIONS
# ==================================================

def get_shipments():

    conn = get_connection()

    try:

        shipments = pd.read_sql_query(
            "SELECT * FROM shipments ORDER BY shipment_id DESC",
            conn
        )

        return shipments

    finally:

        conn.close()


def ensure_delivery_status_column():

    """Add the delivery_status column to existing databases (safe to re-run)."""

    conn = get_connection()

    try:

        columns = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(shipments)"
            ).fetchall()
        ]

        # The shipments table does not exist yet, nothing to migrate.
        if not columns:
            return

        if "delivery_status" not in columns:

            conn.execute(
                """
                ALTER TABLE shipments
                ADD COLUMN delivery_status TEXT DEFAULT 'In Transit'
                """
            )

        conn.execute(
            """
            UPDATE shipments
            SET delivery_status = 'In Transit'
            WHERE delivery_status IS NULL OR TRIM(delivery_status) = ''
            """
        )

        conn.commit()

    finally:

        conn.close()


def get_shipment_delivery_status(shipment_id):

    conn = get_connection()

    try:

        row = conn.execute(
            """
            SELECT delivery_status
            FROM shipments
            WHERE shipment_id = ?
            """,
            (int(shipment_id),)
        ).fetchone()

        return row[0] if row and row[0] else "In Transit"

    finally:

        conn.close()


def update_shipment_delivery_status(shipment_id, status):

    allowed_statuses = {
        "In Transit",
        "Delivered",
        "Cancelled",
        "Returned",
        "Unknown"
    }

    if status not in allowed_statuses:
        raise ValueError("Invalid delivery status.")

    conn = get_connection()

    try:

        conn.execute(
            """
            UPDATE shipments
            SET delivery_status = ?
            WHERE shipment_id = ?
            """,
            (status, int(shipment_id))
        )

        conn.commit()

    finally:

        conn.close()


# Add the new column when the application starts.
ensure_delivery_status_column()


@st.cache_data(ttl=3600, show_spinner=False)
def load_ml_shipment_route_distance(
    source_lat,
    source_lon,
    destination_lat,
    destination_lon
):
    """Get the planned OSRM road distance for a registered shipment."""

    route = generate_route(
        (float(source_lat), float(source_lon)),
        (float(destination_lat), float(destination_lon))
    )

    return float(route["distance_km"])


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_historical_weather_for_route(
    source_lat,
    source_lon,
    destination_lat,
    destination_lon,
    observation_date
):
    """
    Retrieve historical daily weather for the midpoint
    of a registered shipment's planned OSRM route.
    """

    route = generate_route(
        (float(source_lat), float(source_lon)),
        (float(destination_lat), float(destination_lon))
    )

    route_points = route.get("points", [])

    if not route_points:
        raise ValueError(
            "Could not obtain route points for weather lookup."
        )

    # Use the middle point of the planned road route.
    midpoint = route_points[len(route_points) // 2]

    latitude = float(midpoint[0])
    longitude = float(midpoint[1])

    api_url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": observation_date.isoformat(),
        "end_date": observation_date.isoformat(),
        "daily": (
            "temperature_2m_mean,"
            "precipitation_sum,"
            "wind_speed_10m_max"
        ),
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(
        api_url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):
        raise ValueError(
            data.get("reason", "Historical weather data unavailable.")
        )

    daily = data.get("daily", {})

    temperatures = daily.get("temperature_2m_mean", [])
    rainfall_values = daily.get("precipitation_sum", [])
    wind_values = daily.get("wind_speed_10m_max", [])

    if not temperatures or not rainfall_values or not wind_values:
        raise ValueError(
            "The historical weather API returned incomplete data."
        )

    temperature = temperatures[0]
    rainfall = rainfall_values[0]
    wind_speed = wind_values[0]

    if any(
        value is None
        for value in (temperature, rainfall, wind_speed)
    ):
        raise ValueError(
            "One or more historical weather values are unavailable."
        )

    temperature = float(temperature)
    rainfall = float(rainfall)
    wind_speed = float(wind_speed)

    if not all(
        math.isfinite(value)
        for value in (temperature, rainfall, wind_speed)
    ):
        raise ValueError(
            "The API returned invalid weather values."
        )

    return {
        "date": observation_date.isoformat(),
        "temperature": round(temperature, 1),
        "rainfall": round(rainfall, 1),
        "wind_speed": round(wind_speed, 1),
        "latitude": latitude,
        "longitude": longitude,
        "source": "Open-Meteo Historical Weather API"
    }


def get_gps_update_count():

    conn = get_connection()

    try:

        result = conn.execute(
            "SELECT COUNT(*) FROM tracking_updates"
        ).fetchone()

        return result[0]

    finally:

        conn.close()


def register_shipment(
    source,
    destination,
    deadline,
    source_lat,
    source_lon,
    destination_lat,
    destination_lon
):

    conn = get_connection()

    try:

        conn.execute(
            """
            INSERT INTO shipments (
                source,
                destination,
                delivery_deadline,
                created_at,
                source_latitude,
                source_longitude,
                destination_latitude,
                destination_longitude
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                destination,
                deadline.isoformat(),
                datetime.now().isoformat(),
                source_lat,
                source_lon,
                destination_lat,
                destination_lon
            )
        )

        conn.commit()

    finally:

        conn.close()


# ==================================================
# 5. AUTOMATIC LOCATION GEOCODING
# ==================================================

@st.cache_data(ttl=86400, show_spinner=False)
def load_geocoded_location(location_name):

    """
    Find latitude and longitude automatically
    using OpenStreetMap Nominatim.
    """

    return geocode_location(location_name.strip())


@st.cache_data(ttl=86400, show_spinner=False)
def load_disturbance_locality(place, district=None, state=None):

    """Cache RSS locality geocoding so alternative routes reuse lookups."""

    return geocode_disturbance_locality(
        str(place).strip(),
        district=district,
        state=state
    )


@st.cache_data(ttl=86400, show_spinner=False)
def load_location_name(latitude, longitude):

    """
    Reverse-geocode route checkpoints.
    """

    return reverse_geocode(
        float(latitude),
        float(longitude)
    )


# ==================================================
# 6. WEATHER FORECAST CACHE
# ==================================================

@st.cache_data(ttl=1800, show_spinner=False)
def load_weather(latitude, longitude):

    return get_weather_forecast(
        latitude,
        longitude
    )


@st.cache_data(ttl=900, show_spinner=False)
def refresh_ml_rss_events():
    """Fetch RSS feeds at most once every 15 minutes across Streamlit reruns."""
    return fetch_disturbance_feeds()


@st.cache_data(ttl=3600, show_spinner=False)
def load_cached_route(source, destination):

    """Cache the planned route while the live GPS panel refreshes."""

    return generate_route(source, destination)


# ==================================================
# 7. DISTANCE CALCULATION
# ==================================================

def calculate_distance_km(point1, point2):

    """
    Calculate distance between two GPS coordinates
    using the Haversine formula.
    """

    lat1, lon1 = point1
    lat2, lon2 = point2

    earth_radius = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius * c


def assess_shipment_outcome(
    shipment,
    gps_history,
    delivery_status
):
    """
    Suggest an operational outcome using delivery status
    and reliable GPS records.

    This is supporting evidence, NOT a verified ML label.
    """

    result = {
        "suggestion": "Insufficient evidence",
        "details": [],
        "destination_distance_km": None,
        "reliable_gps_records": 0,
        "arrival_evidence": False
    }

    try:
        destination_lat = float(
            shipment["destination_latitude"]
        )
        destination_lon = float(
            shipment["destination_longitude"]
        )

        if not (
            -90 <= destination_lat <= 90
            and -180 <= destination_lon <= 180
        ):
            raise ValueError("Invalid destination coordinates.")

    except (KeyError, TypeError, ValueError):
        result["details"].append(
            "Destination coordinates are missing or invalid."
        )

        if delivery_status == "Delivered":
            result["suggestion"] = (
                "Delivery status says Delivered; GPS cannot verify arrival."
            )

        return result

    reliable_records = []

    for record in gps_history or []:

        try:
            latitude = float(record[0])
            longitude = float(record[1])
            accuracy = float(record[2])
            recorded_at = str(record[3])

            if not (
                -90 <= latitude <= 90
                and -180 <= longitude <= 180
                and math.isfinite(accuracy)
                and 0 <= accuracy <= MAX_GPS_ACCURACY
            ):
                continue

            timestamp = datetime.fromisoformat(
                recorded_at.replace("Z", "+00:00")
            )

            distance = calculate_distance_km(
                (latitude, longitude),
                (destination_lat, destination_lon)
            )

            reliable_records.append({
                "timestamp": timestamp,
                "distance_km": distance,
                "accuracy": accuracy
            })

        except (
            IndexError,
            TypeError,
            ValueError,
            OverflowError
        ):
            continue

    reliable_records.sort(
        key=lambda item: item["timestamp"]
    )

    result["reliable_gps_records"] = len(reliable_records)

    if reliable_records:

        latest = reliable_records[-1]

        result["destination_distance_km"] = round(
            latest["distance_km"], 3
        )

        result["details"].append(
            f"Latest reliable GPS fix is "
            f"{latest['distance_km']:.2f} km from the destination."
        )

    else:

        result["details"].append(
            "No GPS records with acceptable accuracy were found."
        )

    # Require two recent reliable fixes near the destination.
    recent_records = reliable_records[-3:]

    arrival_fixes = [
        item for item in recent_records
        if item["distance_km"] <= 0.5
    ]

    arrival_confirmed_by_gps = len(arrival_fixes) >= 2

    result["arrival_evidence"] = arrival_confirmed_by_gps

    if arrival_confirmed_by_gps:

        result["details"].append(
            "At least two of the latest three reliable GPS fixes "
            "were within 500 metres of the destination."
        )

    if delivery_status == "Delivered":

        if arrival_confirmed_by_gps:
            result["suggestion"] = (
                "Delivered status and GPS arrival evidence agree."
            )
        else:
            result["suggestion"] = (
                "Delivery status says Delivered, but GPS arrival "
                "has not been independently confirmed."
            )

    elif delivery_status == "In Transit":

        if arrival_confirmed_by_gps:
            result["suggestion"] = (
                "GPS suggests destination reached, but delivery "
                "status is still In Transit. Review the status."
            )
        elif reliable_records:
            result["suggestion"] = (
                "Latest reliable GPS record is away from the "
                "destination. Arrival is not confirmed."
            )
        else:
            result["suggestion"] = (
                "Shipment is marked In Transit, but reliable "
                "GPS evidence is unavailable."
            )

    elif delivery_status == "Cancelled":
        result["suggestion"] = (
            "Shipment is marked Cancelled. Confirm the cancellation."
        )

    elif delivery_status == "Returned":
        result["suggestion"] = (
            "Shipment is marked Returned. Review the return evidence."
        )

    else:
        result["suggestion"] = (
            "Delivery status is unknown. More evidence is required."
        )

    return result


# ==================================================
# 8. ROUTE AND DISTANCE HELPERS
# ==================================================

def select_route_weather_locations(
    route,
    source,
    destination,
    max_points=5
):

    """
    Select up to five representative intermediate
    locations along the road route.
    """

    route_points = route.get("points", []) or []
    checkpoints = route.get("checkpoints", []) or []

    def normalize_points(points):

        normalized = []

        for point in points:

            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue

            try:

                latitude = float(point[0])
                longitude = float(point[1])

            except (TypeError, ValueError):

                continue

            if (
                -90 <= latitude <= 90
                and -180 <= longitude <= 180
            ):

                normalized.append(
                    (latitude, longitude)
                )

        return normalized

    candidates = normalize_points(checkpoints)

    if not candidates:

        normalized_route = normalize_points(route_points)

        if len(normalized_route) > 2:

            sample_count = min(
                max_points,
                len(normalized_route) - 2
            )

            indexes = []

            for sample_number in range(1, sample_count + 1):

                index = round(
                    sample_number
                    * (len(normalized_route) - 1)
                    / (sample_count + 1)
                )

                index = max(
                    1,
                    min(len(normalized_route) - 2, index)
                )

                if index not in indexes:
                    indexes.append(index)

            candidates = [
                normalized_route[index]
                for index in indexes
            ]

    intermediate = []

    for point in candidates:

        if (
            calculate_distance_km(point, source) < 0.5
            or calculate_distance_km(point, destination) < 0.5
        ):
            continue

        if all(
            calculate_distance_km(point, existing) >= 1.0
            for existing in intermediate
        ):
            intermediate.append(point)

    if len(intermediate) > max_points:

        selected = []

        for sample_number in range(max_points):

            index = (
                round(
                    sample_number
                    * (len(intermediate) - 1)
                    / (max_points - 1)
                )
                if max_points > 1
                else 0
            )

            if intermediate[index] not in selected:
                selected.append(intermediate[index])

        intermediate = selected

    return intermediate


def calculate_remaining_route(
    current_location,
    route_points
):

    """
    Find the nearest point on the planned route
    and calculate the remaining route distance.
    """

    if not route_points:
        return None, None

    nearest_index = min(
        range(len(route_points)),
        key=lambda index: calculate_distance_km(
            current_location,
            route_points[index]
        )
    )

    remaining_distance = 0.0

    for index in range(
        nearest_index,
        len(route_points) - 1
    ):

        remaining_distance += calculate_distance_km(
            route_points[index],
            route_points[index + 1]
        )

    distance_to_route = calculate_distance_km(
        current_location,
        route_points[nearest_index]
    )

    remaining_distance += distance_to_route

    return remaining_distance, nearest_index


def validate_gps_movement(
    shipment_id,
    latitude,
    longitude,
    route_points
):
    """
    Reject GPS jumps that imply an implausible vehicle speed.

    Ignore the previous record when it is not near the current route,
    so an old phone-test location does not block a valid new reading.
    """

    history = get_tracking_history(shipment_id)

    if not history:
        return True, None

    previous = history[-1]

    try:
        previous_latitude = float(previous[0])
        previous_longitude = float(previous[1])

        previous_timestamp = datetime.fromisoformat(
            str(previous[3]).replace("Z", "+00:00")
        )

        if previous_timestamp.tzinfo is None:
            previous_timestamp = previous_timestamp.replace(
                tzinfo=timezone.utc
            )

    except (TypeError, ValueError, IndexError):
        # An unreadable previous record should not permanently block GPS.
        return True, None

    previous_coordinates = (
        previous_latitude,
        previous_longitude
    )

    # Ignore old test/device locations that are not near this route.
    previous_route_distance = distance_from_location_to_route_km(
        previous_coordinates,
        route_points
    )

    if (
        previous_route_distance is None
        or previous_route_distance > MAX_GPS_DISTANCE_FROM_ROUTE_KM
    ):
        return True, None

    current_time = datetime.now(timezone.utc)

    elapsed_seconds = (
        current_time - previous_timestamp.astimezone(timezone.utc)
    ).total_seconds()

    if elapsed_seconds <= 0:
        return False, (
            "GPS reading rejected because its timestamp did not advance."
        )

    # Very short intervals can exaggerate GPS jitter.
    if elapsed_seconds < 5:
        return True, None

    # Do not infer speed across a long period without GPS updates.
    if elapsed_seconds > GPS_JUMP_CHECK_MAX_GAP_MINUTES * 60:
        return True, None

    distance_km = calculate_distance_km(
        previous_coordinates,
        (float(latitude), float(longitude))
    )

    speed_kmh = distance_km / (elapsed_seconds / 3600)

    if speed_kmh > MAX_GPS_SPEED_KMH:
        return False, (
            f"GPS jump rejected: the implied speed is "
            f"{speed_kmh:.1f} km/h. The configured maximum is "
            f"{MAX_GPS_SPEED_KMH:.0f} km/h. Waiting for the next GPS fix."
        )

    return True, None


def calculate_gps_aware_route_progress(shipment_id, selected_route):
    """
    Calculate remaining distance and driving time from a recent,
    accurate GPS fix that is close to the selected route.
    """

    latest = get_latest_location(shipment_id)

    if not latest:
        return {
            "available": False,
            "reason": "No GPS location is available for this shipment."
        }

    try:
        latitude = float(latest[0])
        longitude = float(latest[1])
        accuracy = float(latest[2])
        recorded_at = datetime.fromisoformat(
            str(latest[3]).replace("Z", "+00:00")
        )

        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=timezone.utc)

        recorded_at = recorded_at.astimezone(timezone.utc)
        now_utc = datetime.now(timezone.utc)

        age_minutes = (
            now_utc - recorded_at
        ).total_seconds() / 60

    except (TypeError, ValueError, OverflowError):
        return {
            "available": False,
            "reason": "The latest GPS record contains invalid data."
        }

    if not all(math.isfinite(value) for value in (
        latitude, longitude, accuracy, age_minutes
    )):
        return {
            "available": False,
            "reason": "The latest GPS record contains non-finite values."
        }

    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return {
            "available": False,
            "reason": "The GPS coordinates are outside valid limits."
        }

    if accuracy < 0 or accuracy > MAX_GPS_ACCURACY:
        return {
            "available": False,
            "reason": (
                f"GPS accuracy is {accuracy:.1f} metres. "
                "A reading of 200 metres or better is required."
            )
        }

    if age_minutes < -1:
        return {
            "available": False,
            "reason": "The GPS timestamp is in the future."
        }

    if age_minutes > MAX_GPS_STALENESS_MINUTES:
        return {
            "available": False,
            "reason": (
                f"The latest GPS record is {age_minutes:.1f} minutes old. "
                f"A record no older than {MAX_GPS_STALENESS_MINUTES} "
                "minutes is required."
            )
        }

    route_points = (
        selected_route.get("points")
        or selected_route.get("geometry")
        or []
    )

    if len(route_points) < 2:
        return {
            "available": False,
            "reason": "The selected route has no usable route geometry."
        }

    current_location = (latitude, longitude)

    distance_from_route = distance_from_location_to_route_km(
        current_location,
        route_points
    )

    if (
        distance_from_route is None
        or distance_from_route > MAX_GPS_DISTANCE_FOR_ETA_KM
    ):
        return {
            "available": False,
            "reason": (
                "The latest GPS position is more than "
                f"{MAX_GPS_DISTANCE_FOR_ETA_KM:.1f} km from the selected "
                "route. A closer GPS fix is required for ETA calculations."
            )
        }

    remaining_distance, nearest_index = calculate_remaining_route(
        current_location,
        route_points
    )

    if remaining_distance is None or nearest_index is None:
        return {
            "available": False,
            "reason": "Could not calculate the remaining route distance."
        }

    total_distance = float(selected_route["distance_km"])
    total_duration = float(selected_route["duration_minutes"])

    if (
        not math.isfinite(total_distance)
        or not math.isfinite(total_duration)
        or total_distance <= 0
        or total_duration < 0
    ):
        return {
            "available": False,
            "reason": "The selected route has invalid distance or duration."
        }

    remaining_ratio = min(
        1.0,
        max(0.0, remaining_distance / total_distance)
    )

    remaining_minutes = total_duration * remaining_ratio

    if remaining_distance < 0.2:
        remaining_distance = 0.0
        remaining_minutes = 0.0

    progress = (
        1.0 - remaining_ratio
    ) * 100

    return {
        "available": True,
        "latitude": latitude,
        "longitude": longitude,
        "accuracy": accuracy,
        "recorded_at": recorded_at,
        "age_minutes": age_minutes,
        "distance_from_route_km": distance_from_route,
        "remaining_distance_km": remaining_distance,
        "remaining_minutes": remaining_minutes,
        "progress_percent": max(0.0, min(100.0, progress))
    }


# ==================================================
# 9. FESTIVAL MONITORING HELPER FUNCTIONS
# ==================================================

def get_festival_impact_color(impact_level):

    """Return a display color for the event impact level."""

    colors = {
        "High": "🔴",
        "Medium": "🟠",
        "Low": "🟢"
    }

    return colors.get(impact_level, "⚪")


def display_festival_table(events):

    """Display festival records in a readable table."""

    if not events:

        st.info("No festival or holiday records found.")

        return

    festival_df = pd.DataFrame(events)

    display_columns = [
        "event_id",
        "name",
        "event_type",
        "start_date",
        "end_date",
        "state",
        "affected_location",
        "impact_level",
        "description"
    ]

    festival_df = festival_df[display_columns]

    festival_df.columns = [
        "ID",
        "Event Name",
        "Event Type",
        "Start Date",
        "End Date",
        "State",
        "Affected Location",
        "Impact",
        "Description"
    ]

    st.dataframe(
        festival_df,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# 10. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("🚚 Shipment Management")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Register Shipment",
        "GPS Tracking",
        "Weather Monitoring",
        "Festival Monitoring",
        "Disturbance Monitoring",
        "ML Dataset Collection",
        "ML Model Training",
        "ML Prediction Dashboard"
    ]
)


# ==================================================
# GLOBAL ACTIVE SHIPMENT
# Select the shipment once; every shipment-specific
# module uses this shared selection.
# ==================================================

active_shipments_df = get_shipments()

active_shipment = None
active_shipment_id = None

if active_shipments_df is not None and not active_shipments_df.empty:

    active_shipment_lookup = {
        int(row["shipment_id"]): row.to_dict()
        for _, row in active_shipments_df.iterrows()
    }

    active_shipment_ids = list(active_shipment_lookup.keys())

    # Keep the stored selection valid if the shipment list changes.
    if st.session_state.get("global_active_shipment_id") not in active_shipment_ids:
        st.session_state["global_active_shipment_id"] = active_shipment_ids[0]

    st.sidebar.divider()
    st.sidebar.subheader("📦 Active Shipment")

    active_shipment_id = st.sidebar.selectbox(
        "Select active shipment",
        options=active_shipment_ids,
        format_func=lambda shipment_id: (
            f"Shipment {shipment_id} | "
            f"{active_shipment_lookup[shipment_id].get('source', 'Unknown')} "
            f"→ "
            f"{active_shipment_lookup[shipment_id].get('destination', 'Unknown')}"
        ),
        key="global_active_shipment_id"
    )

    active_shipment_id = int(active_shipment_id)
    active_shipment = active_shipment_lookup[active_shipment_id]

    st.sidebar.caption(
        f"All modules will use Shipment {active_shipment_id} "
        "until you change it here."
    )

else:
    st.sidebar.info(
        "Register a shipment to enable the global shipment selector."
    )

# Show the shared shipment context consistently in every module.
if active_shipment is not None:
    st.caption(
        f"📦 Active shipment: **{active_shipment_id}** | "
        f"{active_shipment.get('source', 'Unknown')} → "
        f"{active_shipment.get('destination', 'Unknown')}"
    )


# ==================================================
# 11. DASHBOARD
# ==================================================

if page == "Dashboard":

    st.header("📊 Shipment Dashboard")

    shipments = get_shipments()

    total_shipments = len(shipments)

    total_gps_updates = get_gps_update_count()

    total_festivals = len(get_festival_events())

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Shipments",
            total_shipments
        )

    with col2:

        st.metric(
            "Total GPS Updates",
            total_gps_updates
        )

    with col3:

        st.metric(
            "Festival / Holiday Events",
            total_festivals
        )

    st.divider()

    st.subheader("📦 All Registered Shipments")

    if not shipments.empty:

        st.dataframe(
            shipments,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No shipments found. Register your first shipment."
        )

    st.divider()

    st.subheader("🎉 Upcoming Festivals and Holidays")

    today = date.today()

    upcoming_events = get_festival_events(
        start_date=today.isoformat()
    )

    if upcoming_events:

        for event in upcoming_events[:5]:

            icon = get_festival_impact_color(
                event["impact_level"]
            )

            st.write(
                f"{icon} **{event['name']}** — "
                f"{event['start_date']} to {event['end_date']} "
                f"({event['affected_location']})"
            )

    else:

        st.info("No upcoming events in the festival database.")



    st.divider()

    st.subheader("🚧 Potential Shipment Disturbance Alerts")

    if shipments.empty:

        st.info("Select an active shipment in the sidebar to check its route disturbances.")

    else:

        if st.button(
            "🔍 Check Active Shipment Route",
            use_container_width=True,
            key="dashboard_check_disturbances"
        ):

            dashboard_alerts = []
            route_errors = []
            shipments_checked = 0

            try:
                saved_events = get_disturbance_events(limit=500)
            except Exception as error:
                saved_events = []
                route_errors.append(f"Could not load saved RSS events: {error}")

            eligible_events = [
                event for event in saved_events
                if event.get("route_matchable") is True
                and str(event.get("location_confidence", "")).lower() == "high"
            ]

            with st.spinner(
                "Checking the active shipment route against saved RSS announcements..."
            ):

                for _, shipment_row in shipments[shipments["shipment_id"] == active_shipment_id].iterrows():

                    shipment_id = int(shipment_row["shipment_id"])

                    try:

                        source = (
                            float(shipment_row["source_latitude"]),
                            float(shipment_row["source_longitude"])
                        )

                        destination = (
                            float(shipment_row["destination_latitude"]),
                            float(shipment_row["destination_longitude"])
                        )

                        route = generate_route(source, destination)

                        if not route or not route.get("points"):
                            raise ValueError("The routing service returned no route points.")

                        shipments_checked += 1

                        route_matches = find_shipment_disturbance_matches(
                            route.get("points", []),
                            saved_events,
                            radius_km=DISTURBANCE_MATCH_RADIUS_KM
                        )

                        for match in route_matches:

                            dashboard_alerts.append({
                                "Shipment ID": shipment_id,
                                "Route": (
                                    f"{shipment_row['source']} → "
                                    f"{shipment_row['destination']}"
                                ),
                                "Disturbance": match["title"],
                                "Location": match["place"],
                                "Distance from Route (km)": round(
                                    match["distance_km"], 2
                                ),
                                "Severity": match["severity"],
                                "Match": match["match_label"]
                            })

                    except Exception as error:
                        route_errors.append(f"Shipment {shipment_id}: {error}")

            st.session_state["dashboard_disturbance_alerts"] = dashboard_alerts
            st.session_state["dashboard_disturbance_check_completed"] = True
            st.session_state["dashboard_disturbance_check_summary"] = {
                "shipments_checked": shipments_checked,
                "total_events": len(saved_events),
                "eligible_events": len(eligible_events),
                "route_errors": route_errors
            }

        dashboard_alerts = st.session_state.get(
            "dashboard_disturbance_alerts",
            []
        )
        check_completed = st.session_state.get(
            "dashboard_disturbance_check_completed",
            False
        )
        check_summary = st.session_state.get(
            "dashboard_disturbance_check_summary",
            {}
        )

        if check_completed:

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric(
                "Shipment Routes Checked",
                check_summary.get("shipments_checked", 0)
            )
            metric2.metric(
                "Saved RSS Announcements",
                check_summary.get("total_events", 0)
            )
            metric3.metric(
                "Locations Eligible for Matching",
                check_summary.get("eligible_events", 0)
            )

            if dashboard_alerts:

                st.warning(
                    f"{len(dashboard_alerts)} potential route disturbance "
                    "match(es) found. Review the announcements before dispatch."
                )

                st.dataframe(
                    pd.DataFrame(dashboard_alerts),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "Route checking completed, but no potential geographic "
                    "matches were found within 15 km of the registered routes."
                )

                if check_summary.get("eligible_events", 0) == 0:
                    st.warning(
                        "None of the saved RSS announcements currently has a "
                        "high-confidence, precise settlement location. The "
                        "announcements are still visible in Disturbance Monitoring, "
                        "but broad or ambiguous locations are excluded from route alerts."
                    )

            for message in check_summary.get("route_errors", []):
                st.warning(message)

        else:

            st.info(
                "Click **Check All Shipment Routes** to run the route comparison. "
                "The results and diagnostic counts will appear here."
            )

        st.caption(
            "These alerts indicate geographic proximity only. They do not "
            "confirm a road closure, traffic disruption, or actual shipment impact."
        )
    # ==================================================
# 12. REGISTER SHIPMENT
# ==================================================

elif page == "Register Shipment":

    st.header("📦 Register New Shipment")

    st.write(
        "Enter the source and destination city or address. "
        "The application will automatically find their GPS coordinates."
    )

    with st.form("shipment_registration"):

        source = st.text_input(
            "Source City / Address",
            placeholder="Example: Howrah Railway Station, West Bengal"
        )

        destination = st.text_input(
            "Destination City / Address",
            placeholder="Example: New Delhi Railway Station"
        )

        st.subheader("📅 Delivery Deadline")

        col1, col2 = st.columns(2)

        with col1:

            delivery_date = st.date_input(
                "Delivery Date"
            )

        with col2:

            delivery_time = st.time_input(
                "Delivery Time"
            )

        st.divider()

        st.info(
            "📍 Latitude and longitude will be detected "
            "automatically from the locations you enter. "
            "You do not need to enter GPS coordinates."
        )

        submit = st.form_submit_button(
            "Register Shipment",
            use_container_width=True
        )

        if submit:

            if not source.strip() or not destination.strip():

                st.error(
                    "Please enter both source and destination."
                )

            else:

                deadline = datetime.combine(
                    delivery_date,
                    delivery_time
                )

                if deadline <= datetime.now():

                    st.error(
                        "Delivery deadline must be in the future."
                    )

                else:

                    try:

                        with st.spinner(
                            "Finding source and destination coordinates..."
                        ):

                            source_location = load_geocoded_location(
                                source.strip()
                            )

                            destination_location = load_geocoded_location(
                                destination.strip()
                            )

                        if source_location is None:

                            st.error(
                                f"Could not find the source location: "
                                f"{source}. Please enter a more specific "
                                "city, landmark, or address."
                            )

                        elif destination_location is None:

                            st.error(
                                f"Could not find the destination location: "
                                f"{destination}. Please enter a more specific "
                                "city, landmark, or address."
                            )

                        else:

                            source_lat = float(
                                source_location["latitude"]
                            )

                            source_lon = float(
                                source_location["longitude"]
                            )

                            destination_lat = float(
                                destination_location["latitude"]
                            )

                            destination_lon = float(
                                destination_location["longitude"]
                            )

                            st.subheader(
                                "📍 Automatically Detected Coordinates"
                            )

                            source_col, destination_col = st.columns(2)

                            with source_col:

                                st.write("**Source**")

                                st.write(
                                    source_location["display_name"]
                                )

                                st.caption(
                                    f"Latitude: {source_lat:.6f}"
                                )

                                st.caption(
                                    f"Longitude: {source_lon:.6f}"
                                )

                            with destination_col:

                                st.write("**Destination**")

                                st.write(
                                    destination_location["display_name"]
                                )

                                st.caption(
                                    f"Latitude: {destination_lat:.6f}"
                                )

                                st.caption(
                                    f"Longitude: {destination_lon:.6f}"
                                )

                            distance = calculate_distance_km(
                                (source_lat, source_lon),
                                (destination_lat, destination_lon)
                            )

                            if distance < 0.1:

                                st.error(
                                    "The source and destination appear "
                                    "to be the same location. Please check "
                                    "the addresses."
                                )

                            else:

                                register_shipment(
                                    source.strip(),
                                    destination.strip(),
                                    deadline,
                                    source_lat,
                                    source_lon,
                                    destination_lat,
                                    destination_lon
                                )

                                st.success(
                                    "Shipment registered successfully! "
                                    "GPS coordinates were detected and "
                                    "saved automatically."
                                )

                                st.metric(
                                    "Approximate Straight-Line Distance",
                                    f"{distance:.2f} km"
                                )

                                st.balloons()

                                st.rerun()

                    except requests.RequestException as error:

                        st.error(
                            f"Location service connection error: {error}"
                        )

                    except (KeyError, ValueError, TypeError) as error:

                        st.error(
                            f"Location processing error: {error}"
                        )

                    except sqlite3.Error as error:

                        st.error(
                            f"Database Error: {error}"
                        )

                    except Exception as error:

                        st.error(
                            f"Unexpected registration error: {error}"
                        )


# ==================================================
# 13. GPS TRACKING
# ==================================================

elif page == "GPS Tracking":

    # Refresh only the GPS page every five seconds. The Android phone writes
    # coordinates to SQLite; the laptop's independent session reads those rows.
    if hasattr(st, "fragment"):
        gps_fragment = st.fragment(run_every=5)
    else:
        gps_fragment = lambda function: function

    @gps_fragment
    def render_gps_tracking_page():

        if not hasattr(st, "fragment"):
            st.warning(
                "Automatic five-second dashboard refresh requires a newer "
                "Streamlit version. In your activated virtual environment run: "
                "python -m pip install --upgrade streamlit"
            )


        st.header("📍 Automatic Shipment GPS Tracking")

        st.caption(
            "Live dashboard refresh: every 5 seconds | "
            f"Last refreshed: {datetime.now().astimezone().strftime('%H:%M:%S')}"
        )

        shipment = active_shipment

        if shipment is None:

            st.warning(
                "Please register a shipment and select the active shipment "
                "in the sidebar before GPS tracking."
            )

        else:

            shipment_id = int(active_shipment_id)

            st.info(
                f"Tracking Shipment ID: {shipment_id}"
            )

            st.write(
                f"**Source:** {shipment['source']}"
            )

            st.write(
                f"**Destination:** {shipment['destination']}"
            )

            coordinate_columns = [
                "source_latitude",
                "source_longitude",
                "destination_latitude",
                "destination_longitude"
            ]

            missing_coordinates = any(
                pd.isna(shipment[column])
                for column in coordinate_columns
            )

            if missing_coordinates:

                st.warning(
                    "Route coordinates are missing. "
                    "The application can find them automatically "
                    "using the saved source and destination names."
                )

                if st.button(
                    "📍 Find and Save Coordinates Automatically",
                    key=f"auto_fix_gps_coordinates_{shipment_id}"
                ):

                    try:

                        with st.spinner(
                            "Finding shipment coordinates..."
                        ):

                            source_location = load_geocoded_location(
                                shipment["source"]
                            )

                            destination_location = load_geocoded_location(
                                shipment["destination"]
                            )

                        if (
                            source_location is None
                            or destination_location is None
                        ):

                            st.error(
                                "Could not find one or both shipment locations. "
                                "Check the saved source and destination names."
                            )

                        else:

                            update_shipment_coordinates(
                                shipment_id,
                                source_location["latitude"],
                                source_location["longitude"],
                                destination_location["latitude"],
                                destination_location["longitude"]
                            )

                            st.success(
                                "Coordinates found and saved automatically."
                            )

                            st.rerun()

                    except (sqlite3.Error, ValueError, TypeError) as error:

                        st.error(
                            f"Could not update coordinates: {error}"
                        )

                    except requests.RequestException as error:

                        st.error(
                            f"Location service error: {error}"
                        )

            else:

                source = (
                    float(shipment["source_latitude"]),
                    float(shipment["source_longitude"])
                )

                destination = (
                    float(shipment["destination_latitude"]),
                    float(shipment["destination_longitude"])
                )

                st.subheader("🛣️ Planned Road Route")

                try:

                    with st.spinner(
                        "Generating shipment road route..."
                    ):

                        route = load_cached_route(
                            source,
                            destination
                        )

                    if not route.get("points"):

                        st.error(
                            "No road route was returned. Check the shipment "
                            "coordinates and try again."
                        )

                        st.stop()

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Total Route Distance",
                            f"{route['distance_km']:.2f} km"
                        )

                    with col2:

                        st.metric(
                            "Estimated Driving Time",
                            f"{route['duration_minutes']:.0f} minutes"
                        )

                    # ==============================================
                    # DAY 5 - ALTERNATIVE ROUTE GENERATION
                    # ==============================================

                    st.divider()
                    st.subheader("🧠 Intelligent Alternative Route Planning")

                    st.write(
                        "Generate alternative driving routes between the "
                        "registered shipment locations and compare their "
                        "distance and estimated driving time."
                    )

                    candidate_key = (
                        f"day5_route_candidates_{shipment_id}"
                    )

                    candidate_shipment_key = (
                        "day5_route_candidates_active_shipment"
                    )
                    candidate_signature = (
                        int(shipment_id),
                        tuple(round(float(value), 6) for value in source),
                        tuple(round(float(value), 6) for value in destination),
                    )

                    if st.button(
                        "🔄 Generate Alternative Routes",
                        use_container_width=True,
                        key=f"generate_day5_routes_{shipment_id}"
                    ):

                        try:

                            with st.spinner(
                                "Generating alternative road routes..."
                            ):

                                candidate_routes = generate_route_candidates(
                                    source,
                                    destination,
                                    alternatives=3,
                                    timeout=15
                                )

                            if not candidate_routes:

                                st.warning(
                                    "The routing service did not return "
                                    "any alternative routes."
                                )

                            else:

                                st.session_state[candidate_key] = (
                                    candidate_routes
                                )

                                st.session_state[
                                    candidate_shipment_key
                                ] = candidate_signature

                                st.success(
                                    f"{len(candidate_routes)} distinct road route(s) "
                                    "generated successfully."
                                )

                                if len(candidate_routes) == 1:
                                    st.info(
                                        "OSRM provided only one distinct routable corridor "
                                        "for these endpoints. The system did not invent "
                                        "duplicate routes. Try again later or verify the "
                                        "registered endpoint coordinates if you expected "
                                        "more road choices."
                                    )

                        except Exception as error:

                            st.error(
                                f"Alternative route generation failed: {error}"
                            )

                    saved_candidate_routes = st.session_state.get(
                        candidate_key,
                        []
                    )

                    saved_candidate_shipment = st.session_state.get(
                        candidate_shipment_key
                    )

                    if (
                        saved_candidate_routes
                        and saved_candidate_shipment != candidate_signature
                    ):
                        # Endpoint coordinates changed: discard stale route
                        # geometries and require regeneration for this shipment.
                        st.session_state[candidate_key] = []
                        saved_candidate_routes = []

                    if (
                        saved_candidate_routes
                        and saved_candidate_shipment == candidate_signature
                    ):

                        shortest_route = min(
                            saved_candidate_routes,
                            key=lambda item: item["distance_km"]
                        )

                        fastest_route = min(
                            saved_candidate_routes,
                            key=lambda item: item["duration_minutes"]
                        )

                        route_comparison_rows = []

                        for candidate in saved_candidate_routes:

                            route_comparison_rows.append({

                                "Route": candidate["route_name"],

                                "Generation Method": candidate.get(
                                    "route_method", "OSRM alternative"
                                ),

                                "Distance (km)": round(
                                    candidate["distance_km"],
                                    2
                                ),

                                "Driving Time (minutes)": round(
                                    candidate["duration_minutes"],
                                    1
                                ),

                                "Shortest Route": (
                                    "Yes"
                                    if candidate["route_id"]
                                    == shortest_route["route_id"]
                                    else "No"
                                ),

                                "Fastest Route": (
                                    "Yes"
                                    if candidate["route_id"]
                                    == fastest_route["route_id"]
                                    else "No"
                                )

                            })

                        st.markdown("#### 📊 Route Comparison")

                        st.dataframe(
                            pd.DataFrame(route_comparison_rows),
                            use_container_width=True,
                            hide_index=True
                        )

                        metric_col1, metric_col2 = st.columns(2)

                        with metric_col1:

                            st.metric(
                                "Shortest Available Route",
                                f"{shortest_route['distance_km']:.2f} km"
                            )

                        with metric_col2:

                            st.metric(
                                "Fastest Available Route",
                                f"{fastest_route['duration_minutes']:.1f} minutes"
                            )

                        st.caption(
                            "Routes and travel times are supplied by OSRM. "
                            "Via-point corridor routes are real road-network routes "
                            "forced through a nearby snapped waypoint. Driving times "
                            "are estimates, not live traffic predictions."
                        )

                        # Compare the actual route geometries on one map.
                        st.markdown("#### 🗺️ Alternative Route Map")
                        route_palette = ["blue", "purple", "darkgreen", "orange", "cadetblue"]
                        comparison_map = folium.Map(
                            location=source,
                            zoom_start=6,
                            control_scale=True
                        )
                        all_route_points = []
                        for route_index, candidate in enumerate(saved_candidate_routes):
                            route_points = candidate.get("points") or []
                            if len(route_points) < 2:
                                continue
                            all_route_points.extend(route_points)
                            route_layer = folium.FeatureGroup(
                                name=(
                                    f"{candidate['route_name']} - "
                                    f"{candidate['distance_km']:.1f} km / "
                                    f"{candidate['duration_minutes']:.0f} min"
                                ),
                                show=True
                            )
                            folium.PolyLine(
                                route_points,
                                color=route_palette[route_index % len(route_palette)],
                                weight=5,
                                opacity=0.82,
                                tooltip=(
                                    f"{candidate['route_name']} | "
                                    f"{candidate['distance_km']:.1f} km | "
                                    f"{candidate['duration_minutes']:.0f} min | "
                                    f"{candidate.get('route_method', 'OSRM route')}"
                                )
                            ).add_to(route_layer)
                            route_layer.add_to(comparison_map)

                        folium.Marker(
                            source,
                            tooltip="Shipment source",
                            icon=folium.Icon(color="green", icon="play")
                        ).add_to(comparison_map)
                        folium.Marker(
                            destination,
                            tooltip="Shipment destination",
                            icon=folium.Icon(color="red", icon="flag")
                        ).add_to(comparison_map)

                        if all_route_points:
                            comparison_map.fit_bounds(all_route_points)

                        folium.LayerControl(collapsed=False).add_to(comparison_map)

                        st_folium(
                            comparison_map,
                            width=None,
                            height=470,
                            key=f"alternative_route_map_{shipment_id}"
                        )

                        # ==============================================
                        # DAY 5 - STEP 7 PART 3
                        # ROUTE CONDITION COLLECTION AND RISK EVALUATION
                        # ==============================================

                        st.divider()
                        st.subheader("🌦️ Disturbance-Aware Route Evaluation")

                        st.write(
                            "Evaluate each alternative route using its own "
                            "weather forecast, saved festival calendar and "
                            "geocoded RSS disturbance records."
                        )

                        st.caption(
                            "Weather is sampled at up to three locations per route. "
                            "Festival matches use a 30 km radius and RSS matches "
                            "use a 15 km radius. Nearby events are potential signals, "
                            "not confirmed road closures."
                        )

                        evaluate_key = (
                            f"day5_evaluate_conditions_{shipment_id}"
                        )

                        evaluated_key = (
                            f"day5_evaluated_routes_{shipment_id}"
                        )

                        st.caption(
                            "Click once to start the evaluation. A progress spinner will remain visible "
                            "while weather, festival and RSS conditions are checked for each route."
                        )

                        if st.button(
                            "🧠 Evaluate Route Conditions and Risk",
                            use_container_width=True,
                            key=evaluate_key
                        ):

                            try:

                                with st.spinner(
                                    "Evaluating weather, festival and RSS conditions "
                                    "for every alternative route. This may take a little time "
                                    "while location matches are checked..."
                                ):


                                    evaluated_conditions = {}
                                    route_condition_messages = {}

                                    # ----------------------------------
                                    # LOAD SAVED EVENT DATA
                                    # ----------------------------------

                                    festival_lookup_ok = False
                                    rss_lookup_ok = False

                                    festival_events = []
                                    rss_events = []

                                    try:
                                        festival_events = (
                                            get_festival_events() or []
                                        )
                                        festival_lookup_ok = True

                                    except Exception as error:
                                        route_condition_messages[
                                            "Festival calendar"
                                        ] = str(error)

                                    try:
                                        rss_events = (
                                            get_disturbance_events(limit=1000) or []
                                        )
                                        rss_lookup_ok = True

                                    except Exception as error:
                                        route_condition_messages[
                                            "RSS records"
                                        ] = str(error)

                                    # ----------------------------------
                                    # EVALUATE EACH ALTERNATIVE ROUTE
                                    # ----------------------------------

                                    for candidate in saved_candidate_routes:

                                        route_id = candidate["route_id"]

                                        route_points = (
                                            candidate.get("points") or []
                                        )

                                        route_errors = []

                                        if not route_points:
                                            route_condition_messages[
                                                route_id
                                            ] = "Route geometry is unavailable."
                                            continue

                                        # Use three representative locations.
                                        sample_locations = (
                                            select_route_weather_locations(
                                                {
                                                    "points": route_points,
                                                    "checkpoints": []
                                                },
                                                source,
                                                destination,
                                                max_points=3
                                            )
                                        )

                                        if not sample_locations:
                                            route_errors.append(
                                                "No intermediate weather locations "
                                                "could be selected."
                                            )

                                        # ----------------------------------
                                        # WEATHER COLLECTION
                                        # ----------------------------------

                                        route_forecasts = []

                                        for sample_number, point in enumerate(
                                            sample_locations,
                                            start=1
                                        ):

                                            try:

                                                forecast = load_weather(
                                                    point[0],
                                                    point[1]
                                                )

                                                if not forecast:
                                                    raise ValueError(
                                                        "Weather service returned no forecast."
                                                    )

                                                route_forecasts.append({
                                                    "sample": sample_number,
                                                    "coordinates": point,
                                                    "forecast": forecast
                                                })

                                            except Exception as error:

                                                route_errors.append(
                                                    f"Weather point {sample_number}: {error}"
                                                )

                                        weather_values = []

                                        if (
                                            route_forecasts
                                            and len(route_forecasts)
                                            == len(sample_locations)
                                        ):

                                            forecast_dates_by_point = [
                                                [
                                                    str(day.get("date", ""))[:10]
                                                    for day in item["forecast"]
                                                ]
                                                for item in route_forecasts
                                            ]

                                            common_dates = (
                                                forecast_dates_by_point[0]
                                                if forecast_dates_by_point
                                                else []
                                            )

                                            if any(
                                                dates != common_dates
                                                for dates in forecast_dates_by_point
                                            ):
                                                route_errors.append(
                                                    "Weather forecast dates do not "
                                                    "match across route locations."
                                                )

                                            elif not common_dates:
                                                route_errors.append(
                                                    "Weather forecast dates are unavailable."
                                                )

                                            else:

                                                # Evaluate the first forecast day.
                                                evaluation_date = common_dates[0]

                                                for item in route_forecasts:

                                                    matching_days = [
                                                        day
                                                        for day in item["forecast"]
                                                        if str(
                                                            day.get("date", "")
                                                        )[:10] == evaluation_date
                                                    ]

                                                    if not matching_days:
                                                        route_errors.append(
                                                            "A route location is missing "
                                                            "the selected forecast date."
                                                        )
                                                        continue

                                                    weather_values.append(
                                                        matching_days[0]
                                                    )

                                        else:
                                            evaluation_date = None

                                        # ----------------------------------
                                        # AGGREGATE WEATHER METRICS
                                        # ----------------------------------

                                        weather_complete = (
                                            bool(weather_values)
                                            and len(weather_values)
                                            == len(sample_locations)
                                            and not any(
                                                "Weather" in error
                                                for error in route_errors
                                            )
                                        )

                                        temperature = None
                                        rainfall = None
                                        wind_speed = None
                                        severe_weather = None
                                        weather_risk = None

                                        if weather_complete:

                                            temperature = max(
                                                float(day["temperature_max"])
                                                for day in weather_values
                                            )

                                            rainfall = max(
                                                float(day["rainfall"])
                                                for day in weather_values
                                            )

                                            wind_speed = max(
                                                float(day["wind_speed"])
                                                for day in weather_values
                                            )

                                            severe_weather = int(
                                                any(
                                                    str(
                                                        day.get("severity", "")
                                                    ).lower() == "severe"
                                                    for day in weather_values
                                                )
                                            )

                                            # Operational screening score,
                                            # not a calibrated probability.
                                            weather_risk = 0

                                            if severe_weather:
                                                weather_risk = 70

                                            if rainfall >= 50:
                                                weather_risk = max(
                                                    weather_risk, 70
                                                )
                                            elif rainfall >= 20:
                                                weather_risk = max(
                                                    weather_risk, 50
                                                )
                                            elif rainfall >= 5:
                                                weather_risk = max(
                                                    weather_risk, 25
                                                )

                                            if wind_speed >= 60:
                                                weather_risk = max(
                                                    weather_risk, 55
                                                )
                                            elif wind_speed >= 40:
                                                weather_risk = max(
                                                    weather_risk, 25
                                                )

                                        # ----------------------------------
                                        # FESTIVAL / HOLIDAY MATCHING
                                        # ----------------------------------

                                        festival_nearby = None
                                        festival_risk = None
                                        route_festival_matches = []

                                        if (
                                            festival_lookup_ok
                                            and evaluation_date
                                        ):

                                            festival_nearby = 0
                                            festival_risk = 0

                                            for event in festival_events:

                                                event_start = str(
                                                    event.get("start_date") or ""
                                                )[:10]

                                                event_end = str(
                                                    event.get("end_date")
                                                    or event_start
                                                )[:10]

                                                if (
                                                    not event_start
                                                    or not event_end
                                                    or not (
                                                        event_start
                                                        <= evaluation_date
                                                        <= event_end
                                                    )
                                                ):
                                                    continue

                                                locality = str(
                                                    event.get("affected_location")
                                                    or ""
                                                ).strip()

                                                state_name = str(
                                                    event.get("state") or ""
                                                ).strip()

                                                if not locality or locality.lower() in {
                                                    "india",
                                                    "all india",
                                                    "statewide",
                                                    "unknown",
                                                    "location not identified"
                                                }:
                                                    continue

                                                search_name = ", ".join(
                                                    part
                                                    for part in [
                                                        locality,
                                                        state_name,
                                                        "India"
                                                    ]
                                                    if part
                                                )

                                                try:

                                                    event_coordinates = (
                                                        load_geocoded_location(
                                                            search_name
                                                        )
                                                    )

                                                    if not event_coordinates:
                                                        continue

                                                    distance = (
                                                        distance_from_location_to_route_km(
                                                            event_coordinates,
                                                            route_points
                                                        )
                                                    )

                                                except Exception:
                                                    continue

                                                if (
                                                    distance is None
                                                    or distance > 30.0
                                                ):
                                                    continue

                                                festival_nearby = 1

                                                impact = str(
                                                    event.get("impact_level")
                                                    or "Unknown"
                                                ).lower()

                                                if (
                                                    "high" in impact
                                                    or "severe" in impact
                                                ):
                                                    event_risk = 60
                                                elif (
                                                    "medium" in impact
                                                    or "moderate" in impact
                                                ):
                                                    event_risk = 35
                                                elif "low" in impact:
                                                    event_risk = 15
                                                else:
                                                    event_risk = 25

                                                festival_risk = max(
                                                    festival_risk,
                                                    event_risk
                                                )

                                                route_festival_matches.append({
                                                    "name": event.get("name"),
                                                    "location": locality,
                                                    "distance_km": round(
                                                        float(distance), 1
                                                    ),
                                                    "impact": event.get(
                                                        "impact_level",
                                                        "Unknown"
                                                    )
                                                })

                                        # ----------------------------------
                                        # RSS DISTURBANCE MATCHING
                                        # ----------------------------------

                                        rss_disturbance = None
                                        strike_risk = None
                                        road_disruption_risk = None
                                        route_rss_matches = []

                                        if (
                                            rss_lookup_ok
                                            and evaluation_date
                                        ):

                                            all_route_rss_matches = (
                                                find_shipment_disturbance_matches(
                                                    route_points,
                                                    rss_events,
                                                    radius_km=15.0
                                                )
                                            )

                                            unknown_date_matches = [
                                                event
                                                for event in all_route_rss_matches
                                                if not str(
                                                    event.get("event_date") or ""
                                                )[:10]
                                            ]

                                            # RSS articles often do not state an
                                            # explicit event date. An undated nearby
                                            # article is still shown as an unverified
                                            # signal, but it must not invalidate the
                                            # entire route evaluation. Only dated
                                            # matches for the forecast evaluation day
                                            # contribute to the date-specific risk score.
                                            route_rss_matches = [
                                                event
                                                for event in all_route_rss_matches
                                                if str(
                                                    event.get("event_date") or ""
                                                )[:10] == evaluation_date
                                            ]

                                            rss_disturbance = int(
                                                bool(route_rss_matches)
                                            )

                                            strike_risk = 0
                                            road_disruption_risk = 0

                                            for event in route_rss_matches:

                                                severity = str(
                                                    event.get("severity")
                                                    or "Unknown"
                                                ).lower()

                                                if (
                                                    "severe" in severity
                                                    or "critical" in severity
                                                ):
                                                    event_risk = 75
                                                elif "high" in severity:
                                                    event_risk = 60
                                                elif (
                                                    "moderate" in severity
                                                    or "medium" in severity
                                                ):
                                                    event_risk = 35
                                                elif "low" in severity:
                                                    event_risk = 15
                                                else:
                                                    event_risk = 25

                                                road_disruption_risk = max(
                                                    road_disruption_risk,
                                                    event_risk
                                                )

                                                event_type = str(
                                                    event.get("event_type")
                                                    or ""
                                                ).lower()

                                                if any(
                                                    word in event_type
                                                    for word in [
                                                        "strike",
                                                        "protest",
                                                        "labour",
                                                        "labor",
                                                        "union"
                                                    ]
                                                ):
                                                    strike_risk = max(
                                                        strike_risk,
                                                        event_risk
                                                    )
                                        # ----------------------------------
                                        # SAVE ONLY COMPLETE CONDITIONS
                                        # ----------------------------------

                                        if (
                                            weather_complete
                                            and festival_nearby is not None
                                            and rss_disturbance is not None
                                            and not route_errors
                                        ):

                                            evaluated_conditions[route_id] = {

                                                "temperature": temperature,
                                                "rainfall": rainfall,
                                                "wind_speed": wind_speed,
                                                "severe_weather": severe_weather,
                                                "festival_nearby": festival_nearby,
                                                "rss_disturbance": rss_disturbance,

                                                "weather_risk": weather_risk,
                                                "festival_risk": festival_risk,
                                                "strike_risk": strike_risk,
                                                "road_disruption_risk": (
                                                    road_disruption_risk
                                                ),

                                                "evaluation_date": evaluation_date,
                                                "festival_matches": (
                                                    route_festival_matches
                                                ),
                                                "rss_matches": route_rss_matches,
                                                "undated_rss_matches": unknown_date_matches if (
                                                    rss_lookup_ok and evaluation_date
                                                ) else []
                                            }

                                        else:

                                            route_condition_messages[
                                                route_id
                                            ] = (
                                                "; ".join(route_errors)
                                                if route_errors
                                                else "One or more condition sources "
                                                "could not be checked."
                                            )

                                    # ----------------------------------
                                    # INTEGRATE CONDITIONS AND RANK ROUTES
                                    # ----------------------------------

                                    if not evaluated_conditions:
                                        st.session_state[evaluated_key] = {
                                            "routes": [],
                                            "ranked": [],
                                            "conditions": {},
                                            "messages": route_condition_messages
                                        }

                                        st.error(
                                            "No alternative route has complete condition data yet. "
                                            "Review the route-specific reasons below. "
                                            "Undated RSS reports are retained as unverified signals "
                                            "and no longer block route evaluation."
                                        )

                                        for failed_route, failure_reason in route_condition_messages.items():
                                            st.warning(
                                                f"{failed_route}: {failure_reason}"
                                            )

                                        st.stop()

                                    enriched_routes = integrate_route_conditions(
                                        saved_candidate_routes,
                                        evaluated_conditions,
                                        use_verified_ml=True
                                    )

                                    ranked_routes = rank_routes_intelligently(
                                        enriched_routes,
                                        risk_weight=0.40,
                                        time_weight=0.35,
                                        distance_weight=0.25,
                                        delay_weight=0.0
                                    )

                                    st.session_state[evaluated_key] = {
                                        "routes": enriched_routes,
                                        "ranked": ranked_routes,
                                        "conditions": evaluated_conditions,
                                        "messages": route_condition_messages
                                    }

                                    st.success(
                                        "Route-condition evaluation completed."
                                    )

                            except Exception as error:

                                st.error(
                                    "Route-condition evaluation failed: "
                                    f"{error}"
                                )

                        # ----------------------------------
                        # DISPLAY ROUTE RISK RESULTS
                        # ----------------------------------

                        saved_evaluation = st.session_state.get(
                            evaluated_key
                        )

                        if saved_evaluation:

                            st.markdown("#### 📊 Disturbance-Aware Route Ranking")

                            ranked = saved_evaluation["ranked"]
                            enriched = saved_evaluation["routes"]
                            conditions = saved_evaluation["conditions"]

                            if ranked:

                                ranking_rows = []

                                for route_item in ranked:

                                    condition = conditions.get(
                                        route_item["route_id"],
                                        {}
                                    )

                                    ranking_rows.append({

                                        "Rank": route_item.get("rank"),

                                        "Route": route_item["route_name"],

                                        "Distance (km)": round(
                                            route_item["distance_km"], 2
                                        ),

                                        "Driving time (min)": round(
                                            route_item["duration_minutes"], 1
                                        ),

                                        "Risk score": route_item.get(
                                            "risk_score"
                                        ),

                                        "Risk level": route_item.get(
                                            "risk_level"
                                        ),

                                        "Weather risk": condition.get(
                                            "weather_risk"
                                        ),

                                        "Festival risk": condition.get(
                                            "festival_risk"
                                        ),

                                        "Strike risk": condition.get(
                                            "strike_risk"
                                        ),

                                        "Road disruption risk": condition.get(
                                            "road_disruption_risk"
                                        ),

                                        "Verified ML used": (
                                            "Yes"
                                            if route_item.get("ml_model_used")
                                            else "No"
                                        )

                                    })

                                st.dataframe(
                                    pd.DataFrame(ranking_rows),
                                    use_container_width=True,
                                    hide_index=True
                                )

                                recommended = ranked[0]

                                st.success(
                                    f"Recommended route: "
                                    f"{recommended['route_name']} "
                                    f"— risk score "
                                    f"{recommended['risk_score']:.2f}/100."
                                )

                                st.caption(
                                    "Recommendation balances the available "
                                    "risk score (40%), estimated driving time "
                                    "(35%) and distance (25%). The score is an "
                                    "operational ranking measure, not a calibrated "
                                    "probability of disruption."
                                )

                            else:

                                st.warning(
                                    "No route has complete enough condition "
                                    "information for a risk-based recommendation. "
                                    "Review the missing source information below."
                                )

                            if saved_evaluation["messages"]:

                                st.markdown("#### ⚠️ Incomplete Route Conditions")

                                for route_name, message in (
                                    saved_evaluation["messages"].items()
                                ):

                                    st.warning(
                                        f"{route_name}: {message}"
                                    )

                            st.caption(
                                "A festival match indicates a nearby calendar "
                                "event, not confirmed traffic impact. RSS reports "
                                "are unverified. No match does not prove that "
                                "a road is clear. Verified ML is used only when "
                                "the verified-data model is trained; synthetic "
                                "models are not used for this route recommendation."
                            )

                            # ==============================================
                            # DAY 5 - STEP 8: PLANNED DELIVERY DELAY
                            # AND REVISED ETA (DOES NOT REQUIRE GPS)
                            # ==============================================

                            st.divider()
                            st.subheader("⏱️ Planned Delivery Delay and Revised ETA")
                            st.write(
                                "Estimate the delivery time using the recommended route and its "
                                "evaluated weather, festival and disturbance conditions. This "
                                "planning estimate is available even before the first GPS update."
                            )

                            planned_ranked_routes = saved_evaluation.get("ranked", [])
                            planned_conditions = saved_evaluation.get("conditions", {})

                            if not planned_ranked_routes:
                                st.warning(
                                    "A planned ETA cannot be calculated until at least one route "
                                    "has complete condition data."
                                )
                            else:
                                planned_route = planned_ranked_routes[0]
                                planned_route_id = planned_route.get("route_id")
                                planned_route_conditions = planned_conditions.get(
                                    planned_route_id, {}
                                )

                                try:
                                    planned_duration = float(
                                        planned_route.get("duration_minutes", 0)
                                    )
                                    planned_weather_risk = float(
                                        planned_route_conditions.get("weather_risk", 0)
                                    )
                                    planned_rainfall = float(
                                        planned_route_conditions.get("rainfall", 0)
                                    )
                                    planned_wind = float(
                                        planned_route_conditions.get("wind_speed", 0)
                                    )
                                    planned_festival_risk = float(
                                        planned_route_conditions.get("festival_risk", 0)
                                    )
                                    planned_strike_risk = float(
                                        planned_route_conditions.get("strike_risk", 0)
                                    )
                                    planned_road_risk = float(
                                        planned_route_conditions.get(
                                            "road_disruption_risk", 0
                                        )
                                    )
                                    planned_severe_weather = bool(
                                        planned_route_conditions.get("severe_weather", 0)
                                    )

                                    if planned_severe_weather:
                                        planned_weather_severity = "Severe"
                                    elif planned_weather_risk >= 70:
                                        planned_weather_severity = "High"
                                    elif planned_weather_risk >= 25:
                                        planned_weather_severity = "Moderate"
                                    else:
                                        planned_weather_severity = "Normal"

                                    planned_delay = predict_additional_delay(
                                        base_duration_minutes=planned_duration,
                                        weather_severity=planned_weather_severity,
                                        rainfall_mm=planned_rainfall,
                                        wind_speed_kmh=planned_wind,
                                        disturbance_risk=0,
                                        festival_risk=planned_festival_risk,
                                        strike_risk=planned_strike_risk,
                                        road_disruption_risk=planned_road_risk,
                                        historical_delay_minutes=0.0
                                    )

                                    planned_departure = datetime.now().astimezone()
                                    planned_deadline = None
                                    planned_deadline_value = shipment.get(
                                        "delivery_deadline"
                                    )

                                    if (
                                        planned_deadline_value is not None
                                        and not pd.isna(planned_deadline_value)
                                    ):
                                        try:
                                            planned_deadline = datetime.fromisoformat(
                                                str(planned_deadline_value).replace(
                                                    "Z", "+00:00"
                                                )
                                            )
                                            if planned_deadline.tzinfo is None:
                                                planned_deadline = planned_deadline.replace(
                                                    tzinfo=planned_departure.tzinfo
                                                )
                                            else:
                                                planned_deadline = planned_deadline.astimezone(
                                                    planned_departure.tzinfo
                                                )
                                        except (TypeError, ValueError):
                                            st.warning(
                                                "The registered delivery deadline could not be "
                                                "parsed. Check the shipment's deadline value."
                                            )
                                            planned_deadline = None

                                    planned_eta = calculate_revised_eta(
                                        departure_time=planned_departure,
                                        base_duration_minutes=planned_duration,
                                        additional_delay_minutes=planned_delay[
                                            "additional_delay_minutes"
                                        ],
                                        delivery_deadline=planned_deadline
                                    )

                                    planned_eta_datetime = datetime.fromisoformat(
                                        planned_eta["revised_eta"]
                                    )
                                    planned_status = planned_eta["delivery_status"]
                                    planned_minutes_to_deadline = None

                                    if planned_deadline is not None:
                                        planned_minutes_to_deadline = (
                                            planned_deadline - planned_eta_datetime
                                        ).total_seconds() / 60
                                        if planned_minutes_to_deadline < 0:
                                            planned_status = "Delayed"
                                        elif (
                                            planned_minutes_to_deadline
                                            <= DEADLINE_AT_RISK_BUFFER_MINUTES
                                        ):
                                            planned_status = "At Risk"
                                        else:
                                            planned_status = "On Time"

                                    planned_col1, planned_col2, planned_col3 = st.columns(3)
                                    with planned_col1:
                                        st.metric(
                                            "Recommended Route",
                                            planned_route.get("route_name", "Route")
                                        )
                                    with planned_col2:
                                        st.metric(
                                            "Estimated Additional Delay",
                                            f"{planned_delay['additional_delay_minutes']:.1f} min"
                                        )
                                    with planned_col3:
                                        st.metric(
                                            "Revised Journey Duration",
                                            f"{planned_eta['revised_duration_minutes']:.1f} min"
                                        )

                                    planned_eta_col1, planned_eta_col2 = st.columns(2)
                                    with planned_eta_col1:
                                        st.write(
                                            "**ETA without additional delay:** "
                                            + datetime.fromisoformat(
                                                planned_eta["original_eta"]
                                            ).astimezone().strftime("%d %b %Y, %I:%M %p")
                                        )
                                    with planned_eta_col2:
                                        st.write(
                                            "**Revised delivery ETA:** "
                                            + planned_eta_datetime.astimezone().strftime(
                                                "%d %b %Y, %I:%M %p"
                                            )
                                        )

                                    if planned_status == "On Time":
                                        st.success("Expected delivery status: On Time")
                                    elif planned_status == "At Risk":
                                        st.warning(
                                            "Expected delivery status: At Risk — the revised ETA "
                                            f"is within {DEADLINE_AT_RISK_BUFFER_MINUTES} minutes "
                                            "of the registered deadline."
                                        )
                                    elif planned_status == "Delayed":
                                        st.error("Expected delivery status: Delayed")
                                    else:
                                        st.info("No delivery deadline is registered for this shipment.")

                                    st.markdown("##### Delay estimate breakdown")
                                    delay_breakdown = pd.DataFrame([
                                        {
                                            "Factor": "Weather",
                                            "Estimated delay (min)": planned_delay[
                                                "weather_delay_minutes"
                                            ]
                                        },
                                        {
                                            "Factor": "Festival / crowd signal",
                                            "Estimated delay (min)": planned_delay[
                                                "festival_delay_minutes"
                                            ]
                                        },
                                        {
                                            "Factor": "Strike signal",
                                            "Estimated delay (min)": planned_delay[
                                                "strike_delay_minutes"
                                            ]
                                        },
                                        {
                                            "Factor": "Road disruption signal",
                                            "Estimated delay (min)": planned_delay[
                                                "road_disruption_delay_minutes"
                                            ]
                                        }
                                    ])
                                    st.dataframe(
                                        delay_breakdown,
                                        use_container_width=True,
                                        hide_index=True
                                    )

                                    st.caption(
                                        "This is a route-planning estimate from OSRM duration and "
                                        "rule-based weather / event factors, not a live-traffic ETA. "
                                        "Nearby festivals and RSS announcements are unverified signals. "
                                        "The current Random Forest predicts disruption probability, "
                                        "not delay minutes, so it is not converted into a fabricated "
                                        "ML delay value. GPS-aware remaining-time prediction appears "
                                        "in the section below when a fresh phone GPS fix is available."
                                    )

                                except Exception as planned_eta_error:
                                    st.error(
                                        "Could not calculate the planned revised ETA: "
                                        f"{planned_eta_error}"
                                    )

                            # ==============================================
                            # DAY 5 - STEP 9
                            # GPS-AWARE SHIPMENT DELAY AND ETA MONITORING
                            # ==============================================

                            st.divider()
                            st.subheader("⏱️ GPS-Aware Shipment Delay Prediction")

                            st.write(
                                "Automatically estimate the remaining journey time and revised "
                                "delivery ETA using the latest valid GPS position."
                            )

                            ranked_routes = saved_evaluation.get("ranked", [])

                            conditions_by_route = saved_evaluation.get("conditions", {})

                            delay_result_key = f"day5_delay_result_{shipment_id}"

                            if not ranked_routes:

                                st.warning(
                                    "No recommended route is available for GPS-aware delay prediction."
                                )

                                st.session_state.pop(delay_result_key, None)

                            else:

                                # Use the highest-ranked route from the disturbance-aware evaluation.
                                selected_route = ranked_routes[0]

                                route_id = selected_route["route_id"]

                                route_conditions = conditions_by_route.get(route_id)

                                if not route_conditions:

                                    st.warning(
                                        "Condition data is unavailable for the recommended route."
                                    )

                                    st.session_state.pop(delay_result_key, None)

                                else:

                                    # ------------------------------------------
                                    # GET THE LATEST VALID GPS POSITION
                                    # ------------------------------------------

                                    gps_progress = calculate_gps_aware_route_progress(
                                        shipment_id,
                                        selected_route
                                    )

                                    if not gps_progress["available"]:

                                        st.warning(
                                            "GPS-aware ETA is currently unavailable: "
                                            + gps_progress["reason"]
                                        )

                                        st.info(
                                            "Start or continue GPS tracking on the Android phone "
                                            "carried with the shipment. The app requires a fresh "
                                            "GPS record before calculating the remaining journey ETA."
                                        )

                                        st.session_state.pop(delay_result_key, None)

                                    else:

                                        # ------------------------------------------
                                        # REMAINING JOURNEY DETAILS
                                        # ------------------------------------------

                                        remaining_distance = float(
                                            gps_progress["remaining_distance_km"]
                                        )

                                        remaining_minutes = float(
                                            gps_progress["remaining_minutes"]
                                        )

                                        departure_time = datetime.now().astimezone()

                                        # ------------------------------------------
                                        # GET ROUTE WEATHER AND DISTURBANCE CONDITIONS
                                        # ------------------------------------------

                                        weather_risk = float(
                                            route_conditions.get("weather_risk", 0)
                                        )

                                        rainfall = float(
                                            route_conditions.get("rainfall", 0)
                                        )

                                        wind_speed = float(
                                            route_conditions.get("wind_speed", 0)
                                        )

                                        severe_weather = int(
                                            route_conditions.get("severe_weather", 0)
                                        )

                                        if severe_weather:
                                            weather_severity = "Severe"
                                        elif weather_risk >= 70:
                                            weather_severity = "High"
                                        elif weather_risk >= 25:
                                            weather_severity = "Moderate"
                                        else:
                                            weather_severity = "Normal"

                                        festival_risk = float(
                                            route_conditions.get("festival_risk", 0)
                                        )

                                        strike_risk = float(
                                            route_conditions.get("strike_risk", 0)
                                        )

                                        road_disruption_risk = float(
                                            route_conditions.get("road_disruption_risk", 0)
                                        )

                                        # The current ML model predicts disturbance risk,
                                        # not historical delay minutes.
                                        historical_delay_minutes = 0.0

                                        # ------------------------------------------
                                        # PREDICT DELAY FOR THE REMAINING JOURNEY
                                        # ------------------------------------------

                                        if remaining_minutes <= 0:

                                            delay_prediction = {
                                                "base_duration_minutes": 0.0,
                                                "weather_delay_minutes": 0.0,
                                                "disturbance_delay_minutes": 0.0,
                                                "festival_delay_minutes": 0.0,
                                                "strike_delay_minutes": 0.0,
                                                "road_disruption_delay_minutes": 0.0,
                                                "historical_ml_delay_minutes": 0.0,
                                                "additional_delay_minutes": 0.0,
                                                "revised_duration_minutes": 0.0,
                                                "weather_severity": weather_severity,
                                                "festival_risk": festival_risk,
                                                "strike_risk": strike_risk,
                                                "road_disruption_risk": road_disruption_risk,
                                                "prediction_method": "Completed journey"
                                            }

                                        else:

                                            delay_prediction = predict_additional_delay(
                                                base_duration_minutes=float(
                                                    remaining_minutes
                                                ),
                                                weather_severity=weather_severity,
                                                rainfall_mm=rainfall,
                                                wind_speed_kmh=wind_speed,

                                                # Pass each disturbance type separately.
                                                disturbance_risk=0,
                                                festival_risk=festival_risk,
                                                strike_risk=strike_risk,
                                                road_disruption_risk=road_disruption_risk,

                                                historical_delay_minutes=historical_delay_minutes
                                            )

                                        additional_delay = float(
                                            delay_prediction["additional_delay_minutes"]
                                        )

                                        # ------------------------------------------
                                        # PARSE THE REGISTERED DELIVERY DEADLINE
                                        # ------------------------------------------

                                        deadline_value = shipment.get("delivery_deadline")

                                        delivery_deadline = None

                                        if (
                                            deadline_value is not None
                                            and not pd.isna(deadline_value)
                                        ):

                                            try:

                                                delivery_deadline = datetime.fromisoformat(
                                                    str(deadline_value).replace("Z", "+00:00")
                                                )

                                                if delivery_deadline.tzinfo is None:

                                                    delivery_deadline = delivery_deadline.replace(
                                                        tzinfo=departure_time.tzinfo
                                                    )

                                                else:

                                                    delivery_deadline = delivery_deadline.astimezone(
                                                        departure_time.tzinfo
                                                    )

                                            except (TypeError, ValueError):

                                                st.warning(
                                                    "The registered delivery deadline could not "
                                                    "be interpreted. Please check the shipment record."
                                                )

                                                delivery_deadline = None

                                        # ------------------------------------------
                                        # CALCULATE GPS-AWARE REVISED ETA
                                        # ------------------------------------------

                                        eta_prediction = calculate_revised_eta(
                                            departure_time=departure_time,
                                            base_duration_minutes=remaining_minutes,
                                            additional_delay_minutes=additional_delay,
                                            delivery_deadline=delivery_deadline
                                        )

                                        # ------------------------------------------
                                        # CLASSIFY DEADLINE STATUS
                                        # ------------------------------------------

                                        if delivery_deadline is None:

                                            deadline_status = "Deadline not provided"

                                        else:

                                            revised_eta_datetime = (
                                                departure_time
                                                + timedelta(
                                                    minutes=remaining_minutes + additional_delay
                                                )
                                            )

                                            minutes_until_deadline = (
                                                delivery_deadline - revised_eta_datetime
                                            ).total_seconds() / 60

                                            if minutes_until_deadline < 0:

                                                deadline_status = "Delayed"

                                            elif (
                                                minutes_until_deadline
                                                <= DEADLINE_AT_RISK_BUFFER_MINUTES
                                            ):

                                                deadline_status = "At Risk"

                                            else:

                                                deadline_status = "On Time"

                                        eta_prediction["delivery_status"] = deadline_status

                                        # ------------------------------------------
                                        # SAVE THE LATEST GPS-AWARE RESULT
                                        # ------------------------------------------

                                        st.session_state[delay_result_key] = {
                                            "route": selected_route,
                                            "conditions": route_conditions,
                                            "gps_progress": gps_progress,
                                            "delay": delay_prediction,
                                            "eta": eta_prediction,
                                            "calculated_at": departure_time.isoformat()
                                        }

                            # ==============================================
                            # DISPLAY GPS-AWARE DELAY PREDICTION
                            # ==============================================

                            delay_result = st.session_state.get(delay_result_key)

                            if delay_result:

                                selected_route = delay_result["route"]
                                gps_progress = delay_result["gps_progress"]
                                delay_prediction = delay_result["delay"]
                                eta_prediction = delay_result["eta"]
                                route_conditions = delay_result["conditions"]

                                st.markdown("#### 📍 Current Shipment Progress")

                                progress_col1, progress_col2, progress_col3 = st.columns(3)

                                with progress_col1:

                                    st.metric(
                                        "Remaining Distance",
                                        f"{gps_progress['remaining_distance_km']:.2f} km"
                                    )

                                with progress_col2:

                                    st.metric(
                                        "Remaining Driving Time",
                                        f"{gps_progress['remaining_minutes']:.1f} min"
                                    )

                                with progress_col3:

                                    st.metric(
                                        "Route Progress",
                                        f"{gps_progress['progress_percent']:.1f}%"
                                    )

                                st.caption(
                                    f"Latest GPS accuracy: {gps_progress['accuracy']:.1f} metres | "
                                    f"GPS age: {gps_progress['age_minutes']:.1f} minutes | "
                                    f"Distance from route: "
                                    f"{gps_progress['distance_from_route_km']:.2f} km"
                                )

                                st.write(
                                    f"**GPS recorded at:** "
                                    f"{gps_progress['recorded_at'].astimezone().isoformat()}"
                                )

                                st.write(
                                    f"**Evaluated route:** {selected_route['route_name']}"
                                )

                                st.markdown("#### 📊 Remaining Journey Delay Results")

                                delay_col1, delay_col2, delay_col3 = st.columns(3)

                                with delay_col1:

                                    st.metric(
                                        "Remaining Driving Time",
                                        f"{delay_prediction['base_duration_minutes']:.1f} min"
                                    )

                                with delay_col2:

                                    st.metric(
                                        "Predicted Additional Delay",
                                        f"{delay_prediction['additional_delay_minutes']:.1f} min"
                                    )

                                with delay_col3:

                                    st.metric(
                                        "Revised Remaining Journey",
                                        f"{delay_prediction['revised_duration_minutes']:.1f} min"
                                    )

                                st.markdown("#### 🌦️ Conditions Used")

                                st.write(
                                    f"Weather severity: {route_conditions['weather_risk']}/100"
                                )

                                st.write(
                                    f"Rainfall: {route_conditions['rainfall']:.1f} mm"
                                )

                                st.write(
                                    f"Wind speed: {route_conditions['wind_speed']:.1f} km/h"
                                )

                                st.write(
                                    f"Festival risk: {route_conditions['festival_risk']}/100"
                                )

                                st.write(
                                    f"Strike risk: {route_conditions['strike_risk']}/100"
                                )

                                st.write(
                                    f"Road disruption risk: "
                                    f"{route_conditions['road_disruption_risk']}/100"
                                )

                                st.markdown("#### ⏱️ Delay Estimate Breakdown")

                                st.write(
                                    f"Weather-related estimate: "
                                    f"{delay_prediction.get('weather_delay_minutes', 0.0):.1f} minutes"
                                )

                                st.write(
                                    f"Potential festival-related estimate: "
                                    f"{delay_prediction.get('festival_delay_minutes', 0.0):.1f} minutes"
                                )

                                st.write(
                                    f"Strike-related estimate: "
                                    f"{delay_prediction.get('strike_delay_minutes', 0.0):.1f} minutes"
                                )

                                st.write(
                                    f"Road-disruption estimate: "
                                    f"{delay_prediction.get('road_disruption_delay_minutes', 0.0):.1f} minutes"
                                )

                                st.caption(
                                    "These are rule-based planning estimates. A nearby festival or "
                                    "unverified announcement does not establish that a road is delayed "
                                    "or closed. Actual traffic conditions may differ."
                                )

                                st.markdown("#### 🕒 GPS-Aware Revised ETA")

                                original_eta = eta_prediction.get("original_eta")
                                revised_eta = eta_prediction.get("revised_eta")

                                if original_eta:

                                    st.write(
                                        f"**ETA without additional delay:** {original_eta}"
                                    )

                                if revised_eta:

                                    st.write(
                                        f"**GPS-aware revised ETA:** {revised_eta}"
                                    )

                                deadline_status = eta_prediction.get(
                                    "delivery_status",
                                    "Not available"
                                )

                                if deadline_status == "On Time":

                                    st.success("Delivery status: On Time")

                                elif deadline_status == "At Risk":

                                    st.warning(
                                        "Delivery status: At Risk — the predicted ETA is within "
                                        f"{DEADLINE_AT_RISK_BUFFER_MINUTES} minutes of the deadline."
                                    )

                                elif deadline_status == "Delayed":

                                    st.error("Delivery status: Delayed")

                                else:

                                    st.info(f"Delivery status: {deadline_status}")

                                st.caption(
                                    "The GPS-aware ETA uses the latest valid GPS position and the "
                                    "remaining fraction of the planned OSRM route. Remaining driving "
                                    "time is estimated proportionally from the original OSRM duration. "
                                    "The delay estimate applies route conditions to the remaining "
                                    "journey; it is not a live traffic prediction. The At Risk threshold "
                                    f"is {DEADLINE_AT_RISK_BUFFER_MINUTES} minutes."
                                )

                                st.caption(
                                    f"Last recalculated: {delay_result['calculated_at']}"
                                )

                    # ----------------------------------------------
                    # DAY 3 STEP 4 - POTENTIAL ROUTE DISTURBANCES
                    # ----------------------------------------------
                    st.subheader("🚧 Potential Route Disturbances")
                    st.caption(
                        "RSS-based geographic matches are unverified signals. "
                        "A nearby event does not prove that the road is blocked."
                    )

                    try:
                        saved_disturbances = get_disturbance_events(limit=500)
                        route_disturbance_matches = find_shipment_disturbance_matches(
                            route.get("points", []),
                            saved_disturbances,
                            radius_km=DISTURBANCE_MATCH_RADIUS_KM
                        )
                    except Exception as match_error:
                        route_disturbance_matches = []
                        st.warning(f"Could not check route disturbances: {match_error}")

                    if route_disturbance_matches:
                        st.warning(
                            f"{len(route_disturbance_matches)} potential disturbance "
                            "match(es) found within 15 km of this route. Verify before dispatch."
                        )
                        for match in route_disturbance_matches:
                            with st.expander(
                                f"⚠️ {match['title']} | {match['place']} | "
                                f"{match['distance_km']:.1f} km from route"
                            ):
                                st.write(f"**Match:** {match['match_label']} — Potential Match, Unverified")
                                st.write(f"**Event type:** {match['event_type']}")
                                st.write(f"**Severity from RSS classifier:** {match['severity']}")
                                st.write(f"**Event date:** {match.get('event_date') or 'Not identified'}")
                                st.write(f"**Publisher:** {match.get('source_name') or 'Unknown'}")
                                if match.get("source_url", "").startswith(("http://", "https://")):
                                    st.markdown(f"[Read original announcement]({match['source_url']})")
                    else:
                        st.info(
                            "No geocoded RSS disturbance locations were matched "
                            "within 15 km of this route. This does not guarantee "
                            "that the route is disruption-free."
                        )
                        with st.expander("🔎 Why were no RSS route matches found?"):
                            match_diagnostics = st.session_state.get(
                                "last_route_disturbance_diagnostics", {}
                            )
                            if match_diagnostics:
                                st.json(match_diagnostics)
                                if match_diagnostics.get("events_checked", 0) == 0:
                                    st.warning(
                                        "No saved RSS events were supplied to the route matcher. "
                                        "Open Disturbance Monitoring and fetch the RSS feeds first."
                                    )
                                elif match_diagnostics.get("events_rejected_by_confidence", 0):
                                    st.warning(
                                        "Some saved events did not contain a recognized, specific "
                                        "city/locality in the article title or event-related summary."
                                    )
                                elif match_diagnostics.get("geocoding_attempts", 0) == 0:
                                    st.warning(
                                        "No specific locality could be extracted from the saved RSS events."
                                    )
                                elif match_diagnostics.get("geocoding_successes", 0) == 0:
                                    st.warning(
                                        "Specific locations were extracted, but geocoding returned no "
                                        "usable locality. Check internet connectivity and Nominatim access."
                                    )
                                else:
                                    st.caption(
                                        "Localities were geocoded, but none fell within the 15 km route corridor."
                                    )
                            else:
                                st.caption(
                                    "No route-matching diagnostics are available yet. Refresh the GPS Tracking page."
                                )

                    st.divider()

                    st.subheader("📡 Automatic GPS Tracking")

                    st.caption(
                        "Start tracking to allow your browser to "
                        "send updated GPS coordinates automatically."
                    )

                    tracking_state_key = (
                        f"gps_tracking_enabled_{shipment_id}"
                    )

                    last_saved_key = (
                        f"last_saved_gps_timestamp_{shipment_id}"
                    )

                    last_processed_key = (
                        f"last_processed_gps_timestamp_{shipment_id}"
                    )

                    if tracking_state_key not in st.session_state:
                        st.session_state[tracking_state_key] = False

                    if last_saved_key not in st.session_state:
                        st.session_state[last_saved_key] = None

                    if last_processed_key not in st.session_state:
                        st.session_state[last_processed_key] = None

                    button_col1, button_col2 = st.columns(2)

                    with button_col1:

                        if st.button(
                            "▶️ Start GPS Tracking",
                            use_container_width=True,
                            disabled=st.session_state[tracking_state_key]
                        ):

                            st.session_state[tracking_state_key] = True

                            st.rerun()

                    with button_col2:

                        if st.button(
                            "⏹️ Stop GPS Tracking",
                            use_container_width=True,
                            disabled=not st.session_state[tracking_state_key]
                        ):

                            st.session_state[tracking_state_key] = False

                            st.rerun()

                    tracking_enabled = st.session_state[
                        tracking_state_key
                    ]

                    if tracking_enabled:

                        st.success(
                            "Automatic GPS tracking is ON in this browser session."
                        )

                        st.caption(
                            "Keep this browser tab open and allow "
                            "location permission."
                        )

                    else:

                        st.warning(
                            "Automatic GPS tracking is OFF in this browser session."
                        )
                        st.info(
                            "If your Android phone is the device sending GPS, this "
                            "OFF message on the laptop is expected. The laptop reads "
                            "the phone's saved GPS records from the shared database "
                            "and refreshes this page automatically every 5 seconds."
                        )

                    st.info(
                        "ANDROID PHONE SETUP: Open this same shipment dashboard on the phone "
                        "that travels with the vehicle, select the same Shipment ID, and press "
                        "Start GPS Tracking on the phone. The phone and laptop must connect to "
                        "the same running Streamlit app/database. Browser GPS requires a secure "
                        "HTTPS address (or localhost); a plain http://192.168.x.x address is "
                        "blocked by Android Chrome. If the phone shows a secure-context error, "
                        "open the app through an HTTPS deployment or HTTPS tunnel."
                    )

                    st.caption(
                        "Live dashboard refresh: every 5 seconds. Keep the phone page open, "
                        "screen awake, location services enabled, and Chrome location permission "
                        "set to Allow. The laptop reads the saved GPS records automatically."
                    )

                    location = automatic_gps_tracker(
                        enabled=tracking_enabled,
                        key=f"automatic_gps_{shipment_id}"
                    )

                    if tracking_enabled and isinstance(location, dict):

                        if location.get("status") == "error":

                            st.warning(
                                location.get(
                                    "message",
                                    "Unable to retrieve GPS location."
                                )
                            )

                        elif location.get("status") == "success":

                            try:

                                current_lat = float(
                                    location["latitude"]
                                )

                                current_lon = float(
                                    location["longitude"]
                                )

                                raw_accuracy = location.get("accuracy")

                                gps_timestamp = location.get("timestamp")

                                coordinates_valid = (
                                    math.isfinite(current_lat)
                                    and math.isfinite(current_lon)
                                    and -90 <= current_lat <= 90
                                    and -180 <= current_lon <= 180
                                )

                                if not coordinates_valid:

                                    st.error(
                                        "Invalid GPS coordinates received."
                                    )

                                elif gps_timestamp is None:

                                    st.warning(
                                        "GPS update received without a timestamp. "
                                        "Waiting for the next update."
                                    )

                                elif (
                                    gps_timestamp
                                    != st.session_state[last_processed_key]
                                ):

                                    try:

                                        accuracy = float(raw_accuracy)

                                    except (TypeError, ValueError):

                                        accuracy = None

                                    if (
                                        accuracy is None
                                        or not math.isfinite(accuracy)
                                        or accuracy < 0
                                        or accuracy > MAX_GPS_ACCURACY
                                    ):

                                        st.warning(
                                            "GPS reading rejected. Accuracy is "
                                            + (
                                                f"{accuracy:.1f} metres. "
                                                if accuracy is not None
                                                and math.isfinite(accuracy)
                                                else "unavailable. "
                                            )
                                            + "Accuracy must be 200 metres "
                                            "or better. Waiting for a better "
                                            "GPS reading."
                                        )

                                        st.session_state[last_processed_key] = (
                                            gps_timestamp
                                        )

                                    else:

                                        # The phone may be in a different city
                                        # while we test the connection. Do not
                                        # save that device position as this
                                        # shipment's truck location.
                                        phone_coordinates = (
                                            current_lat,
                                            current_lon
                                        )
                                        _, nearest_route_index = calculate_remaining_route(
                                            phone_coordinates,
                                            route["points"]
                                        )
                                        distance_from_route_km = (
                                            distance_from_location_to_route_km(
                                                phone_coordinates,
                                                route["points"]
                                            )
                                        )
                                        if distance_from_route_km is None:
                                            distance_from_route_km = float("inf")

                                        st.session_state[last_processed_key] = (
                                            gps_timestamp
                                        )

                                        if distance_from_route_km > MAX_GPS_DISTANCE_FROM_ROUTE_KM:

                                            st.info(
                                                "Phone GPS connection is working. "
                                                f"Accuracy: {accuracy:.1f} metres. "
                                                f"This reading is {distance_from_route_km:.1f} km "
                                                "from the selected shipment route, so it is "
                                                "being treated as a test/device location and "
                                                "was NOT saved as the truck's location. "
                                                "Take the phone with the vehicle to record "
                                                "actual shipment tracking."
                                            )

                                        else:

                                            movement_valid, movement_message = validate_gps_movement(
                                                shipment_id,
                                                current_lat,
                                                current_lon,
                                                route["points"]
                                            )

                                            if not movement_valid:

                                                st.warning(movement_message)

                                            else:

                                                saved = save_gps_location(
                                                    shipment_id,
                                                    current_lat,
                                                    current_lon,
                                                    accuracy
                                                )

                                                if saved:

                                                    st.session_state[last_saved_key] = gps_timestamp

                                                    st.success(
                                                        "Accurate GPS location passed route and movement "
                                                        "validation and was saved!"
                                                    )

                                                else:

                                                    st.warning(
                                                        f"GPS reading could not be saved. "
                                                        f"Accuracy is {accuracy:.1f} metres; "
                                                        "check the shipment record and database."
                                                    )

                            except (TypeError, ValueError, KeyError) as error:

                                st.warning(
                                    f"Invalid GPS data received: {error}"
                                )

                            except sqlite3.Error as error:

                                st.error(
                                    f"GPS Database Error: {error}"
                                )

                    latest = get_latest_location(
                        shipment_id
                    )

                    history = get_tracking_history(
                        shipment_id
                    )

                    # A saved record is not automatically a live position.
                    # Require acceptable accuracy, freshness, and proximity to
                    # the selected route before using it for progress, ETA or map marker.
                    gps_progress_result = calculate_gps_aware_route_progress(
                        shipment_id,
                        route
                    )
                    latest_is_on_route = bool(
                        gps_progress_result.get("available", False)
                    )
                    latest_distance_from_route_km = (
                        gps_progress_result.get("distance_from_route_km")
                        if latest_is_on_route else None
                    )

                    st.divider()

                    st.subheader("📊 Live Shipment Status")

                    if latest:

                        current_location = (
                            float(latest[0]),
                            float(latest[1])
                        )

                        remaining_distance, nearest_index = (
                            calculate_remaining_route(
                                current_location,
                                route["points"]
                            )
                        )

                        if latest_is_on_route:
                            # Reuse the validated, freshness-checked calculation.
                            remaining_distance = gps_progress_result[
                                "remaining_distance_km"
                            ]
                            remaining_minutes = gps_progress_result[
                                "remaining_minutes"
                            ]
                            progress = gps_progress_result[
                                "progress_percent"
                            ]
                            total_distance = float(route["distance_km"])

                            eta_hours = int(
                                remaining_minutes // 60
                            )

                            eta_minutes = int(
                                remaining_minutes % 60
                            )

                            metric1, metric2, metric3 = st.columns(3)

                            with metric1:

                                st.metric(
                                    "Remaining Distance",
                                    f"{remaining_distance:.2f} km"
                                )

                            with metric2:

                                st.metric(
                                    "Estimated Remaining Time",
                                    f"{eta_hours} hr {eta_minutes} min"
                                )

                            with metric3:

                                st.metric(
                                    "Approximate Route Progress",
                                    f"{progress:.1f}%"
                                )

                                st.progress(
                                    progress / 100
                                )

                        st.caption(
                            "Remaining distance and ETA are approximate. "
                            "They use the planned route and do not account "
                            "for live traffic or automatic route changes."
                        )

                        st.write(
                            f"**Latest Latitude:** {current_location[0]:.6f}"
                        )

                        st.write(
                            f"**Latest Longitude:** {current_location[1]:.6f}"
                        )

                        if latest[2] is not None:

                            latest_accuracy = float(latest[2])

                            st.write(
                                f"**GPS Accuracy:** "
                                f"{latest_accuracy:.1f} metres"
                            )

                            if latest_accuracy > MAX_GPS_ACCURACY:

                                st.warning(
                                    "The latest historical GPS record has "
                                    "accuracy worse than 100 metres. "
                                    "It was recorded before the new filtering "
                                    "was applied."
                                )

                            else:

                                st.success(
                                    "The latest saved GPS record meets "
                                    "the 100-metre accuracy requirement."
                                )

                        st.write(
                            f"**Last Recorded:** {latest[3]}"
                        )

                        if not latest_is_on_route:
                            reason = gps_progress_result.get(
                                "reason",
                                "The latest GPS record is not suitable for live tracking."
                            )
                            st.warning(
                                f"GPS record is saved, but it is NOT being treated "
                                f"as the live truck position. {reason} "
                                "Remaining distance, ETA, route progress and the "
                                "current-position map marker are hidden until a "
                                "fresh, accurate GPS fix is received near this route."
                            )
                        else:
                            st.success(
                                "Latest GPS location is close to the selected "
                                "shipment route and can be used for approximate "
                                "remaining-distance and ETA calculations."
                            )

                    else:

                        st.info(
                            "No GPS location available yet. "
                            "Start GPS tracking to receive the first location."
                        )

                    st.divider()

                    st.subheader("🗺️ Live Shipment Map")

                    if latest and latest_is_on_route:

                        map_center = (
                            float(latest[0]),
                            float(latest[1])
                        )

                    else:

                        # Keep the map focused on the shipment route when the
                        # phone is being tested from a different city.
                        map_center = source

                    shipment_map = folium.Map(
                        location=map_center,
                        zoom_start=6,
                        control_scale=True
                    )

                    folium.Marker(
                        location=source,
                        popup=f"Source: {shipment['source']}",
                        tooltip="Shipment Source",
                        icon=folium.Icon(
                            color="green",
                            icon="play"
                        )
                    ).add_to(shipment_map)

                    folium.Marker(
                        location=destination,
                        popup=f"Destination: {shipment['destination']}",
                        tooltip="Shipment Destination",
                        icon=folium.Icon(
                            color="red",
                            icon="flag"
                        )
                    ).add_to(shipment_map)

                    folium.PolyLine(
                        route["points"],
                        color="blue",
                        weight=5,
                        opacity=0.8,
                        tooltip="Planned Shipment Road Route"
                    ).add_to(shipment_map)

                    # ----------------------------------------------
                    # POTENTIAL DISTURBANCE MARKERS ON ROUTE MAP
                    # ----------------------------------------------
                    disturbance_layer = folium.FeatureGroup(
                        name="Potential Disturbances (Unverified)",
                        show=True
                    )

                    disturbance_map_points = []

                    for match in route_disturbance_matches:

                        try:
                            latitude = float(match["latitude"])
                            longitude = float(match["longitude"])
                            distance_km = float(match["distance_km"])
                        except (KeyError, TypeError, ValueError):
                            continue

                        # Escape all announcement text before inserting it into HTML.
                        place = escape(str(match.get("place") or "Unknown location"))
                        title = escape(str(match.get("title") or "Untitled announcement"))
                        event_type = escape(str(match.get("event_type") or "Unknown"))
                        severity = escape(str(match.get("severity") or "Unknown"))
                        event_date = escape(str(match.get("event_date") or "Not identified"))
                        publisher = escape(str(match.get("source_name") or "Unknown"))
                        status = escape(str(match.get("status") or "Unverified"))

                        source_url = str(match.get("source_url") or "").strip()
                        source_link_html = ""
                        if source_url.startswith(("https://", "http://")):
                            safe_url = escape(source_url, quote=True)
                            source_link_html = (
                                f'<br><a href="{safe_url}" target="_blank" '
                                f'rel="noopener noreferrer">Read original announcement</a>'
                            )

                        popup_html = f"""
                        <div style="font-family:Arial,sans-serif;min-width:240px;max-width:340px;line-height:1.5;">
                            <div style="background:#b91c1c;color:#fff;padding:7px;border-radius:5px;font-weight:bold;margin-bottom:8px;">
                                &#9888; POTENTIAL DISTURBANCE - UNVERIFIED
                            </div>
                            <b>{title}</b>
                            <hr style="margin:6px 0;">
                            <b>Location:</b> {place}<br>
                            <b>Event type:</b> {event_type}<br>
                            <b>RSS severity:</b> {severity}<br>
                            <b>Record status:</b> {status}<br>
                            <b>Event date:</b> {event_date}<br>
                            <b>Publisher:</b> {publisher}<br>
                            <b>Distance from route:</b> {distance_km:.1f} km
                            {source_link_html}
                            <hr style="margin:6px 0;">
                            <div style="background:#fff7ed;color:#9a3412;padding:8px;border-radius:5px;font-size:12px;">
                                This marker represents a geocoded locality reference point,
                                not necessarily the exact incident location. It does not
                                confirm a road closure or traffic disruption.
                            </div>
                        </div>
                        """

                        marker_label = f"""
                        <div style="background:#b91c1c;color:white;border:2px solid white;
                                    border-radius:6px;padding:5px 8px;font-size:12px;
                                    font-weight:bold;white-space:nowrap;
                                    box-shadow:0 2px 6px rgba(0,0,0,0.35);">
                            &#9888; {place} - UNVERIFIED
                        </div>
                        """

                        folium.Marker(
                            location=(latitude, longitude),
                            popup=folium.Popup(popup_html, max_width=380),
                            tooltip=f"Potential disturbance (unverified): {place}",
                            icon=folium.DivIcon(
                                html=marker_label,
                                icon_size=(250, 42),
                                icon_anchor=(5, 21)
                            )
                        ).add_to(disturbance_layer)

                        disturbance_map_points.append((latitude, longitude))

                    if route_disturbance_matches:
                        disturbance_layer.add_to(shipment_map)

                    for index, point in enumerate(
                        route["checkpoints"],
                        start=1
                    ):

                        checkpoint_name = load_location_name(
                            point[0],
                            point[1]
                        )

                        display_name = (
                            checkpoint_name
                            or "Nearby place name unavailable"
                        )

                        popup_text = (
                            f"<b>Checkpoint {index}</b><br>"
                            f"{display_name}<br>"
                            f"Latitude: {point[0]:.6f}<br>"
                            f"Longitude: {point[1]:.6f}"
                        )

                        folium.Marker(
                            location=point,
                            popup=folium.Popup(
                                popup_text,
                                max_width=350
                            ),
                            tooltip=(
                                f"Checkpoint {index}: {display_name}"
                            ),
                            icon=folium.Icon(
                                color="orange",
                                icon="info-sign"
                            )
                        ).add_to(shipment_map)

                    if latest and latest_is_on_route:

                        folium.Marker(
                            location=(
                                float(latest[0]),
                                float(latest[1])
                            ),
                            popup=(
                                f"Current Truck Location<br>"
                                f"Latitude: {latest[0]}<br>"
                                f"Longitude: {latest[1]}<br>"
                                f"Accuracy: {latest[2]} metres<br>"
                                f"Recorded: {latest[3]}"
                            ),
                            tooltip="🚚 Current Truck Location",
                            icon=folium.Icon(
                                color="purple",
                                icon="truck",
                                prefix="fa"
                            )
                        ).add_to(shipment_map)

                    # Plot only recent historical fixes that are near this route.
                    # Old test readings from another city must not stretch the map.
                    history_points = []
                    for history_row in history[-200:]:
                        history_coordinate = (
                            float(history_row[0]),
                            float(history_row[1])
                        )
                        history_route_distance = distance_from_location_to_route_km(
                            history_coordinate,
                            route["points"]
                        )
                        if history_route_distance is None:
                            continue
                        if history_route_distance <= MAX_GPS_DISTANCE_FROM_ROUTE_KM:
                            history_points.append(history_coordinate)

                    if len(history_points) > 1:
                        folium.PolyLine(
                            history_points,
                            color="purple",
                            weight=3,
                            opacity=0.7,
                            tooltip="GPS Tracking History (near selected route)"
                        ).add_to(shipment_map)

                    # ----------------------------------------------
                    # FIT MAP TO ROUTE, GPS AND DISTURBANCE MARKERS
                    # ----------------------------------------------
                    map_bounds = list(route["points"])

                    if latest and latest_is_on_route:
                        map_bounds.append(
                            (
                                float(latest[0]),
                                float(latest[1])
                            )
                        )

                    # Include disturbance reference points so markers are not
                    # accidentally outside the initial map viewport.
                    map_bounds.extend(disturbance_map_points)

                    if map_bounds:
                        shipment_map.fit_bounds(
                            map_bounds,
                            padding=(40, 40)
                        )

                    # Toggle the disturbance layer from the map controls.
                    if route_disturbance_matches:
                        folium.LayerControl(collapsed=False).add_to(shipment_map)

                    st_folium(
                        shipment_map,
                        width=None,
                        height=600,
                        key=f"shipment_map_{shipment_id}"
                    )

                    st.caption(
                        "Map legend: Green = Source | Red = Destination | "
                        "Blue = Planned route | Orange = Intermediate checkpoints | "
                        "Purple = Current truck location (shown only when the GPS "
                        f"fix is fresh, accurate and within {MAX_GPS_DISTANCE_FOR_ETA_KM:.0f} km of this route) | "
                        "Red warning label = Potential unverified disturbance."
                    )

                    if latest and not latest_is_on_route:
                        st.caption(
                            "The latest saved GPS record is stale, inaccurate, or "
                            "too far from this route. It is intentionally not shown "
                            "as the current truck position."
                        )

                    if route_disturbance_matches:
                        st.warning(
                            "Disturbance markers show approximate geocoded locality "
                            "positions. They do not identify the exact incident site "
                            "or prove that a road is blocked. Verify announcements "
                            "with the publisher or official authorities before dispatch."
                        )

                    st.subheader("📍 Intermediate Checkpoints")

                    checkpoint_data = []

                    for index, point in enumerate(
                        route["checkpoints"],
                        start=1
                    ):

                        checkpoint_name = load_location_name(
                            point[0],
                            point[1]
                        )

                        checkpoint_data.append({
                            "Checkpoint": index,
                            "Nearby Location / Road": (
                                checkpoint_name or "Name unavailable"
                            ),
                            "Latitude": round(point[0], 6),
                            "Longitude": round(point[1], 6)
                        })

                    st.dataframe(
                        pd.DataFrame(checkpoint_data),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader("📋 GPS Tracking History")

                    if history:

                        history_df = pd.DataFrame(
                            history,
                            columns=[
                                "Latitude",
                                "Longitude",
                                "Accuracy (metres)",
                                "Recorded At"
                            ]
                        )

                        st.dataframe(
                            history_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.success(
                            f"Total GPS records: {len(history)}"
                        )

                    else:

                        st.info(
                            "No GPS tracking history available yet."
                        )

                except (RuntimeError, ValueError, KeyError, TypeError) as error:

                    st.error(
                        f"Route or Tracking Error: {error}"
                    )

                except requests.RequestException as error:

                    st.error(
                        f"Network Error: {error}"
                    )

                except sqlite3.Error as error:

                    st.error(
                        f"Database Error: {error}"
                    )


    # ==================================================
    # 14. DAY 3 - WEATHER MONITORING
    # ==================================================

    render_gps_tracking_page()


elif page == "Weather Monitoring":

    st.header("🌦️ Real-Time Weather Monitoring")

    st.write(
        "Seven-day weather forecasts for the shipment source, destination, "
        "and selected intermediate points along its planned OSRM road route."
    )

    shipment = active_shipment

    if shipment is None:

        st.warning(
            "Please register a shipment and select the active shipment "
            "in the sidebar before weather monitoring."
        )

    else:

        shipment_id = int(active_shipment_id)

        st.info(
            f"Weather monitoring for Shipment #{shipment_id}"
        )

        coordinate_columns = [
            "source_latitude",
            "source_longitude",
            "destination_latitude",
            "destination_longitude"
        ]

        missing_coordinates = any(
            pd.isna(shipment[column])
            for column in coordinate_columns
        )

        if missing_coordinates:

            st.error(
                "Shipment coordinates are missing. "
                "Use GPS Tracking to find and save them automatically."
            )

        else:

            source_coordinates = (
                float(shipment["source_latitude"]),
                float(shipment["source_longitude"])
            )

            destination_coordinates = (
                float(shipment["destination_latitude"]),
                float(shipment["destination_longitude"])
            )

            straight_line_distance = calculate_distance_km(
                source_coordinates,
                destination_coordinates
            )

            if (
                straight_line_distance < 25
                and shipment["source"].strip().lower()
                != shipment["destination"].strip().lower()
            ):

                st.warning(
                    f"Source and destination coordinates are only "
                    f"{straight_line_distance:.1f} km apart. If these are "
                    "different cities or states, please verify the saved "
                    "locations."
                )

            with st.expander(
                "🛠️ Re-detect shipment coordinates automatically",
                expanded=False
            ):

                st.write(
                    "If the saved coordinates are incorrect, the application "
                    "can look them up again using the shipment's saved "
                    "source and destination names."
                )

                if st.button(
                    "Find Coordinates Again",
                    key=f"regeocode_weather_{shipment_id}"
                ):

                    try:

                        with st.spinner(
                            "Looking up source and destination..."
                        ):

                            corrected_source = load_geocoded_location(
                                shipment["source"]
                            )

                            corrected_destination = load_geocoded_location(
                                shipment["destination"]
                            )

                        if (
                            corrected_source is None
                            or corrected_destination is None
                        ):

                            st.error(
                                "Could not find one or both locations. "
                                "Check the saved shipment names."
                            )

                        else:

                            update_shipment_coordinates(
                                shipment_id,
                                corrected_source["latitude"],
                                corrected_source["longitude"],
                                corrected_destination["latitude"],
                                corrected_destination["longitude"]
                            )

                            load_weather.clear()

                            st.success(
                                "Coordinates were detected and updated "
                                "automatically."
                            )

                            st.rerun()

                    except (
                        sqlite3.Error,
                        ValueError,
                        TypeError
                    ) as error:

                        st.error(
                            f"Could not update coordinates: {error}"
                        )

                    except requests.RequestException as error:

                        st.error(
                            f"Location service error: {error}"
                        )

            try:

                with st.spinner(
                    "Retrieving seven-day weather forecasts..."
                ):

                    source_weather = load_weather(
                        source_coordinates[0],
                        source_coordinates[1]
                    )

                    destination_weather = load_weather(
                        destination_coordinates[0],
                        destination_coordinates[1]
                    )

                    route = generate_route(
                        source_coordinates,
                        destination_coordinates
                    )

                    if not route.get("points"):

                        raise ValueError(
                            "The road-routing service returned no route points. "
                            "Check the shipment locations and try again."
                        )

                    route_weather_locations = (
                        select_route_weather_locations(
                            route,
                            source_coordinates,
                            destination_coordinates,
                            max_points=5
                        )
                    )

                    route_weather_results = []

                    for checkpoint_number, checkpoint in enumerate(
                        route_weather_locations,
                        start=1
                    ):

                        checkpoint_forecast = load_weather(
                            checkpoint[0],
                            checkpoint[1]
                        )

                        severe_days = [
                            day for day in checkpoint_forecast
                            if day.get("severity") == "Severe"
                        ]

                        checkpoint_alerts = sorted({
                            alert
                            for day in severe_days
                            for alert in day.get("alerts", [])
                        })

                        checkpoint_place_name = load_location_name(
                            checkpoint[0],
                            checkpoint[1]
                        )

                        route_weather_results.append({

                            "name": f"Route Point {checkpoint_number}",

                            "place_name": (
                                checkpoint_place_name
                                or "Nearby place name unavailable"
                            ),

                            "coordinates": checkpoint,

                            "forecast": checkpoint_forecast,

                            "severe_days": severe_days,

                            "alerts": checkpoint_alerts
                        })

                st.success(
                    "Seven-day weather forecasts retrieved for the shipment "
                    "endpoints and selected route locations!"
                )

                st.divider()

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("📍 Source Location")

                    st.write(shipment["source"])

                    st.caption(
                        f"Latitude: {source_coordinates[0]:.6f}"
                    )

                    st.caption(
                        f"Longitude: {source_coordinates[1]:.6f}"
                    )

                with col2:

                    st.subheader("📍 Destination Location")

                    st.write(shipment["destination"])

                    st.caption(
                        f"Latitude: {destination_coordinates[0]:.6f}"
                    )

                    st.caption(
                        f"Longitude: {destination_coordinates[1]:.6f}"
                    )

                def display_weather_table(
                    weather_data,
                    location_name
                ):

                    st.subheader(
                        f"🌤️ Seven-Day Forecast - {location_name}"
                    )

                    weather_rows = []

                    for day in weather_data:

                        weather_rows.append({

                            "Date": day["date"],

                            "Min Temperature (°C)": (
                                day["temperature_min"]
                            ),

                            "Max Temperature (°C)": (
                                day["temperature_max"]
                            ),

                            "Rainfall (mm)": day["rainfall"],

                            "Wind Speed (km/h)": (
                                day["wind_speed"]
                            ),

                            "Weather": day["condition"],

                            "Severity": day["severity"]
                        })

                    weather_df = pd.DataFrame(
                        weather_rows
                    )

                    st.dataframe(
                        weather_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    severe_days = [
                        day for day in weather_data
                        if day["severity"] == "Severe"
                    ]

                    if severe_days:

                        st.error(
                            f"⚠️ {len(severe_days)} potentially severe "
                            "weather day(s) detected!"
                        )

                        for day in severe_days:

                            alert_text = ", ".join(
                                day.get("alerts", [])
                            )

                            st.warning(
                                f"{day['date']}: {alert_text}"
                            )

                    else:

                        st.success(
                            "No severe weather detected using the "
                            "configured thresholds."
                        )

                st.divider()

                display_weather_table(
                    source_weather,
                    shipment["source"]
                )

                st.divider()

                display_weather_table(
                    destination_weather,
                    shipment["destination"]
                )

                st.divider()

                st.subheader(
                    "🛣️ Weather Along the Complete Shipment Route"
                )

                st.caption(
                    "The application checks up to five representative "
                    "intermediate locations from the OSRM road route. "
                    "Each location has a seven-day forecast. These are "
                    "sampled route locations, not a weather reading at "
                    "every road coordinate."
                )

                route_affected_count = sum(
                    bool(item["severe_days"])
                    for item in route_weather_results
                )

                if route_weather_results:

                    route_rows = []

                    for item in route_weather_results:

                        forecast = item["forecast"]

                        route_rows.append({

                            "Route Location": item["name"],

                            "Nearby Location / Road": (
                                item["place_name"]
                            ),

                            "Latitude": round(
                                item["coordinates"][0],
                                6
                            ),

                            "Longitude": round(
                                item["coordinates"][1],
                                6
                            ),

                            "Severe Forecast Days": len(
                                item["severe_days"]
                            ),

                            "Maximum Rainfall (mm)": max(
                                (
                                    day["rainfall"]
                                    for day in forecast
                                ),
                                default=0
                            ),

                            "Maximum Wind (km/h)": max(
                                (
                                    day["wind_speed"]
                                    for day in forecast
                                ),
                                default=0
                            ),

                            "Alerts": (
                                "; ".join(item["alerts"])
                                or "None"
                            )
                        })

                    st.dataframe(
                        pd.DataFrame(route_rows),
                        use_container_width=True,
                        hide_index=True
                    )

                    for item in route_weather_results:

                        with st.expander(
                            f"{item['name']} – {item['place_name']} "
                            f"({item['coordinates'][0]:.4f}, "
                            f"{item['coordinates'][1]:.4f})"
                        ):

                            detail_rows = [

                                {

                                    "Date": day["date"],

                                    "Min Temperature (°C)": (
                                        day["temperature_min"]
                                    ),

                                    "Max Temperature (°C)": (
                                        day["temperature_max"]
                                    ),

                                    "Rainfall (mm)": day["rainfall"],

                                    "Wind Speed (km/h)": (
                                        day["wind_speed"]
                                    ),

                                    "Weather": day["condition"],

                                    "Severity": day["severity"],

                                    "Alerts": (
                                        ", ".join(
                                            day.get("alerts", [])
                                        )
                                        or "None"
                                    )
                                }

                                for day in item["forecast"]
                            ]

                            st.dataframe(
                                pd.DataFrame(detail_rows),
                                use_container_width=True,
                                hide_index=True
                            )

                            if item["severe_days"]:

                                st.error(
                                    "Potential severe weather detected at "
                                    f"{item['name']}. Review the dates and "
                                    "alerts above."
                                )

                            else:

                                st.success(
                                    "No severe weather detected at this "
                                    "sampled route location using the "
                                    "configured thresholds."
                                )

                    route_map_center = route["points"][
                        len(route["points"]) // 2
                    ]

                    weather_route_map = folium.Map(
                        location=route_map_center,
                        zoom_start=6,
                        control_scale=True
                    )

                    folium.PolyLine(
                        route["points"],
                        color="blue",
                        weight=5,
                        opacity=0.8,
                        tooltip="Planned Shipment Road Route"
                    ).add_to(weather_route_map)

                    folium.Marker(
                        location=source_coordinates,
                        popup=f"Source: {shipment['source']}",
                        tooltip="Shipment Source",
                        icon=folium.Icon(
                            color="green",
                            icon="play"
                        )
                    ).add_to(weather_route_map)

                    folium.Marker(
                        location=destination_coordinates,
                        popup=f"Destination: {shipment['destination']}",
                        tooltip="Shipment Destination",
                        icon=folium.Icon(
                            color="red",
                            icon="flag"
                        )
                    ).add_to(weather_route_map)

                    for item in route_weather_results:

                        has_severe_weather = bool(
                            item["severe_days"]
                        )

                        marker_color = (
                            "red"
                            if has_severe_weather
                            else "orange"
                        )

                        popup_text = (
                            f"<b>{item['name']}</b><br>"
                            f"{item['place_name']}<br>"
                            f"Latitude: "
                            f"{item['coordinates'][0]:.6f}<br>"
                            f"Longitude: "
                            f"{item['coordinates'][1]:.6f}<br>"
                            f"Severe forecast days: "
                            f"{len(item['severe_days'])}<br>"
                            f"Alerts: "
                            f"{', '.join(item['alerts']) or 'None'}"
                        )

                        folium.Marker(
                            location=item["coordinates"],
                            popup=folium.Popup(
                                popup_text,
                                max_width=350
                            ),
                            tooltip=(
                                f"Severe weather: {item['place_name']}"
                                if has_severe_weather
                                else (
                                    f"{item['place_name']} – "
                                    "No severe weather detected"
                                )
                            ),
                            icon=folium.Icon(
                                color=marker_color,
                                icon=(
                                    "exclamation-triangle"
                                    if has_severe_weather
                                    else "cloud"
                                ),
                                prefix="fa"
                            )
                        ).add_to(weather_route_map)

                    weather_route_map.fit_bounds(
                        route["points"],
                        padding=(25, 25)
                    )

                    st.subheader(
                        "🗺️ Shipment Route Weather Map"
                    )

                    st.caption(
                        "Blue line: planned road route | Green: source | "
                        "Red: destination | Orange: sampled route point "
                        "with no severe forecast | Red marker: sampled "
                        "route point with at least one severe forecast day"
                    )

                    st_folium(
                        weather_route_map,
                        width=None,
                        height=550,
                        key=f"weather_route_map_{shipment_id}"
                    )

                else:

                    st.info(
                        "No intermediate route locations were available "
                        "from the generated route. The source and "
                        "destination forecasts are still displayed above."
                    )

                st.divider()

                st.subheader(
                    "📊 Weather Monitoring Summary"
                )

                source_severe = sum(
                    day["severity"] == "Severe"
                    for day in source_weather
                )

                destination_severe = sum(
                    day["severity"] == "Severe"
                    for day in destination_weather
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Forecast Days",
                    7
                )

                col2.metric(
                    "Source Severe Days",
                    source_severe
                )

                col3.metric(
                    "Destination Severe Days",
                    destination_severe
                )

                col4.metric(
                    "Route Points Affected",
                    route_affected_count
                )

                any_route_severe = (
                    route_affected_count > 0
                )

                if (
                    source_severe
                    or destination_severe
                    or any_route_severe
                ):

                    st.warning(
                        "Potential weather disturbance detected at one "
                        "or more shipment locations. Review the forecast "
                        "dates and route map before dispatching."
                    )

                else:

                    st.success(
                        "No severe weather warnings detected at the "
                        "shipment endpoints or the sampled intermediate "
                        "route locations."
                    )

                st.caption(
                    "Coverage note: forecasts are checked at the source, "
                    "destination, and up to five sampled intermediate route "
                    "locations. Conditions between sampled points may differ. "
                    "This is a planning aid, not a live road-closure or "
                    "official emergency-warning service."
                )

            except requests.RequestException as error:

                st.error(
                    f"Weather or route API network error: {error}"
                )

            except RuntimeError as error:

                st.error(
                    f"Route generation error: {error}"
                )

            except (ValueError, KeyError, TypeError) as error:

                st.error(
                    f"Weather data processing error: {error}"
                )


# ==================================================
# 15. DAY 3 - FESTIVAL AND HOLIDAY MONITORING
# ==================================================

elif page == "Festival Monitoring":

    st.header("🎉 Festival and Holiday Monitoring")

    st.write(
        "Monitor upcoming festivals and public holidays that may affect "
        "shipment operations, local traffic, and delivery schedules."
    )

    # ----------------------------------------------
    # LOAD FESTIVAL EVENTS
    # ----------------------------------------------

    today = date.today()

    all_events = get_festival_events()

    upcoming_events = get_festival_events(
        start_date=today.isoformat()
    )

    # ----------------------------------------------
    # SUMMARY METRICS
    # ----------------------------------------------

    total_events = len(all_events)

    upcoming_count = len(upcoming_events)

    high_impact_count = sum(
        event["impact_level"] == "High"
        for event in upcoming_events
    )

    medium_impact_count = sum(
        event["impact_level"] == "Medium"
        for event in upcoming_events
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Total Events",
        total_events
    )

    metric2.metric(
        "Upcoming Events",
        upcoming_count
    )

    metric3.metric(
        "High Impact Events",
        high_impact_count
    )

    metric4.metric(
        "Medium Impact Events",
        medium_impact_count
    )

    st.divider()

    # ----------------------------------------------
    # FILTER FESTIVAL EVENTS
    # ----------------------------------------------

    st.subheader("🔎 Search and Filter Events")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:

        filter_state = st.selectbox(
            "Filter by State",
            [
                "All States",
                "West Bengal",
                "Odisha",
                "Jharkhand",
                "Bihar",
                "Delhi",
                "Maharashtra",
                "Uttar Pradesh",
                "Other"
            ],
            key="festival_filter_state"
        )

    with filter_col2:

        filter_type = st.selectbox(
            "Filter by Event Type",
            [
                "All Types",
                "Festival",
                "Public Holiday",
                "Other"
            ],
            key="festival_filter_type"
        )

    with filter_col3:

        filter_period = st.selectbox(
            "Date Range",
            [
                "All Events",
                "Upcoming Events",
                "Next 30 Days",
                "Next 90 Days",
                "Past Events"
            ],
            key="festival_filter_period"
        )

    filtered_events = list(all_events)

    if filter_state != "All States":

        if filter_state == "Other":

            known_states = [
                "West Bengal",
                "Odisha",
                "Jharkhand",
                "Bihar",
                "Delhi",
                "Maharashtra",
                "Uttar Pradesh"
            ]

            filtered_events = [
                event for event in filtered_events
                if event["state"] not in known_states
            ]

        else:

            filtered_events = [
                event for event in filtered_events
                if event["state"].lower() == filter_state.lower()
            ]

    if filter_type != "All Types":

        if filter_type == "Other":

            filtered_events = [
                event for event in filtered_events
                if event["event_type"] not in [
                    "Festival",
                    "Public Holiday"
                ]
            ]

        else:

            filtered_events = [
                event for event in filtered_events
                if event["event_type"] == filter_type
            ]

    if filter_period == "Upcoming Events":

        filtered_events = [
            event for event in filtered_events
            if event["end_date"] >= today.isoformat()
        ]

    elif filter_period == "Next 30 Days":

        last_date = (
            today + timedelta(days=30)
        ).isoformat()

        filtered_events = [
            event for event in filtered_events
            if (
                event["end_date"] >= today.isoformat()
                and event["start_date"] <= last_date
            )
        ]

    elif filter_period == "Next 90 Days":

        last_date = (
            today + timedelta(days=90)
        ).isoformat()

        filtered_events = [
            event for event in filtered_events
            if (
                event["end_date"] >= today.isoformat()
                and event["start_date"] <= last_date
            )
        ]

    elif filter_period == "Past Events":

        filtered_events = [
            event for event in filtered_events
            if event["end_date"] < today.isoformat()
        ]

    st.subheader(
        f"📅 Festival and Holiday Calendar ({len(filtered_events)} events)"
    )

    display_festival_table(filtered_events)

    # ----------------------------------------------
    # UPCOMING EVENT DETAILS
    # ----------------------------------------------

    st.divider()

    st.subheader("⚠️ Upcoming Event Details")

    if upcoming_events:

        for event in upcoming_events:

            icon = get_festival_impact_color(
                event["impact_level"]
            )

            with st.expander(
                f"{icon} {event['name']} | "
                f"{event['start_date']} | "
                f"{event['impact_level']} Impact"
            ):

                st.write(
                    f"**Event Type:** {event['event_type']}"
                )

                st.write(
                    f"**Start Date:** {event['start_date']}"
                )

                st.write(
                    f"**End Date:** {event['end_date']}"
                )

                st.write(
                    f"**State:** {event['state']}"
                )

                st.write(
                    f"**Affected Location:** "
                    f"{event['affected_location']}"
                )

                st.write(
                    f"**Impact Level:** {event['impact_level']}"
                )

                st.write(
                    f"**Description:** "
                    f"{event['description'] or 'No description available.'}"
                )

                if event["source_url"]:

                    st.markdown(
                        f"**Reference:** [View source]({event['source_url']})"
                    )

    else:

        st.info(
            "No upcoming festival or holiday records are available."
        )

    # ----------------------------------------------
    # ADD NEW EVENT
    # ----------------------------------------------

    st.divider()

    st.subheader("➕ Add Festival or Holiday")

    with st.form("add_festival_form", clear_on_submit=True):

        new_name = st.text_input(
            "Festival / Holiday Name",
            placeholder="Example: Durga Puja"
        )

        add_col1, add_col2 = st.columns(2)

        with add_col1:

            new_type = st.selectbox(
                "Event Type",
                [
                    "Festival",
                    "Public Holiday",
                    "Other"
                ],
                key="new_festival_type"
            )

            new_start_date = st.date_input(
                "Start Date",
                value=today,
                key="new_festival_start"
            )

            new_state = st.text_input(
                "State",
                value="West Bengal",
                placeholder="Example: West Bengal"
            )

            new_impact = st.selectbox(
                "Impact Level",
                [
                    "Low",
                    "Medium",
                    "High"
                ],
                index=1,
                key="new_festival_impact"
            )

        with add_col2:

            new_end_date = st.date_input(
                "End Date",
                value=today,
                key="new_festival_end"
            )

            new_location = st.text_input(
                "Affected Location",
                placeholder="Example: Kolkata, Howrah"
            )

            new_source_url = st.text_input(
                "Reference URL (optional)",
                placeholder="https://example.com"
            )

        new_description = st.text_area(
            "Description / Possible Traffic Impact",
            placeholder=(
                "Describe how the event may affect traffic or deliveries."
            )
        )

        add_event_submit = st.form_submit_button(
            "➕ Add Event",
            use_container_width=True
        )

        if add_event_submit:

            if (
                not new_name.strip()
                or not new_state.strip()
                or not new_location.strip()
            ):

                st.error(
                    "Please enter the event name, state, "
                    "and affected location."
                )

            elif new_end_date < new_start_date:

                st.error(
                    "End date cannot be earlier than start date."
                )

            elif (
                new_source_url.strip()
                and not new_source_url.strip().startswith(
                    ("http://", "https://")
                )
            ):

                st.error(
                    "Reference URL must start with http:// or https://."
                )

            else:

                try:

                    add_festival_event(
                        name=new_name,
                        event_type=new_type,
                        start_date=new_start_date.isoformat(),
                        end_date=new_end_date.isoformat(),
                        state=new_state,
                        affected_location=new_location,
                        impact_level=new_impact,
                        description=new_description,
                        source_url=new_source_url
                    )

                    st.success(
                        f"{new_name} added successfully!"
                    )

                    st.rerun()

                except sqlite3.IntegrityError:

                    st.error(
                        "This event already exists with the same name, "
                        "start date, state, and affected location."
                    )

                except (ValueError, sqlite3.Error) as error:

                    st.error(
                        f"Could not add event: {error}"
                    )

    # ----------------------------------------------
    # EDIT EXISTING EVENT
    # ----------------------------------------------

    st.divider()

    st.subheader("✏️ Edit Existing Event")

    if all_events:

        edit_options = {
            (
                f"ID {event['event_id']} | "
                f"{event['name']} | "
                f"{event['start_date']}"
            ): event
            for event in all_events
        }

        selected_edit_label = st.selectbox(
            "Select Event to Edit",
            list(edit_options.keys()),
            key="edit_festival_selection"
        )

        selected_edit_event = edit_options[
            selected_edit_label
        ]

        with st.form("edit_festival_form"):

            edit_name = st.text_input(
                "Event Name",
                value=selected_edit_event["name"]
            )

            edit_col1, edit_col2 = st.columns(2)

            with edit_col1:

                edit_type = st.selectbox(
                    "Event Type",
                    [
                        "Festival",
                        "Public Holiday",
                        "Other"
                    ],
                    index=(
                        [
                            "Festival",
                            "Public Holiday",
                            "Other"
                        ].index(selected_edit_event["event_type"])
                        if selected_edit_event["event_type"]
                        in ["Festival", "Public Holiday", "Other"]
                        else 2
                    ),
                    key="edit_festival_type"
                )

                edit_start_date = st.date_input(
                    "Start Date",
                    value=date.fromisoformat(
                        selected_edit_event["start_date"]
                    ),
                    key="edit_festival_start"
                )

                edit_state = st.text_input(
                    "State",
                    value=selected_edit_event["state"],
                    key="edit_festival_state"
                )

                impact_options = [
                    "Low",
                    "Medium",
                    "High"
                ]

                current_impact = selected_edit_event["impact_level"]

                edit_impact = st.selectbox(
                    "Impact Level",
                    impact_options,
                    index=(
                        impact_options.index(current_impact)
                        if current_impact in impact_options
                        else 0
                    ),
                    key="edit_festival_impact"
                )

            with edit_col2:

                edit_end_date = st.date_input(
                    "End Date",
                    value=date.fromisoformat(
                        selected_edit_event["end_date"]
                    ),
                    key="edit_festival_end"
                )

                edit_location = st.text_input(
                    "Affected Location",
                    value=selected_edit_event["affected_location"],
                    key="edit_festival_location"
                )

                edit_source_url = st.text_input(
                    "Reference URL",
                    value=selected_edit_event["source_url"] or "",
                    key="edit_festival_url"
                )

            edit_description = st.text_area(
                "Description",
                value=selected_edit_event["description"] or "",
                key="edit_festival_description"
            )

            edit_submit = st.form_submit_button(
                "💾 Save Changes",
                use_container_width=True
            )

            if edit_submit:

                if (
                    not edit_name.strip()
                    or not edit_state.strip()
                    or not edit_location.strip()
                ):

                    st.error(
                        "Event name, state, and affected location are required."
                    )

                elif edit_end_date < edit_start_date:

                    st.error(
                        "End date cannot be earlier than start date."
                    )

                elif (
                    edit_source_url.strip()
                    and not edit_source_url.strip().startswith(
                        ("http://", "https://")
                    )
                ):

                    st.error(
                        "Reference URL must start with http:// or https://."
                    )

                else:

                    try:

                        updated = update_festival_event(
                            event_id=selected_edit_event["event_id"],
                            name=edit_name,
                            event_type=edit_type,
                            start_date=edit_start_date.isoformat(),
                            end_date=edit_end_date.isoformat(),
                            state=edit_state,
                            affected_location=edit_location,
                            impact_level=edit_impact,
                            description=edit_description,
                            source_url=edit_source_url
                        )

                        if updated:

                            st.success(
                                "Festival event updated successfully!"
                            )

                            st.rerun()

                        else:

                            st.warning(
                                "No event was updated. Please check the selected record."
                            )

                    except sqlite3.IntegrityError:

                        st.error(
                            "An event with the same name, start date, "
                            "state, and location already exists."
                        )

                    except (ValueError, sqlite3.Error) as error:

                        st.error(
                            f"Could not update event: {error}"
                        )

    else:

        st.info(
            "There are no events to edit."
        )

    # ----------------------------------------------
    # DELETE EVENT
    # ----------------------------------------------

    st.divider()

    st.subheader("🗑️ Delete Event")

    if all_events:

        delete_options = {
            (
                f"ID {event['event_id']} | "
                f"{event['name']} | "
                f"{event['start_date']}"
            ): event["event_id"]
            for event in all_events
        }

        selected_delete_label = st.selectbox(
            "Select Event to Delete",
            list(delete_options.keys()),
            key="delete_festival_selection"
        )

        confirm_delete = st.checkbox(
            "I confirm that I want to delete this event.",
            key="confirm_delete_festival"
        )

        if st.button(
            "🗑️ Delete Selected Event",
            use_container_width=True,
            disabled=not confirm_delete
        ):

            try:

                deleted = delete_festival_event(
                    delete_options[selected_delete_label]
                )

                if deleted:

                    st.success(
                        "Festival event deleted successfully!"
                    )

                    st.rerun()

                else:

                    st.warning(
                        "The selected event was not found."
                    )

            except sqlite3.Error as error:

                st.error(
                    f"Could not delete event: {error}"
                )

    else:

        st.info(
            "There are no events to delete."
        )

    # ----------------------------------------------
    # IMPORTANT INFORMATION
    # ----------------------------------------------

    st.divider()

    st.subheader("ℹ️ Festival Monitoring Information")

    st.info(
        """
        **How this module works:**

        1. Festival and holiday records are stored in SQLite.
        2. You can add, edit, and delete event information.
        3. You can filter events by state, event type, and date range.
        4. Upcoming events are displayed on the Dashboard.

        **Important:** An event's impact level is a planning indicator.
        It does not confirm an actual road closure or traffic disruption.
        Verify local traffic restrictions with official authorities.
        """
    )


# ==================================================
# 16. DAY 3 - RSS DISTURBANCE MONITORING
# ==================================================

elif page == "Disturbance Monitoring":

    st.header("📢 RSS-Based Disturbance Monitoring")
    st.caption("RSS diagnostics build: LOCATION-CONFIDENCE-2026-11-10")

    st.write(
        "Fetch public RSS announcements about protests, strikes, "
        "weather incidents, transport disruption, and public events. "
        "Potential disturbances are saved in the local SQLite database."
    )

    st.warning(
        "RSS announcements are unverified information signals. "
        "They do not prove that a road is closed or that a shipment is affected. "
        "Confirm important information with official authorities or the source publisher."
    )

    # ----------------------------------------------
    # FETCH LATEST RSS FEEDS
    # ----------------------------------------------

    st.subheader("🔄 Fetch Latest RSS Announcements")

    fetch_col, info_col = st.columns([1, 2])

    with fetch_col:
        fetch_clicked = st.button(
            "🔄 Fetch Latest RSS Announcements",
            use_container_width=True,
            key="fetch_disturbance_rss",
        )

    with info_col:
        st.caption(
            "Each configured RSS publisher is checked. Existing articles are "
            "not inserted again; new relevant announcements are saved."
        )

    if fetch_clicked:
        try:
            with st.spinner("Fetching and analysing RSS announcements..."):
                fetch_result = fetch_disturbance_feeds()

            st.session_state["disturbance_fetch_result"] = fetch_result

            scanned_now = fetch_result.get(
                "articles_scanned", fetch_result.get("total_scanned", 0)
            )
            failed_now = fetch_result.get("failed_feeds", 0)
            empty_now = fetch_result.get("empty_feeds", 0)

            if scanned_now > 0:
                st.success(
                    f"RSS scan completed: {scanned_now} feed entries were read."
                )
            elif failed_now:
                st.warning(
                    "No RSS entries were read. One or more publishers could not "
                    "be reached. Check the feed-by-feed diagnostics below."
                )
            elif empty_now:
                st.warning(
                    "The publishers responded, but no RSS entries were parsed. "
                    "Check the feed format and diagnostics below."
                )
            else:
                st.warning(
                    "The scan returned no entries. Review the feed diagnostics below."
                )

        except requests.RequestException as error:
            st.error(f"RSS network error: {error}")
        except sqlite3.Error as error:
            st.error(f"Disturbance database error: {error}")
        except Exception as error:
            st.error(f"Could not fetch RSS announcements: {error}")

    fetch_result = st.session_state.get("disturbance_fetch_result")

    if fetch_result:
        st.subheader("📡 Latest Feed Scan Results")

        # The service returns articles_scanned/relevant_articles/
        # new_disturbances_saved. The fallback keys support older service versions.
        scanned_value = fetch_result.get(
            "articles_scanned", fetch_result.get("total_scanned", 0)
        )
        relevant_value = fetch_result.get(
            "relevant_articles", fetch_result.get("total_relevant", 0)
        )
        saved_value = fetch_result.get(
            "new_disturbances_saved", fetch_result.get("total_saved", 0)
        )

        duplicate_value = fetch_result.get("duplicate_articles", 0)

        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        result_col1.metric("Articles Scanned", scanned_value)
        result_col2.metric("Relevant Articles", relevant_value)
        result_col3.metric("New Disturbances Saved", saved_value)
        result_col4.metric("Already Saved / Duplicates", duplicate_value)

        if relevant_value > 0 and saved_value == 0 and duplicate_value >= relevant_value:
            st.info(
                "All relevant articles from this scan matched announcements already "
                "stored in the database (or repeated across feeds). No new database "
                "rows were needed. The matching article titles are listed below."
            )
        elif duplicate_value > 0:
            st.info(
                f"{duplicate_value} relevant article(s) were skipped as duplicates. "
                f"{saved_value} new announcement(s) were added."
            )

        # Display scan details directly (not inside collapsed expanders), so
        # the duplicate count can always be checked against its actual titles.
        
        newly_saved_details = fetch_result.get("newly_saved_details") or []
        duplicate_details = fetch_result.get("duplicate_details") or []

        st.markdown("#### 🆕 Newly saved announcements")

        if newly_saved_details:
            st.dataframe(
                pd.DataFrame(newly_saved_details),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No new announcements were inserted during this scan.")

        st.markdown("#### ♻️ Relevant articles skipped as duplicates")

        # Temporary diagnostic information
        st.write("Duplicate count:", duplicate_value)
        st.write("Duplicate details received:", len(duplicate_details))

        if duplicate_details:
            duplicate_df = pd.DataFrame(duplicate_details)

            st.dataframe(
                duplicate_df,
                use_container_width=True,
                hide_index=True,
            )

            st.success(
                f"Successfully displayed {len(duplicate_details)} "
                "duplicate article details."
            )

        elif duplicate_value > 0:
            st.error(
                "The scan reports duplicates, but the Streamlit page "
                "did not receive their details."
            )
            st.json(fetch_result)

        else:
            st.info("No relevant duplicate articles were found.")


        feed_rows = fetch_result.get(
            "feed_statuses", fetch_result.get("feeds", [])
        )

        if feed_rows:
            feed_df = pd.DataFrame(feed_rows)
            expected_feed_columns = [
                "source", "status", "http_status", "scanned", "relevant",
                "new_saved", "duplicates", "content_type", "message",
            ]
            for column in expected_feed_columns:
                if column not in feed_df.columns:
                    feed_df[column] = ""

            feed_df = feed_df[expected_feed_columns]
            feed_df.columns = [
                "RSS Source", "Fetch Status", "HTTP Status", "Entries Read",
                "Relevant", "New Records Saved", "Duplicates",
                "Response Content Type", "Diagnostic Message",
            ]
            st.dataframe(feed_df, use_container_width=True, hide_index=True)

            if scanned_value == 0:
                st.error(
                    "Articles Scanned is zero because no feed entries were "
                    "successfully parsed. The diagnostic message for each feed "
                    "above identifies whether it timed out, returned an HTTP "
                    "error, or returned an empty/unrecognised feed."
                )
        else:
            st.error(
                "No feed diagnostics were returned by disturbance_service.py. "
                "Make sure the updated service file is installed in the same "
                "folder as app.py and restart Streamlit."
            )

    st.divider()

    # ----------------------------------------------
    # LOAD SAVED DISTURBANCE EVENTS
    # ----------------------------------------------

    st.subheader("🚧 Saved Disturbance Announcements")

    try:

        disturbance_events = get_disturbance_events(limit=500)

    except (sqlite3.Error, TypeError, ValueError) as error:

        st.error(f"Could not load disturbance records: {error}")
        disturbance_events = []

    total_disturbances = len(disturbance_events)

    high_count = sum(
        str(event.get("severity", "")).title() == "High"
        for event in disturbance_events
    )

    medium_count = sum(
        str(event.get("severity", "")).title() == "Medium"
        for event in disturbance_events
    )

    low_count = sum(
        str(event.get("severity", "")).title() == "Low"
        for event in disturbance_events
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Total Records", total_disturbances)
    metric2.metric("High Severity", high_count)
    metric3.metric("Medium Severity", medium_count)
    metric4.metric("Low Severity", low_count)

    if disturbance_events:

        # ----------------------------------------------
        # FILTERS
        # ----------------------------------------------

        st.subheader("🔎 Filter Announcements")

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        event_types = sorted({
            str(event.get("event_type") or "Unknown")
            for event in disturbance_events
        })

        severities = sorted({
            str(event.get("severity") or "Unknown").title()
            for event in disturbance_events
        })

        statuses = sorted({
            str(event.get("status") or "Unknown")
            for event in disturbance_events
        })

        with filter_col1:

            selected_event_type = st.selectbox(
                "Event Type",
                ["All Types"] + event_types,
                key="disturbance_filter_type"
            )

        with filter_col2:

            selected_severity = st.selectbox(
                "Severity",
                ["All Severities"] + severities,
                key="disturbance_filter_severity"
            )

        with filter_col3:

            selected_status = st.selectbox(
                "Verification Status",
                ["All Statuses"] + statuses,
                key="disturbance_filter_status"
            )

        search_text = st.text_input(
            "Search title, location, or description",
            key="disturbance_search_text"
        ).strip().lower()

        filtered_events = list(disturbance_events)

        if selected_event_type != "All Types":

            filtered_events = [
                event for event in filtered_events
                if str(event.get("event_type") or "Unknown") == selected_event_type
            ]

        if selected_severity != "All Severities":

            filtered_events = [
                event for event in filtered_events
                if str(event.get("severity") or "Unknown").title() == selected_severity
            ]

        if selected_status != "All Statuses":

            filtered_events = [
                event for event in filtered_events
                if str(event.get("status") or "Unknown") == selected_status
            ]

        if search_text:

            filtered_events = [
                event for event in filtered_events
                if search_text in " ".join([
                    str(event.get("title") or ""),
                    str(event.get("location") or ""),
                    str(event.get("summary") or ""),
                    str(event.get("event_type") or "")
                ]).lower()
            ]

        st.subheader(
            f"📋 Announcements ({len(filtered_events)} matching records)"
        )

        if filtered_events:

            display_rows = []

            for event in filtered_events:

                display_rows.append({
                    "ID": event.get("disturbance_id"),
                    "Title": event.get("title"),
                    "Event Date": event.get("event_date"),
                    "Location": event.get("location") or "Not identified",
                    "Location Confidence": event.get("location_confidence", "Unknown"),
                    "Route Match Eligible": "Yes" if event.get("route_matchable") is True else "No",
                    "Event Type": event.get("event_type"),
                    "Severity": event.get("severity"),
                    "Status": event.get("status"),
                    "Published": event.get("published_at"),
                    "Source": event.get("source_name")
                })

            st.dataframe(
                pd.DataFrame(display_rows),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("📰 Announcement Details")

            for event in filtered_events:

                severity = str(event.get("severity") or "Unknown").title()

                severity_icon = {
                    "High": "🔴",
                    "Medium": "🟠",
                    "Low": "🟢"
                }.get(severity, "⚪")

                event_title = event.get("title") or "Untitled announcement"

                with st.expander(
                    f"{severity_icon} {event_title} | "
                    f"{event.get('location') or 'Location not identified'} | "
                    f"{severity}"
                ):

                    st.write(f"**Event type:** {event.get('event_type') or 'Unknown'}")
                    st.write(f"**Event date:** {event.get('event_date') or 'Not identified'}")
                    st.write(f"**Location:** {event.get('location') or 'Not identified'}")
                    st.write(f"**Location confidence:** {event.get('location_confidence', 'Unknown')}")
                    st.write(f"**Location evidence:** {event.get('location_evidence', 'Not available')}")
                    st.write(f"**Eligible for automatic route matching:** {'Yes' if event.get('route_matchable') is True else 'No'}")
                    if event.get("location_note"):
                        st.caption(event["location_note"])
                    st.write(f"**Severity:** {severity}")
                    st.write(f"**Verification status:** {event.get('status') or 'Unknown'}")
                    st.write(f"**Publisher:** {event.get('source_name') or 'Unknown'}")
                    st.write(f"**Published at:** {event.get('published_at') or 'Unknown'}")

                    summary = event.get("summary")

                    if summary:
                        st.write("**Summary**")
                        st.write(summary)

                    source_url = event.get("source_url")

                    if source_url and str(source_url).startswith(("http://", "https://")):
                        st.markdown(f"[Read original announcement]({source_url})")

        else:

            st.info("No announcements match the selected filters.")

    else:

        st.info(
            "No RSS disturbance records are saved yet. Click **Fetch Latest RSS Feeds** "
            "to collect announcements."
        )

    st.divider()

    st.subheader("ℹ️ How RSS Disturbance Monitoring Works")

    st.markdown(
        """
        1. RSS feeds are checked for public announcements.
        2. Relevant articles are classified into disturbance categories.
        3. The service attempts to extract event dates and locations.
        4. Records are stored in SQLite and duplicate articles are not repeatedly inserted.
        5. Saved event locations are compared with registered shipment routes.

        **Route matching:** Potential geographic matches are displayed in GPS Tracking.
        Matches are based on geocoded locations and their distance from the planned route.

        **Location confidence and matching:**
        - High-confidence settlement names can be checked against shipment routes.
        - Broad, district-level or ambiguous names remain visible but are excluded from automatic route alerts.
        - Geocoding must return a locality-level result whose address matches the requested place name.

        **Important limitations:**
        - A geographic match does not prove that a road is closed.
        - State-level locations are not treated as precise route matches.
        - Exact affected road segments are not independently verified.
        - RSS announcements remain unverified until confirmed by an official source.
        """
    )


# ==================================================
# DAY 4 - ML DATASET COLLECTION
# IMPORT EXISTING SHIPMENT AND GPS INFORMATION
# ==================================================

elif page == "ML Dataset Collection":

    st.header("🤖 Automatic Machine Learning Dataset Collection")
    st.write(
        "Automatically collect historical route, GPS, weather, festival and RSS "
        "features from registered shipments. No manual observation form is required."
    )

    verified_summary = get_dataset_summary()
    auto_summary = get_auto_dataset_summary()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Automatic Observations", auto_summary["total_records"])
    m2.metric("Verified Training Records", verified_summary["total_records"])
    m3.metric("Verified Disruptions", verified_summary["disrupted"])
    m4.metric("Verified No Disruptions", verified_summary["not_disrupted"])

    st.info(
        "Automatic collection stores weather and event features without guessing the "
        "shipment outcome. An RSS report, festival, weather condition, delivery status, "
        "or GPS position alone cannot prove that a shipment was disrupted or unaffected. "
        "Unverified observations are kept separate and are not used to train the classifier."
    )

    st.subheader("🔄 Automatic collection from all registered shipments")
    st.caption(
        "The collector groups reliable, on-route GPS records by local observation date, "
        "fetches Open-Meteo historical weather, and checks saved festival and RSS records. "
        "Repeated runs update the same shipment/date record instead of creating duplicates."
    )

    if st.button("🚀 Run Automatic Dataset Collection", use_container_width=True, type="primary"):
        shipments_df = get_shipments()
        try:
            festival_events = get_festival_events() or []
        except Exception as error:
            festival_events = []
            st.warning(f"Festival records could not be loaded: {error}")
        try:
            rss_events = get_disturbance_events(limit=1000) or []
        except Exception as error:
            rss_events = []
            st.warning(f"RSS records could not be loaded: {error}")

        inserted = 0
        skipped_no_gps = 0
        skipped_off_route = 0
        weather_failures = 0
        errors = []
        gps_diagnostics = []
        progress = st.progress(0.0)
        status_line = st.empty()

        if shipments_df.empty:
            st.warning("No registered shipments were found.")
        else:
            for shipment_index, (_, shipment_row) in enumerate(shipments_df.iterrows()):
                shipment = shipment_row.to_dict()
                shipment_id = int(shipment["shipment_id"])
                source = str(shipment.get("source") or "Unknown")
                destination = str(shipment.get("destination") or "Unknown")
                route_segment = f"{source} → {destination}"
                coordinate_names = (
                    "source_latitude", "source_longitude",
                    "destination_latitude", "destination_longitude"
                )
                try:
                    coords = [float(shipment.get(name)) for name in coordinate_names]
                    if not all(math.isfinite(value) for value in coords):
                        raise ValueError("Shipment coordinates are invalid.")
                    route = generate_route((coords[0], coords[1]), (coords[2], coords[3]))
                    route_points = route.get("points") or []
                    if not route_points:
                        raise ValueError("OSRM returned no route geometry.")
                    route_distance = float(route.get("distance_km", 0))
                    gps_history = get_tracking_history(shipment_id) or []
                except Exception as error:
                    errors.append(f"Shipment {shipment_id}: {error}")
                    skipped_no_gps += 1
                    progress.progress((shipment_index + 1) / len(shipments_df))
                    continue

                if not gps_history:
                    skipped_no_gps += 1
                    progress.progress((shipment_index + 1) / len(shipments_df))
                    continue

                # Validate each fix against the complete route polyline and keep
                # an audit breakdown. The 15 km outer corridor is intentionally
                # retained for compatibility with legitimate short detours; it
                # is already broad, so do not increase it just to reduce rejects.
                valid_dates = set()
                off_route_fixes = 0
                invalid_accuracy_fixes = 0
                valid_accuracy_fixes = 0
                within_2km_fixes = 0
                within_2_to_15km_fixes = 0
                measured_distances = []

                for record in gps_history:
                    try:
                        lat = float(record[0])
                        lon = float(record[1])
                        accuracy = float(record[2])
                        recorded_at = datetime.fromisoformat(
                            str(record[3]).replace("Z", "+00:00")
                        )

                        if not (
                            math.isfinite(lat) and math.isfinite(lon)
                            and -90 <= lat <= 90 and -180 <= lon <= 180
                        ):
                            continue

                        if not (math.isfinite(accuracy) and 0 <= accuracy <= MAX_GPS_ACCURACY):
                            invalid_accuracy_fixes += 1
                            continue

                        valid_accuracy_fixes += 1

                        # Segment-based distance avoids false rejection between
                        # sparse route vertices and works for curved routes.
                        nearest_km = distance_from_location_to_route_km(
                            (lat, lon), route_points
                        )
                        if nearest_km is None or not math.isfinite(nearest_km):
                            continue

                        measured_distances.append(float(nearest_km))
                        if nearest_km <= 2.0:
                            within_2km_fixes += 1
                        elif nearest_km <= MAX_GPS_DISTANCE_FROM_ROUTE_KM:
                            within_2_to_15km_fixes += 1

                        if nearest_km <= MAX_GPS_DISTANCE_FROM_ROUTE_KM:
                            valid_dates.add(recorded_at.date())
                        else:
                            off_route_fixes += 1

                    except (ValueError, TypeError, IndexError, OverflowError):
                        continue

                gps_diagnostics.append({
                    "Shipment ID": shipment_id,
                    "Route": route_segment,
                    "GPS records": len(gps_history),
                    "Accuracy-valid fixes": valid_accuracy_fixes,
                    "Within 2 km": within_2km_fixes,
                    "2–15 km (review band)": within_2_to_15km_fixes,
                    "Over 15 km": off_route_fixes,
                    "Invalid accuracy": invalid_accuracy_fixes,
                    "Nearest route distance (km)": (
                        round(min(measured_distances), 2) if measured_distances else None
                    ),
                    "Median route distance (km)": (
                        round(float(pd.Series(measured_distances).median()), 2)
                        if measured_distances else None
                    ),
                })

                skipped_off_route += off_route_fixes
                if not valid_dates:
                    skipped_no_gps += 1
                    progress.progress((shipment_index + 1) / len(shipments_df))
                    continue

                # Route names and state/region names used for conservative event matching.
                route_terms = {source.casefold(), destination.casefold()}
                combined_route = f"{source} {destination}".casefold()
                region_terms = [
                    state for state in (
                        "Karnataka", "West Bengal", "Tamil Nadu", "Kerala", "Maharashtra",
                        "Odisha", "Andhra Pradesh", "Telangana", "Bihar", "Jharkhand"
                    ) if state.casefold() in combined_route
                ]
                route_terms.update(term.casefold() for term in region_terms)

                for observation_day in sorted(valid_dates):
                    if observation_day >= date.today():
                        continue
                    try:
                        weather = fetch_historical_weather_for_route(
                            coords[0], coords[1], coords[2], coords[3], observation_day
                        )
                    except Exception as error:
                        weather_failures += 1
                        errors.append(f"Shipment {shipment_id} on {observation_day}: historical weather unavailable ({error})")
                        continue

                    day_iso = observation_day.isoformat()
                    matched_festivals = []
                    festival_data_available = bool(festival_events)
                    for event in festival_events:
                        event_start = str(event.get("start_date") or "")[:10]
                        event_end = str(event.get("end_date") or event_start)[:10]
                        location = str(event.get("affected_location") or event.get("location") or "").casefold()
                        date_match = bool(event_start and event_end and event_start <= day_iso <= event_end)
                        location_match = any(term and term in location for term in route_terms)
                        if date_match and location_match:
                            matched_festivals.append(event)

                    matched_rss = []
                    rss_data_available = bool(rss_events)
                    for event in rss_events:
                        event_day = str(event.get("event_date") or "")[:10]
                        location = str(event.get("location") or "").casefold()
                        date_match = event_day == day_iso
                        location_match = any(term and term in location for term in route_terms)
                        if date_match and location_match:
                            matched_rss.append(event)

                    # A weather severity feature is derived from actual archive values,
                    # not from a user-entered checkbox.
                    severe_weather = int(
                        float(weather["rainfall"]) >= 50.0
                        or float(weather["wind_speed"]) >= 60.0
                    )
                    festival_flag = int(bool(matched_festivals)) if festival_data_available else None
                    rss_flag = int(bool(matched_rss)) if rss_data_available else None
                    severities = [str(event.get("severity") or "Unknown") for event in matched_rss]
                    severity_rank = {"high": 3, "medium": 2, "low": 1}
                    rss_severity = max(severities, key=lambda value: severity_rank.get(value.casefold(), 0)) if severities else "None"
                    disturbance_types = sorted({str(event.get("event_type") or "Unknown") for event in matched_rss})
                    festival_names = [str(event.get("name") or "Festival") for event in matched_festivals]
                    rss_titles = [str(event.get("title") or "RSS announcement") for event in matched_rss]
                    notes = (
                        "GPS evidence: at least one fix with accuracy <= 100 m within "
                        f"{MAX_GPS_DISTANCE_FROM_ROUTE_KM:.0f} km of the planned route. "
                        "Weather source: Open-Meteo Historical Weather API (reanalysis). "
                        f"Festival matches: {', '.join(festival_names) if festival_names else ('none found in stored calendar' if festival_data_available else 'calendar unavailable')}. "
                        f"RSS matches: {', '.join(rss_titles) if rss_titles else ('none found in stored RSS records' if rss_data_available else 'RSS database unavailable')}. "
                        "Outcome label intentionally left unverified; these signals do not prove shipment disruption or absence of disruption."
                    )
                    save_auto_observation({
                        "observation_key": f"{shipment_id}:{day_iso}",
                        "observation_date": day_iso,
                        "shipment_id": shipment_id,
                        "route_segment": route_segment,
                        "distance_km": route_distance,
                        "temperature": weather["temperature"],
                        "rainfall": weather["rainfall"],
                        "wind_speed": weather["wind_speed"],
                        "severe_weather": severe_weather,
                        "festival_nearby": festival_flag,
                        "rss_disturbance": rss_flag,
                        "rss_severity": rss_severity,
                        "disturbance_type": ", ".join(disturbance_types) if disturbance_types else "None",
                        "evidence_notes": notes,
                    })
                    inserted += 1

                progress.progress((shipment_index + 1) / len(shipments_df))
                status_line.write(f"Processed shipment {shipment_id} ({shipment_index + 1}/{len(shipments_df)}).")

        st.session_state["ml_auto_collection_result"] = {
            "saved": inserted,
            "no_gps": skipped_no_gps,
            "off_route": skipped_off_route,
            "weather_failures": weather_failures,
            "gps_diagnostics": gps_diagnostics,
            "errors": errors[:10],
        }
        st.rerun()

    collection_result = st.session_state.get("ml_auto_collection_result")
    if collection_result:
        st.subheader("📊 Latest automatic collection result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Observations saved / updated", collection_result["saved"])
        c2.metric("Shipments skipped (no valid on-route GPS)", collection_result["no_gps"])
        c3.metric("GPS fixes rejected as off-route", collection_result["off_route"])
        if collection_result["weather_failures"]:
            st.warning(f"Historical weather lookup failed for {collection_result['weather_failures']} shipment/date combinations.")

        diagnostics = collection_result.get("gps_diagnostics", [])
        if diagnostics:
            st.markdown("**GPS validation breakdown by shipment**")
            st.dataframe(
                pd.DataFrame(diagnostics),
                use_container_width=True,
                hide_index=True
            )
            st.caption(
                "Within 2 km = close route match. 2–15 km = broad review band "
                "that is still included in collection. Over 15 km = excluded as "
                "off-route. The nearest and median distances help identify a "
                "wrong shipment route or a phone that was not travelling with the vehicle."
            )
            st.info(
                "The 15 km limit has deliberately not been increased: it is already "
                "a generous corridor. If most fixes are over 15 km, check the saved "
                "shipment coordinates and confirm that the Android phone was with "
                "the vehicle when those readings were recorded."
            )

        for message in collection_result["errors"]:
            st.caption(message)

    st.divider()
    st.subheader("📚 Automatically collected observations")
    auto_data = get_auto_observations()
    if not auto_data.empty:
        st.dataframe(auto_data, use_container_width=True, hide_index=True)
        st.caption(
            "These observations contain automatically collected features. Their outcome label is intentionally blank until independent evidence verifies it."
        )
    else:
        st.info("No automatic observations have been collected yet. Run automatic collection above.")

    st.divider()
    st.subheader("🧾 Review and verify automatic observations")
    st.write(
        "Review the shipment evidence for a date, then record the actual outcome. "
        "Only observations with an independently supported outcome are promoted into training data."
    )
    review_data = get_auto_observations()
    pending_data = review_data[review_data["collection_status"].fillna("Unverified") != "Verified"] if not review_data.empty else review_data
    if pending_data.empty:
        st.success("There are no unverified automatic observations waiting for review.")
    else:
        choices = {
            f"Observation #{int(row.observation_id)} | Shipment #{int(row.shipment_id)} | {row.observation_date} | {row.route_segment}": int(row.observation_id)
            for row in pending_data.itertuples()
        }
        selected_label = st.selectbox("Choose an observation to review", list(choices.keys()), key="ml_review_observation")
        selected_id = choices[selected_label]
        selected_row = pending_data[pending_data["observation_id"] == selected_id].iloc[0]
        st.dataframe(pd.DataFrame([selected_row]), use_container_width=True, hide_index=True)
        st.caption("Do not mark an observation as disrupted merely because it has severe weather, a festival, an RSS alert, or unusual GPS readings. Use independent shipment records, carrier confirmation, road authority notices, or other evidence of the actual outcome.")
        with st.form(f"verify_auto_observation_{selected_id}"):
            outcome = st.radio("Verified shipment outcome", ["Disruption occurred", "No disruption occurred"], horizontal=True)
            verification_source = st.text_input("Evidence source (required)", placeholder="Example: carrier incident report / delivery log reference")
            verification_notes = st.text_area("Evidence details and review notes", placeholder="Describe what the evidence confirms, including dates or report references.")
            st.markdown("**Confirm or complete the model features**")
            feature_defaults = {
                "distance_km": 0.0, "temperature": 0.0, "rainfall": 0.0,
                "wind_speed": 0.0, "severe_weather": 0, "festival_nearby": 0,
                "rss_disturbance": 0
            }
            feature_inputs = {}
            cols = st.columns(2)
            for idx, (field, default) in enumerate(feature_defaults.items()):
                raw = selected_row.get(field)
                value = default if pd.isna(raw) else raw
                with cols[idx % 2]:
                    if field in ("severe_weather", "festival_nearby", "rss_disturbance"):
                        feature_inputs[field] = int(st.selectbox(field.replace("_", " ").title(), [0, 1], index=int(value), key=f"verify_{selected_id}_{field}"))
                    else:
                        feature_inputs[field] = st.number_input(field.replace("_", " ").title(), value=float(value), min_value=-100.0 if field == "temperature" else 0.0, step=0.5, key=f"verify_{selected_id}_{field}")
            submitted = st.form_submit_button("✅ Verify and add to training dataset", type="primary", use_container_width=True)
            if submitted:
                try:
                    verify_auto_observation(
                        selected_id,
                        1 if outcome == "Disruption occurred" else 0,
                        verification_source,
                        verification_notes,
                        feature_inputs
                    )
                    st.success("Observation verified and promoted to the training dataset.")
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not verify this observation: {error}")

    st.divider()
    st.subheader("🧠 Verified records available for Random Forest training")
    verified_data = get_training_data()
    if verified_data is not None and not verified_data.empty:
        st.dataframe(verified_data, use_container_width=True, hide_index=True)
    else:
        st.info("No verified training labels are available yet. Automatic feature collection does not manufacture Yes/No outcomes.")

    if not verified_summary["ready"]:
        st.warning(
            "The model still needs at least 10 independently verified disruption records "
            "and 10 independently verified non-disruption records. Automatic weather, "
            "festival, RSS and GPS features alone cannot safely create those labels."
        )

# ==================================================
# DAY 4 - STEP 2
# RANDOM FOREST MODEL TRAINING PAGE
# ==================================================

elif page == "ML Model Training":

    st.header("🤖 Random Forest Model Training")
    st.write("Train and evaluate a separate experimental Random Forest model for the currently selected shipment.")

    if active_shipment is None or active_shipment_id is None:
        st.warning("Register and select a shipment before generating its ML dataset.")
    else:
        shipment_id = int(active_shipment_id)
        source_name = str(active_shipment.get("source", "Unknown"))
        destination_name = str(active_shipment.get("destination", "Unknown"))
        source_point = (float(active_shipment.get("source_latitude") or 0), float(active_shipment.get("source_longitude") or 0))
        destination_point = (float(active_shipment.get("destination_latitude") or 0), float(active_shipment.get("destination_longitude") or 0))
        route_distance_km = calculate_distance_km(source_point, destination_point) if all(source_point + destination_point) else 100.0
        if not math.isfinite(route_distance_km) or route_distance_km <= 0:
            route_distance_km = 100.0

        st.info(f"Active shipment: **{shipment_id}** | {source_name} → {destination_name} | Approx. straight-line distance: {route_distance_km:.1f} km")
        st.warning(
            "This is a shipment-specific synthetic demonstration dataset. Its records and labels are simulated, "
            "not observed shipment history or verified outcomes. The results must not guide real dispatch decisions."
        )

        synthetic_summary = get_synthetic_dataset_summary(shipment_id)
        if synthetic_summary["total_records"] == 0:
            with st.spinner(f"Creating the first synthetic dataset for Shipment {shipment_id}..."):
                synthetic_summary = generate_synthetic_dataset(1000, random_state=42, shipment_id=shipment_id, route_distance_km=route_distance_km)
            st.success(f"Created a separate synthetic dataset for Shipment {shipment_id}.")

        syn1, syn2, syn3 = st.columns(3)
        syn1.metric("Shipment Synthetic Records", synthetic_summary["total_records"])
        syn2.metric("Synthetic Disruptions", synthetic_summary["disrupted"])
        syn3.metric("Synthetic No Disruptions", synthetic_summary["not_disrupted"])

        synthetic_count = st.number_input(
            "Number of synthetic scenarios for this shipment", min_value=100, max_value=20000,
            value=1000, step=100, key=f"synthetic_record_count_{shipment_id}"
        )
        st.caption("Each shipment has its own saved dataset and model. Switching the active shipment loads that shipment's records; it does not overwrite another shipment's dataset.")

        if st.button("🧪 Regenerate this shipment's synthetic dataset", use_container_width=True, key=f"generate_synthetic_ml_data_{shipment_id}"):
            try:
                with st.spinner(f"Generating simulated scenarios for Shipment {shipment_id}..."):
                    generated_summary = generate_synthetic_dataset(int(synthetic_count), random_state=42, shipment_id=shipment_id, route_distance_km=route_distance_km)
                st.session_state.pop(f"synthetic_rf_training_result_{shipment_id}", None)
                st.success(f"Shipment {shipment_id}: generated {generated_summary['total_records']} records ({generated_summary['disrupted']} disruptions, {generated_summary['not_disrupted']} no disruptions). Retrain this shipment's model.")
                st.rerun()
            except Exception as error:
                st.error(f"Synthetic dataset generation failed: {error}")

        synthetic_df = get_synthetic_training_data(shipment_id)
        if not synthetic_df.empty:
            with st.expander(f"Preview Shipment {shipment_id} synthetic records"):
                st.dataframe(synthetic_df.head(20), use_container_width=True)

        if st.button("🌳 Train this shipment's Synthetic Random Forest", use_container_width=True, disabled=not synthetic_summary["ready"], key=f"train_synthetic_random_forest_{shipment_id}"):
            try:
                with st.spinner(f"Training Shipment {shipment_id}'s model..."):
                    synthetic_result = train_synthetic_random_forest_model(shipment_id=shipment_id)
                st.session_state[f"synthetic_rf_training_result_{shipment_id}"] = synthetic_result
                st.success(f"Synthetic model for Shipment {shipment_id} trained and saved separately.")
            except Exception as error:
                st.error(f"Synthetic model training failed: {error}")

        synthetic_result = st.session_state.get(f"synthetic_rf_training_result_{shipment_id}")
        if synthetic_result is None:
            model_status_for_shipment = get_random_forest_model_status(model_type="synthetic", shipment_id=shipment_id)
            if model_status_for_shipment.get("trained"):
                synthetic_result = model_status_for_shipment
                st.session_state[f"synthetic_rf_training_result_{shipment_id}"] = synthetic_result
        if synthetic_result:
            st.markdown(f"**Shipment {shipment_id} synthetic hold-out test results — demonstration only**")
            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("Accuracy", f"{synthetic_result['accuracy'] * 100:.1f}%")
            sm2.metric("Precision", f"{synthetic_result['precision'] * 100:.1f}%")
            sm3.metric("Recall", f"{synthetic_result['recall'] * 100:.1f}%")
            sm4.metric("F1-score", f"{synthetic_result['f1'] * 100:.1f}%")
            sm5.metric("Test rows", synthetic_result["test_records"])
            st.code(synthetic_result["classification_report"])
            synthetic_matrix = pd.DataFrame(synthetic_result["confusion_matrix"], index=["Actual: No disruption", "Actual: Disruption"], columns=["Predicted: No disruption", "Predicted: Disruption"])
            st.dataframe(synthetic_matrix, use_container_width=True)
            st.caption("These metrics measure agreement with this shipment's synthetic simulation labels only. They are not real-world accuracy.")
            importance_df = pd.DataFrame(list(synthetic_result["feature_importance"].items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
            st.markdown("**Shipment-specific synthetic model feature importance**")
            st.dataframe(importance_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Verified real-world training dataset")
    st.caption(
        "The section below remains separate and continues to require independently verified shipment outcomes."
    )

    # --------------------------------------
    # DATASET READINESS
    # --------------------------------------

    summary = get_dataset_summary()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Training Records",
        summary["total_records"]
    )

    col2.metric(
        "Verified Disruptions",
        summary["disrupted"]
    )

    col3.metric(
        "No Disruptions",
        summary["not_disrupted"]
    )

    st.divider()

    if summary["ready"]:

        st.success(
            "Dataset meets the minimum record-count requirement."
        )

    else:

        st.warning(
            "The dataset is not ready. Add at least 10 verified "
            "disruptions and 10 verified non-disruptions."
        )

    # --------------------------------------
    # TRAINING BUTTON
    # --------------------------------------

    st.subheader("🌳 Train Random Forest Classifier")

    if st.button(
        "🚀 Train Random Forest Model",
        use_container_width=True,
        disabled=not summary["ready"]
    ):

        try:

            with st.spinner(
                "Training and evaluating Random Forest model..."
            ):

                result = train_random_forest_model()

            st.session_state["rf_training_result"] = result

            st.success(
                "Random Forest model trained and saved successfully!"
            )

        except Exception as error:

            st.error(
                f"Model training failed: {error}"
            )

    st.divider()

    # --------------------------------------
    # TRAINING RESULTS
    # --------------------------------------

    result = st.session_state.get(
        "rf_training_result"
    )

    if result:

        st.subheader("📊 Model Evaluation Results")

        metric1, metric2 = st.columns(2)

        metric1.metric(
            "Test Accuracy",
            f"{result['accuracy'] * 100:.2f}%"
        )

        metric2.metric(
            "Balanced Accuracy",
            f"{result['balanced_accuracy'] * 100:.2f}%"
        )

        st.caption(
            "These scores are calculated on the held-out test split. "
            "They are preliminary estimates and may be unreliable "
            "with a small dataset."
        )

        st.subheader("📋 Classification Report")

        st.code(
            result["classification_report"]
        )

        st.subheader("Confusion Matrix")

        matrix = pd.DataFrame(
            result["confusion_matrix"],
            index=["Actual: No Disruption", "Actual: Disruption"],
            columns=["Predicted: No Disruption", "Predicted: Disruption"]
        )

        st.dataframe(
            matrix,
            use_container_width=True
        )

        st.caption(
            "Rows represent actual labels; columns represent model predictions."
        )

        st.write(
            f"Training records: {result['training_records']}"
        )

        st.write(
            f"Model saved at: {result['model_path']}"
        )

        st.write(
            f"Training time: {result['trained_at']}"
        )

    # --------------------------------------
    # SAVED MODEL STATUS
    # --------------------------------------

    st.divider()

    st.subheader("💾 Saved Random Forest Model")

    model_status = get_random_forest_model_status()

    if model_status["trained"]:

        st.success("A saved Random Forest model is available.")

        st.write(
            f"Training records: {model_status['training_records']}"
        )

        st.write(
            f"Last trained: {model_status['trained_at']}"
        )

        st.write(
            f"Saved model accuracy: "
            f"{model_status['accuracy'] * 100:.2f}%"
        )

    else:

        st.info(
            model_status.get(
                "message",
                "No trained Random Forest model is available yet."
            )
        )


# ==================================================
# DAY 4 - STEP 3
# SEVEN-DAY ML PREDICTION DASHBOARD
# ==================================================

elif page == "ML Prediction Dashboard":

    st.header("🔮 Seven-Day Shipment Disturbance Prediction")

    st.write(
        "Use the trained Random Forest model, seven-day weather "
        "forecasts, and reviewed festival and RSS information "
        "to estimate disturbance risk at sampled route checkpoints."
    )

    st.warning(
        "These are preliminary model estimates, not confirmed road "
        "closures or guaranteed shipment outcomes. Risk thresholds "
        "are development thresholds and probabilities are not calibrated."
    )

    # ----------------------------------------------
    # CHOOSE VERIFIED OR EXPERIMENTAL MODEL
    # ----------------------------------------------

    prediction_model_choice = st.radio(
        "Choose prediction model",
        ["Verified real-data model", "Synthetic-data experimental model"],
        horizontal=True,
        key="ml_prediction_model_choice"
    )
    prediction_model_type = (
        "synthetic" if prediction_model_choice == "Synthetic-data experimental model" else "verified"
    )
    if prediction_model_type == "synthetic":
        st.warning(
            "Experimental mode: this model was trained on simulated labels. "
            "Its output is for software demonstration only and must not guide real shipment decisions."
        )
    else:
        st.info(
            "Verified mode uses only independently verified shipment outcome records."
        )

    # ----------------------------------------------
    # CHECK SAVED MODEL
    # ----------------------------------------------

    model_status = get_random_forest_model_status(model_type=prediction_model_type, shipment_id=active_shipment_id if prediction_model_type == "synthetic" else None)

    if not model_status.get("trained"):

        st.error(
            model_status.get(
                "message",
                "Train the Random Forest model before generating predictions."
            )
        )

    else:

        st.success(
            "A saved Random Forest model is available."
        )

        st.caption(
            f"Model training records: "
            f"{model_status.get('training_records', 'Unknown')} | "
            f"Last trained: {model_status.get('trained_at', 'Unknown')}"
        )

        # ------------------------------------------
        # CHECK TRAINING FEATURE QUALITY
        # ------------------------------------------

        training_data = (
            get_synthetic_training_data(active_shipment_id)
            if prediction_model_type == "synthetic"
            else get_training_data()
        )

        if training_data.empty:

            st.error(
                ("No synthetic records are available. Generate the synthetic dataset first."
                 if prediction_model_type == "synthetic"
                 else "No verified training records are available. Add independently verified observations first.")
            )

        else:

            distance_values = pd.to_numeric(
                training_data["distance_km"],
                errors="coerce"
            ).fillna(0)

            if not (distance_values > 0).any():

                st.error(
                    "Your training records currently contain no positive "
                    "route-segment distances. Update the historical records, "
                    "then retrain the model before using this dashboard."
                )

            else:

                # ----------------------------------
                # SELECT SHIPMENT
                # ----------------------------------

                if active_shipment is None:

                    st.info(
                        "Register a shipment and select it in the sidebar "
                        "before generating predictions."
                    )

                else:

                    # The prediction dashboard uses the same active shipment
                    # selected once in the sidebar.
                    shipment = active_shipment
                    shipment_id = int(active_shipment_id)

                    st.subheader(
                        f"📦 Active Shipment {shipment_id}"
                    )

                    st.write(
                        f"**Source:** {shipment['source']}"
                    )

                    st.write(
                        f"**Destination:** {shipment['destination']}"
                    )

                    # ----------------------------------
                    # LOAD SHIPMENT COORDINATES
                    # ----------------------------------

                    try:

                        source_coordinates = (
                            float(shipment["source_latitude"]),
                            float(shipment["source_longitude"])
                        )

                        destination_coordinates = (
                            float(shipment["destination_latitude"]),
                            float(shipment["destination_longitude"])
                        )

                    except (KeyError, TypeError, ValueError):

                        st.error(
                            "This shipment does not have valid source and "
                            "destination coordinates. Correct its coordinates "
                            "before generating predictions."
                        )

                        source_coordinates = None
                        destination_coordinates = None

                    if (
                        source_coordinates is not None
                        and destination_coordinates is not None
                    ):

                        # ----------------------------------
                        # GENERATE ROAD ROUTE
                        # ----------------------------------

                        try:

                            with st.spinner(
                                "Loading the shipment road route..."
                            ):

                                route = load_cached_route(
                                    source_coordinates,
                                    destination_coordinates
                                )

                            if not route or not route.get("points"):

                                raise ValueError(
                                    "The routing service returned no route."
                                )

                            route_points = route["points"]

                            route_distance_km = float(
                                route.get("distance_km", 0)
                            )

                            if route_distance_km <= 0:

                                raise ValueError(
                                    "The road route has no valid distance."
                                )

                        except Exception as error:

                            st.error(
                                f"Could not load the shipment route: {error}"
                            )

                            route = None

                        if route:

                            st.metric(
                                "Planned Road Distance",
                                f"{route_distance_km:.2f} km"
                            )

                            # ----------------------------------
                            # SELECT ROUTE WEATHER LOCATIONS
                            # ----------------------------------

                            route_locations = (
                                select_route_weather_locations(
                                    route,
                                    source_coordinates,
                                    destination_coordinates,
                                    max_points=5
                                )
                            )

                            # If no intermediate checkpoints are available,
                            # use the shipment endpoints instead.

                            if not route_locations:

                                route_locations = [
                                    source_coordinates,
                                    destination_coordinates
                                ]

                                route_location_names = [
                                    str(shipment["source"]),
                                    str(shipment["destination"])
                                ]

                            else:

                                route_location_names = []

                                for index, coordinates in enumerate(
                                    route_locations,
                                    start=1
                                ):

                                    place_name = load_location_name(
                                        coordinates[0],
                                        coordinates[1]
                                    )

                                    route_location_names.append(
                                        place_name
                                        or f"Route Checkpoint {index}"
                                    )

                            # ----------------------------------
                            # LOAD SEVEN-DAY WEATHER
                            # ----------------------------------

                            route_weather_data = []

                            weather_error = None

                            with st.spinner(
                                "Retrieving seven-day forecasts "
                                "for the shipment route..."
                            ):

                                for index, coordinates in enumerate(
                                    route_locations
                                ):

                                    try:

                                        forecast = load_weather(
                                            coordinates[0],
                                            coordinates[1]
                                        )

                                        if not forecast or len(forecast) < 7:

                                            raise ValueError(
                                                "Fewer than seven forecast "
                                                "days were returned."
                                            )

                                        route_weather_data.append({

                                            "name": route_location_names[index],

                                            "coordinates": coordinates,

                                            "forecast": forecast[:7]

                                        })

                                    except Exception as error:

                                        weather_error = (
                                            f"{route_location_names[index]}: "
                                            f"{error}"
                                        )

                                        break

                            if weather_error:

                                st.error(
                                    "Could not load all route forecasts: "
                                    + weather_error
                                )

                            elif route_weather_data:

                                # ----------------------------------
                                # CHECK FORECAST DATE CONSISTENCY
                                # ----------------------------------

                                forecast_dates = [
                                    str(day.get("date", ""))[:10]
                                    for day in route_weather_data[0]["forecast"]
                                ]

                                dates_valid = (
                                    len(forecast_dates) == 7
                                    and all(forecast_dates)
                                    and len(set(forecast_dates)) == 7
                                )

                                for location in route_weather_data[1:]:

                                    location_dates = [
                                        str(day.get("date", ""))[:10]
                                        for day in location["forecast"]
                                    ]

                                    if location_dates != forecast_dates:

                                        dates_valid = False

                                if not dates_valid:

                                    st.error(
                                        "The route locations did not return "
                                        "matching seven-day forecast dates. "
                                        "Predictions have been stopped."
                                    )

                                else:

                                    # ----------------------------------
                                    # AUTOMATIC FESTIVAL + RSS COLLECTION
                                    # ----------------------------------

                                    first_date = forecast_dates[0]
                                    last_date = forecast_dates[-1]
                                    daily_auto_conditions = {
                                        day: {
                                            "festival_nearby": "Unknown",
                                            "rss_disturbance": "Unknown",
                                            "festival_note": "Festival calendar could not be checked.",
                                            "rss_note": "RSS feeds could not be checked.",
                                        }
                                        for day in forecast_dates
                                    }

                                    st.divider()
                                    st.subheader("🎉 Automatic Festival and Holiday Check")
                                    festival_lookup_ok = False
                                    festival_matches = []

                                    try:
                                        festival_events = get_festival_events() or []
                                        festival_lookup_ok = True
                                        route_festival_radius_km = 30.0

                                        for event in festival_events:
                                            event_start = str(event.get("start_date") or "")[:10]
                                            event_end = str(event.get("end_date") or event_start)[:10]
                                            if not event_start or not event_end:
                                                continue
                                            if event_start > last_date or event_end < first_date:
                                                continue

                                            locality = str(
                                                event.get("affected_location") or ""
                                            ).strip()
                                            state_name = str(event.get("state") or "").strip()
                                            if not locality or locality.lower() in {
                                                "india", "all india", "statewide", "unknown",
                                                "location not identified"
                                            }:
                                                continue

                                            # Geocode a specific event locality and compare it
                                            # with the actual OSRM route, not just the shipment state.
                                            search_name = ", ".join(
                                                part for part in [locality, state_name, "India"] if part
                                            )
                                            try:
                                                festival_geo = load_geocoded_location(search_name)
                                                if not festival_geo:
                                                    continue
                                                route_distance = distance_from_location_to_route_km(
                                                    festival_geo, route_points
                                                )
                                            except Exception:
                                                continue

                                            if route_distance is None or route_distance > route_festival_radius_km:
                                                continue

                                            festival_matches.append({
                                                "Event": event.get("name", "Festival / holiday"),
                                                "Start": event_start,
                                                "End": event_end,
                                                "Location": locality,
                                                "Distance from route (km)": round(float(route_distance), 1),
                                                "Impact": event.get("impact_level", "Unknown"),
                                            })

                                            for forecast_date in forecast_dates:
                                                if event_start <= forecast_date <= event_end:
                                                    daily_auto_conditions[forecast_date]["festival_nearby"] = "Yes"
                                                    daily_auto_conditions[forecast_date]["festival_note"] = (
                                                        f"Calendar event {event.get('name', 'Festival')} "
                                                        f"is listed within {route_distance:.1f} km of the route."
                                                    )

                                        if festival_lookup_ok:
                                            for forecast_date in forecast_dates:
                                                if daily_auto_conditions[forecast_date]["festival_nearby"] != "Yes":
                                                    daily_auto_conditions[forecast_date]["festival_nearby"] = "No"
                                                    daily_auto_conditions[forecast_date]["festival_note"] = (
                                                        "No date-and-location match was found in the saved festival calendar; "
                                                        "the calendar may be incomplete."
                                                    )

                                    except Exception as error:
                                        st.warning(f"Could not automatically check the festival calendar: {error}")

                                    if festival_matches:
                                        st.dataframe(
                                            pd.DataFrame(festival_matches),
                                            use_container_width=True,
                                            hide_index=True
                                        )
                                        st.caption(
                                            "Events are matched by date and geocoded locality within 30 km of the planned route. "
                                            "This indicates a nearby calendar event, not a confirmed traffic impact."
                                        )
                                    elif festival_lookup_ok:
                                        st.info(
                                            "No saved festival-calendar entry matched the route and forecast dates. "
                                            "Automatic status is No for calendar matches only; it does not prove that no local event exists."
                                        )

                                    st.divider()
                                    st.subheader("📢 Automatic RSS Disturbance Check")
                                    if st.button(
                                        "🔄 Refresh RSS feeds now",
                                        key=f"ml_refresh_rss_{shipment_id}",
                                        help="Bypass the 15-minute cache and fetch the configured public RSS feeds again."
                                    ):
                                        refresh_ml_rss_events.clear()

                                    rss_scan_ok = False
                                    rss_scan_result = None
                                    rss_matches = []
                                    rss_date_unknown_matches = []

                                    try:
                                        with st.spinner("Refreshing public RSS feeds (cached for 15 minutes)..."):
                                            rss_scan_result = refresh_ml_rss_events()
                                        rss_scan_ok = True
                                    except Exception as error:
                                        st.warning(f"Automatic RSS refresh failed: {error}")

                                    try:
                                        rss_events = get_disturbance_events(limit=1000) or []
                                        route_rss_matches = find_shipment_disturbance_matches(
                                            route_points, rss_events, radius_km=DISTURBANCE_MATCH_RADIUS_KM
                                        )
                                        for event in route_rss_matches:
                                            event_date = str(event.get("event_date") or "")[:10]
                                            if event_date in forecast_dates:
                                                rss_matches.append(event)
                                            elif not event_date:
                                                rss_date_unknown_matches.append(event)
                                    except Exception as error:
                                        rss_scan_ok = False
                                        st.warning(f"Could not match saved RSS announcements to this route: {error}")

                                    if rss_scan_result:
                                        scanned = rss_scan_result.get(
                                            "articles_scanned", rss_scan_result.get("total_scanned", 0)
                                        )
                                        failed = rss_scan_result.get("failed_feeds", 0)
                                        if isinstance(failed, list):
                                            failed_count = len(failed)
                                        else:
                                            failed_count = int(failed or 0)
                                        st.caption(
                                            f"Automatic feed refresh: {scanned} entries scanned; "
                                            f"{failed_count} feed(s) reported failure. Saved announcements are still unverified."
                                        )

                                    if rss_matches:
                                        rss_display = [{
                                            "Announcement": item.get("title"),
                                            "Event date": item.get("event_date"),
                                            "Location": item.get("place") or item.get("display_name"),
                                            "Distance from route (km)": round(item.get("distance_km", 0), 1),
                                            "Severity": item.get("severity"),
                                            "Review status": item.get("status", "Unverified"),
                                        } for item in rss_matches]
                                        st.dataframe(pd.DataFrame(rss_display), use_container_width=True, hide_index=True)
                                    elif rss_date_unknown_matches:
                                        st.warning(
                                            f"{len(rss_date_unknown_matches)} route-near announcement(s) have no usable event date. "
                                            "Their forecast-day relevance remains Unknown."
                                        )
                                    elif rss_scan_ok:
                                        st.info(
                                            "No dated, high-confidence RSS announcement matched this route and forecast window. "
                                            "Automatic No means no matching announcement was found in the scanned feeds; it does not prove that roads are clear."
                                        )

                                    for forecast_date in forecast_dates:
                                        dated_matches = [
                                            item for item in rss_matches
                                            if str(item.get("event_date") or "")[:10] == forecast_date
                                        ]
                                        if dated_matches:
                                            daily_auto_conditions[forecast_date]["rss_disturbance"] = "Yes"
                                            daily_auto_conditions[forecast_date]["rss_note"] = (
                                                f"{len(dated_matches)} dated, high-confidence announcement(s) were matched near the route; all remain unverified."
                                            )
                                        elif rss_scan_ok and not rss_date_unknown_matches:
                                            daily_auto_conditions[forecast_date]["rss_disturbance"] = "No"
                                            daily_auto_conditions[forecast_date]["rss_note"] = (
                                                "No matching dated route announcement was found in the scanned RSS records; this is not proof of no disruption."
                                            )

                                    st.caption(
                                        "RSS matches are unverified reports. No match is not proof of an open road. "
                                        "Festival and RSS statuses can be manually overridden below."
                                    )

                                    # ----------------------------------
                                    # DAILY EVENT CONFIRMATION
                                    # ----------------------------------

                                    st.divider()

                                    st.subheader(
                                        "📝 Confirm Route Conditions for Each Day"
                                    )

                                    st.write(
                                        "Weather is collected automatically from the seven-day forecast. "
                                        "Festival and RSS values are pre-filled from the saved calendar and a cached feed scan; "
                                        "you can override either suggestion. Keep Unknown whenever the source could not be checked "
                                        "or the evidence is ambiguous."
                                    )

                                    daily_conditions = {}

                                    for day_index, forecast_date in enumerate(
                                        forecast_dates
                                    ):

                                        st.markdown(
                                            f"**{forecast_date}**"
                                        )

                                        condition_col1, condition_col2 = (
                                            st.columns(2)
                                        )

                                        with condition_col1:

                                            festival_options = ["Unknown", "No", "Yes"]
                                            festival_auto = daily_auto_conditions[forecast_date]["festival_nearby"]
                                            festival_choice = st.selectbox(
                                                "Festival / holiday near route (automatic suggestion)",
                                                festival_options,
                                                index=(festival_options.index(festival_auto) if festival_auto in festival_options else 0),
                                                key=f"ml_festival_{shipment_id}_{day_index}_{forecast_date}"
                                            )
                                            st.caption(daily_auto_conditions[forecast_date]["festival_note"])

                                        with condition_col2:

                                            rss_options = ["Unknown", "No", "Yes"]
                                            rss_auto = daily_auto_conditions[forecast_date]["rss_disturbance"]
                                            rss_choice = st.selectbox(
                                                "Route-related RSS announcement (automatic suggestion)",
                                                rss_options,
                                                index=(rss_options.index(rss_auto) if rss_auto in rss_options else 0),
                                                key=f"ml_rss_{shipment_id}_{day_index}_{forecast_date}"
                                            )
                                            st.caption(daily_auto_conditions[forecast_date]["rss_note"])

                                        daily_conditions[forecast_date] = {

                                            "festival_nearby": festival_choice,

                                            "rss_disturbance": rss_choice

                                        }

                                    # ----------------------------------
                                    # GENERATE PREDICTIONS
                                    # ----------------------------------

                                    st.divider()

                                    if st.button(
                                        "🔮 Generate Seven-Day Risk Predictions",
                                        use_container_width=True,
                                        key="generate_ml_route_predictions"
                                    ):

                                        unknown_conditions = [

                                            day for day, values
                                            in daily_conditions.items()

                                            if (
                                                values["festival_nearby"] == "Unknown"
                                                or values["rss_disturbance"] == "Unknown"
                                            )

                                        ]

                                        if unknown_conditions:

                                            st.error(
                                                "Some event conditions remain Unknown because the source could not be checked reliably. "
                                                "Review the source or manually confirm the condition before generating predictions. "
                                                "Unknown values are never converted to No."
                                            )

                                        else:

                                            prediction_rows = []

                                            try:

                                                with st.spinner(
                                                    "Calculating route checkpoint "
                                                    "risk predictions..."
                                                ):

                                                    # Approximate distance per
                                                    # sampled route segment.
                                                    segment_distance_km = (
                                                        route_distance_km
                                                        / (
                                                            len(route_weather_data) + 1
                                                        )
                                                    )

                                                    for location in route_weather_data:

                                                        forecast_records = []

                                                        for day in location["forecast"]:

                                                            forecast_date = str(
                                                                day["date"]
                                                            )[:10]

                                                            temperature_min = float(
                                                                day["temperature_min"]
                                                            )

                                                            temperature_max = float(
                                                                day["temperature_max"]
                                                            )

                                                            # Use mean daily
                                                            # temperature to match
                                                            # the dataset definition.
                                                            mean_temperature = (
                                                                temperature_min
                                                                + temperature_max
                                                            ) / 2

                                                            conditions = (
                                                                daily_conditions[
                                                                    forecast_date
                                                                ]
                                                            )

                                                            forecast_records.append({

                                                                "date": forecast_date,

                                                                "distance_km": (
                                                                    segment_distance_km
                                                                ),

                                                                "temperature": (
                                                                    mean_temperature
                                                                ),

                                                                "rainfall": float(
                                                                    day["rainfall"]
                                                                ),

                                                                "wind_speed": float(
                                                                    day["wind_speed"]
                                                                ),

                                                                "severe_weather": int(
                                                                    day.get("severity")
                                                                    == "Severe"
                                                                ),

                                                                "festival_nearby": int(
                                                                    conditions[
                                                                        "festival_nearby"
                                                                    ] == "Yes"
                                                                ),

                                                                "rss_disturbance": int(
                                                                    conditions[
                                                                        "rss_disturbance"
                                                                    ] == "Yes"
                                                                )

                                                            })

                                                        predictions = (
                                                            predict_seven_day_risk(
                                                                forecast_records,
                                                                model_type=prediction_model_type
                                                            )
                                                        )

                                                        for day_index, prediction in enumerate(
                                                            predictions
                                                        ):

                                                            weather_day = (
                                                                location["forecast"][
                                                                    day_index
                                                                ]
                                                            )

                                                            prediction_rows.append({

                                                                "Shipment ID": shipment_id,

                                                                "Date": prediction["date"],

                                                                "Route Checkpoint": (
                                                                    location["name"]
                                                                ),

                                                                "Segment Distance (km)": round(
                                                                    segment_distance_km,
                                                                    2
                                                                ),

                                                                "Temperature (°C)": round(
                                                                    forecast_records[
                                                                        day_index
                                                                    ]["temperature"],
                                                                    1
                                                                ),

                                                                "Rainfall (mm)": round(
                                                                    float(
                                                                        weather_day["rainfall"]
                                                                    ),
                                                                    1
                                                                ),

                                                                "Wind Speed (km/h)": round(
                                                                    float(
                                                                        weather_day["wind_speed"]
                                                                    ),
                                                                    1
                                                                ),

                                                                "Risk (%)": prediction[
                                                                    "risk_percentage"
                                                                ],

                                                                "Risk Level": prediction[
                                                                    "risk_level"
                                                                ],

                                                                "Latitude": location[
                                                                    "coordinates"
                                                                ][0],

                                                                "Longitude": location[
                                                                    "coordinates"
                                                                ][1]

                                                            })

                                                st.session_state[
                                                    "ml_route_prediction_rows"
                                                ] = prediction_rows

                                                st.session_state[
                                                    "ml_route_prediction_shipment"
                                                ] = shipment_id

                                                st.success(
                                                    "Seven-day route checkpoint "
                                                    "predictions generated successfully."
                                                )

                                            except Exception as error:

                                                st.error(
                                                    f"Prediction failed: {error}"
                                                )

                                    # ----------------------------------
                                    # DISPLAY SAVED PREDICTIONS
                                    # ----------------------------------

                                    saved_predictions = st.session_state.get(
                                        "ml_route_prediction_rows",
                                        []
                                    )

                                    saved_prediction_shipment = (
                                        st.session_state.get(
                                            "ml_route_prediction_shipment"
                                        )
                                    )

                                    if (
                                        saved_predictions
                                        and saved_prediction_shipment == shipment_id
                                    ):

                                        prediction_df = pd.DataFrame(
                                            saved_predictions
                                        )

                                        st.divider()

                                        st.subheader(
                                            "📊 Seven-Day Risk Results"
                                        )

                                        selected_result_date = st.selectbox(
                                            "Select forecast date",
                                            forecast_dates,
                                            key=(
                                                f"ml_result_date_{shipment_id}"
                                            )
                                        )

                                        selected_day_df = prediction_df[
                                            prediction_df["Date"]
                                            == selected_result_date
                                        ].copy()

                                        high_count = int(
                                            (
                                                selected_day_df["Risk Level"]
                                                == "High"
                                            ).sum()
                                        )

                                        medium_count = int(
                                            (
                                                selected_day_df["Risk Level"]
                                                == "Medium"
                                            ).sum()
                                        )

                                        low_count = int(
                                            (
                                                selected_day_df["Risk Level"]
                                                == "Low"
                                            ).sum()
                                        )

                                        metric1, metric2, metric3 = (
                                            st.columns(3)
                                        )

                                        metric1.metric(
                                            "High-Risk Checkpoints",
                                            high_count
                                        )

                                        metric2.metric(
                                            "Medium-Risk Checkpoints",
                                            medium_count
                                        )

                                        metric3.metric(
                                            "Low-Risk Checkpoints",
                                            low_count
                                        )

                                        st.dataframe(
                                            selected_day_df[
                                                [
                                                    "Date",
                                                    "Route Checkpoint",
                                                    "Segment Distance (km)",
                                                    "Temperature (°C)",
                                                    "Rainfall (mm)",
                                                    "Wind Speed (km/h)",
                                                    "Risk (%)",
                                                    "Risk Level"
                                                ]
                                            ],
                                            use_container_width=True,
                                            hide_index=True
                                        )

                                        st.subheader(
                                            "🗺️ Route Checkpoint Risk Map"
                                        )

                                        risk_map = folium.Map(
                                            location=source_coordinates,
                                            zoom_start=7,
                                            control_scale=True
                                        )

                                        folium.PolyLine(
                                            route_points,
                                            color="blue",
                                            weight=4,
                                            opacity=0.8,
                                            tooltip="Planned shipment route"
                                        ).add_to(risk_map)

                                        risk_colors = {
                                            "High": "red",
                                            "Medium": "orange",
                                            "Low": "green"
                                        }

                                        for _, result_row in (
                                            selected_day_df.iterrows()
                                        ):

                                            risk_level = result_row[
                                                "Risk Level"
                                            ]

                                            marker_color = risk_colors.get(
                                                risk_level,
                                                "gray"
                                            )

                                            popup_text = (
                                                f"Checkpoint: "
                                                f"{result_row['Route Checkpoint']}<br>"
                                                f"Date: {result_row['Date']}<br>"
                                                f"Risk: {result_row['Risk (%)']}%<br>"
                                                f"Level: {risk_level}"
                                            )

                                            folium.CircleMarker(

                                                location=[
                                                    result_row["Latitude"],
                                                    result_row["Longitude"]
                                                ],

                                                radius=9,

                                                color=marker_color,

                                                fill=True,

                                                fill_color=marker_color,

                                                fill_opacity=0.85,

                                                popup=folium.Popup(
                                                    popup_text,
                                                    max_width=300
                                                )

                                            ).add_to(risk_map)

                                        st_folium(
                                            risk_map,
                                            use_container_width=True,
                                            height=500,
                                            key=(
                                                f"ml_risk_map_{shipment_id}_"
                                                f"{selected_result_date}"
                                            )
                                        )

                                        st.subheader(
                                            "📅 Full Seven-Day Prediction Table"
                                        )

                                        st.dataframe(
                                            prediction_df[
                                                [
                                                    "Date",
                                                    "Route Checkpoint",
                                                    "Risk (%)",
                                                    "Risk Level"
                                                ]
                                            ],
                                            use_container_width=True,
                                            hide_index=True
                                        )

                                        st.caption(
                                            "Each marker represents a sampled "
                                            "weather checkpoint. Segment distances "
                                            "are approximate equal divisions of "
                                            "the OSRM route, not surveyed road "
                                            "segment boundaries. A checkpoint risk "
                                            "does not establish a road closure."
                                        )


# ==================================================
# 17. FOOTER
# ==================================================

st.sidebar.divider()

st.sidebar.caption(
    "Shipment Management System | Day 1–Day 5 + Day 4"
)

st.sidebar.caption(
    "OpenStreetMap | OSRM | SQLite | Automatic GPS | "
    "Open-Meteo | Festival Monitoring | RSS Monitoring | "
    "Route Disturbance Matching | Random Forest ML | "
    "Seven-Day Risk Prediction"
)
