import streamlit as st
import pandas as pd
import folium
import sqlite3
import requests
import math
import re
from html import escape

from pathlib import Path
from datetime import datetime, date, timedelta

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
        return []

    matches = []

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
            continue
        if str(event.get("location_confidence", "")).lower() != "high":
            continue

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
                geo = geocode_disturbance_locality(
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

MAX_GPS_ACCURACY = 100.0


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
        "Disturbance Monitoring"
    ]
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

        st.info("Register shipments to check potential route disturbances.")

    else:

        if st.button(
            "🔍 Check All Shipment Routes",
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
                "Checking shipment routes against saved RSS announcements..."
            ):

                for _, shipment_row in shipments.iterrows():

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

    st.header("📍 Automatic Shipment GPS Tracking")

    shipments = get_shipments()

    if shipments.empty:

        st.warning(
            "Please register a shipment before GPS tracking."
        )

    else:

        shipment_options = {}

        for _, row in shipments.iterrows():

            label = (
                f"Shipment {row['shipment_id']} | "
                f"{row['source']} → {row['destination']}"
            )

            shipment_options[label] = int(
                row["shipment_id"]
            )

        selected_shipment = st.selectbox(
            "Select Shipment",
            list(shipment_options.keys())
        )

        shipment_id = shipment_options[selected_shipment]

        shipment = shipments[
            shipments["shipment_id"] == shipment_id
        ].iloc[0]

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

                    route = generate_route(
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
                    st.success(
                        "No geocoded RSS disturbance locations were matched "
                        "within 15 km of this route. This does not guarantee "
                        "that the route is disruption-free."
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
                        "Automatic GPS tracking is ON."
                    )

                    st.caption(
                        "Keep this browser tab open and allow "
                        "location permission."
                    )

                else:

                    st.warning(
                        "Automatic GPS tracking is currently OFF."
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
                                        + "Accuracy must be 100 metres "
                                        "or better. Waiting for a better "
                                        "GPS reading."
                                    )

                                    st.session_state[last_processed_key] = (
                                        gps_timestamp
                                    )

                                else:

                                    saved = save_gps_location(
                                        shipment_id,
                                        current_lat,
                                        current_lon,
                                        accuracy
                                    )

                                    st.session_state[last_processed_key] = (
                                        gps_timestamp
                                    )

                                    if saved:

                                        st.session_state[last_saved_key] = (
                                            gps_timestamp
                                        )

                                        st.success(
                                            "Accurate GPS location saved "
                                            "automatically!"
                                        )

                                    else:

                                        st.warning(
                                            f"GPS reading rejected. "
                                            f"Accuracy is {accuracy:.1f} "
                                            "metres. Accuracy must be "
                                            "100 metres or better."
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

                    if remaining_distance is not None:

                        total_distance = float(
                            route["distance_km"]
                        )

                        total_duration = float(
                            route["duration_minutes"]
                        )

                        if total_distance > 0:

                            remaining_ratio = min(
                                1.0,
                                remaining_distance / total_distance
                            )

                            remaining_minutes = (
                                total_duration * remaining_ratio
                            )

                        else:

                            remaining_minutes = 0.0

                        if remaining_distance < 0.5:

                            remaining_distance = 0.0
                            remaining_minutes = 0.0

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

                            if total_distance > 0:

                                progress = (
                                    1 - remaining_distance / total_distance
                                ) * 100

                                progress = max(
                                    0,
                                    min(100, progress)
                                )

                            else:

                                progress = 100

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

                else:

                    st.info(
                        "No GPS location available yet. "
                        "Start GPS tracking to receive the first location."
                    )

                st.divider()

                st.subheader("🗺️ Live Shipment Map")

                if latest:

                    map_center = (
                        float(latest[0]),
                        float(latest[1])
                    )

                else:

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

                if latest:

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

                if len(history) > 1:

                    history_points = [
                        (row[0], row[1])
                        for row in history
                    ]

                    folium.PolyLine(
                        history_points,
                        color="purple",
                        weight=3,
                        opacity=0.7,
                        tooltip="GPS Tracking History"
                    ).add_to(shipment_map)

                # ----------------------------------------------
                # FIT MAP TO ROUTE, GPS AND DISTURBANCE MARKERS
                # ----------------------------------------------
                map_bounds = list(route["points"])

                if latest:
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
                    "Purple = Current truck location | "
                    "Red warning label = Potential unverified disturbance."
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

elif page == "Weather Monitoring":

    st.header("🌦️ Real-Time Weather Monitoring")

    st.write(
        "Seven-day weather forecasts for the shipment source, destination, "
        "and selected intermediate points along its planned OSRM road route."
    )

    shipments = get_shipments()

    if shipments.empty:

        st.warning(
            "Please register a shipment before weather monitoring."
        )

    else:

        shipment_options = {}

        for _, row in shipments.iterrows():

            label = (
                f"Shipment {row['shipment_id']} | "
                f"{row['source']} → {row['destination']}"
            )

            shipment_options[label] = int(
                row["shipment_id"]
            )

        selected_shipment = st.selectbox(
            "Select Shipment",
            list(shipment_options.keys()),
            key="weather_shipment_select"
        )

        shipment_id = shipment_options[selected_shipment]

        shipment = shipments[
            shipments["shipment_id"] == shipment_id
        ].iloc[0]

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
# 17. FOOTER
# ==================================================

st.sidebar.divider()

st.sidebar.caption(
    "Shipment Management System | Day 1 + Day 2 + Day 3"
)

st.sidebar.caption(
    "OpenStreetMap | OSRM | SQLite | Automatic GPS | "
    "Open-Meteo | Festival Monitoring | RSS Monitoring | Route Disturbance Matching"
)
