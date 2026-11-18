"""
Day 5 - Intelligent Routing Engine
Shipment Management System

Contents
--------
Step 1: OSRM route candidates (distance, duration, coordinates, shortest route)
Step 2: Delay prediction engine (rule-based baseline)
Step 3: Revised ETA calculation
Step 4: Shortest vs fastest route comparison
Step 5: Disturbance-aware route risk engine
Step 6: Intelligent multi-criteria route ranking + full pipeline
"""

import math
from datetime import datetime, timedelta, timezone  # noqa: F401 (timezone re-exported for callers)
from typing import Optional  # noqa: F401

import requests


OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving"

REQUEST_HEADERS = {
    "User-Agent": "ShipmentManagementSystem/1.0"
}


# ============================================================
# DAY 5 - STEP 1: ROUTE CANDIDATE GENERATION
# ============================================================

def validate_coordinates(location, location_name="Location"):
    """
    Validate coordinates supplied as (latitude, longitude).
    """

    if not isinstance(location, (tuple, list)) or len(location) != 2:
        raise ValueError(
            f"{location_name} must contain latitude and longitude."
        )

    latitude = float(location[0])
    longitude = float(location[1])

    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError(f"{location_name} contains invalid coordinates.")

    if not -90 <= latitude <= 90:
        raise ValueError(f"Invalid latitude for {location_name}.")

    if not -180 <= longitude <= 180:
        raise ValueError(f"Invalid longitude for {location_name}.")

    return latitude, longitude


def haversine_distance(point1, point2):
    """
    Calculate the straight-line distance between two coordinates.
    Returns kilometers.
    """

    lat1, lon1 = validate_coordinates(point1, "Point 1")
    lat2, lon2 = validate_coordinates(point2, "Point 2")

    radius = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(a))


def _route_geometry_from_osrm(route):
    """Convert one OSRM GeoJSON route to validated (latitude, longitude) points."""
    raw_coordinates = route["geometry"]["coordinates"]
    points = []
    for coordinate in raw_coordinates:
        if len(coordinate) < 2:
            continue
        points.append(validate_coordinates(
            (float(coordinate[1]), float(coordinate[0])),
            "Route point"
        ))
    if len(points) < 2:
        raise ValueError("OSRM route geometry contains too few points.")
    return points


def _route_is_distinct(candidate_points, existing_routes,
                       minimum_separation_km=1.5,
                       minimum_different_fraction=0.15):
    """Reject near-identical geometries without requiring GIS dependencies."""
    if not existing_routes:
        return True

    def sampled(points, count=24):
        if len(points) <= count:
            return points
        last = len(points) - 1
        return [points[round(i * last / (count - 1))] for i in range(count)]

    candidate_samples = sampled(candidate_points)
    for existing in existing_routes:
        existing_samples = sampled(existing["points"])

        def different_fraction(points_a, points_b):
            different = 0
            for point in points_a:
                nearest = min(haversine_distance(point, other) for other in points_b)
                if nearest >= minimum_separation_km:
                    different += 1
            return different / max(1, len(points_a))

        fraction_a = different_fraction(candidate_samples, existing_samples)
        fraction_b = different_fraction(existing_samples, candidate_samples)
        if max(fraction_a, fraction_b) < minimum_different_fraction:
            return False

    return True


def _make_route_candidate(route, source, destination, route_index,
                          route_method, via_point=None):
    distance_meters = float(route["distance"])
    duration_seconds = float(route["duration"])
    if distance_meters <= 0 or duration_seconds < 0:
        raise ValueError("OSRM returned invalid route metrics.")
    points = _route_geometry_from_osrm(route)
    return {
        "route_id": f"route_{route_index}",
        "route_name": f"Route {route_index}",
        "distance_km": round(distance_meters / 1000, 2),
        "duration_minutes": round(duration_seconds / 60, 2),
        "duration_seconds": round(duration_seconds, 2),
        "geometry": points,
        "points": points,
        "route_method": route_method,
        "via_point": via_point,
        "is_shortest": False,
        "is_fastest": False,
        "source": source,
        "destination": destination,
    }


def _request_osrm_routes(coordinates, alternatives_value, timeout):
    """Call OSRM for a coordinate sequence; coordinates are (lat, lon)."""
    coordinate_string = ";".join(
        f"{longitude},{latitude}" for latitude, longitude in coordinates
    )
    url = f"{OSRM_ROUTE_URL}/{coordinate_string}"
    params = {
        "alternatives": alternatives_value,
        "steps": "false",
        "overview": "full",
        "geometries": "geojson",
    }
    response = requests.get(
        url, params=params, headers=REQUEST_HEADERS, timeout=timeout
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") != "Ok":
        raise RuntimeError(data.get("message") or data.get("code") or "OSRM error")
    return data


def _offset_waypoint(source, destination, fraction, offset_km, side):
    """Create a waypoint offset from the straight source-destination corridor."""
    lat1, lon1 = map(math.radians, source)
    lat2, lon2 = map(math.radians, destination)
    mean_lat = (lat1 + lat2) / 2.0
    x1, y1 = lon1 * math.cos(mean_lat), lat1
    x2, y2 = lon2 * math.cos(mean_lat), lat2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        raise ValueError("Source and destination are identical.")
    # Unit perpendicular in local equirectangular coordinates.
    px, py = -dy / length, dx / length
    earth_km = 6371.0
    offset_radians = (offset_km / earth_km) * side
    x = x1 + fraction * dx + px * offset_radians
    y = y1 + fraction * dy + py * offset_radians
    waypoint_lat = math.degrees(y)
    waypoint_lon = math.degrees(x / max(0.01, math.cos(mean_lat)))
    return validate_coordinates((waypoint_lat, waypoint_lon), "Alternative waypoint")


def generate_route_candidates(
    source,
    destination,
    alternatives=3,
    timeout=15
):
    """Generate up to ``alternatives`` distinct, real OSRM driving routes.

    OSRM's native alternatives are requested first. If OSRM returns too few,
    this function explores a set of offset corridor waypoints. Each waypoint
    is sent through OSRM, which snaps it to the road network and returns a
    complete road route. Routes with near-identical geometries are discarded.

    The service/network may not contain three genuinely different corridors;
    in that case fewer than three routes are returned rather than inventing
    straight-line or duplicate routes.
    """
    source = validate_coordinates(source, "Source")
    destination = validate_coordinates(destination, "Destination")
    if source == destination:
        raise ValueError("Source and destination coordinates cannot be identical.")
    if not isinstance(alternatives, int) or not 1 <= alternatives <= 5:
        raise ValueError("Alternatives must be an integer between 1 and 5.")

    candidates = []

    # 1) Ask OSRM for its own alternative routes.
    try:
        data = _request_osrm_routes([source, destination], str(alternatives), timeout)
        for route in data.get("routes", []):
            try:
                candidate = _make_route_candidate(
                    route, source, destination, len(candidates) + 1,
                    "OSRM native alternative"
                )
                if _route_is_distinct(candidate["points"], candidates):
                    candidates.append(candidate)
                if len(candidates) >= alternatives:
                    break
            except (KeyError, TypeError, ValueError, IndexError):
                continue
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("OSRM routing request timed out. Please try again.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Unable to connect to OSRM routing service: {exc}") from exc
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(f"OSRM could not generate a valid route: {exc}") from exc

    if not candidates:
        raise RuntimeError("OSRM could not produce a valid driving route.")

    # 2) Explore different corridors when native alternatives are insufficient.
    # The offsets scale with journey length but are capped to avoid extreme
    # detours. We vary both the side and the position along the corridor.
    if len(candidates) < alternatives:
        straight_km = haversine_distance(source, destination)
        offsets = sorted(set([
            round(max(4.0, min(12.0, straight_km * 0.02)), 1),
            round(max(8.0, min(25.0, straight_km * 0.05)), 1),
            round(max(15.0, min(40.0, straight_km * 0.09)), 1),
        ]))
        fractions = (0.25, 0.50, 0.75)
        waypoint_specs = []
        # Interleave positions/sides so the first attempts explore genuinely
        # different corridors rather than neighboring points on one side.
        for offset in offsets:
            for fraction in fractions:
                for side in (1, -1):
                    waypoint_specs.append((fraction, offset, side))

        # Keep public-service usage bounded. Usually only a few attempts are
        # needed; stop immediately once the requested count is reached.
        max_corridor_attempts = 14
        for fraction, offset, side in waypoint_specs[:max_corridor_attempts]:
            if len(candidates) >= alternatives:
                break
            try:
                waypoint = _offset_waypoint(
                    source, destination, fraction, offset, side
                )
                data = _request_osrm_routes(
                    [source, waypoint, destination], "false", timeout
                )
                routes = data.get("routes", [])
                if not routes:
                    continue

                snapped_waypoints = data.get("waypoints", [])
                snap_distance_km = None
                if len(snapped_waypoints) >= 3:
                    snap_distance_km = float(
                        snapped_waypoints[1].get("distance", 1e9)
                    ) / 1000.0
                    # A waypoint far from any routable road does not represent
                    # the intended corridor, so do not accept that candidate.
                    if snap_distance_km > 7.0:
                        continue

                candidate = _make_route_candidate(
                    routes[0], source, destination, len(candidates) + 1,
                    f"OSRM via-road corridor ({offset:.1f} km offset)",
                    via_point=waypoint,
                )
                candidate["via_point_snap_distance_km"] = (
                    round(snap_distance_km, 2)
                    if snap_distance_km is not None else None
                )
                candidate["corridor_fraction"] = fraction
                if _route_is_distinct(candidate["points"], candidates):
                    candidates.append(candidate)
            except requests.exceptions.RequestException:
                # A failed corridor attempt should not discard routes already
                # generated; continue trying the remaining corridors.
                continue
            except (RuntimeError, KeyError, TypeError, ValueError, IndexError):
                continue

    # 3) Stable IDs and shortest/fastest flags after duplicate filtering.
    for index, candidate in enumerate(candidates, start=1):
        candidate["route_id"] = f"route_{index}"
        candidate["route_name"] = f"Route {index}"
        candidate["is_shortest"] = False
        candidate["is_fastest"] = False

    shortest_index = min(
        range(len(candidates)), key=lambda i: candidates[i]["distance_km"]
    )
    fastest_index = min(
        range(len(candidates)), key=lambda i: candidates[i]["duration_minutes"]
    )
    candidates[shortest_index]["is_shortest"] = True
    candidates[fastest_index]["is_fastest"] = True

    return candidates

def generate_shortest_route(source, destination):
    """
    Return the shortest-distance route from available OSRM candidates.
    """

    candidates = generate_route_candidates(
        source, destination, alternatives=3
    )

    return min(candidates, key=lambda route: route["distance_km"])


def calculate_route_summary(routes):
    """
    Create a summary of generated route candidates.
    """

    if not routes:
        return {
            "total_routes": 0,
            "shortest_distance_km": None,
            "fastest_duration_minutes": None
        }

    return {
        "total_routes": len(routes),
        "shortest_distance_km": min(
            route["distance_km"] for route in routes
        ),
        "fastest_duration_minutes": min(
            route["duration_minutes"] for route in routes
        )
    }


# ============================================================
# DAY 5 - STEP 2: DELAY PREDICTION ENGINE
# ============================================================

def predict_additional_delay(
    base_duration_minutes,
    weather_severity="Normal",
    rainfall_mm=0.0,
    wind_speed_kmh=0.0,
    disturbance_risk=0,
    festival_risk=0,
    strike_risk=0,
    road_disruption_risk=0,
    historical_delay_minutes=0.0
):
    """
    Estimate additional shipment delay in minutes.

    Inputs:
        base_duration_minutes: OSRM estimated journey time.
        weather_severity: Normal, Moderate, High, or Severe.
        rainfall_mm: Forecast rainfall in mm.
        wind_speed_kmh: Forecast wind speed in km/h.
        disturbance_risk: General disturbance risk score from 0 to 100.
        festival_risk: Festival/crowd risk score from 0 to 100.
        strike_risk: Strike/bandh risk score from 0 to 100.
        road_disruption_risk: Road disruption risk score from 0 to 100.
        historical_delay_minutes: Optional ML-predicted delay.

    Returns:
        Dictionary containing delay and prediction details.

    Note:
        This is a baseline rule-based estimator, not a trained ML
        prediction. A trained model can replace it in a later step.
    """

    base_duration_minutes = float(base_duration_minutes)
    rainfall_mm = float(rainfall_mm)
    wind_speed_kmh = float(wind_speed_kmh)
    disturbance_risk = float(disturbance_risk)
    festival_risk = float(festival_risk)
    strike_risk = float(strike_risk)
    road_disruption_risk = float(road_disruption_risk)
    historical_delay_minutes = float(historical_delay_minutes)

    values = [
        base_duration_minutes, rainfall_mm, wind_speed_kmh,
        disturbance_risk, festival_risk, strike_risk,
        road_disruption_risk, historical_delay_minutes
    ]

    if not all(math.isfinite(value) for value in values):
        raise ValueError("All numeric inputs must be finite.")

    if base_duration_minutes <= 0:
        raise ValueError("Base journey duration must be positive.")

    if rainfall_mm < 0 or wind_speed_kmh < 0:
        raise ValueError("Weather values cannot be negative.")

    for risk_name, risk_value in {
        "disturbance_risk": disturbance_risk,
        "festival_risk": festival_risk,
        "strike_risk": strike_risk,
        "road_disruption_risk": road_disruption_risk
    }.items():
        if not 0 <= risk_value <= 100:
            raise ValueError(f"{risk_name} must be between 0 and 100.")

    if historical_delay_minutes < 0:
        raise ValueError("Historical delay cannot be negative.")

    severity = str(weather_severity).strip().lower()

    weather_factors = {
        "normal": 0.0,
        "low": 0.0,
        "moderate": 0.08,
        "medium": 0.08,
        "high": 0.18,
        "severe": 0.30,
        "extreme": 0.40
    }

    if severity not in weather_factors:
        raise ValueError(f"Unknown weather severity: {weather_severity}")

    weather_factor = weather_factors[severity]

    # Rainfall contribution.
    if rainfall_mm >= 50:
        rainfall_factor = 0.20
    elif rainfall_mm >= 20:
        rainfall_factor = 0.12
    elif rainfall_mm >= 5:
        rainfall_factor = 0.06
    else:
        rainfall_factor = 0.0

    # Wind contribution.
    if wind_speed_kmh >= 80:
        wind_factor = 0.15
    elif wind_speed_kmh >= 60:
        wind_factor = 0.10
    elif wind_speed_kmh >= 40:
        wind_factor = 0.05
    else:
        wind_factor = 0.0

    # Combine individual disturbance signals without adding each one at
    # full strength (which could double-count the same underlying event).
    specific_risk_weights = {
        "festival": 0.20,
        "strike": 0.25,
        "road_disruption": 0.25
    }
    weight_total = sum(specific_risk_weights.values())
    specific_disturbance_risk = (
        festival_risk * specific_risk_weights["festival"]
        + strike_risk * specific_risk_weights["strike"]
        + road_disruption_risk * specific_risk_weights["road_disruption"]
    ) / weight_total
    combined_disturbance_risk = max(
        disturbance_risk, specific_disturbance_risk
    )

    weather_delay = base_duration_minutes * (
        weather_factor + rainfall_factor + wind_factor
    )

    # Split the disturbance estimate into the categories displayed by app.py.
    festival_delay = (
        base_duration_minutes * 0.40 * (festival_risk / 100)
        * specific_risk_weights["festival"] / weight_total
    )
    strike_delay = (
        base_duration_minutes * 0.40 * (strike_risk / 100)
        * specific_risk_weights["strike"] / weight_total
    )
    road_disruption_delay = (
        base_duration_minutes * 0.40 * (road_disruption_risk / 100)
        * specific_risk_weights["road_disruption"] / weight_total
    )
    specific_delay_total = (
        festival_delay + strike_delay + road_disruption_delay
    )
    general_disturbance_delay = max(
        0.0,
        base_duration_minutes * combined_disturbance_risk / 100 * 0.40
        - specific_delay_total
    )
    disturbance_delay = (
        specific_delay_total + general_disturbance_delay
    )

    # The historical ML prediction is an additional signal. It is not
    # added blindly to weather and disturbance estimates, which could
    # double-count the same event.
    estimated_environmental_delay = weather_delay + disturbance_delay

    additional_delay = round(
        max(estimated_environmental_delay, historical_delay_minutes), 2
    )

    revised_duration = round(base_duration_minutes + additional_delay, 2)

    return {
        "base_duration_minutes": round(base_duration_minutes, 2),
        "weather_delay_minutes": round(weather_delay, 2),
        "disturbance_delay_minutes": round(disturbance_delay, 2),
        "festival_delay_minutes": round(festival_delay, 2),
        "strike_delay_minutes": round(strike_delay, 2),
        "road_disruption_delay_minutes": round(road_disruption_delay, 2),
        "historical_ml_delay_minutes": round(historical_delay_minutes, 2),
        "additional_delay_minutes": additional_delay,
        "revised_duration_minutes": revised_duration,
        "weather_severity": weather_severity,
        "disturbance_risk": round(combined_disturbance_risk, 2),
        "festival_risk": round(festival_risk, 2),
        "strike_risk": round(strike_risk, 2),
        "road_disruption_risk": round(road_disruption_risk, 2),
        "prediction_method": "Rule-based baseline estimator"
    }


# ============================================================
# DAY 5 - STEP 3: REVISED ETA CALCULATION
# ============================================================

def calculate_revised_eta(
    departure_time,
    base_duration_minutes,
    additional_delay_minutes=0,
    delivery_deadline=None
):
    """
    Calculate original ETA and revised ETA after predicted delay.

    Parameters
    ----------
    departure_time:
        Timezone-aware datetime representing shipment departure.

    base_duration_minutes:
        Original estimated driving duration.

    additional_delay_minutes:
        Predicted additional delay in minutes.

    delivery_deadline:
        Optional timezone-aware datetime for the delivery deadline.

    Returns
    -------
    Dictionary containing original ETA, revised ETA and status.
    """

    if not isinstance(departure_time, datetime):
        raise TypeError("departure_time must be a datetime object.")

    if departure_time.tzinfo is None:
        raise ValueError("departure_time must include timezone information.")

    base_duration_minutes = float(base_duration_minutes)
    additional_delay_minutes = float(additional_delay_minutes)

    if not math.isfinite(base_duration_minutes):
        raise ValueError("Base duration must be finite.")

    if not math.isfinite(additional_delay_minutes):
        raise ValueError("Additional delay must be finite.")

    if base_duration_minutes < 0:
        raise ValueError("Base duration cannot be negative.")

    if additional_delay_minutes < 0:
        raise ValueError("Additional delay cannot be negative.")

    if delivery_deadline is not None:

        if not isinstance(delivery_deadline, datetime):
            raise TypeError("delivery_deadline must be a datetime object.")

        if delivery_deadline.tzinfo is None:
            raise ValueError(
                "delivery_deadline must include timezone information."
            )

        if delivery_deadline.utcoffset() != departure_time.utcoffset():
            delivery_deadline = delivery_deadline.astimezone(
                departure_time.tzinfo
            )

    original_eta = departure_time + timedelta(minutes=base_duration_minutes)

    revised_eta = departure_time + timedelta(
        minutes=base_duration_minutes + additional_delay_minutes
    )

    if delivery_deadline is None:
        delivery_status = "Deadline not provided"
        deadline_delay_minutes = None

    elif revised_eta <= delivery_deadline:
        delivery_status = "On Time"
        deadline_delay_minutes = 0

    else:
        delivery_status = "Delayed"
        deadline_delay_minutes = round(
            (revised_eta - delivery_deadline).total_seconds() / 60, 2
        )

    return {
        "departure_time": departure_time.isoformat(),
        "original_eta": original_eta.isoformat(),
        "revised_eta": revised_eta.isoformat(),
        "base_duration_minutes": round(base_duration_minutes, 2),
        "additional_delay_minutes": round(additional_delay_minutes, 2),
        "revised_duration_minutes": round(
            base_duration_minutes + additional_delay_minutes, 2
        ),
        "delivery_deadline": (
            delivery_deadline.isoformat() if delivery_deadline else None
        ),
        "delivery_status": delivery_status,
        "deadline_delay_minutes": deadline_delay_minutes
    }


# ============================================================
# DAY 5 - STEP 4: SHORTEST AND FASTEST ROUTE COMPARISON
# ============================================================

def generate_route_comparison(source, destination):
    """
    Generate available OSRM routes and compare:
    1. Shortest-distance route
    2. Fastest-time route

    Returns a dictionary containing both routes and comparison.
    """

    routes = generate_route_candidates(
        source, destination, alternatives=3
    )

    if not routes:
        raise RuntimeError("No routes available for comparison.")

    shortest_route = min(routes, key=lambda route: route["distance_km"])

    fastest_route = min(routes, key=lambda route: route["duration_minutes"])

    for route in routes:
        route["is_fastest"] = route["route_id"] == fastest_route["route_id"]
        route["is_shortest"] = route["route_id"] == shortest_route["route_id"]

    return {
        "routes": routes,
        "shortest_route": shortest_route,
        "fastest_route": fastest_route,
        "comparison": {
            "shortest_distance_km": shortest_route["distance_km"],
            "shortest_duration_minutes": shortest_route["duration_minutes"],
            "fastest_distance_km": fastest_route["distance_km"],
            "fastest_duration_minutes": fastest_route["duration_minutes"],
            "available_route_count": len(routes)
        }
    }


def calculate_route_difference(route1, route2):
    """
    Compare two route dictionaries (route2 relative to route1).
    """

    if not isinstance(route1, dict) or not isinstance(route2, dict):
        raise TypeError("Both routes must be dictionaries.")

    for key in ("distance_km", "duration_minutes"):
        if key not in route1 or key not in route2:
            raise ValueError(f"Both routes must contain {key}.")

        value1 = float(route1[key])
        value2 = float(route2[key])

        if not math.isfinite(value1) or not math.isfinite(value2):
            raise ValueError(f"Invalid {key} value.")

    distance_difference = (
        float(route2["distance_km"]) - float(route1["distance_km"])
    )

    time_difference = (
        float(route2["duration_minutes"])
        - float(route1["duration_minutes"])
    )

    return {
        "distance_difference_km": round(distance_difference, 2),
        "time_difference_minutes": round(time_difference, 2),
        "route2_distance_change_percent": round(
            distance_difference / float(route1["distance_km"]) * 100, 2
        ) if float(route1["distance_km"]) > 0 else None
    }


# ============================================================
# DAY 5 - STEP 5: DISTURBANCE-AWARE ROUTE RISK ENGINE
# ============================================================

def calculate_route_risk(
    weather_risk=0,
    festival_risk=0,
    strike_risk=0,
    road_disruption_risk=0
):
    """
    Calculate route disturbance risk on a 0-100 scale.

    Each input must be between 0 and 100.

    Returns a risk score, risk category and components.
    """

    risks = {
        "weather": float(weather_risk),
        "festival": float(festival_risk),
        "strike": float(strike_risk),
        "road_disruption": float(road_disruption_risk)
    }

    for name, value in risks.items():
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"{name} risk must be between 0 and 100.")

    weights = {
        "weather": 0.30,
        "festival": 0.20,
        "strike": 0.25,
        "road_disruption": 0.25
    }

    weighted_risk = sum(risks[name] * weights[name] for name in risks)

    risk_score = round(weighted_risk, 2)

    if risk_score < 20:
        risk_level = "Low"
    elif risk_score < 40:
        risk_level = "Moderate"
    elif risk_score < 70:
        risk_level = "High"
    else:
        risk_level = "Severe"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_components": risks,
        "risk_weights": weights
    }


def evaluate_route_candidates(routes, route_risk_data):
    """
    Evaluate available routes using their disturbance risk.

    route_risk_data format:

    {
        "route_1": {
            "weather_risk": 20,
            "festival_risk": 30,
            "strike_risk": 0,
            "road_disruption_risk": 10
        }
    }

    Missing route risk data is treated as unknown, not safe.
    """

    if not routes:
        raise ValueError("At least one route is required.")

    if not isinstance(route_risk_data, dict):
        raise TypeError("route_risk_data must be a dictionary.")

    evaluated_routes = []

    for route in routes:

        route_id = route.get("route_id")

        if not route_id:
            raise ValueError("Every route must have a route_id.")

        if route_id not in route_risk_data:
            evaluation = {
                "risk_score": None,
                "risk_level": "Unknown",
                "risk_components": None
            }

        else:
            risk_data = route_risk_data[route_id]

            if not isinstance(risk_data, dict):
                raise ValueError(
                    f"Risk data for {route_id} must be a dictionary."
                )

            evaluation = calculate_route_risk(
                weather_risk=risk_data.get("weather_risk", 0),
                festival_risk=risk_data.get("festival_risk", 0),
                strike_risk=risk_data.get("strike_risk", 0),
                road_disruption_risk=risk_data.get(
                    "road_disruption_risk", 0
                )
            )

        evaluated_routes.append({
            **route,
            "risk_score": evaluation["risk_score"],
            "risk_level": evaluation["risk_level"],
            "risk_components": evaluation["risk_components"]
        })

    return evaluated_routes


def recommend_safest_route(routes):
    """
    Recommend a route using risk, travel time and distance.

    Routes without risk data cannot be recommended as safe.
    """

    if not routes:
        raise ValueError("No routes available.")

    known_routes = [
        route for route in routes
        if route.get("risk_score") is not None
    ]

    if not known_routes:
        raise ValueError(
            "Cannot recommend a safest route: "
            "risk information is unavailable for every route."
        )

    return min(
        known_routes,
        key=lambda route: (
            route["risk_score"],
            route["duration_minutes"],
            route["distance_km"]
        )
    )


# ============================================================
# DAY 5 - STEP 6: INTELLIGENT ROUTE RANKING
# ============================================================

def integrate_route_conditions(routes, conditions_by_route, use_verified_ml=False):
    """Attach route-specific disturbance conditions and risk scores.

    The current app supplies verified weather/festival/RSS condition
    dictionaries keyed by route_id. This function deliberately does not
    claim to use a trained ML model: no verified model/prediction is passed
    into this interface. Rule-based condition scores are used instead.
    """
    if not isinstance(routes, (list, tuple)) or not routes:
        raise ValueError("At least one route is required.")
    if not isinstance(conditions_by_route, dict):
        raise TypeError("conditions_by_route must be a dictionary.")

    enriched = []
    for route in routes:
        if not isinstance(route, dict):
            raise TypeError("Each route must be a dictionary.")
        route_id = route.get("route_id")
        if not route_id:
            raise ValueError("Every route must contain a route_id.")
        if route_id not in conditions_by_route:
            # Do not silently treat missing evidence as a safe route.
            continue

        condition = conditions_by_route[route_id]
        if not isinstance(condition, dict):
            raise TypeError(f"Conditions for {route_id} must be a dictionary.")

        risk = calculate_route_risk(
            weather_risk=condition.get("weather_risk", 0),
            festival_risk=condition.get("festival_risk", 0),
            strike_risk=condition.get("strike_risk", 0),
            road_disruption_risk=condition.get("road_disruption_risk", 0),
        )
        item = {
            **route,
            "risk_score": risk["risk_score"],
            "risk_level": risk["risk_level"],
            "risk_components": risk["risk_components"],
            "condition_data": condition,
            "ml_model_used": False,
        }
        enriched.append(item)

    if not enriched:
        raise ValueError(
            "No routes have complete condition data; cannot rank routes as safe."
        )
    return enriched


def rank_routes_intelligently(
    routes,
    risk_weight=0.40,
    time_weight=0.35,
    distance_weight=0.25,
    delay_weight=0.0
):
    """
    Rank routes using:
    - Disturbance risk
    - Driving time
    - Distance
    - Additional predicted delay

    All weights must sum to 1.

    Missing risk information makes a route ineligible.
    """

    if not routes:
        raise ValueError("No routes available.")

    weights = {
        "risk": float(risk_weight),
        "time": float(time_weight),
        "distance": float(distance_weight),
        "delay": float(delay_weight)
    }

    for name, weight in weights.items():
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid {name} weight.")

    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Optimization weights must sum to 1.")

    eligible_routes = []

    for route in routes:

        risk = route.get("risk_score")

        if risk is None:
            continue

        distance = float(route["distance_km"])
        duration = float(route["duration_minutes"])
        delay = float(route.get("additional_delay_minutes", 0))

        values = [risk, distance, duration, delay]

        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Route contains non-finite values.")

        if not 0 <= float(risk) <= 100:
            raise ValueError("Risk score must be between 0 and 100.")

        if distance <= 0 or duration < 0 or delay < 0:
            raise ValueError("Invalid distance, time or delay.")

        eligible_routes.append({
            **route,
            "distance_km": distance,
            "duration_minutes": duration,
            "risk_score": float(risk),
            "additional_delay_minutes": delay
        })

    if not eligible_routes:
        raise ValueError("No routes have sufficient risk information.")

    minimum_distance = min(r["distance_km"] for r in eligible_routes)
    minimum_duration = min(r["duration_minutes"] for r in eligible_routes)
    maximum_delay = max(
        r["additional_delay_minutes"] for r in eligible_routes
    )

    ranked_routes = []

    for route in eligible_routes:

        distance_score = route["distance_km"] / minimum_distance * 100

        time_score = (
            route["duration_minutes"] / minimum_duration * 100
            if minimum_duration > 0 else 0
        )

        delay_score = (
            route["additional_delay_minutes"] / maximum_delay * 100
            if maximum_delay > 0 else 0
        )

        # Lower overall score means a better route.
        optimization_score = (
            route["risk_score"] * weights["risk"]
            + time_score * weights["time"]
            + distance_score * weights["distance"]
            + delay_score * weights["delay"]
        )

        ranked_routes.append({
            **route,
            "distance_score": round(distance_score, 2),
            "time_score": round(time_score, 2),
            "delay_score": round(delay_score, 2),
            "optimization_score": round(optimization_score, 2)
        })

    ranked_routes.sort(
        key=lambda r: (
            r["optimization_score"],
            r["risk_score"],
            r["duration_minutes"],
            r["distance_km"]
        )
    )

    for index, route in enumerate(ranked_routes):
        route["optimization_rank"] = index + 1
        route["rank"] = index + 1
        route["recommended"] = index == 0

    return ranked_routes


def optimize_shipment_route(
    source,
    destination,
    route_risk_data,
    risk_weight=0.40,
    time_weight=0.35,
    distance_weight=0.25,
    delay_weight=0.0
):
    """
    Complete route optimization pipeline.

    Generates routes, evaluates risk and ranks them.
    """

    routes = generate_route_candidates(
        source, destination, alternatives=3
    )

    evaluated_routes = evaluate_route_candidates(routes, route_risk_data)

    ranked_routes = rank_routes_intelligently(
        evaluated_routes,
        risk_weight=risk_weight,
        time_weight=time_weight,
        distance_weight=distance_weight,
        delay_weight=delay_weight
    )

    return {
        "routes": ranked_routes,
        "recommended_route": ranked_routes[0],
        "total_routes_generated": len(routes),
        "total_routes_evaluated": len(ranked_routes),
        "optimization_method": "Weighted multi-criteria ranking"
    }