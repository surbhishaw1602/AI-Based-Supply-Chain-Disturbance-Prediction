"""Routing, geocoding and route-distance helpers for Shipment Management System."""

import math
import threading
import time

import requests

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
GEOCODING_HEADERS = {
    "User-Agent": "ShipmentManagementDashboard/1.2 (shipment route geocoding)"
}
ROUTING_HEADERS = {"User-Agent": "ShipmentManagementDashboard/1.2"}

# Nominatim's public service requires a maximum of one request per second.
_nominatim_lock = threading.Lock()
_last_nominatim_request = 0.0


def _nominatim_get(url, *, params, timeout=20):
    global _last_nominatim_request
    with _nominatim_lock:
        wait = 1.05 - (time.monotonic() - _last_nominatim_request)
        if wait > 0:
            time.sleep(wait)
        response = requests.get(
            url, params=params, headers=GEOCODING_HEADERS, timeout=timeout
        )
        _last_nominatim_request = time.monotonic()
    response.raise_for_status()
    return response


def _valid_coordinates(latitude, longitude):
    lat, lon = float(latitude), float(longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("Coordinates must be finite numbers.")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Coordinates are outside valid latitude/longitude ranges.")
    return lat, lon


def search_locations(query, limit=5):
    """Return specific Indian settlement/locality/street geocoding candidates."""
    query = str(query or "").strip()
    if len(query) < 2:
        return []
    if query.casefold() in {
        "odisha", "orissa", "west bengal", "bihar", "jharkhand",
        "sikkim", "assam", "india"
    }:
        return []
    try:
        response = _nominatim_get(
            NOMINATIM_SEARCH_URL,
            params={
                "q": query, "format": "jsonv2", "addressdetails": 1,
                "limit": min(max(int(limit), 1), 5), "countrycodes": "in",
                "accept-language": "en"
            },
            timeout=20
        )
        candidates = []
        for item in response.json():
            address = item.get("address") or {}
            if not isinstance(address, dict):
                continue
            specific = any(address.get(k) for k in (
                "city", "town", "village", "municipality", "suburb",
                "city_district", "neighbourhood", "road", "pedestrian", "hamlet"
            ))
            if not specific:
                continue
            try:
                lat, lon = _valid_coordinates(item["lat"], item["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            display_name = item.get("display_name")
            if not display_name:
                continue
            candidates.append({
                "display_name": display_name,
                "latitude": lat,
                "longitude": lon,
                "type": item.get("type", ""),
                "class": item.get("class", ""),
                "address": address,
            })
        return candidates
    except (requests.RequestException, ValueError, TypeError):
        return []


def geocode_location(location_name):
    """Return the top specific Indian place candidate, or None."""
    candidates = search_locations(location_name, limit=5)
    return candidates[0] if candidates else None


def geocode_disturbance_locality(place, district=None, state=None):
    parts = [str(place or "").strip()]
    if district:
        parts.append(str(district).strip())
    if state:
        parts.append(str(state).strip())
    parts.append("India")
    candidates = search_locations(", ".join(p for p in parts if p), limit=5)
    if not candidates:
        return None
    result = dict(candidates[0])
    result["is_locality"] = True
    return result


def _point_to_segment_distance_km(point, start, end):
    """Approximate point-to-segment distance using a local projection."""
    lat0, lon0 = map(math.radians, _valid_coordinates(*point))
    cos_lat = max(abs(math.cos(lat0)), 1e-8)

    def project(coord):
        lat, lon = map(math.radians, _valid_coordinates(*coord))
        return (
            6371.0088 * (lon - lon0) * cos_lat,
            6371.0088 * (lat - lat0),
        )

    ax, ay = project(start)
    bx, by = project(end)
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / length_sq))
    return math.hypot(ax + t * dx, ay + t * dy)


def distance_from_location_to_route_km(location, route_points):
    """Shortest approximate distance from a GPS point to any route segment."""
    try:
        if isinstance(location, dict):
            point = _valid_coordinates(location["latitude"], location["longitude"])
        else:
            point = _valid_coordinates(location[0], location[1])
        points = [
            _valid_coordinates(p[0], p[1])
            for p in (route_points or [])
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]
        if not points:
            return None
        if len(points) == 1:
            return _point_to_segment_distance_km(point, points[0], points[0])
        return min(
            _point_to_segment_distance_km(point, points[i], points[i + 1])
            for i in range(len(points) - 1)
        )
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return None


def reverse_geocode(latitude, longitude):
    """Return the nearest mapped road/locality label for route checkpoints."""
    try:
        lat, lon = _valid_coordinates(latitude, longitude)
        response = _nominatim_get(
            NOMINATIM_REVERSE_URL,
            params={
                "format": "jsonv2", "lat": lat, "lon": lon,
                "zoom": 16, "addressdetails": 1, "accept-language": "en"
            },
            timeout=15
        )
        data = response.json()
        address = data.get("address") or {}
        if not isinstance(address, dict):
            address = {}
        primary = next((address.get(k) for k in (
            "road", "pedestrian", "neighbourhood", "suburb", "village",
            "hamlet", "quarter"
        ) if address.get(k)), None)
        locality = next((address.get(k) for k in (
            "city", "town", "municipality", "county", "state_district"
        ) if address.get(k)), None)
        if primary and locality and primary.casefold() != locality.casefold():
            return f"{primary}, {locality}"
        return primary or locality or data.get("display_name") or None
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return None


def generate_route(source, destination):
    """Generate an OSRM driving route. Inputs are (latitude, longitude)."""
    source = _valid_coordinates(source[0], source[1])
    destination = _valid_coordinates(destination[0], destination[1])
    if source == destination:
        raise ValueError("Source and destination coordinates cannot be identical.")
    coordinates = f"{source[1]},{source[0]};{destination[1]},{destination[0]}"
    try:
        response = requests.get(
            f"{OSRM_URL}/{coordinates}",
            params={"overview": "full", "geometries": "geojson", "steps": "true"},
            headers=ROUTING_HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        raise RuntimeError(f"Routing service connection failed: {error}") from error
    except ValueError as error:
        raise RuntimeError("Routing service returned an invalid response.") from error
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(data.get("message") or "No driving route found.")
    route = data["routes"][0]
    points = []
    for coordinate in route.get("geometry", {}).get("coordinates", []):
        if len(coordinate) >= 2:
            points.append(_valid_coordinates(coordinate[1], coordinate[0]))
    if len(points) < 2:
        raise RuntimeError("OSRM returned a route without usable geometry.")
    return {
        "points": points,
        "distance_km": float(route["distance"]) / 1000.0,
        "duration_minutes": float(route["duration"]) / 60.0,
        "checkpoints": extract_checkpoints(points),
    }


def extract_checkpoints(route_points, number_of_checkpoints=5):
    """Select evenly spaced route-geometry points, excluding endpoints."""
    if not route_points:
        return []
    count = max(0, int(number_of_checkpoints))
    if count == 0 or len(route_points) <= 2:
        return []
    last_index = len(route_points) - 1
    actual_count = min(count, max(0, len(route_points) - 2))
    indices = sorted({
        max(1, min(last_index - 1, round(i * last_index / (actual_count + 1))))
        for i in range(1, actual_count + 1)
    })
    return [route_points[i] for i in indices]
