import requests
import time
import re
import math
from functools import lru_cache

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"

GEOCODING_HEADERS = {
    "User-Agent": "ShipmentManagementDashboard/1.4 (shipment location and route lookup)"
}
EARTH_RADIUS_KM = 6371.0

BROAD_LOCATIONS = {
    "west bengal", "india", "odisha", "bihar", "jharkhand",
    "assam", "sikkim", "uttar pradesh", "delhi"
}


@lru_cache(maxsize=500)
def _geocode_location_cached(location_name):
    """Geocode in India, preferring settlement results over admin areas."""
    try:
        time.sleep(1.05)
        response = requests.get(
            NOMINATIM_SEARCH_URL,
            params={
                "q": location_name,
                "format": "jsonv2",
                "limit": 8,
                "countrycodes": "in",
                "addressdetails": 1,
                "accept-language": "en",
            },
            headers=GEOCODING_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            return None

        preferred_types = {
            "city", "town", "village", "suburb", "neighbourhood",
            "hamlet", "municipality",
        }
        valid = []
        for result in results:
            address = result.get("address", {})
            if not isinstance(address, dict):
                continue
            if str(address.get("country_code", "")).lower() != "in":
                continue
            result_type = str(result.get("type", "")).lower()
            try:
                candidate = {
                    "latitude": float(result["lat"]),
                    "longitude": float(result["lon"]),
                    "display_name": result.get("display_name", location_name),
                    "result_type": result_type,
                    "is_locality": result_type in preferred_types,
                    "address": address,
                }
            except (KeyError, TypeError, ValueError):
                continue
            valid.append(candidate)

        if not valid:
            return None

        # A locality is preferable to a state/district/admin boundary result.
        valid.sort(key=lambda item: (not item["is_locality"],))
        return valid[0]

    except (requests.RequestException, ValueError, TypeError, KeyError) as error:
        print(f"Location lookup failed for '{location_name}': {error}")
        return None


def geocode_location(location_name):
    if not location_name:
        return None
    location_name = str(location_name).strip()
    if not location_name:
        return None
    return _geocode_location_cached(location_name)


def geocode_disturbance_locality(
    locality,
    district=None,
    state=None
):
    """
    Geocode an event locality using its geographic context.

    Examples:
    Velagaleru, NTR district, Andhra Pradesh, India
    Kolkata, West Bengal, India
    """

    locality = str(locality or "").strip()

    if not locality:
        return None

    broad_locations = {
        "india",
        "andhra pradesh",
        "west bengal",
        "odisha",
        "bihar",
        "jharkhand",
        "karnataka",
        "tamil nadu",
        "kerala",
        "maharashtra",
        "punjab",
        "delhi",
        "telangana",
        "rajasthan",
        "gujarat"
    }

    if locality.lower() in broad_locations:
        return None

    parts = [locality]

    if district and district.lower() != locality.lower():
        parts.append(str(district).strip())

    if state and state.lower() != locality.lower():
        parts.append(str(state).strip())

    parts.append("India")

    search_name = ", ".join(
        part for part in parts if part
    )

    result = geocode_location(search_name)

    if not result:
        return None

    # Do not use district/state/admin centroids as if they were incident sites.
    if not result.get("is_locality"):
        return None

    address = result.get("address", {})
    if str(address.get("country_code", "")).lower() != "in":
        return None

    # Require the requested name to appear in a locality-level address field.
    # This prevents a loose search result from matching an unrelated place.
    locality_fields = (
        "city", "town", "village", "municipality", "suburb",
        "neighbourhood", "hamlet", "quarter",
    )
    requested = re.sub(r"\s+", " ", locality).strip().casefold()
    address_names = {
        re.sub(r"\s+", " ", str(address.get(field, ""))).strip().casefold()
        for field in locality_fields
        if address.get(field)
    }
    if requested not in address_names:
        return None

    result["matched_locality"] = locality
    result["search_query"] = search_name
    result["match_quality"] = "locality_exact_address_match"
    return result



def geocode_affected_locations(affected_location, state=None, max_locations=6):
    if not affected_location:
        return []
    parts = re.split(r",|;|\||\band\b", str(affected_location), flags=re.I)
    results, seen = [], set()
    for part in parts:
        locality = part.strip()
        normalized = locality.lower()
        if not locality or normalized in BROAD_LOCATIONS:
            continue
        if state and normalized == str(state).strip().lower():
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        search_name = f"{locality}, {state}, India" if state else f"{locality}, India"
        location = geocode_location(search_name)
        if location:
            location["matched_locality"] = locality
            results.append(location)
        if len(results) >= max_locations:
            break
    return results


def reverse_geocode(latitude, longitude):
    try:
        time.sleep(1.05)
        response = requests.get(
            NOMINATIM_REVERSE_URL,
            params={
                "format": "jsonv2", "lat": float(latitude), "lon": float(longitude),
                "zoom": 16, "addressdetails": 1, "accept-language": "en"
            },
            headers=GEOCODING_HEADERS,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        address = data.get("address", {})
        if not isinstance(address, dict):
            address = {}
        primary = (
            address.get("road") or address.get("pedestrian")
            or address.get("neighbourhood") or address.get("suburb")
            or address.get("village") or address.get("hamlet")
            or address.get("quarter")
        )
        locality = (
            address.get("city") or address.get("town")
            or address.get("municipality") or address.get("county")
            or address.get("state_district")
        )
        if primary and locality and primary.lower() != locality.lower():
            return f"{primary}, {locality}"
        return primary or locality or data.get("display_name") or None
    except (requests.RequestException, ValueError, TypeError, KeyError) as error:
        print(f"Reverse geocoding failed for ({latitude}, {longitude}): {error}")
        return None


def generate_route(source, destination):
    if not source or not destination:
        raise ValueError("Source and destination coordinates are required.")
    source_lat, source_lon = map(float, source)
    destination_lat, destination_lon = map(float, destination)
    if not (-90 <= source_lat <= 90 and -90 <= destination_lat <= 90
            and -180 <= source_lon <= 180 and -180 <= destination_lon <= 180):
        raise ValueError("Invalid source or destination coordinates.")
    coordinates = f"{source_lon},{source_lat};{destination_lon},{destination_lat}"
    url = f"{OSRM_URL}/{coordinates}"
    try:
        response = requests.get(
            url,
            params={"overview": "full", "geometries": "geojson", "steps": "true"},
            headers={"User-Agent": "ShipmentManagementDashboard/1.4"},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            raise ValueError("No road route found between the source and destination.")
        route = data["routes"][0]
        route_points = [
            (point[1], point[0])
            for point in route["geometry"]["coordinates"]
        ]
        return {
            "points": route_points,
            "distance_km": route["distance"] / 1000,
            "duration_minutes": route["duration"] / 60,
            "checkpoints": extract_checkpoints(route_points, 5)
        }
    except requests.RequestException as error:
        raise RuntimeError(f"Routing service connection failed: {error}") from error


def extract_checkpoints(route_points, number_of_checkpoints=5):
    if not route_points or number_of_checkpoints <= 0 or len(route_points) <= 2:
        return []
    count = min(number_of_checkpoints, len(route_points) - 2)
    last_index = len(route_points) - 1
    indices = [
        round(i * last_index / (count + 1))
        for i in range(1, count + 1)
    ]
    return [route_points[index] for index in dict.fromkeys(indices)]


def distance_point_to_segment_km(point, segment_start, segment_end):
    point_lat, point_lon = map(float, point)
    start_lat, start_lon = map(float, segment_start)
    end_lat, end_lon = map(float, segment_end)
    lat0 = math.radians(point_lat)

    def project(latitude, longitude):
        x = EARTH_RADIUS_KM * math.radians(longitude - point_lon) * math.cos(lat0)
        y = EARTH_RADIUS_KM * math.radians(latitude - point_lat)
        return x, y

    ax, ay = project(start_lat, start_lon)
    bx, by = project(end_lat, end_lon)
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / length_squared))
    return math.hypot(ax + t * dx, ay + t * dy)


def distance_from_location_to_route_km(location, route_points):
    if not location or not route_points:
        return None
    event_point = (float(location["latitude"]), float(location["longitude"]))
    if len(route_points) == 1:
        return distance_point_to_segment_km(event_point, route_points[0], route_points[0])
    distances = [
        distance_point_to_segment_km(event_point, route_points[i], route_points[i + 1])
        for i in range(len(route_points) - 1)
    ]
    return min(distances) if distances else None


def find_event_route_impacts(route_points, events, state=None, max_distance_km=15):
    if not route_points or not events:
        return []
    impacts = []
    for event in events:
        affected_location = event.get("affected_location", "")
        event_state = event.get("state") or state
        locations = geocode_affected_locations(affected_location, state=event_state)
        best_location, best_distance = None, None
        for location in locations:
            distance = distance_from_location_to_route_km(location, route_points)
            if distance is not None and (best_distance is None or distance < best_distance):
                best_distance, best_location = distance, location
        if best_distance is None or best_distance > float(max_distance_km):
            continue
        impacts.append({
            "event_id": event.get("event_id"),
            "name": event.get("name", "Unnamed event"),
            "event_type": event.get("event_type", ""),
            "start_date": event.get("start_date", ""),
            "end_date": event.get("end_date", ""),
            "state": event.get("state", ""),
            "affected_location": affected_location,
            "impact_level": event.get("impact_level", "Unknown"),
            "description": event.get("description", ""),
            "source_url": event.get("source_url", ""),
            "matched_locality": best_location.get("matched_locality"),
            "matched_display_name": best_location.get("display_name"),
            "latitude": best_location.get("latitude"),
            "longitude": best_location.get("longitude"),
            "distance_km": round(best_distance, 2),
            "match_type": "Near route" if best_distance <= 5 else "Possible nearby impact"
        })
    impacts.sort(key=lambda item: item["distance_km"])
    return impacts
