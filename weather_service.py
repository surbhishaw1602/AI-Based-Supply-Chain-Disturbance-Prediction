
"""
Day 3 - Weather Monitoring Service

Features:
1. Connect to Open-Meteo API.
2. Retrieve seven-day forecasts.
3. Read temperature, rainfall, wind speed and weather conditions.
4. Identify potentially severe weather.
"""

import requests


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# WMO weather condition descriptions
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Severe thunderstorm with hail"
}


def get_weather_condition(weather_code):
    """Convert WMO weather code into readable text."""

    return WEATHER_CODES.get(
        int(weather_code),
        "Unknown weather condition"
    )


def identify_weather_severity(
    weather_code,
    rainfall,
    wind_speed,
    temperature
):
    """
    Identify weather conditions that may affect shipments.

    These are configurable operational warning thresholds,
    not official meteorological danger classifications.
    """

    alerts = []

    code = int(weather_code)

    # Thunderstorms
    if code in [95, 96, 99]:
        alerts.append("Thunderstorm warning")

    # Heavy rain
    if code in [65, 67, 82]:
        alerts.append("Heavy rain warning")

    # Heavy snowfall
    if code in [75, 86]:
        alerts.append("Heavy snowfall warning")

    # Heavy rainfall threshold
    if rainfall >= 50:
        alerts.append("Very heavy rainfall")

    # Strong wind threshold (km/h)
    if wind_speed >= 60:
        alerts.append("Strong wind warning")

    # Extreme heat threshold
    if temperature >= 40:
        alerts.append("Extreme heat warning")

    if alerts:
        return "Severe", alerts

    return "Normal", []


def get_weather_forecast(latitude, longitude):
    """
    Retrieve a seven-day forecast for given coordinates.

    Returns:
        A list of daily weather dictionaries.

    Raises:
        ValueError: Invalid coordinates or invalid API response.
        requests.RequestException: Network/API errors.
    """

    latitude = float(latitude)
    longitude = float(longitude)

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": (
            "weather_code,"
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "wind_speed_10m_max"
        ),
        "forecast_days": 7,
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm"
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if "daily" not in data:
        raise ValueError("Weather API did not return daily forecast data.")

    daily = data["daily"]

    required_fields = [
        "time",
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max"
    ]

    for field in required_fields:
        if field not in daily:
            raise ValueError(
                f"Weather API response is missing: {field}"
            )

    forecasts = []

    for index, forecast_date in enumerate(daily["time"]):

        weather_code = daily["weather_code"][index]

        max_temperature = daily["temperature_2m_max"][index]

        min_temperature = daily["temperature_2m_min"][index]

        rainfall = daily["precipitation_sum"][index]

        wind_speed = daily["wind_speed_10m_max"][index]

        condition = get_weather_condition(weather_code)

        severity, alerts = identify_weather_severity(
            weather_code,
            rainfall,
            wind_speed,
            max_temperature
        )

        forecasts.append({
            "date": forecast_date,
            "temperature_max": max_temperature,
            "temperature_min": min_temperature,
            "rainfall": rainfall,
            "wind_speed": wind_speed,
            "weather_code": weather_code,
            "condition": condition,
            "severity": severity,
            "alerts": alerts
        })

    if len(forecasts) != 7:
        raise ValueError(
            f"Expected 7 forecast days, received {len(forecasts)}."
        )

    return forecasts
