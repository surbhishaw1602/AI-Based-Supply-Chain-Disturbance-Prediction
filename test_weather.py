
from weather_service import get_weather_forecast

# Kolkata coordinates
latitude = 22.5726
longitude = 88.3639

try:

    forecasts = get_weather_forecast(
        latitude,
        longitude
    )

    print("\n========== WEATHER API TEST ==========")

    print("Location: Kolkata")
    print("Forecast days:", len(forecasts))

    for day in forecasts:

        print("\nDate:", day["date"])

        print(
            "Temperature:",
            day["temperature_min"],
            "-",
            day["temperature_max"],
            "°C"
        )

        print("Rainfall:", day["rainfall"], "mm")

        print("Wind:", day["wind_speed"], "km/h")

        print("Condition:", day["condition"])

        print("Severity:", day["severity"])

        if day["alerts"]:
            print("Warnings:", ", ".join(day["alerts"]))

    print("\nWeather API connected successfully!")

except Exception as error:

    print("\nWeather API test failed:", error)
