import os
import requests
from datetime import datetime
from collections import defaultdict

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required

weather_bp = Blueprint("weather", __name__, url_prefix="/api")


WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


# ═══════════════════════════════════════
# DISTRICTS
# ═══════════════════════════════════════
DISTRICTS = {
    "sankhuwasabha": {"lat": 27.5333, "lon": 87.1833, "name": "Sankhuwasabha"},
    "ilam": {"lat": 26.9125, "lon": 87.9258, "name": "Ilam"},
    "taplejung": {"lat": 27.3548, "lon": 87.6694, "name": "Taplejung"},
    "panchthar": {"lat": 27.1452, "lon": 87.7958, "name": "Panchthar"},
}


# ═══════════════════════════════════════
# RISK CALCULATION
# ═══════════════════════════════════════
def calc_risk(temp, humidity, rain):
    risks = []

    # Chhirke (virus - aphid)
    if temp > 24 and humidity < 70:
        risks.append({
            "disease": "Chhirke",
            "level": "High",
            "message": "Hot + dry → high aphid activity",
            "action": "Use neem oil spray"
        })
    elif temp > 20 and humidity < 80:
        risks.append({
            "disease": "Chhirke",
            "level": "Medium",
            "message": "Moderate risk",
            "action": "Monitor plants regularly"
        })
    else:
        risks.append({
            "disease": "Chhirke",
            "level": "Low",
            "message": "Low risk",
            "action": "Normal monitoring"
        })

    # Leaf blight (fungal)
    if 20 <= temp <= 28 and humidity > 85 and rain > 5:
        risks.append({
            "disease": "Leaf Blight",
            "level": "High",
            "message": "Warm + humid + rain → fungus risk",
            "action": "Apply fungicide"
        })
    elif humidity > 75:
        risks.append({
            "disease": "Leaf Blight",
            "level": "Medium",
            "message": "Moderate fungal risk",
            "action": "Preventive spray"
        })
    else:
        risks.append({
            "disease": "Leaf Blight",
            "level": "Low",
            "message": "Safe conditions",
            "action": "Continue monitoring"
        })

    levels = [r["level"] for r in risks]
    overall = "High" if "High" in levels else (
        "Medium" if "Medium" in levels else "Low")

    return {"risks": risks, "overall": overall}


# ═══════════════════════════════════════
# WEATHER ROUTE
# ═══════════════════════════════════════
@weather_bp.route("/weather", methods=["GET"])
@jwt_required()
def weather():
    district = request.args.get("district", "ilam").lower()
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if lat and lon:
        location = f"{lat:.2f}, {lon:.2f}"
    elif district in DISTRICTS:
        info = DISTRICTS[district]
        lat, lon = info["lat"], info["lon"]
        location = info["name"]
    else:
        return jsonify({"error": "Invalid district"}), 400

    # Demo mode
    if not WEATHER_API_KEY:
        temp, humidity, rain = 24.5, 82, 8.2

        return jsonify({
            "demo": True,
            "location": location,
            "temperature": temp,
            "humidity": humidity,
            "rain_mm": rain,
            "risk": calc_risk(temp, humidity, rain),
            "note": "Add WEATHER_API_KEY for live data"
        })

    try:
        r = requests.get(WEATHER_URL, params={
            "lat": lat,
            "lon": lon,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }, timeout=5)

        r.raise_for_status()
        d = r.json()

        temp = d["main"]["temp"]
        humidity = d["main"]["humidity"]
        rain = d.get("rain", {}).get("1h", 0)

        return jsonify({
            "demo": False,
            "location": location,
            "temperature": round(temp, 1),
            "humidity": humidity,
            "rain_mm": rain,
            "description": d["weather"][0]["description"],
            "wind_kmh": round(d["wind"]["speed"] * 3.6, 1),
            "risk": calc_risk(temp, humidity, rain)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════
# FORECAST ROUTE
# ═══════════════════════════════════════
@weather_bp.route("/forecast", methods=["GET"])
@jwt_required()
def forecast():
    district = request.args.get("district", "ilam").lower()

    if district not in DISTRICTS:
        return jsonify({"error": "Invalid district"}), 400

    info = DISTRICTS[district]
    lat, lon = info["lat"], info["lon"]

    # ── Demo mode (no API key) ───────────────────────────────────────────
    if not WEATHER_API_KEY:
        from datetime import timedelta
        today = datetime.now()
        demo_days = []
        for i in range(1, 4):
            day = today + timedelta(days=i)
            temp = round(22 + i * 1.5, 1)
            humidity = 80 + i * 2
            rain = 3.0 * i
            risk = calc_risk(temp, humidity, rain)
            demo_days.append({
                "date": day.strftime("%b %d"),
                "avg_temp": temp,
                "avg_humidity": humidity,
                "avg_rain": rain,
                "overall": risk["overall"]
            })
        return jsonify({
            "demo": True,
            "location": info["name"],
            "daily_summary": demo_days,
            "risk_timeline": demo_days,
            "alert": "⚠️ Demo data — add WEATHER_API_KEY for live forecast"
        })

    # ── Live mode ────────────────────────────────────────────────────────
    try:
        resp = requests.get(FORECAST_URL, params={
            "lat": lat,
            "lon": lon,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        # Group 3-hour slots by date
        buckets = defaultdict(
            lambda: {"temps": [], "humidities": [], "rains": []})
        for item in data.get("list", []):
            date_str = datetime.fromtimestamp(item["dt"]).strftime("%b %d")
            buckets[date_str]["temps"].append(item["main"]["temp"])
            buckets[date_str]["humidities"].append(item["main"]["humidity"])
            buckets[date_str]["rains"].append(
                item.get("rain", {}).get("3h", 0))

        # Build daily_summary (next 3 days only)
        daily_summary = []
        risk_timeline = []
        for date_str, vals in list(buckets.items())[:3]:
            avg_temp = round(sum(vals["temps"]) / len(vals["temps"]), 1)
            avg_humidity = round(
                sum(vals["humidities"]) / len(vals["humidities"]), 1)
            avg_rain = round(sum(vals["rains"]), 1)
            risk = calc_risk(avg_temp, avg_humidity, avg_rain)

            daily_summary.append({
                "date":         date_str,
                "avg_temp":     avg_temp,
                "avg_humidity": avg_humidity,
                "avg_rain":     avg_rain,
            })
            risk_timeline.append({
                "date":    date_str,
                "overall": risk["overall"]
            })

        # Overall alert
        all_levels = [d["overall"] for d in risk_timeline]
        if "High" in all_levels:
            alert = "⚠️ High disease risk expected in the next 3 days. Take preventive action."
        elif "Medium" in all_levels:
            alert = "🟡 Moderate risk forecast. Monitor your crops closely."
        else:
            alert = "🟢 Low risk expected over the next 3 days."

        return jsonify({
            "demo":          False,
            "location":      info["name"],
            "daily_summary": daily_summary,
            "risk_timeline": risk_timeline,
            "alert":         alert
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════
# DISTRICTS LIST
# ═══════════════════════════════════════


@weather_bp.route("/weather/districts", methods=["GET"])
def districts():
    return jsonify({
        "districts": [
            {"key": k, "name": v["name"]}
            for k, v in DISTRICTS.items()
        ]
    })
