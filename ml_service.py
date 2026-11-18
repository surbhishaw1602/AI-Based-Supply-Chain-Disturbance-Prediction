
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd


# ==========================================
# DATABASE CONFIGURATION
# ==========================================

DB_PATH = Path(__file__).resolve().parent / "shipments.db"


# ==========================================
# INITIALIZE ML TRAINING TABLE
# ==========================================

def initialize_ml_database():

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_training_data (

                record_id INTEGER PRIMARY KEY AUTOINCREMENT,

                observation_date TEXT NOT NULL,

                shipment_id INTEGER,

                route_segment TEXT NOT NULL,

                distance_km REAL DEFAULT 0,

                temperature REAL DEFAULT 0,

                rainfall REAL DEFAULT 0,

                wind_speed REAL DEFAULT 0,

                severe_weather INTEGER DEFAULT 0,

                festival_nearby INTEGER DEFAULT 0,

                rss_disturbance INTEGER DEFAULT 0,

                rss_severity TEXT DEFAULT 'None',

                disturbance_type TEXT DEFAULT 'None',

                verified_disruption INTEGER NOT NULL,

                verification_source TEXT NOT NULL,

                notes TEXT,

                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_ml_observation_date
            ON ml_training_data(observation_date)
        """)

        conn.commit()


# ==========================================
# VALIDATE TRAINING RECORD
# ==========================================

def validate_training_record(data):

    required_fields = [
        "observation_date",
        "route_segment",
        "verified_disruption",
        "verification_source"
    ]

    for field in required_fields:

        if data.get(field) is None or str(data[field]).strip() == "":

            raise ValueError(
                f"{field} is required."
            )

    try:

        datetime.strptime(
            str(data["observation_date"]),
            "%Y-%m-%d"
        )

    except ValueError:

        raise ValueError(
            "Date must use YYYY-MM-DD format."
        )

    label = int(data["verified_disruption"])

    if label not in (0, 1):

        raise ValueError(
            "verified_disruption must be 0 or 1."
        )

    numeric_fields = [
        "distance_km",
        "temperature",
        "rainfall",
        "wind_speed"
    ]

    for field in numeric_fields:

        value = float(data.get(field, 0))

        if value != value:
            raise ValueError(f"{field} cannot be NaN.")

        if value < 0 and field != "temperature":
            raise ValueError(
                f"{field} cannot be negative."
            )

    for field in [
        "severe_weather",
        "festival_nearby",
        "rss_disturbance"
    ]:

        if int(data.get(field, 0)) not in (0, 1):

            raise ValueError(
                f"{field} must be 0 or 1."
            )


# ==========================================
# SAVE VERIFIED TRAINING RECORD
# ==========================================

def add_training_record(data):

    validate_training_record(data)

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            INSERT INTO ml_training_data (

                observation_date,
                shipment_id,
                route_segment,
                distance_km,
                temperature,
                rainfall,
                wind_speed,
                severe_weather,
                festival_nearby,
                rss_disturbance,
                rss_severity,
                disturbance_type,
                verified_disruption,
                verification_source,
                notes,
                created_at

            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            str(data["observation_date"]),

            data.get("shipment_id"),

            str(data["route_segment"]).strip(),

            float(data.get("distance_km", 0)),

            float(data.get("temperature", 0)),

            float(data.get("rainfall", 0)),

            float(data.get("wind_speed", 0)),

            int(data.get("severe_weather", 0)),

            int(data.get("festival_nearby", 0)),

            int(data.get("rss_disturbance", 0)),

            str(data.get("rss_severity", "None")),

            str(data.get("disturbance_type", "None")),

            int(data["verified_disruption"]),

            str(data["verification_source"]).strip(),

            str(data.get("notes", "")),

            datetime.now().isoformat()

        ))

        conn.commit()


# ==========================================
# RETRIEVE TRAINING DATA
# ==========================================

def get_training_data():

    initialize_ml_database()

    with sqlite3.connect(DB_PATH) as conn:

        df = pd.read_sql_query("""
            SELECT *
            FROM ml_training_data
            ORDER BY observation_date DESC
        """, conn)

    return df


# ==========================================
# TRAINING DATA SUMMARY
# ==========================================

def get_dataset_summary():

    df = get_training_data()

    if df.empty:

        return {
            "total_records": 0,
            "disrupted": 0,
            "not_disrupted": 0,
            "ready": False
        }

    disrupted = int(
        (df["verified_disruption"] == 1).sum()
    )

    not_disrupted = int(
        (df["verified_disruption"] == 0).sum()
    )

    return {

        "total_records": len(df),

        "disrupted": disrupted,

        "not_disrupted": not_disrupted,

        "ready": (
            disrupted >= 10
            and not_disrupted >= 10
        )
    }


# ==========================================
# INITIALIZE ON IMPORT
# ==========================================

initialize_ml_database()


# ==========================================
# DAY 4 - RANDOM FOREST MODEL TRAINING
# ==========================================

MODEL_PATH = Path(__file__).resolve().parent / "random_forest_model.joblib"


def train_random_forest_model():

    """
    Evaluate and train a Random Forest classifier
    using verified historical training records.
    """

    import numpy as np
    import joblib

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix
    )

    # --------------------------------------
    # LOAD DATASET
    # --------------------------------------

    df = get_training_data()

    if df.empty:
        raise ValueError(
            "No training records are available."
        )

    # --------------------------------------
    # CHECK CLASS BALANCE
    # --------------------------------------

    class_counts = df["verified_disruption"].value_counts()

    disrupted_count = int(class_counts.get(1, 0))
    normal_count = int(class_counts.get(0, 0))

    if disrupted_count < 10 or normal_count < 10:

        raise ValueError(
            "Insufficient training data. You need at least "
            "10 verified disruptions and 10 verified "
            "non-disruptions."
        )

    # --------------------------------------
    # SELECT MODEL FEATURES
    # --------------------------------------

    feature_columns = [
        "distance_km",
        "temperature",
        "rainfall",
        "wind_speed",
        "severe_weather",
        "festival_nearby",
        "rss_disturbance"
    ]

    missing_columns = [
        column for column in feature_columns + ["verified_disruption"]
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing dataset columns: "
            + ", ".join(missing_columns)
        )

    X = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    y = pd.to_numeric(
        df["verified_disruption"],
        errors="coerce"
    )

    # --------------------------------------
    # VALIDATE MODEL INPUTS
    # --------------------------------------

    if X.isnull().any().any() or y.isnull().any():

        raise ValueError(
            "Training data contains missing or invalid numeric values."
        )

    if not np.isfinite(X.to_numpy(dtype=float)).all():

        raise ValueError(
            "Training features contain infinite or invalid values."
        )

    if not set(y.unique()).issubset({0, 1}):

        raise ValueError(
            "Training labels must contain only 0 and 1."
        )

    y = y.astype(int)

    # --------------------------------------
    # SPLIT DATA FOR EVALUATION
    # --------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # --------------------------------------
    # CREATE RANDOM FOREST CLASSIFIER
    # --------------------------------------

    evaluation_model = RandomForestClassifier(

        n_estimators=300,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    )

    # --------------------------------------
    # EVALUATE MODEL
    # --------------------------------------

    evaluation_model.fit(
        X_train,
        y_train
    )

    predictions = evaluation_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[
            "No Disruption",
            "Disruption"
        ],
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    # --------------------------------------
    # TRAIN FINAL MODEL ON ALL VERIFIED DATA
    # --------------------------------------

    final_model = RandomForestClassifier(

        n_estimators=300,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    )

    final_model.fit(
        X,
        y
    )

    # --------------------------------------
    # SAVE MODEL
    # --------------------------------------

    model_data = {

        "model": final_model,

        "feature_columns": feature_columns,

        "training_records": len(df),

        "disrupted_records": disrupted_count,

        "non_disrupted_records": normal_count,

        "trained_at": datetime.now().isoformat(),

        "accuracy": float(accuracy),

        "balanced_accuracy": float(balanced_accuracy)

    }

    joblib.dump(
        model_data,
        MODEL_PATH
    )

    return {

        "success": True,

        "accuracy": float(accuracy),

        "balanced_accuracy": float(balanced_accuracy),

        "classification_report": report,

        "confusion_matrix": matrix.tolist(),

        "training_records": len(df),

        "disrupted_records": disrupted_count,

        "non_disrupted_records": normal_count,

        "model_path": str(MODEL_PATH),

        "trained_at": model_data["trained_at"]

    }


# ==========================================
# CHECK SAVED RANDOM FOREST MODEL
# ==========================================

def get_random_forest_model_status(model_type="verified", shipment_id=None):

    """Return status for either the verified-data or synthetic demo model."""
    import joblib

    selected_path = _synthetic_model_path(shipment_id) if model_type == "synthetic" else MODEL_PATH

    if not selected_path.exists():

        return {
            "trained": False,
            "message": "Random Forest model has not been trained yet."
        }

    try:

        model_data = joblib.load(selected_path)

        return {

            "trained": True,

            "model_type": model_data.get("model_type", "verified"),

            "training_records": model_data["training_records"],

            "trained_at": model_data["trained_at"],

            "accuracy": model_data["accuracy"],

            "balanced_accuracy": model_data["balanced_accuracy"],
            "precision": model_data.get("precision"),
            "recall": model_data.get("recall"),
            "f1": model_data.get("f1"),
            "test_records": model_data.get("test_records"),
            "classification_report": model_data.get("classification_report"),
            "confusion_matrix": model_data.get("confusion_matrix"),
            "feature_importance": model_data.get("feature_importance", {})

        }

    except Exception as error:

        return {

            "trained": False,

            "message": f"Could not load saved model: {error}"

        }


# ==================================================
# AUTOMATIC FEATURE COLLECTION (LABELS MAY BE UNKNOWN)
# ==================================================

def initialize_auto_observation_database():
    """Store automatically collected features separately from verified labels."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_auto_observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_key TEXT NOT NULL UNIQUE,
                observation_date TEXT NOT NULL,
                shipment_id INTEGER NOT NULL,
                route_segment TEXT NOT NULL,
                distance_km REAL,
                temperature REAL,
                rainfall REAL,
                wind_speed REAL,
                severe_weather INTEGER,
                festival_nearby INTEGER,
                rss_disturbance INTEGER,
                rss_severity TEXT,
                disturbance_type TEXT,
                verified_disruption INTEGER,
                verification_source TEXT,
                collection_status TEXT NOT NULL DEFAULT 'Unverified',
                evidence_notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_auto_date
            ON ml_auto_observations(observation_date)
        """)
        conn.commit()


def save_auto_observation(data):
    """Idempotently insert/update features; never invent a binary label."""
    initialize_auto_observation_database()
    required = ('observation_key', 'observation_date', 'shipment_id', 'route_segment')
    for key in required:
        if data.get(key) in (None, ''):
            raise ValueError(f'{key} is required for an automatic observation.')
    now = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO ml_auto_observations (
                observation_key, observation_date, shipment_id, route_segment,
                distance_km, temperature, rainfall, wind_speed, severe_weather,
                festival_nearby, rss_disturbance, rss_severity, disturbance_type,
                verified_disruption, verification_source, collection_status,
                evidence_notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 'Unverified', ?, ?, ?)
            ON CONFLICT(observation_key) DO UPDATE SET
                distance_km=excluded.distance_km,
                temperature=excluded.temperature,
                rainfall=excluded.rainfall,
                wind_speed=excluded.wind_speed,
                severe_weather=excluded.severe_weather,
                festival_nearby=excluded.festival_nearby,
                rss_disturbance=excluded.rss_disturbance,
                rss_severity=excluded.rss_severity,
                disturbance_type=excluded.disturbance_type,
                evidence_notes=excluded.evidence_notes,
                updated_at=excluded.updated_at
        """, (
            str(data['observation_key']), str(data['observation_date']),
            int(data['shipment_id']), str(data['route_segment']),
            data.get('distance_km'), data.get('temperature'), data.get('rainfall'),
            data.get('wind_speed'), data.get('severe_weather'),
            data.get('festival_nearby'), data.get('rss_disturbance'),
            data.get('rss_severity'), data.get('disturbance_type'),
            'Automatic weather/event signals only; outcome not independently verified',
            str(data.get('evidence_notes', '')), now, now
        ))
        conn.commit()


def get_auto_observations():
    initialize_auto_observation_database()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            'SELECT * FROM ml_auto_observations ORDER BY observation_date DESC, shipment_id DESC',
            conn
        )



def verify_auto_observation(observation_id, verified_disruption, verification_source, notes="", feature_values=None):
    """Promote one reviewed automatic observation into verified training data once."""
    label = int(verified_disruption)
    if label not in (0, 1):
        raise ValueError("Outcome must be Disruption or No disruption.")
    source = str(verification_source or "").strip()
    if len(source) < 5:
        raise ValueError("Enter a specific evidence source (at least 5 characters).")

    feature_columns = [
        "distance_km", "temperature", "rainfall", "wind_speed",
        "severe_weather", "festival_nearby", "rss_disturbance"
    ]
    values = dict(feature_values or {})
    initialize_auto_observation_database()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM ml_auto_observations WHERE observation_id=?",
            (int(observation_id),)
        ).fetchone()
        if row is None:
            raise ValueError("The selected automatic observation no longer exists.")
        if row["collection_status"] == "Verified":
            raise ValueError("This observation has already been verified and promoted.")

        features = {}
        for column in feature_columns:
            raw = values.get(column, row[column])
            if raw is None or str(raw).strip() == "":
                raise ValueError(f"{column} is missing. Supply or confirm this feature before promotion.")
            number = float(raw)
            if number != number:
                raise ValueError(f"{column} must be a valid number.")
            if column != "temperature" and number < 0:
                raise ValueError(f"{column} cannot be negative.")
            if column in ("severe_weather", "festival_nearby", "rss_disturbance") and number not in (0, 1):
                raise ValueError(f"{column} must be 0 or 1.")
            features[column] = int(number) if column in ("severe_weather", "festival_nearby", "rss_disturbance") else number

        now = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO ml_training_data (
                observation_date, shipment_id, route_segment, distance_km,
                temperature, rainfall, wind_speed, severe_weather, festival_nearby,
                rss_disturbance, rss_severity, disturbance_type, verified_disruption,
                verification_source, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["observation_date"], row["shipment_id"], row["route_segment"],
            features["distance_km"], features["temperature"], features["rainfall"],
            features["wind_speed"], features["severe_weather"], features["festival_nearby"],
            features["rss_disturbance"], row["rss_severity"] or "Unknown",
            row["disturbance_type"] or "Unknown", label, source,
            str(notes or "").strip(), now
        ))
        conn.execute("""
            UPDATE ml_auto_observations SET
                distance_km=?, temperature=?, rainfall=?, wind_speed=?, severe_weather=?,
                festival_nearby=?, rss_disturbance=?, verified_disruption=?,
                verification_source=?, collection_status='Verified', evidence_notes=?, updated_at=?
            WHERE observation_id=?
        """, (
            features["distance_km"], features["temperature"], features["rainfall"],
            features["wind_speed"], features["severe_weather"], features["festival_nearby"],
            features["rss_disturbance"], label, source, str(notes or "").strip(), now,
            int(observation_id)
        ))
        conn.commit()
    return True

def get_auto_dataset_summary():
    df = get_auto_observations()
    return {
        'total_records': int(len(df)),
        'unverified': int((df['verified_disruption'].isna()).sum()) if not df.empty else 0,
        'verified_disruptions': int((df['verified_disruption'] == 1).sum()) if not df.empty else 0,
        'verified_no_disruption': int((df['verified_disruption'] == 0).sum()) if not df.empty else 0,
    }


initialize_auto_observation_database()


# ==========================================
# BACKWARD-COMPATIBLE PREDICTION FUNCTIONS
# ==========================================

def _load_trained_model(model_type="verified", shipment_id=None):
    """Load the requested model without mixing verified and synthetic models."""
    import joblib

    selected_path = _synthetic_model_path(shipment_id) if model_type == "synthetic" else MODEL_PATH
    if model_type not in ("verified", "synthetic"):
        raise ValueError("model_type must be 'verified' or 'synthetic'.")

    if not selected_path.exists():
        if model_type == "synthetic":
            raise ValueError(
                "The synthetic demonstration model has not been trained. "
                "Generate the synthetic dataset and train the experimental model first."
            )
        raise ValueError(
            "The verified-data Random Forest model has not been trained yet. "
            "Collect independently verified records for both classes and train the model first."
        )

    try:
        model_data = joblib.load(selected_path)
    except Exception as error:
        raise ValueError(f"Could not load the saved Random Forest model: {error}")

    if not isinstance(model_data, dict) or "model" not in model_data:
        raise ValueError("The saved model file has an unsupported format. Retrain the model.")

    return model_data


def _prepare_prediction_frame(records, feature_columns):
    """Validate and order prediction features exactly as during training."""
    if isinstance(records, dict):
        records = [records]

    if not records:
        raise ValueError("No prediction records were supplied.")

    frame = pd.DataFrame(records)
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError("Missing prediction features: " + ", ".join(missing))

    frame = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    if frame.isnull().any().any():
        raise ValueError("Prediction features contain missing or non-numeric values.")

    return frame


def predict_disturbance_risk(features, model_type="verified", shipment_id=None):
    """Predict disruption risk for one feature dictionary.

    Returns the probability as a percentage plus a development risk band.
    """
    model_data = _load_trained_model(model_type=model_type, shipment_id=shipment_id)
    feature_columns = model_data.get("feature_columns", [
        "distance_km", "temperature", "rainfall", "wind_speed",
        "severe_weather", "festival_nearby", "rss_disturbance"
    ])
    frame = _prepare_prediction_frame(features, feature_columns)
    model = model_data["model"]

    probabilities = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    disruption_probability = (
        float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    )
    percentage = round(disruption_probability * 100, 2)

    if percentage < 30:
        level = "Low"
    elif percentage < 60:
        level = "Medium"
    else:
        level = "High"

    return {
        "risk_probability": disruption_probability,
        "risk_percentage": percentage,
        "risk_level": level,
        "prediction": int(model.predict(frame)[0]),
        "is_calibrated": False,
    }


def predict_seven_day_risk(forecast_records, model_type="verified", shipment_id=None):
    """Predict risk for each supplied forecast row, preserving its date."""
    if isinstance(forecast_records, dict):
        forecast_records = [forecast_records]

    if not forecast_records:
        return []

    model_data = _load_trained_model(model_type=model_type, shipment_id=shipment_id)
    feature_columns = model_data.get("feature_columns", [
        "distance_km", "temperature", "rainfall", "wind_speed",
        "severe_weather", "festival_nearby", "rss_disturbance"
    ])
    frame = _prepare_prediction_frame(forecast_records, feature_columns)
    model = model_data["model"]

    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)
    disruption_index = classes.index(1) if 1 in classes else None
    predictions = []

    for index, record in enumerate(forecast_records):
        probability = (
            float(probabilities[index][disruption_index])
            if disruption_index is not None else 0.0
        )
        percentage = round(probability * 100, 2)
        if percentage < 30:
            level = "Low"
        elif percentage < 60:
            level = "Medium"
        else:
            level = "High"

        predictions.append({
            "date": str(record.get("date", "")),
            "risk_probability": probability,
            "risk_percentage": percentage,
            "risk_level": level,
            "prediction": int(model.predict(frame.iloc[[index]])[0]),
            "is_calibrated": False,
        })

    return predictions


# ==================================================
# HYBRID ML: ISOLATED SYNTHETIC DEVELOPMENT DATASET
# ==================================================

SYNTHETIC_MODEL_PATH = Path(__file__).resolve().parent / "random_forest_synthetic_model.joblib"
SYNTHETIC_FEATURE_COLUMNS = [
    "distance_km", "temperature", "rainfall", "wind_speed",
    "severe_weather", "festival_nearby", "rss_disturbance"
]


def _synthetic_model_path(shipment_id=None):
    """Return a distinct persisted model path for each shipment."""
    if shipment_id is None:
        return SYNTHETIC_MODEL_PATH
    return Path(__file__).resolve().parent / f"random_forest_synthetic_shipment_{int(shipment_id)}.joblib"


def initialize_synthetic_database():
    """Create/migrate a synthetic-only table, isolated by shipment_id."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_synthetic_training_data (
                synthetic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                shipment_id INTEGER,
                distance_km REAL NOT NULL,
                temperature REAL NOT NULL,
                rainfall REAL NOT NULL,
                wind_speed REAL NOT NULL,
                severe_weather INTEGER NOT NULL,
                festival_nearby INTEGER NOT NULL,
                rss_disturbance INTEGER NOT NULL,
                synthetic_disruption INTEGER NOT NULL,
                generation_seed INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(ml_synthetic_training_data)")}
        if "shipment_id" not in columns:
            conn.execute("ALTER TABLE ml_synthetic_training_data ADD COLUMN shipment_id INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ml_synthetic_shipment ON ml_synthetic_training_data(shipment_id)")
        conn.commit()


def get_synthetic_training_data(shipment_id=None):
    initialize_synthetic_database()
    with sqlite3.connect(DB_PATH) as conn:
        if shipment_id is None:
            query = "SELECT * FROM ml_synthetic_training_data ORDER BY synthetic_id"
            return pd.read_sql_query(query, conn)
        return pd.read_sql_query(
            "SELECT * FROM ml_synthetic_training_data WHERE shipment_id = ? ORDER BY synthetic_id",
            conn, params=(int(shipment_id),)
        )


def get_synthetic_dataset_summary(shipment_id=None):
    df = get_synthetic_training_data(shipment_id)
    if df.empty:
        return {"total_records": 0, "disrupted": 0, "not_disrupted": 0, "ready": False}
    disrupted = int((df["synthetic_disruption"] == 1).sum())
    not_disrupted = int((df["synthetic_disruption"] == 0).sum())
    return {
        "total_records": int(len(df)), "disrupted": disrupted,
        "not_disrupted": not_disrupted,
        "ready": disrupted >= 10 and not_disrupted >= 10,
    }


def generate_synthetic_dataset(n_records=1000, random_state=42, shipment_id=None, route_distance_km=None):
    """Generate a separate simulated dataset for one shipment; never real outcomes."""
    import numpy as np
    if shipment_id is None:
        raise ValueError("shipment_id is required for shipment-specific synthetic data.")
    if int(n_records) < 100 or int(n_records) > 20000:
        raise ValueError("Generate between 100 and 20,000 synthetic records.")

    shipment_id = int(shipment_id)
    seed = (int(random_state) + shipment_id * 1009) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    n = int(n_records)
    route_distance = float(route_distance_km or 100.0)
    if not np.isfinite(route_distance) or route_distance <= 0:
        route_distance = 100.0

    # Shipment-specific route distance is the anchor; each row simulates a
    # scenario around that shipment's route rather than using one global range.
    distance = np.clip(rng.normal(route_distance, max(route_distance * 0.12, 2.0), n), 1, 3000)
    temperature = np.clip(rng.normal(29, 7, n), 5, 48)
    rainfall = np.clip(rng.gamma(shape=1.4, scale=13, size=n), 0, 180)
    wind = np.clip(rng.gamma(shape=2.0, scale=9, size=n), 0, 110)
    severe_probability = np.clip(0.02 + rainfall / 240 + wind / 260, 0.02, 0.78)
    severe = rng.binomial(1, severe_probability)
    festival = rng.binomial(1, 0.18, n)
    rss = rng.binomial(1, np.clip(0.07 + severe * 0.13, 0.03, 0.35))
    logit = (-2.2 + 0.027 * rainfall + 0.023 * wind + 1.40 * severe
             + 0.95 * rss + 0.30 * festival + 0.00015 * distance
             + 0.55 * ((rainfall > 45) & (wind > 28)))
    probability = np.clip(1.0 / (1.0 + np.exp(-logit)), 0.025, 0.92)
    labels = rng.binomial(1, probability)
    frame = pd.DataFrame({
        "shipment_id": shipment_id,
        "distance_km": np.round(distance, 2),
        "temperature": np.round(temperature, 1),
        "rainfall": np.round(rainfall, 1),
        "wind_speed": np.round(wind, 1),
        "severe_weather": severe.astype(int),
        "festival_nearby": festival.astype(int),
        "rss_disturbance": rss.astype(int),
        "synthetic_disruption": labels.astype(int),
        "generation_seed": seed,
        "created_at": datetime.now().isoformat(),
    })
    initialize_synthetic_database()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM ml_synthetic_training_data WHERE shipment_id = ?", (shipment_id,))
        frame.to_sql("ml_synthetic_training_data", conn, if_exists="append", index=False)
        conn.commit()
    model_path = _synthetic_model_path(shipment_id)
    if model_path.exists():
        model_path.unlink()
    return get_synthetic_dataset_summary(shipment_id)


def train_synthetic_random_forest_model(shipment_id=None):
    """Train/evaluate a shipment-specific experimental model on simulated rows."""
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
        classification_report, confusion_matrix, precision_score, recall_score, f1_score)
    if shipment_id is None:
        raise ValueError("shipment_id is required to train a shipment-specific model.")
    shipment_id = int(shipment_id)
    df = get_synthetic_training_data(shipment_id)
    if len(df) < 100:
        raise ValueError("Generate at least 100 synthetic records for this shipment before training.")
    counts = df["synthetic_disruption"].value_counts()
    if int(counts.get(0, 0)) < 10 or int(counts.get(1, 0)) < 10:
        raise ValueError("This shipment's synthetic sample needs at least 10 examples of each class. Regenerate it.")
    X = df[SYNTHETIC_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["synthetic_disruption"], errors="coerce").astype(int)
    if X.isnull().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Synthetic features contain missing or invalid values.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    evaluation_model = RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
    evaluation_model.fit(X_train, y_train)
    predicted = evaluation_model.predict(X_test)
    final_model = RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
    final_model.fit(X, y)
    model_data = {
        "model": final_model, "model_type": "synthetic", "shipment_id": shipment_id,
        "feature_columns": SYNTHETIC_FEATURE_COLUMNS, "training_records": int(len(df)),
        "disrupted_records": int((y == 1).sum()), "non_disrupted_records": int((y == 0).sum()),
        "trained_at": datetime.now().isoformat(),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "test_records": int(len(y_test)),
        "classification_report": classification_report(y_test, predicted, labels=[0, 1], target_names=["No Disruption (synthetic)", "Disruption (synthetic)"], zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "feature_importance": {name: float(value) for name, value in zip(SYNTHETIC_FEATURE_COLUMNS, final_model.feature_importances_)},
        "disclaimer": "Synthetic simulation only; not evidence of real-world predictive performance.",
    }
    model_path = _synthetic_model_path(shipment_id)
    joblib.dump(model_data, model_path)
    return {
        "success": True, "model_type": "synthetic", "shipment_id": shipment_id,
        "accuracy": model_data["accuracy"], "balanced_accuracy": model_data["balanced_accuracy"],
        "precision": model_data["precision"], "recall": model_data["recall"], "f1": model_data["f1"],
        "classification_report": classification_report(y_test, predicted, labels=[0, 1], target_names=["No Disruption (synthetic)", "Disruption (synthetic)"], zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "training_records": int(len(df)), "disrupted_records": int((y == 1).sum()),
        "non_disrupted_records": int((y == 0).sum()), "test_records": int(len(y_test)),
        "feature_importance": model_data["feature_importance"], "trained_at": model_data["trained_at"],
        "model_path": str(model_path),
    }


initialize_synthetic_database()
