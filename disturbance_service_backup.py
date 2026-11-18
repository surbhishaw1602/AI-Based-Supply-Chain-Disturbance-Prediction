"""
RSS-based disturbance monitoring for the Shipment Management System.

Safety rules:
- RSS items are unverified information signals, not proof of a road closure.
- A place is recorded only when it is explicitly present in the article text
  and appears in an event-related context.
- Institutional/venue mentions (for example, "Madras High Court") are not
  automatically treated as the location of the reported strike.
- Existing SQLite schema and records are preserved.
"""

import hashlib
import html
import re
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser
import requests


DB_PATH = Path(__file__).resolve().parent / "shipments.db"
INDIA_TZ = ZoneInfo("Asia/Kolkata")
REQUEST_TIMEOUT = 20
MAX_ENTRIES_PER_FEED = 50

RSS_FEEDS = [
    {
        "name": "Press Information Bureau",
        "url": "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=1",
    },
    {
        "name": "Amar Ujala West Bengal",
        "url": "https://www.amarujala.com/rss/west-bengal.xml",
    },
    {
        "name": "The Hindu National",
        "url": "https://www.thehindu.com/news/national/feeder/default.rss",
    },
    {
        "name": "The Indian Express",
        "url": "https://indianexpress.com/feed/",
    },
]

DISTURBANCE_KEYWORDS = {
    "Strike / Bandh": [
        "strike", "strikes", "workers strike", "employees strike",
        "bandh", "shutdown", "shut down", "work stoppage", "walkout",
        "proposed strike", "on strike",
    ],
    "Road Closure / Traffic": [
        "road closed", "road closure", "traffic diversion", "traffic jam",
        "highway closed", "bridge closed", "blocked road", "route blocked",
        "traffic disruption", "road blocked",
    ],
    "Protest / Demonstration": [
        "protest", "protests", "demonstration", "rally", "agitation",
        "march", "sit-in", "sit in", "road blockade", "blockade",
    ],
    "Weather / Natural Disaster": [
        "flood", "flooding", "cyclone", "landslide", "heavy rain",
        "extreme rainfall", "storm", "heatwave", "heat wave",
        "cloudburst", "natural disaster",
    ],
    "Transport Disruption": [
        "train cancelled", "trains cancelled", "flight cancelled",
        "flight cancellation", "rail disruption", "train services",
        "bus services suspended", "transport disruption",
        "services suspended", " ferry services",
    ],
    "Public Event / Crowd Disruption": [
        "large gathering", "public event", "festival procession",
        "religious procession", "crowd", "mass gathering",
        "marathon", "public meeting",
    ],
}

IRRELEVANT_PATTERNS = [
    r"\b(stock market|share price|earnings|quarterly results)\b",
    r"\b(movie review|film review|celebrity|fashion)\b",
    r"\b(cricket score|match result|football score)\b",
    r"\b(places to visit|travel guide|holiday destination|tourist attractions)\b",
    r"\b(strikes? a chord|strikes? gold|striking distance|strike rate|strike price)\b",
    r"\b(legal awareness rally|poll bugle|election campaign)\b",
]

HISTORICAL_CUES = [
    "years ago", "last year", "in 2019", "in 2020", "in 2021",
    "in 2022", "in 2023", "in 2024", "in 2025", "anniversary of",
    "remembering the", "on this day",
]

RESOLVED_CUES = [
    "strike called off", "strike withdrawn", "strike ends",
    "protest ends", "road reopened", "traffic restored",
    "normal services resumed", "blockade lifted",
]

# Specific places first. These are used only when the article text itself
# mentions them; no place is inferred from a publisher's location.
KNOWN_LOCATIONS = [
    # West Bengal
    "Kolkata", "Howrah", "Hooghly", "Durgapur", "Asansol", "Siliguri",
    "Darjeeling", "Kharagpur", "Haldia", "Bardhaman", "Burdwan",
    "Bankura", "Purulia", "Medinipur", "Paschim Medinipur",
    "Purba Medinipur", "Jalpaiguri", "Cooch Behar", "Malda",
    "Murshidabad", "Nadia", "North 24 Parganas", "South 24 Parganas",
    "Alipurduar", "Birbhum", "Kalimpong",
    # Odisha
    "Bhubaneswar", "Cuttack", "Puri", "Rourkela", "Sambalpur",
    "Berhampur", "Balasore", "Baleswar", "Jajpur", "Angul",
    "Jharsuguda", "Koraput", "Kendrapara", "Khordha", "Prakasam", "West Godavari", "East Godavari",
    # Andhra Pradesh
    "Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Nellore",
    "Kurnool", "Anantapur", "Rajahmundry", "Kakinada", "NTR district",
    # Other Indian cities and districts
    "New Delhi", "Delhi", "Mumbai", "Pune", "Nagpur", "Nashik",
    "Chennai", "Bengaluru", "Bangalore", "Hyderabad", "Patna",
    "Ranchi", "Leh", "Kargil", "Belagavi", "Belgaum", "Sangrur",
    # Geographic names that may be broad, ambiguous or non-city references.
    "Siang", "East Siang", "West Siang", "Upper Siang", "Lower Siang",
    "Amritsar", "Ludhiana", "Jalandhar", "Chandigarh", "Jaipur",
    "Jodhpur", "Udaipur", "Ahmedabad", "Surat", "Vadodara",
    "Bhopal", "Indore", "Gwalior", "Lucknow", "Kanpur", "Varanasi",
    "Prayagraj", "Agra", "Dehradun", "Haridwar", "Kochi",
    "Thiruvananthapuram", "Kozhikode", "Coimbatore", "Madurai",
    "Mysuru", "Mangalore", "Guwahati", "Shillong", "Imphal",
    "Aizawl", "Kohima", "Itanagar", "Gangtok", "Srinagar", "Jammu",
    # Broad regions/states are recognized but deliberately not used as
    # precise route-match locations.
    "West Bengal", "Odisha", "Andhra Pradesh", "Karnataka",
    "Tamil Nadu", "Kerala", "Maharashtra", "Punjab", "Uttarakhand",
    "Ladakh", "Madhya Pradesh", "Rajasthan", "Gujarat", "Jharkhand",
    "Bihar", "Telangana", "Himachal Pradesh", "Uttar Pradesh",
    "Assam", "Goa",
]

LOCATION_ALIASES = {
    "Bangalore": "Bengaluru",
    "Belgaum": "Belagavi",
    "Burdwan": "Bardhaman",
    "Baleswar": "Balasore",
}

BROAD_LOCATIONS = {
    "West Bengal", "Odisha", "Andhra Pradesh", "Karnataka",
    "Tamil Nadu", "Kerala", "Maharashtra", "Punjab", "Uttarakhand",
    "Ladakh", "Madhya Pradesh", "Rajasthan", "Gujarat", "Jharkhand",
    "Bihar", "Telangana", "Himachal Pradesh", "Uttar Pradesh",
    "Assam", "Goa",
}

# These names may identify a district/large area or an ambiguous geographic
# feature rather than a specific settlement. Keep them visible, but never use
# them alone to generate a shipment-route proximity alert.
LOW_PRECISION_LOCATIONS = {
    "siang", "east siang", "west siang", "upper siang", "lower siang",
    "prakasam", "west godavari", "east godavari", "ntr district",
    "north 24 parganas", "south 24 parganas", "paschim medinipur",
    "purba medinipur", "hooghly", "murshidabad", "nadia", "bankura",
    "purulia", "birbhum", "jalpaiguri", "cooch behar", "alipurduar",
    "kalimpong", "malda", "jajpur", "angul", "jharsuguda", "koraput",
    "kendrapara", "khordha", "sangrur",
}

ROUTE_LOCALITY_TYPES = {
    "city", "town", "village", "suburb", "neighbourhood", "hamlet",
    "municipality",
}

# Court, government-office and publisher names are not event locations.
# In particular, "Madras High Court" must not create a Chennai location.
INSTITUTION_PATTERNS = [
    r"\bMadras High Court\b",
    r"\bCalcutta High Court\b",
    r"\bDelhi High Court\b",
    r"\bSupreme Court\b",
    r"\bHigh Court\b",
    r"\bDistrict Court\b",
    r"\bSessions Court\b",
    r"\bPress Information Bureau\b",
]


def get_connection():
    """Open the existing database without deleting or recreating it."""
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_disturbance_database():
    """Create the disturbance table only if it does not already exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS disturbance_events (
                disturbance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT UNIQUE,
                title TEXT NOT NULL,
                summary TEXT,
                source_name TEXT,
                source_url TEXT,
                published_at TEXT,
                event_date TEXT,
                location TEXT,
                event_type TEXT,
                severity TEXT,
                status TEXT DEFAULT 'Unverified',
                fetched_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_disturbance_published
            ON disturbance_events(published_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_disturbance_external
            ON disturbance_events(external_id)
            """
        )


def clean_text(value):
    if value is None:
        return ""
    value = html.unescape(str(value))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _normalized(value):
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _keyword_category(text):
    """Match complete words/phrases and avoid figurative or incidental uses."""
    normalized = _normalized(text)

    # These expressions use "strike" figuratively, not as industrial action.
    figurative = (
        r"\b(strikes? a chord|strikes? gold|striking distance|"
        r"strike rate|strike price)\b"
    )
    normalized = re.sub(figurative, " ", normalized)

    for category, keywords in DISTURBANCE_KEYWORDS.items():
        for keyword in keywords:
            phrase = _normalized(keyword)
            if not phrase:
                continue
            if re.search(
                rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
                normalized,
            ):
                # A rally/march alone does not establish disruption.
                if phrase in {"rally", "march", "crowd", "public meeting"}:
                    continue
                return category
    return None


def is_irrelevant_report(title, summary=""):
    text = _normalized(f"{title} {summary}")
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in IRRELEVANT_PATTERNS)


def is_historical_report(title, summary="", published_at=None):
    text = _normalized(f"{title} {summary}")
    if any(cue in text for cue in HISTORICAL_CUES):
        return True

    # An article published long ago is not a current disruption signal.
    published = _parse_datetime(published_at)
    if published and published < datetime.now(timezone.utc) - timedelta(days=45):
        return True
    return False


def is_resolved_event(title, summary=""):
    text = _normalized(f"{title} {summary}")
    return any(_normalized(cue) in text for cue in RESOLVED_CUES)


def classify_disturbance(title, summary=""):
    text = _normalized(f"{title} {summary}")
    event_type = _keyword_category(text)
    if not event_type:
        return None, "Low"

    high_terms = [
        "killed", "fatal", "emergency", "evacuation", "major disruption",
        "indefinite strike", "highway closed", "national highway closed",
        "severe flood", "landslide", "cyclone warning",
    ]
    medium_terms = [
        "strike", "bandh", "protest", "road closure", "traffic diversion",
        "heavy rain", "services suspended", "blockade",
    ]
    if any(term in text for term in high_terms):
        severity = "High"
    elif any(term in text for term in medium_terms):
        severity = "Medium"
    else:
        severity = "Low"
    return event_type, severity


def _find_locations_in_text(text):
    """Return explicit known place names in order of first appearance."""
    source = clean_text(text)
    matches = []
    candidates = sorted(KNOWN_LOCATIONS, key=len, reverse=True)

    for place in candidates:
        pattern = rf"(?<![A-Za-z]){re.escape(place)}(?![A-Za-z])"
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            canonical = LOCATION_ALIASES.get(place, place)
            matches.append((match.start(), canonical, place))

    matches.sort(key=lambda item: item[0])
    result = []
    seen = set()
    for _, canonical, _ in matches:
        key = canonical.lower()
        if key not in seen:
            seen.add(key)
            result.append(canonical)
    return result


def _strip_institution_mentions(text):
    """Remove institutional names so their city is not misread as event venue."""
    result = clean_text(text)
    for pattern in INSTITUTION_PATTERNS:
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    return result


def _event_context_text(title, summary):
    """
    Prefer the headline for event-place evidence. Summary places are accepted
    only when they occur near disruption language, reducing incidental mentions.
    """
    title = clean_text(title)
    summary = clean_text(summary)
    title_without_institutions = _strip_institution_mentions(title)
    summary_without_institutions = _strip_institution_mentions(summary)

    title_places = _find_locations_in_text(title_without_institutions)
    if title_places:
        return title_places

    # If the title has no place, inspect summary sentences that contain an
    # event/disruption term. This avoids treating every place in a long article
    # as the location of the incident.
    event_words = (
        "strike", "bandh", "protest", "road", "traffic", "closure",
        "blocked", "flood", "rain", "cyclone", "landslide", "disruption",
        "suspended", "shutdown", "procession", "rally", "demonstration",
        "workers", "services",
    )
    sentences = re.split(r"(?<=[.!?])\s+", summary_without_institutions)
    selected = []
    for sentence in sentences:
        normalized = _normalized(sentence)
        if any(word in normalized for word in event_words):
            selected.extend(_find_locations_in_text(sentence))

    # Preserve order and remove duplicates.
    result = []
    seen = set()
    for place in selected:
        if place.lower() not in seen:
            seen.add(place.lower())
            result.append(place)
    return result


def extract_location_candidates(title, summary=""):
    """Return explicit event locations with a conservative precision score.

    Confidence is derived from the text evidence, not from the RSS publisher.
    Route matching is allowed only for known, non-ambiguous settlement names.
    This metadata is computed at runtime and does not alter the SQLite schema.
    """
    places = _event_context_text(title, summary)
    candidates = []
    for place in places:
        normalized = place.casefold()
        broad = place in BROAD_LOCATIONS
        low_precision = normalized in LOW_PRECISION_LOCATIONS
        district_label = "district" in normalized
        route_matchable = not (broad or low_precision or district_label)
        candidates.append({
            "location": place,
            "confidence": "low" if low_precision else ("medium" if broad or district_label else "high"),
            "evidence_source": "article_headline" if place in _find_locations_in_text(_strip_institution_mentions(title)) else "event_related_summary",
            "is_broad_region": broad,
            "is_route_matchable": route_matchable,
        })
    return candidates[:3]


def extract_location(title, summary="", source_url=""):
    """Return explicit event location(s), or the safe unknown placeholder."""
    candidates = extract_location_candidates(title, summary)
    precise = [
        item["location"] for item in candidates
        if not item["is_broad_region"]
    ]
    if precise:
        return ", ".join(precise[:3])

    # Broad states are useful descriptive metadata, but are not precise enough
    # for route proximity alerts. Keep the existing display format while making
    # it clear that the location cannot be used as a precise route match.
    broad = [item["location"] for item in candidates]
    if broad:
        return ", ".join(broad[:2])
    return "Location not identified"


def normalize_title(title):
    return _normalized(title)


def create_external_id(title, source_url="", published_at=""):
    """Stable headline-based key to avoid duplicates across feed sources."""
    normalized = normalize_title(title)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest


def _parse_datetime(value):
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = parsedate_to_datetime(str(value))
        except Exception:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except Exception:
                return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_datetime(value):
    parsed = _parse_datetime(value)
    if not parsed:
        return None
    return parsed.astimezone(INDIA_TZ).isoformat()


def _extract_event_date(title, summary, published_at):
    text = f"{clean_text(title)} {clean_text(summary)}"
    # Explicit ISO dates in the article text.
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
        except ValueError:
            pass

    # Otherwise use the publication date as a reference date, not as a claim
    # that the event itself occurs on that date.
    return None


def _entry_summary(entry):
    summary = entry.get("summary", "") or entry.get("description", "")
    return clean_text(summary)


def fetch_one_rss_feed(feed):
    """Fetch one RSS/Atom feed and return detailed diagnostics."""
    feed_name = feed["name"]
    feed_url = feed["url"]
    events = []
    status = {
        "source": feed_name,
        "status": "Failed",
        "scanned": 0,
        "relevant": 0,
        "new_saved": 0,
        "http_status": None,
        "content_type": "",
        "message": "",
    }

    try:
        response = requests.get(
            feed_url,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ShipmentManagementDashboard/2.1; RSS reader)"
            },
        )
        status["http_status"] = response.status_code
        status["content_type"] = response.headers.get("Content-Type", "")
        response.raise_for_status()

        parsed = feedparser.parse(response.content)
        entries = parsed.entries or []

        if not entries:
            status["status"] = "Empty"
            if getattr(parsed, "bozo", False):
                status["message"] = (
                    "Publisher responded, but RSS/Atom parsing failed: "
                    + str(getattr(parsed, "bozo_exception", "unknown parser error"))[:300]
                )
            else:
                status["message"] = (
                    "HTTP response received, but no RSS entries were found. "
                    "The publisher may have changed its feed format."
                )
            return events, status

        status["scanned"] = min(len(entries), MAX_ENTRIES_PER_FEED)

        for entry in entries[:MAX_ENTRIES_PER_FEED]:
            title = clean_text(entry.get("title", ""))
            summary = _entry_summary(entry)
            if not title or is_irrelevant_report(title, summary):
                continue

            event_type, severity = classify_disturbance(title, summary)
            if not event_type:
                continue

            published_raw = entry.get("published") or entry.get("updated")
            published_at = _format_datetime(published_raw)
            if is_historical_report(title, summary, published_at):
                continue

            resolved = is_resolved_event(title, summary)
            source_url = clean_text(entry.get("link", ""))
            location = extract_location(title, summary, source_url)
            event_date = _extract_event_date(title, summary, published_at)
            external_id = create_external_id(title, source_url, published_at)

            events.append({
                "external_id": external_id,
                "title": title,
                "summary": summary[:4000],
                "source_name": feed_name,
                "source_url": source_url,
                "published_at": published_at,
                "event_date": event_date,
                "location": location,
                "event_type": event_type,
                "severity": severity,
                "status": "Resolved" if resolved else "Unverified",
            })
            status["relevant"] += 1

        status["status"] = "Success"
        status["message"] = (
            "Feed parsed successfully."
            if not getattr(parsed, "bozo", False)
            else "Entries parsed with a feed-format warning: "
                 + str(getattr(parsed, "bozo_exception", ""))[:200]
        )

    except requests.exceptions.Timeout:
        status["message"] = f"Feed request timed out after {REQUEST_TIMEOUT} seconds."
    except requests.exceptions.HTTPError as exc:
        status["message"] = f"Publisher returned an HTTP error: {exc}"[:500]
    except requests.exceptions.RequestException as exc:
        status["message"] = f"Network/request error: {exc}"[:500]
    except Exception as exc:
        status["message"] = f"Feed processing error: {exc}"[:500]

    return events, status


def save_disturbance_event(event):
    """Insert a new event without overwriting existing records."""
    with get_connection() as connection:
        existing = connection.execute(
            "SELECT disturbance_id FROM disturbance_events WHERE external_id = ?",
            (event["external_id"],),
        ).fetchone()
        if existing:
            return False

        # Secondary duplicate guard for older rows whose external_id may differ.
        existing = connection.execute(
            """
            SELECT disturbance_id
            FROM disturbance_events
            WHERE lower(trim(title)) = lower(trim(?))
            LIMIT 1
            """,
            (event["title"],),
        ).fetchone()
        if existing:
            return False

        connection.execute(
            """
            INSERT INTO disturbance_events (
                external_id, title, summary, source_name, source_url,
                published_at, event_date, location, event_type, severity,
                status, fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.get("external_id"),
                event.get("title"),
                event.get("summary"),
                event.get("source_name"),
                event.get("source_url"),
                event.get("published_at"),
                event.get("event_date"),
                event.get("location", "Location not identified"),
                event.get("event_type"),
                event.get("severity", "Low"),
                event.get("status", "Unverified"),
                datetime.now(INDIA_TZ).isoformat(),
            ),
        )
        return True


def fetch_disturbance_feeds():
    """Fetch feeds, save new events, and explicitly report duplicate matches."""
    total_scanned = 0
    total_relevant = 0
    total_saved = 0
    total_duplicates = 0
    feed_statuses = []
    seen_external_ids = set()
    duplicate_articles = []
    newly_saved_articles = []

    for feed in RSS_FEEDS:
        events, status = fetch_one_rss_feed(feed)
        total_scanned += status["scanned"]
        total_relevant += status["relevant"]
        status["duplicates"] = 0

        for event in events:
            external_id = event["external_id"]

            # The same story can appear in more than one configured feed.
            if external_id in seen_external_ids:
                total_duplicates += 1
                status["duplicates"] += 1
                duplicate_articles.append({
                    "title": event.get("title", "Untitled"),
                    "source": event.get("source_name", feed["name"]),
                    "reason": "Repeated in another configured RSS feed during this scan",
                })
                continue

            seen_external_ids.add(external_id)

            if save_disturbance_event(event):
                total_saved += 1
                status["new_saved"] += 1
                newly_saved_articles.append({
                    "title": event.get("title", "Untitled"),
                    "source": event.get("source_name", feed["name"]),
                    "location": event.get("location", "Location not identified"),
                    "event_type": event.get("event_type", "Unknown"),
                    "severity": event.get("severity", "Low"),
                })
            else:
                # save_disturbance_event checks both stable external_id and
                # normalized headline against the existing SQLite history.
                total_duplicates += 1
                status["duplicates"] += 1
                duplicate_articles.append({
                    "title": event.get("title", "Untitled"),
                    "source": event.get("source_name", feed["name"]),
                    "reason": "Matching article already exists in the SQLite database",
                })

        feed_statuses.append(status)
        # Respect public feed servers and avoid rapid repeated requests.
        time.sleep(0.25)

    return {
        "articles_scanned": total_scanned,
        "relevant_articles": total_relevant,
        "new_disturbances_saved": total_saved,
        "duplicate_articles": total_duplicates,
        "duplicate_details": duplicate_articles,
        "duplicate_details_count": len(duplicate_articles),
        "newly_saved_details": newly_saved_articles,
        "feed_statuses": feed_statuses,
        "successful_feeds": sum(1 for item in feed_statuses if item["status"] == "Success"),
        "empty_feeds": sum(1 for item in feed_statuses if item["status"] == "Empty"),
        "failed_feeds": sum(1 for item in feed_statuses if item["status"] == "Failed"),
    }


def should_display_event(event):
    status = str(event.get("status", "Unverified")).strip().lower()
    return status not in {"resolved", "expired", "dismissed"}


def severity_rank(severity):
    return {"High": 3, "Medium": 2, "Low": 1}.get(str(severity).title(), 0)


def get_disturbance_events(
    event_type=None,
    severity=None,
    status=None,
    limit=200,
    include_resolved=False,
):
    """Retrieve active announcements and attach non-persistent location quality."""
    initialize_disturbance_database()
    query = "SELECT * FROM disturbance_events WHERE 1=1"
    params = []

    if event_type and event_type != "All":
        query += " AND event_type = ?"
        params.append(event_type)
    if severity and severity != "All":
        query += " AND severity = ?"
        params.append(severity)
    if status and status != "All":
        query += " AND status = ?"
        params.append(status)
    elif not include_resolved:
        query += " AND lower(coalesce(status, 'unverified')) NOT IN ('resolved', 'expired', 'dismissed')"

    query += " ORDER BY COALESCE(published_at, fetched_at) DESC LIMIT ?"
    params.append(int(limit) if limit else 2000)

    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()

    results = []
    seen = set()
    for row in rows:
        item = dict(row)
        title = item.get("title", "")
        summary = item.get("summary", "")

        if is_irrelevant_report(title, summary):
            continue

        current_type, current_severity = classify_disturbance(title, summary)
        if current_type is None:
            continue

        item["event_type"] = current_type
        item["severity"] = current_severity

        if is_resolved_event(title, summary):
            item["status"] = "Resolved"
            if not include_resolved and (not status or status == "All"):
                continue

        candidates = extract_location_candidates(title, summary)
        extracted = extract_location(title, summary)
        if extracted != "Location not identified":
            item["location"] = extracted

        # Add quality metadata without adding columns or rewriting old records.
        if candidates:
            primary = candidates[0]
            item["location_confidence"] = primary["confidence"].title()
            item["location_evidence"] = primary["evidence_source"]
            item["route_matchable"] = bool(primary["is_route_matchable"])
            if not primary["is_route_matchable"]:
                item["location_note"] = (
                    "Location is broad or ambiguous; it is shown for review but "
                    "excluded from automatic shipment-route matching."
                )
            else:
                item["location_note"] = "Explicit settlement name found in event-related article text."
        else:
            item["location_confidence"] = "Unknown"
            item["location_evidence"] = "none"
            item["route_matchable"] = False
            item["location_note"] = (
                "No sufficiently specific event location was identified. "
                "Automatic route matching is disabled."
            )
            if not item.get("location"):
                item["location"] = "Location not identified"

        key = normalize_title(title)
        if key in seen:
            continue
        seen.add(key)
        results.append(item)

    return results


def get_disturbance_summary():
    """
    Keep the saved-record total, but calculate severity counts from the
    currently relevant, unresolved display set rather than stale raw labels.
    """
    with get_connection() as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM disturbance_events"
        ).fetchone()[0]

    active_events = get_disturbance_events(
        limit=5000,
        include_resolved=False,
    )
    summary = {"total_records": total, "High": 0, "Medium": 0, "Low": 0}
    for event in active_events:
        severity = event.get("severity")
        if severity in summary:
            summary[severity] += 1
    return summary


# Keep the existing database ready when this module is imported.
initialize_disturbance_database()
