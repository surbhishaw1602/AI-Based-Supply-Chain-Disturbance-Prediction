from disturbance_service import (
    fetch_disturbance_feeds,
    get_disturbance_events
)

print("\nStarting RSS disturbance monitoring...\n")

result = fetch_disturbance_feeds()

print("========== RSS MONITORING RESULTS ==========")

print("Articles scanned:", result["total_scanned"])
print("Relevant articles:", result["total_relevant"])
print("New disturbances saved:", result["total_saved"])

print("\n========== FEED STATUS ==========")

for feed in result["feeds"]:

    print("\nSource:", feed["source"])
    print("Status:", feed["status"])
    print("Scanned:", feed["scanned"])
    print("Relevant:", feed["relevant"])
    print("New saved:", feed["new_saved"])
    print("Message:", feed["message"])


print("\n========== SAVED DISTURBANCES ==========")

events = get_disturbance_events()

if not events:

    print("No relevant disturbance announcements found.")

else:

    for event in events:

        print("\nTitle:", event["title"])
        print("Type:", event["event_type"])
        print("Severity:", event["severity"])
        print("Location:", event["location"])
        print("Published:", event["published_at"])
        print("Source:", event["source_name"])
        print("Link:", event["source_url"])
        print("Status:", event["status"])