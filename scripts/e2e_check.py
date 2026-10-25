"""Score the demo database through Odoo and the running service, and compare with the injected labels.

Run inside `odoo shell` after loading demo data. REVIEW_DAYS sets how far back entries are reviewed.
"""

import json
import os

data = json.load(open(os.environ.get("DEMO_JSON", "/mnt/data/demo.json"), encoding="utf-8"))
labels = {e["id"]: e["anomaly"] for e in data["entries"]}
params = env["ir.config_parameter"].sudo()
params.set_int("journal_anomaly.review_days", int(os.environ.get("REVIEW_DAYS", 120)))
params.set_int("journal_anomaly.max_results", 50)
Review = env["journal.anomaly.review"]
Review.search([]).unlink()
Review.action_score_now()
reviews = Review.search([], order="score desc")
found = []
for review in reviews:
    demo_id = int(review.move_id.ref.split("#")[1].split()[0])
    found.append(labels.get(demo_id))
review_from = max(e["date"] for e in data["entries"])
hits = sum(f is not None for f in found)
print(
    json.dumps(
        {
            "flagged": len(found),
            "true_anomalies": hits,
            "by_type": {t: found.count(t) for t in sorted({f for f in found if f})},
        }
    )
)
print(f"Precision in the review queue: {hits / max(len(found), 1):.3f} ({hits} of {len(found)})")
env.cr.commit()
