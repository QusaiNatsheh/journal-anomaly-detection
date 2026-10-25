# LinkedIn posts: Journal Entry Anomaly Detection (drafts)

Drafts for review before posting. Replace `<repo link>` with the GitHub URL once the repository is public.

---

## Post 1: the idea, explained simply

**Attach:** `docs/screenshots/02_flagged_entry.png`

How could an auditor use machine learning without any labelled fraud data?

Most companies have never labelled their journal entries as "fraud" or "error". So a model can't be trained to
recognise them. What every company *does* have is its own history of normal entries.

That is enough for unsupervised anomaly detection:
1. Learn what normal looks like for this company: which accounts go together, who posts what, at what time, and
   at what amounts.
2. Rank every new entry by how far it is from that normal.
3. Show the reviewer the top of the list, with the reasons in plain words.

For example:
• "Same amount, accounts and partner as another entry in the last 14 days" (a possible duplicate bill)
• "Amount 9,725.56 is just below the approval limit"
• "Posted at night (02:22)"
• "User omar rarely posts to these accounts"

The reviewer marks each flag as a true issue or a false alarm. Those decisions are kept, so a supervised model can
be trained later on real labels.

I built this for Odoo 20 as a portfolio project. Next post: which detector worked, and which didn't.

#Audit #InternalAudit #MachineLearning #Odoo #Accounting

---

## Post 2: what worked, what didn't

**Attach:** a screenshot of the results table from the README

I compared four unsupervised detectors for finding unusual journal entries. The first results were humbling.

The test: 24 months of synthetic accounting data, with 8 kinds of injected anomalies: duplicates, round amounts,
night and weekend postings, amounts just below the approval limit, unusual account pairs, wrong users, and outliers.
That's about 64 anomalies hidden among 5,750 entries in each test year.

**First result:** my hand-written red-flag rules beat every model.

The reason was a detail. Yes/no signals like "duplicate" or "posted at night" were drowned out by large amount
values. After converting every feature into "how rare is this value in this company's data", the picture changed.

Real anomalies among the top 50 flagged entries:
• Red-flag rules: 58%
• Isolation Forest: 46%
• Autoencoder: 63%
• Ensemble (rarity + Local Outlier Factor + autoencoder): 71%

**Second lesson, found in the demo screenshots:** monthly depreciation was flagged as a "duplicate". It posts the
same amount every month, and two month-ends can be exactly 30 days apart. Shortening the duplicate window to 14
days removed 26 of 28 such false alarms. The lesson: look at what your model flags, not just at the metrics.

Honest limits: I wrote both the data and the rules, so the rules know exactly what to look for. Real fraud is
rarer and better hidden. These numbers show the method works, not how it performs on a real company.

<repo link>

#MachineLearning #AnomalyDetection #Audit #Python

---

## Post 3: the review queue

**Attach:** `docs/screenshots/demo.gif`, or a short screen recording

What the reviewer sees in Odoo 20:

1. An *Anomaly Review* queue, most unusual entries first, each with its reasons.
2. One click opens the entry: why it was flagged, the user who posted it, the amount, a field for a note.
3. *True Issue* or *False Alarm*: the decision is recorded with the reviewer's name and the date.

Each run learns "normal" from the company's own last 12 months, so a retailer and a contractor are each compared
with themselves. One test run put 17 of the 19 anomalies of the review period into a 50-entry queue.

Built with Odoo 20 Community, FastAPI, scikit-learn and MLflow. One `make demo` runs it.

<repo link>

#Odoo #Audit #Accounting #Automation
