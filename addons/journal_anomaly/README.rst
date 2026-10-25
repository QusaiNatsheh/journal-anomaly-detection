================================
Journal Entry Anomaly Detection
================================

Flags unusual posted journal entries using unsupervised machine learning, explains why each one was
flagged, and gives reviewers a queue to mark them as *True Issue* or *False Alarm*.

The scoring runs in a separate ML service (see the repository README). On each run, the service learns what
is normal from the company's own history, so no labelled data is needed.

Usage
=====

#. Set the service address, approval limit and weekend days in
   *Invoicing / Accounting → Settings → Journal Entry Anomaly Detection*.
#. Open *Invoicing → Anomaly Review* and click *Score Now*, or wait for the weekly scheduled job.
#. Open a flagged entry, read the reasons, and mark it *True Issue* or *False Alarm*. Reviewer decisions are
   kept with the feature values, for a future supervised model.

Limitations
===========

* The posting time is the entry's creation time in Odoo, shown in the company partner's timezone.
* Scoring sends up to 200,000 journal items per company to the service in one request.

Credits
=======

Author: Qusai Natsheh. License: LGPL-3.
