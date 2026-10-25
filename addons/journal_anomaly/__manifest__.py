{
    "name": "Journal Entry Anomaly Detection",
    "summary": "Flags unusual journal entries with unsupervised ML and lets reviewers triage them",
    "version": "20.0.1.0.0",
    "category": "Accounting/Accounting",
    "author": "Qusai Natsheh",
    "website": "https://github.com/QusaiNatsheh/journal-anomaly-detection",
    "license": "LGPL-3",
    "depends": ["account"],
    "external_dependencies": {"python": ["requests"]},
    "data": [
        "security/ir.access.csv",
        "data/ir_cron.xml",
        "views/journal_anomaly_review_views.xml",
        "views/res_config_settings_views.xml",
    ],
}
