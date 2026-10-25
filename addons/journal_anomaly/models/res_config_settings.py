from odoo import fields, models


class ResConfigSettings(models.TransientModel):
    _inherit = "res.config.settings"

    journal_anomaly_service_url = fields.Char(
        string="Anomaly Service URL", config_parameter="journal_anomaly.service_url", default="http://ml:8000"
    )
    journal_anomaly_approval_limit = fields.Float(
        string="Approval Limit",
        config_parameter="journal_anomaly.approval_limit",
        default=10000.0,
        help="Entries just below this amount are treated as suspicious.",
    )
    journal_anomaly_history_months = fields.Integer(
        string="History (months)",
        config_parameter="journal_anomaly.history_months",
        default=12,
        help="How much history the detector learns 'normal' from.",
    )
    journal_anomaly_review_days = fields.Integer(
        string="Review Period (days)",
        config_parameter="journal_anomaly.review_days",
        default=30,
        help="Only entries dated in this recent period are added to the review queue.",
    )
    journal_anomaly_max_results = fields.Integer(
        string="Entries per Run", config_parameter="journal_anomaly.max_results", default=50
    )
    journal_anomaly_weekend_days = fields.Char(
        string="Weekend Days",
        config_parameter="journal_anomaly.weekend_days",
        default="4,5",
        help="Comma-separated weekday numbers, Monday = 0. Default 4,5 is Friday and Saturday.",
    )

    def action_journal_anomaly_score_now(self):
        self.execute()
        return self.env["journal.anomaly.review"].action_score_now()
