import logging
from datetime import timedelta

import pytz
import requests
from dateutil.relativedelta import relativedelta

from odoo import api, fields, models
from odoo.exceptions import UserError

_logger = logging.getLogger(__name__)

SERVICE_TIMEOUT = 180
MAX_LINES = 200_000


class JournalAnomalyReview(models.Model):
    _name = "journal.anomaly.review"
    _description = "Journal Entry Anomaly Review"
    _order = "state_sequence, score desc, id desc"
    _rec_name = "move_id"

    move_id = fields.Many2one("account.move", string="Journal Entry", required=True, ondelete="cascade", index=True)
    company_id = fields.Many2one(related="move_id.company_id", store=True)
    date = fields.Date(related="move_id.date", store=True)
    journal_id = fields.Many2one(related="move_id.journal_id", store=True)
    posted_by_id = fields.Many2one(related="move_id.create_uid", string="Created By", store=True)
    currency_id = fields.Many2one(related="move_id.company_currency_id")
    amount = fields.Monetary(related="move_id.amount_total_signed", string="Amount")
    ref = fields.Char(related="move_id.ref")
    score = fields.Float(digits=(6, 4), readonly=True, aggregator="max")
    percentile = fields.Float(
        digits=(5, 4),
        readonly=True,
        aggregator="max",
        help="Share of entries in the history window that look more normal than this one.",
    )
    reasons = fields.Text(readonly=True)
    features = fields.Json(readonly=True)
    method = fields.Char(readonly=True)
    scored_on = fields.Datetime(readonly=True)
    state = fields.Selection(
        [("to_review", "To Review"), ("true_issue", "True Issue"), ("false_alarm", "False Alarm")],
        default="to_review",
        required=True,
        index=True,
    )
    state_sequence = fields.Integer(compute="_compute_state_sequence", store=True)
    reviewer_id = fields.Many2one("res.users", readonly=True)
    reviewed_on = fields.Datetime(readonly=True)
    note = fields.Text(string="Reviewer Note")

    _move_uniq = models.Constraint("unique(move_id)", "Each journal entry can be in the review queue only once.")

    @api.depends("state")
    def _compute_state_sequence(self):
        for review in self:
            review.state_sequence = 0 if review.state == "to_review" else 1

    # ------------------------------------------------------------------ review actions
    def _set_state(self, state):
        self.write(
            {
                "state": state,
                "reviewer_id": self.env.user.id if state != "to_review" else False,
                "reviewed_on": fields.Datetime.now() if state != "to_review" else False,
            }
        )

    def action_true_issue(self):
        self._set_state("true_issue")

    def action_false_alarm(self):
        self._set_state("false_alarm")

    def action_reset(self):
        self._set_state("to_review")

    def action_open_move(self):
        self.ensure_one()
        return {
            "type": "ir.actions.act_window",
            "res_model": "account.move",
            "res_id": self.move_id.id,
            "view_mode": "form",
        }

    # ------------------------------------------------------------------ scoring
    @api.model
    def _params(self):
        get = self.env["ir.config_parameter"].sudo()
        weekend = get.get_str("journal_anomaly.weekend_days") or "4,5"
        return {
            "url": (get.get_str("journal_anomaly.service_url") or "http://ml:8000").rstrip("/"),
            "approval_limit": get.get_float("journal_anomaly.approval_limit") or 10000.0,
            "history_months": get.get_int("journal_anomaly.history_months") or 12,
            "review_days": get.get_int("journal_anomaly.review_days") or 30,
            "max_results": get.get_int("journal_anomaly.max_results") or 50,
            "weekend_days": [int(d) for d in weekend.split(",") if d.strip().isdigit()],
        }

    @api.model
    def _payload_lines(self, company, date_from):
        """Posted journal items of the company since date_from, in the service's input format."""
        tz = pytz.timezone(company.partner_id.tz or self.env.user.tz or "UTC")
        lines = (
            self.env["account.move.line"]
            .sudo()
            .search(
                [
                    ("company_id", "=", company.id),
                    ("parent_state", "=", "posted"),
                    ("date", ">=", date_from),
                    ("display_type", "not in", ("line_section", "line_subsection", "line_note")),
                ],
                order="date, move_id, id",
                limit=MAX_LINES,
            )
        )
        result = []
        for line in lines:
            move = line.move_id
            posted_at = pytz.utc.localize(move.create_date).astimezone(tz).replace(tzinfo=None)
            result.append(
                {
                    "entry_id": move.id,
                    "date": fields.Date.to_string(line.date),
                    "posted_at": posted_at.isoformat(),
                    "user": move.create_uid.login or "",
                    "journal": move.journal_id.type,
                    "partner": line.partner_id.commercial_partner_id.name or None,
                    "account": line.account_id.code or str(line.account_id.id),
                    "account_type": line.account_id.account_type or "expense",
                    "debit": line.debit,
                    "credit": line.credit,
                }
            )
        return result

    @api.model
    def _score_company(self, company):
        params = self._params()
        today = fields.Date.context_today(self)
        lines = self._payload_lines(company, today - relativedelta(months=params["history_months"]))
        if not lines:
            return 0
        payload = {
            "lines": lines,
            "score_from": fields.Date.to_string(today - timedelta(days=params["review_days"])),
            "max_results": params["max_results"],
            "approval_limit": params["approval_limit"],
            "weekend_days": params["weekend_days"],
            "manual_journals": ["general"],
        }
        try:
            response = requests.post(f"{params['url']}/score", json=payload, timeout=SERVICE_TIMEOUT)
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            _logger.warning("Journal anomaly service error: %s", exc)
            raise UserError(self.env._("The anomaly service at %(url)s is not available.", url=params["url"])) from exc
        existing = {r.move_id.id: r for r in self.search([("company_id", "=", company.id)])}
        moves = self.env["account.move"].browse([r["entry_id"] for r in body["results"]]).exists()
        valid = set(moves.ids)
        created = 0
        for result in body["results"]:
            if result["entry_id"] not in valid:
                continue
            vals = {
                "score": result["score"],
                "percentile": result["percentile"],
                "reasons": "\n".join(result["reasons"]),
                "features": result["features"],
                "method": f"{body.get('method')} {body.get('version')}",
                "scored_on": fields.Datetime.now(),
            }
            review = existing.get(result["entry_id"])
            if review:
                if review.state == "to_review":
                    review.write(vals)
            else:
                self.create({**vals, "move_id": result["entry_id"]})
                created += 1
        return created

    @api.model
    def action_score_now(self):
        created = sum(self._score_company(company) for company in self.env.companies)
        return {
            "type": "ir.actions.client",
            "tag": "display_notification",
            "params": {
                "title": self.env._("Anomaly scoring finished"),
                "message": self.env._("%(count)s new entries added to the review queue.", count=created),
                "type": "success",
                "next": {"type": "ir.actions.act_window_close"},
            },
        }

    @api.model
    def _cron_score(self):
        for company in self.env["res.company"].search([("active", "=", True)]):
            try:
                self.with_company(company)._score_company(company)
            except UserError as exc:
                _logger.warning("Anomaly scoring skipped for %s: %s", company.name, exc)
