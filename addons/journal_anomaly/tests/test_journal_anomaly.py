from unittest.mock import MagicMock, patch

import requests

from odoo import Command, fields
from odoo.exceptions import UserError
from odoo.tests import tagged

from odoo.addons.account.tests.common import AccountTestInvoicingCommon

REQUESTS_POST = "odoo.addons.journal_anomaly.models.journal_anomaly_review.requests.post"


def _response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


@tagged("post_install", "-at_install")
class TestJournalAnomaly(AccountTestInvoicingCommon):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.company = cls.company_data["company"]
        cls.company.partner_id.tz = "Asia/Riyadh"
        journal = cls.company_data["default_journal_misc"]
        today = fields.Date.context_today(cls.env["journal.anomaly.review"])
        cls.moves = cls.env["account.move"].create(
            [
                {
                    "move_type": "entry",
                    "journal_id": journal.id,
                    "date": today,
                    "line_ids": [
                        Command.create({"account_id": cls.company_data["default_account_expense"].id, "debit": amount}),
                        Command.create(
                            {"account_id": cls.company_data["default_account_revenue"].id, "credit": amount}
                        ),
                    ],
                }
                for amount in (100.0, 9_900.0, 50_000.0)
            ]
        )
        cls.moves.action_post()

    def _score(self, results):
        payload = {"version": "test", "method": "ensemble", "n_entries": 3, "results": results}
        with patch(REQUESTS_POST, return_value=_response(payload)) as post:
            self.env["journal.anomaly.review"].action_score_now()
        return post

    def _result(self, move, score):
        return {
            "entry_id": move.id,
            "score": score,
            "percentile": 0.99,
            "reasons": ["Round amount (50,000)"],
            "features": {"round_zeros": 4.0},
        }

    def test_payload_contains_posted_lines(self):
        post = self._score([])
        sent = post.call_args.kwargs["json"]
        ids = {line["entry_id"] for line in sent["lines"]}
        self.assertTrue(set(self.moves.ids) <= ids)
        line = next(line for line in sent["lines"] if line["entry_id"] == self.moves[2].id and line["debit"])
        self.assertEqual(line["debit"], 50_000.0)
        self.assertEqual(line["journal"], "general")
        self.assertEqual(line["user"], self.moves[2].create_uid.login)
        self.assertEqual(sent["weekend_days"], [4, 5])
        self.assertEqual(sent["approval_limit"], 10_000.0)

    def test_results_create_reviews_once(self):
        self._score([self._result(self.moves[2], 0.9), self._result(self.moves[1], 0.8)])
        reviews = self.env["journal.anomaly.review"].search([("move_id", "in", self.moves.ids)])
        self.assertEqual(len(reviews), 2)
        top = reviews.filtered(lambda r: r.move_id == self.moves[2])
        self.assertEqual(top.reasons, "Round amount (50,000)")
        self.assertEqual(top.amount, 50_000.0)
        # A second run updates instead of duplicating, and does not touch reviewed entries.
        top.action_true_issue()
        self._score([self._result(self.moves[2], 0.5), self._result(self.moves[1], 0.7)])
        self.assertEqual(self.env["journal.anomaly.review"].search_count([("move_id", "in", self.moves.ids)]), 2)
        self.assertEqual(top.score, 0.9)
        self.assertEqual(reviews.filtered(lambda r: r.move_id == self.moves[1]).score, 0.7)

    def test_review_actions(self):
        self._score([self._result(self.moves[0], 0.6)])
        review = self.env["journal.anomaly.review"].search([("move_id", "=", self.moves[0].id)])
        review.action_false_alarm()
        self.assertEqual(review.state, "false_alarm")
        self.assertEqual(review.reviewer_id, self.env.user)
        review.action_reset()
        self.assertEqual(review.state, "to_review")
        self.assertFalse(review.reviewer_id)

    def test_unknown_entries_are_ignored(self):
        self._score([{**self._result(self.moves[0], 0.6), "entry_id": 999999999}])
        self.assertFalse(self.env["journal.anomaly.review"].search([]))

    def test_service_down(self):
        with patch(REQUESTS_POST, side_effect=requests.ConnectionError("down")):
            with self.assertRaises(UserError):
                self.env["journal.anomaly.review"].action_score_now()
            self.env["journal.anomaly.review"]._cron_score()  # logs and continues

    def test_accounting_manager_can_score_and_review(self):
        """Runs as a real accounting administrator (not the superuser), so access rules apply."""
        from odoo.tests import new_test_user

        manager = new_test_user(
            self.env, login="acc_manager", groups="account.group_account_manager", company_id=self.company.id
        )
        payload = {
            "version": "test",
            "method": "ensemble",
            "n_entries": 3,
            "results": [self._result(self.moves[2], 0.9)],
        }
        with patch(REQUESTS_POST, return_value=_response(payload)):
            self.env["journal.anomaly.review"].with_user(manager).action_score_now()
        review = self.env["journal.anomaly.review"].with_user(manager).search([("move_id", "=", self.moves[2].id)])
        review.action_true_issue()
        self.assertEqual(review.state, "true_issue")
        self.assertEqual(review.reviewer_id, manager)

    def test_plain_users_cannot_see_the_queue(self):
        from odoo.exceptions import AccessError
        from odoo.tests import new_test_user

        user = new_test_user(self.env, login="plain", groups="base.group_user", company_id=self.company.id)
        with self.assertRaises(AccessError):
            self.env["journal.anomaly.review"].with_user(user).search([])
