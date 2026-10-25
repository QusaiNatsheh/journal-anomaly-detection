"""Load exported synthetic journal entries into an Odoo 20 database. Run inside `odoo shell`:

    DEMO_JSON=/path/demo.json odoo shell -d <db> < scripts/load_demo_data.py

Creates the demo users, accounts and partners, posts the entries, then sets each entry's creator and creation
time to the synthetic values (stored in UTC), because the detector uses who posted an entry and when.
"""

import json
import os
from datetime import datetime

import pytz

from odoo import Command

data = json.load(open(os.environ.get("DEMO_JSON", "/mnt/data/demo.json"), encoding="utf-8"))
company = env.company
tz = pytz.timezone(data["timezone"])
company.partner_id.tz = data["timezone"]
jod = env["res.currency"].with_context(active_test=False).search([("name", "=", "JOD")])
jod.active = True
if not env["account.account"].search_count([("company_ids", "in", company.ids)], limit=1):
    env["account.chart.template"].try_loading("generic_coa", company, install_demo=False)
if company.currency_id != jod and not env["account.move.line"].search_count([], limit=1):
    company.currency_id = jod

if env["account.move"].search_count([("ref", "=like", "demo #%")], limit=1):
    print("Demo entries already loaded.")
else:
    group = env.ref("account.group_account_invoice")
    users = {}
    for login in data["users"]:
        user = env["res.users"].search([("login", "=", login)])
        users[login] = user or env["res.users"].create(
            {
                "name": login.capitalize(),
                "login": login,
                "group_ids": [Command.link(group.id)],
            }
        )
    Account = env["account.account"]
    accounts = {}
    for acc in data["accounts"]:
        code = f"9{acc['code']}"  # prefix avoids clashing with the generic chart of accounts
        accounts[acc["code"]] = Account.search(
            [("code", "=", code), ("company_ids", "in", company.ids)]
        ) or Account.create(
            {
                "code": code,
                "name": acc["name"],
                "account_type": acc["type"],
                "reconcile": acc["type"] in ("asset_receivable", "liability_payable"),
            }
        )
    journals = {
        t: env["account.journal"].search([("type", "=", t), ("company_id", "=", company.id)], limit=1)
        for t in ("sale", "purchase", "bank", "general")
    }
    partners = {}

    def partner(name):
        if not name:
            return False
        if name not in partners:
            partners[name] = env["res.partner"].create({"name": name, "is_company": True}).id
        return partners[name]

    entries = data["entries"]
    for start in range(0, len(entries), 500):
        batch = entries[start : start + 500]
        moves = env["account.move"].create(
            [
                {
                    "move_type": "entry",
                    "journal_id": journals[e["journal"]].id,
                    "date": e["date"],
                    "ref": f"demo #{e['id']} {e['ref']}",
                    "line_ids": [
                        Command.create(
                            {
                                "account_id": accounts[a].id,
                                "debit": dr,
                                "credit": cr,
                                "partner_id": partner(e["partner"]),
                            }
                        )
                        for a, dr, cr in e["lines"]
                    ],
                }
                for e in batch
            ]
        )
        moves.action_post()
        for move, e in zip(moves, batch, strict=True):
            created = tz.localize(datetime.fromisoformat(e["posted_at"])).astimezone(pytz.utc).replace(tzinfo=None)
            env.cr.execute(
                "UPDATE account_move SET create_uid = %s, create_date = %s WHERE id = %s",
                (users[e["user"]].id, created, move.id),
            )
        env.cr.commit()
        print(f"{start + len(batch)} / {len(entries)} entries")
    env.invalidate_all()
    print(f"Loaded {len(entries)} entries for {len(users)} users")
