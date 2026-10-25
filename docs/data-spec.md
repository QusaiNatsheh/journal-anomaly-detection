# Synthetic Data Specification (v1.0)

All data is synthetic, produced by `ml/src/journal_anomaly_ml/generator.py` from a fixed seed. One company in
Jordan (JOD), 24 months from 2024-10-01. The working week is Sunday to Thursday.

## Normal activity
| Activity | Journal | User | Volume | Accounts |
|---|---|---|---|---|
| Sales invoice | sale | rania | ~150 / month | Dr 1100 Receivable, Cr 4000 Sales, Cr 2100 VAT (16%) |
| Customer payment | bank | rania | one per invoice, 10–60 days later | Dr 1010 Bank, Cr 1100 |
| Vendor bill | purchase | omar | ~80 / month | Dr expense (one per vendor), Dr 2110 VAT, Cr 2000 Payable |
| Vendor payment | bank | omar | one per bill, 15–45 days later | Dr 2000, Cr 1010 |
| Payroll | bank | lina | monthly, +3% per year | Dr 6100, Cr 1010 |
| Rent | bank | omar | monthly, fixed | Dr 6200, Cr 1010 |
| Bank charges | bank | rania | 4 / month | Dr 6900, Cr 1010 |
| Depreciation | general | khaled | monthly, fixed | Dr 6800, Cr 1590 |
| Accruals | general | khaled (85%) / sami | ~12 / month, at month end | Dr expense, Cr 2200 |
| Equipment purchase | purchase | sami | January and July | Dr 1500, Dr 2110, Cr 2000 |

Posting times are usually 08:00–17:00 on working days.

### Normal behaviour that looks odd (on purpose)
These make the data less clean, so a detector cannot rely on one flag:
- 0.6% of entries are posted on a weekend day.
- 3% are posted in overtime (17:00–21:00).
- 15% of accruals use round amounts (500 to 2,500).

## Injected anomalies
1.2% of entries (about 135 per dataset), split evenly across 8 types:

| Type | How it is made |
|---|---|
| duplicate | A vendor bill copied 1–5 days later, same partner, accounts and amount |
| unusual_pair | Manual entry with an account pair never used normally (consulting from bank, cash sale without VAT, equity withdrawal, expense against receivables) |
| weekend | Copy of a normal entry, amount changed, posted on Friday or Saturday |
| night | Copy of a normal entry, amount changed, posted 23:00–04:00 |
| round_amount | Manual entry of 20,000–75,000 to an expense account against bank |
| amount_outlier | Vendor bill multiplied by 8–20 |
| below_limit | Manual accrual of 9,500–9,999 (approval limit 10,000) |
| wrong_user | Sale or purchase entry posted by a user who does not work in that journal |

Copied anomalies get a new amount so that they are not also duplicates.

## Output
`lines.parquet` has one row per journal item: entry_id, date, posted_at, user, journal, partner, ref, account,
account_type, debit, credit, anomaly. The `anomaly` column is used only for evaluation.

## Evaluation protocol
- Features for each entry use only earlier entries.
- Detectors are fitted on and score the last 12 months, as the service does in Odoo.
- Settings are chosen on seed 7. Results are reported on seeds 42, 43 and 44.
