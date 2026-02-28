"""Add 'comments' column to journal schema and migrate existing CSV."""
import csv

# 1. Patch journal.py schema
journal_py = "src/utils/journal.py"
with open(journal_py, "r", encoding="utf-8") as f:
    content = f.read()

old = "        'reasoning',\n    ]"
new = "        'reasoning',\n        'comments',\n    ]"

if "'comments'" not in content:
    content = content.replace(old, new, 1)
    with open(journal_py, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ journal.py: added 'comments' to COLUMNS")
else:
    print("⏭️  journal.py: 'comments' already present")

# 2. Migrate CSV
csv_path = "data/trade_journal.csv"
rows = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames)
    if "comments" not in fieldnames:
        fieldnames.append("comments")
    for row in reader:
        if row.get("trade_id") == "1":
            row["comments"] = (
                "PRUEBA: Test de workflow paper. "
                "Cierre automatico fallo (Error 201/10349). "
                "Posicion sigue abierta."
            )
        else:
            row.setdefault("comments", "")
        rows.append(row)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ trade_journal.csv: migrated ({len(rows)} rows)")
print("   Trade #1 annotated with test comment.")
