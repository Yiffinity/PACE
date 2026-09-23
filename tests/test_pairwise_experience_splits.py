from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from prepare_pairwise_experience_splits import build_pairwise_groups, select_records


def _rows(source: str, count: int) -> list[dict[str, str]]:
    return [
        {"sample_id": f"{source}:{index}", "source": source}
        for index in range(count)
    ]


class PairwiseExperienceSplitTests(unittest.TestCase):
    def test_dataset_selection_is_deterministic(self):
        records = {row["sample_id"]: row for row in _rows("mmsd2", 20)}
        first = select_records(records, 8, seed=42)
        second = select_records(records, 8, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len({row["sample_id"] for row in first}), 8)

    def test_pairwise_groups_reuse_the_same_source_samples(self):
        selected = {
            "docmsu": _rows("docmsu", 5),
            "mmsd2": _rows("mmsd2", 8),
            "sarcnet": _rows("sarcnet", 2),
        }
        all_selected, groups = build_pairwise_groups(selected, seed=42)

        self.assertEqual(len(all_selected), 15)
        self.assertEqual(len(groups["mmsd2_docmsu"]), 13)
        self.assertEqual(len(groups["mmsd2_sarcnet"]), 10)
        self.assertEqual(len(groups["docmsu_sarcnet"]), 7)
        for source, rows in selected.items():
            expected_ids = {row["sample_id"] for row in rows}
            containing_groups = [
                group_rows
                for group_rows in groups.values()
                if any(row["source"] == source for row in group_rows)
            ]
            self.assertEqual(len(containing_groups), 2)
            for group_rows in containing_groups:
                actual_ids = {
                    row["sample_id"]
                    for row in group_rows
                    if row["source"] == source
                }
                self.assertEqual(actual_ids, expected_ids)


if __name__ == "__main__":
    unittest.main()
