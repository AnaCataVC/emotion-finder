#!/usr/bin/env python3
"""
Interactive CLI curation tool for Emotion Finder feedback records.

Allows maintainers to review uncurated feedback records (default: status 'pending'),
and approve them ('verified') or reject them ('rejected') for the active learning
retraining pipeline.

Usage:
    python scripts/curate_feedback.py [--status pending] [--limit 50] [--dry-run]
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from feedback_store import get_feedback_store, FeedbackRecord


def format_record(record: FeedbackRecord, index: int, total: int) -> str:
    """Format a FeedbackRecord into a human-readable CLI review card."""
    lines = [
        "=" * 64,
        f"Record [{index}/{total}]  ID: {record.id[:8]}...  Lang: {record.detected_lang.upper()}  Date: {record.created_at[:19]}",
        f"Status: {record.status.upper()}",
        "-" * 64,
        f'User Text: "{record.user_text}"',
        f"Predicted: {record.predicted_quadrant} ({record.predicted_emotion or 'N/A'}) [Conf: {record.model_confidence:.2f}]",
        f"Rating:    {'👍 Positive' if record.rating == 'positive' else '👎 Negative'}",
    ]
    if record.corrected_quadrant or record.corrected_emotion:
        lines.append(
            f"Correction: {record.corrected_quadrant or 'N/A'} ({record.corrected_emotion or 'N/A'})"
        )
    if record.comments:
        lines.append(f'Comments:  "{record.comments}"')
    lines.append("-" * 64)
    return "\n".join(lines)


def curate_session(status: str = "pending", limit: int = 50, dry_run: bool = False) -> None:
    """Run an interactive feedback review session in terminal."""
    store = get_feedback_store()
    records = store.get_by_status(status=status, limit=limit)

    if not records:
        print(f"\n✨ No feedback records found with status '{status}'. Everything is up to date!\n")
        return

    total = len(records)
    print(f"\n🔍 Found {total} records with status '{status}' (Limit: {limit}).")
    if dry_run:
        print("⚠️  Running in DRY-RUN mode. No database records will be modified.\n")
    else:
        print("Ready for curation. Use keys: [v]erify, [r]eject, [s]kip, [q]uit.\n")

    verified_count = 0
    rejected_count = 0
    skipped_count = 0

    for i, rec in enumerate(records, start=1):
        print(format_record(rec, i, total))

        while True:
            try:
                choice = input("Action [v]erify / [r]eject / [s]kip / [q]uit: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n\nCuration aborted by user.")
                return

            if choice in ("v", "verify", "1"):
                if not dry_run:
                    store.mark_status(rec.id, "verified")
                print("-> Marked as 'verified' (Approved for active learning).\n")
                verified_count += 1
                break
            elif choice in ("r", "reject", "2"):
                if not dry_run:
                    store.mark_status(rec.id, "rejected")
                print("-> Marked as 'rejected' (Discarded from training).\n")
                rejected_count += 1
                break
            elif choice in ("s", "skip", "3", ""):
                print("-> Skipped (Status untouched).\n")
                skipped_count += 1
                break
            elif choice in ("q", "quit"):
                print("-> Exiting curation session.\n")
                print_summary(verified_count, rejected_count, skipped_count, total - (i - 1))
                return
            else:
                print("Invalid key. Please press 'v', 'r', 's', or 'q'.")

    print_summary(verified_count, rejected_count, skipped_count, 0)


def print_summary(verified: int, rejected: int, skipped: int, remaining: int) -> None:
    """Print clean summary table of the curation session."""
    print("=" * 40)
    print("📊 Curation Session Summary")
    print(f"  ✓ Verified (Approved): {verified}")
    print(f"  ✗ Rejected (Discarded): {rejected}")
    print(f"  - Skipped:             {skipped}")
    if remaining > 0:
        print(f"  ⏳ Remaining:           {remaining}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Interactive CLI tool to curate user feedback.")
    parser.add_argument(
        "--status",
        default="pending",
        choices=["pending", "verified", "rejected", "incorporated"],
        help="Target status of records to review (default: pending)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of records to fetch for review (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate review actions without modifying the database",
    )
    args = parser.parse_args()

    curate_session(status=args.status, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
