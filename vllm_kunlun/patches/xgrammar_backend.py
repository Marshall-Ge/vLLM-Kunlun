"""
Usage:
------
    python xgrammar_backend.py          # Apply all patches
    python xgrammar_backend.py --dry-run # Preview changes without modifying files
    python xgrammar_backend.py --revert  # Restore all files from .bak backups
"""

import os
import shutil
import site
import sys

# ============================================================
# Base path for the target Python environment
# ============================================================
# Determine the site-packages directory dynamically to keep this
# script portable across different Python installations.
_site_packages_list = []
try:
    _site_packages_list = site.getsitepackages()
except Exception:
    _site_packages_list = []
if _site_packages_list:
    # Prefer the first returned site-packages directory.
    SITE_PACKAGES = _site_packages_list[0]
else:
    # Fallback: use the current environment prefix.
    SITE_PACKAGES = sys.prefix

# ============================================================
# Patch definitions
# Each patch contains:
#   - file:      target file path (relative to SITE_PACKAGES)
#   - desc:      human-readable description
#   - lines:     original line numbers (for reference only)
#   - old:       the exact code to find and replace
#   - new:       the replacement code
#   - count:     (optional) number of occurrences to replace.
#                Defaults to 1. Use 0 to replace ALL occurrences.
#
# IMPORTANT: When multiple patches target the same file, the
# import/header patches should come BEFORE the body patches to
# ensure correct dependency order.
# ============================================================
PATCHES = [
    {
        "file": "vllm/v1/structured_output/utils.py",
        "desc": "Add backend='torch_native' to xgr.apply_token_bitmask_inplace",
        "lines": "113-117",
        "old": "indices=out_indices if not skip_out_indices else None,",
        "new": 'indices=out_indices if not skip_out_indices else None, backend="torch_native"',
        "count": 1,
    },
]


# ============================================================
# Core logic
# ============================================================


def get_full_path(relative_path: str) -> str:
    """Construct the full file path from SITE_PACKAGES."""
    return os.path.join(SITE_PACKAGES, relative_path)


def apply_patch(patch: dict, dry_run: bool = False) -> str:
    """
    Apply a single patch.
    Returns: 'success', 'skipped', or 'failed'
    """
    file_path = get_full_path(patch["file"])
    desc = patch["desc"]
    # How many occurrences to replace: default 1, use 0 for all
    count = patch.get("count", 1)

    print(
        f"\n{'[DRY RUN] ' if dry_run else ''}"
        f"Patch: {patch['file']} (lines {patch['lines']})"
    )
    print(f"  Description: {desc}")

    # 1. Read the file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  FAIL: File not found: {file_path}")
        return "failed"

    # 2. Check if already patched (idempotent)
    if patch["new"] in content and patch["old"] not in content:
        print("  SKIP: Already patched.")
        return "skipped"

    # 3. Verify the original code exists
    if patch["old"] not in content:
        print(
            "  FAIL: Original code not found. The file may have been "
            "modified or the vLLM version may differ."
        )
        return "failed"

    # Count how many occurrences will be replaced
    num_occurrences = content.count(patch["old"])
    replacements = num_occurrences if count == 0 else min(count, num_occurrences)
    print(
        f"  Found {num_occurrences} occurrence(s), "
        f"will replace {'all' if count == 0 else replacements}."
    )

    if dry_run:
        print("  OK: Would apply patch.")
        return "success"

    # 4. Backup (only once per file, never overwrite existing .bak)
    backup_path = file_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"  Backup created: {backup_path}")
    else:
        print(f"  Backup already exists, not overwriting: {backup_path}")

    # 5. Replace and write back
    if count == 0:
        # Replace ALL occurrences
        new_content = content.replace(patch["old"], patch["new"])
    else:
        new_content = content.replace(patch["old"], patch["new"], count)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"  SUCCESS: Patch applied ({replacements} replacement(s)).")
    return "success"


def revert_all():
    """Restore all patched files from their .bak backups."""
    print("=" * 64)
    print("  Reverting all patches from .bak backups")
    print("=" * 64)

    seen = set()
    for patch in PATCHES:
        file_path = get_full_path(patch["file"])
        if file_path in seen:
            continue
        seen.add(file_path)

        backup_path = file_path + ".bak"
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
            print(f"  Restored: {file_path}")
        else:
            print(f"  No backup found for: {file_path}")

    print("\nRevert complete.")


def main():
    dry_run = "--dry-run" in sys.argv
    revert = "--revert" in sys.argv

    if revert:
        revert_all()
        return

    print("=" * 64)
    print("  vLLM Compatibility Patch for PyTorch 2.5.1")
    print(f"  Total patches: {len(PATCHES)}")
    if dry_run:
        print("  Mode: DRY RUN (no files will be modified)")
    print("")
    print("  NOTE: These patches are TEMPORARY. Remove them after")
    print("        upgrading to PyTorch 2.9+.")
    print("=" * 64)

    results = {"success": 0, "skipped": 0, "failed": 0}

    for patch in PATCHES:
        result = apply_patch(patch, dry_run=dry_run)
        results[result] += 1

    print("\n" + "=" * 64)
    print(
        f"  Results: "
        f"{results['success']} applied  |  "
        f"{results['skipped']} skipped  |  "
        f"{results['failed']} failed  |  "
        f"{len(PATCHES)} total"
    )
    if not dry_run and results["success"] > 0:
        print("  Original files backed up as .bak")
        print("  To revert: python patch_torch251.py --revert")
    print("=" * 64)

    # Exit with non-zero code if any patch failed
    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
