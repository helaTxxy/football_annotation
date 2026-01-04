from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
	return json.loads(path.read_text(encoding="utf-8"))


def merge_segments(split_dir: Path, output_path: Path, manifest_name: str = "manifest.json") -> tuple[int, int]:
	"""Merge <split_dir>/seg_*.json files (ordered by manifest) into one JSON.

	Returns:
		(segment_count, total_items)
	"""
	manifest_path = split_dir / manifest_name
	if not manifest_path.exists():
		raise FileNotFoundError(f"manifest not found: {manifest_path}")

	manifest = _load_json(manifest_path)
	segments = list(manifest.get("segments") or [])
	if not segments:
		raise RuntimeError("manifest has no segments")

	# Sort by seg_idx to be safe.
	segments.sort(key=lambda s: int(s.get("seg_idx", 0)))

	columns = list(manifest.get("columns") or [])
	merged_items: list[dict] = []
	seg_count = 0

	for s in segments:
		rel = s.get("path")
		if not rel:
			continue
		seg_path = (split_dir / str(rel)).resolve()
		if not seg_path.exists():
			raise FileNotFoundError(f"segment file not found: {seg_path}")

		seg = _load_json(seg_path)
		seg_cols = list(seg.get("columns") or [])
		if not columns and seg_cols:
			columns = seg_cols
		elif columns and seg_cols and seg_cols != columns:
			# Keep manifest columns, but warn via exception? keep tolerant.
			pass

		items = seg.get("data")
		if not isinstance(items, list):
			raise RuntimeError(f"segment data is not a list: {seg_path}")

		merged_items.extend(items)
		seg_count += 1

	out_obj = {
		"columns": columns,
		"data": merged_items,
	}
	output_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
	return seg_count, len(merged_items)


def main() -> int:
	parser = argparse.ArgumentParser(description="Merge split tracking JSON segments into one JSON")
	parser.add_argument(
		"--split-dir",
		type=Path,
		default=Path("frame_1128_split"),
		help="Directory that contains manifest.json and seg_*.json files",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("1128_man.json"),
		help="Output merged json path",
	)
	parser.add_argument(
		"--manifest",
		type=str,
		default="manifest.json",
		help="Manifest filename inside split-dir",
	)
	args = parser.parse_args()

	seg_count, total_items = merge_segments(args.split_dir, args.output, manifest_name=args.manifest)
	print(f"Merged {seg_count} segments -> {args.output}  (total items: {total_items})")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
