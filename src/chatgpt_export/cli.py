from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .core import DEFAULT_KEYWORDS, build_spec, extract_archive, preview_matches


COMMANDS = {"extract", "preview", "wizard", "list-default-keywords", "list-presets"}
DEFAULT_PRESET_FILENAMES = [".chatgpt_export_presets.json", "chatgpt_export_presets.json"]


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(cleaned)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "archive_root",
        nargs="?",
        default=".",
        help="Path to the ChatGPT export directory containing conversations-*.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory for extracted conversations",
    )
    parser.add_argument(
        "-k",
        "--keyword",
        action="append",
        dest="keywords",
        help="Keyword to match in title or message text. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-keyword",
        action="append",
        dest="exclude_keywords",
        help="Keyword to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--regex",
        action="append",
        dest="regex_patterns",
        help="Regular expression to match against title/body. Can be repeated.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["any", "all"],
        default=None,
        help="Require any keyword or all keywords to match.",
    )
    parser.add_argument(
        "--scope",
        choices=["title", "body", "title_or_body"],
        default=None,
        help="Where keyword matching should be applied.",
    )
    parser.add_argument(
        "--include-project-enabled",
        action="store_true",
        help="Include conversations whose memory_scope is project_enabled, even without a keyword match.",
    )
    parser.add_argument(
        "--include-shared",
        action="store_true",
        help="Include shared conversations, even without a keyword match.",
    )
    parser.add_argument(
        "--date-from",
        default=None,
        help="Only include conversations created on or after this ISO datetime.",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help="Only include conversations created on or before this ISO datetime.",
    )
    parser.add_argument(
        "--expand-related",
        action="store_true",
        help="Expand from seed matches to related conversations.",
    )
    parser.add_argument(
        "--related-window-days",
        type=int,
        default=None,
        help="Time window used when expanding related conversations.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum score required for inclusion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the extraction without writing any files.",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default=None,
        help="How preview results should be displayed.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        dest="presets",
        help="Load a named search preset from the presets file. Can be repeated.",
    )
    parser.add_argument(
        "--presets-file",
        default=None,
        help="Path to a JSON file containing saved search presets.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract selected ChatGPT conversations from an export bundle."
    )
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract", help="Extract matching conversations")
    add_common_args(extract_parser)

    preview_parser = subparsers.add_parser("preview", help="Preview matching conversations")
    add_common_args(preview_parser)

    wizard_parser = subparsers.add_parser(
        "wizard", help="Interactively guide a first-time extraction"
    )
    wizard_parser.add_argument(
        "archive_root",
        nargs="?",
        default=".",
        help="Path to the ChatGPT export directory containing conversations-*.json",
    )
    wizard_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory for extracted conversations",
    )
    wizard_parser.add_argument(
        "--presets-file",
        default=None,
        help="Path to a JSON file containing saved search presets.",
    )

    subparsers.add_parser(
        "list-default-keywords", help="Print the default keyword list and exit"
    )
    list_presets_parser = subparsers.add_parser("list-presets", help="List saved presets")
    list_presets_parser.add_argument(
        "--presets-file",
        default=None,
        help="Path to a JSON file containing saved search presets.",
    )
    return parser


def _resolve_command(argv: list[str]) -> list[str]:
    if not argv:
        return ["extract"]
    if "--list-default-keywords" in argv:
        return ["list-default-keywords"]
    if argv[0] in COMMANDS or argv[0] in {"-h", "--help"}:
        return argv
    return ["extract", *argv]


def _spec_from_args(args: argparse.Namespace):
    preset_data = _load_presets(args)
    query = _merge_query_sources(preset_data, args)
    return build_spec(
        keywords=query["keywords"],
        exclude_keywords=query["exclude_keywords"],
        regex_patterns=query["regex_patterns"],
        match_mode=query["match_mode"],
        scope=query["scope"],
        include_project_enabled=query["include_project_enabled"],
        include_shared=query["include_shared"],
        date_from=query["date_from"],
        date_to=query["date_to"],
        expand_related=query["expand_related"],
        related_window_days=query["related_window_days"],
        min_score=query["min_score"],
    )


def _resolve_presets_file(args: argparse.Namespace) -> Path | None:
    if getattr(args, "presets_file", None):
        return Path(args.presets_file).expanduser()

    candidates = [Path.cwd() / filename for filename in DEFAULT_PRESET_FILENAMES]
    archive_root = getattr(args, "archive_root", None)
    if archive_root:
        root = Path(archive_root).expanduser()
        candidates.extend(root / filename for filename in DEFAULT_PRESET_FILENAMES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _default_presets_path(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "presets_file", None)
    if explicit:
        return Path(explicit).expanduser()
    archive_root = getattr(args, "archive_root", None)
    if archive_root:
        return Path(archive_root).expanduser() / DEFAULT_PRESET_FILENAMES[0]
    return Path.cwd() / DEFAULT_PRESET_FILENAMES[0]


def _load_presets(args: argparse.Namespace) -> dict[str, dict]:
    path = _resolve_presets_file(args)
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "presets" in data and isinstance(data["presets"], dict):
        data = data["presets"]
    if not isinstance(data, dict):
        return {}
    presets: dict[str, dict] = {}
    for name, value in data.items():
        if isinstance(value, dict):
            presets[str(name)] = value
    return presets


def _merge_list_values(*values: list[str] | None) -> list[str] | None:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        for item in value:
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return merged or None


def _merge_query_sources(presets: dict[str, dict], args: argparse.Namespace) -> dict[str, object]:
    keywords = getattr(args, "keywords", None)
    exclude_keywords = getattr(args, "exclude_keywords", None)
    regex_patterns = getattr(args, "regex_patterns", None)
    match_mode = getattr(args, "match_mode", None)
    scope = getattr(args, "scope", None)
    include_project_enabled = bool(getattr(args, "include_project_enabled", False))
    include_shared = bool(getattr(args, "include_shared", False))
    date_from = getattr(args, "date_from", None)
    date_to = getattr(args, "date_to", None)
    expand_related = bool(getattr(args, "expand_related", False))
    related_window_days = getattr(args, "related_window_days", None)
    min_score = getattr(args, "min_score", None)
    merged: dict[str, object] = {
        "keywords": None,
        "exclude_keywords": None,
        "regex_patterns": None,
        "match_mode": None,
        "scope": None,
        "include_project_enabled": False,
        "include_shared": False,
        "date_from": None,
        "date_to": None,
        "expand_related": False,
        "related_window_days": None,
        "min_score": None,
    }

    preset_names = getattr(args, "presets", None) or []
    for preset_name in preset_names:
        preset = presets.get(preset_name)
        if preset is None:
            raise SystemExit(f"Preset not found: {preset_name}")
        merged["keywords"] = _merge_list_values(merged["keywords"], preset.get("keywords"))
        merged["exclude_keywords"] = _merge_list_values(merged["exclude_keywords"], preset.get("exclude_keywords"))
        merged["regex_patterns"] = _merge_list_values(merged["regex_patterns"], preset.get("regex_patterns"))
        merged["match_mode"] = preset.get("match_mode", merged["match_mode"])
        merged["scope"] = preset.get("scope", merged["scope"])
        merged["include_project_enabled"] = bool(merged["include_project_enabled"] or preset.get("include_project_enabled", False))
        merged["include_shared"] = bool(merged["include_shared"] or preset.get("include_shared", False))
        merged["date_from"] = preset.get("date_from", merged["date_from"])
        merged["date_to"] = preset.get("date_to", merged["date_to"])
        merged["expand_related"] = bool(merged["expand_related"] or preset.get("expand_related", False))
        merged["related_window_days"] = preset.get("related_window_days", merged["related_window_days"])
        merged["min_score"] = preset.get("min_score", merged["min_score"])

    merged["keywords"] = _merge_list_values(merged["keywords"], keywords)
    merged["exclude_keywords"] = _merge_list_values(merged["exclude_keywords"], exclude_keywords)
    merged["regex_patterns"] = _merge_list_values(merged["regex_patterns"], regex_patterns)
    merged["match_mode"] = match_mode or merged["match_mode"] or "any"
    merged["scope"] = scope or merged["scope"] or "title_or_body"
    merged["include_project_enabled"] = bool(merged["include_project_enabled"] or include_project_enabled)
    merged["include_shared"] = bool(merged["include_shared"] or include_shared)
    merged["date_from"] = _parse_datetime(date_from) if date_from else _parse_datetime(merged["date_from"]) if isinstance(merged["date_from"], str) else merged["date_from"]
    merged["date_to"] = _parse_datetime(date_to) if date_to else _parse_datetime(merged["date_to"]) if isinstance(merged["date_to"], str) else merged["date_to"]
    merged["expand_related"] = bool(merged["expand_related"] or expand_related)
    merged["related_window_days"] = related_window_days if related_window_days is not None else merged["related_window_days"] or 14
    merged["min_score"] = min_score if min_score is not None else merged["min_score"] or 0.0
    return merged


def _save_preset(args: argparse.Namespace, name: str, spec) -> Path:
    path = _default_presets_path(args)
    presets: dict[str, dict] = {}
    if path.exists():
        existing_args = argparse.Namespace(**vars(args))
        existing_args.presets_file = str(path)
        presets.update(_load_presets(existing_args))
    presets[name] = {
        "keywords": list(spec.keywords),
        "exclude_keywords": list(spec.exclude_keywords),
        "regex_patterns": list(spec.regex_patterns),
        "match_mode": spec.match_mode,
        "scope": spec.scope,
        "include_project_enabled": spec.include_project_enabled,
        "include_shared": spec.include_shared,
        "date_from": spec.date_from.isoformat() if spec.date_from else None,
        "date_to": spec.date_to.isoformat() if spec.date_to else None,
        "expand_related": spec.expand_related,
        "related_window_days": spec.related_window_days,
        "min_score": spec.min_score,
    }
    payload = {"presets": presets}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return path


def _format_preset_summary(name: str, preset: dict) -> str:
    keywords = ", ".join(preset.get("keywords", [])) or "-"
    scope = preset.get("scope", "title_or_body")
    match_mode = preset.get("match_mode", "any")
    return f"{name}: keywords=[{keywords}] scope={scope} match_mode={match_mode}"


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or (default or "")


def _prompt_yes_no(text: str, default: bool = False) -> bool:
    default_label = "Y/n" if default else "y/N"
    value = input(f"{text} [{default_label}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "true", "1"}


def _prompt_choices(text: str, choices: list[str], default: str) -> str:
    options = "/".join(choices)
    value = input(f"{text} ({options}) [{default}]: ").strip().lower()
    if not value:
        return default
    if value not in choices:
        print(f"Invalid choice, using {default}.")
        return default
    return value


def _prompt_keywords(text: str) -> list[str]:
    raw = input(f"{text} (comma-separated): ").strip()
    return [item.strip() for item in raw.split(",") if item.strip()]


def _print_matches(matches) -> None:
    if not matches:
        print("No matches found.")
        return

    widths = {
        "score": 5,
        "title": max(5, min(50, max(len(match.title) for match in matches))),
        "source": max(6, min(22, max(len(match.source_file) for match in matches))),
    }
    header = f"{'Score':>5}  {'Title':<{widths['title']}}  {'Source':<{widths['source']}}  Reasons"
    print(header)
    print("-" * len(header))
    for match in matches:
        reasons = "; ".join(match.reasons) if match.reasons else ", ".join(match.matched_keywords)
        print(
            f"{match.score:>5.2f}  "
            f"{match.title[:widths['title']]:<{widths['title']}}  "
            f"{match.source_file[:widths['source']]:<{widths['source']}}  "
            f"{reasons}"
        )


def _print_matches_json(matches) -> None:
    payload = [
        {
            "conversation_id": match.conversation_id,
            "title": match.title,
            "score": match.score,
            "matched_keywords": match.matched_keywords,
            "title_keyword_matches": match.title_keyword_matches,
            "body_keyword_matches": match.body_keyword_matches,
            "regex_matches": match.regex_matches,
            "reasons": match.reasons,
            "source_file": match.source_file,
            "project_enabled": match.project_enabled,
            "is_shared": match.is_shared,
            "is_related": match.is_related,
            "related_to": match.related_to,
        }
        for match in matches
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def _run_wizard(args: argparse.Namespace) -> int:
    print("ChatGPT Export Wizard")
    archive_root = _prompt("Export folder", args.archive_root)
    output_dir = _prompt("Output folder", args.output or "")
    keywords = _prompt_keywords("Keywords")
    if not keywords:
        keywords = DEFAULT_KEYWORDS
        print(f"Using default keyword: {', '.join(keywords)}")

    scope = _prompt_choices("Match scope", ["title", "body", "title_or_body"], "title_or_body")
    match_mode = _prompt_choices("Match mode", ["any", "all"], "any")
    include_related = _prompt_yes_no("Expand to related conversations", False)
    include_project_enabled = _prompt_yes_no("Include project-enabled chats", False)
    include_shared = _prompt_yes_no("Include shared chats", False)
    dry_run = _prompt_yes_no("Preview only (no files written)", True)
    output_format = _prompt_choices("Preview format", ["table", "json"], "table")

    spec = build_spec(
        keywords=keywords,
        match_mode=match_mode,
        scope=scope,
        include_project_enabled=include_project_enabled,
        include_shared=include_shared,
        expand_related=include_related,
    )

    if dry_run:
        matches = preview_matches(archive_root, spec)
        if output_format == "json":
            _print_matches_json(matches)
        else:
            _print_matches(matches)
    else:
        manifest = extract_archive(archive_root, output_dir=output_dir or None, spec=spec)
        print(
            f"Wrote {manifest['selected_count']} conversations to {manifest['output_dir']} "
            f"(loaded: {manifest['loaded_count']})"
        )

    if _prompt_yes_no("Save this search as a preset", False):
        preset_name = _prompt("Preset name", "my_search").strip()
        if preset_name:
            preset_path = _save_preset(args, preset_name, spec)
            print(f"Saved preset to {preset_path}")
        return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = _resolve_command(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-default-keywords":
        print("\n".join(DEFAULT_KEYWORDS))
        return 0

    if args.command == "list-presets":
        presets = _load_presets(args)
        if not presets:
            print("No presets found.")
            return 0
        for name, preset in sorted(presets.items()):
            print(_format_preset_summary(name, preset))
        return 0

    if args.command == "preview":
        spec = _spec_from_args(args)
        matches = preview_matches(args.archive_root, spec)
        if (args.output_format or "table") == "json":
            _print_matches_json(matches)
        else:
            _print_matches(matches)
        return 0

    if args.command == "wizard":
        return _run_wizard(args)

    spec = _spec_from_args(args)

    if args.dry_run:
        matches = preview_matches(args.archive_root, spec)
        if (args.output_format or "table") == "json":
            _print_matches_json(matches)
        else:
            _print_matches(matches)
        return 0

    manifest = extract_archive(args.archive_root, output_dir=args.output, spec=spec)
    print(
        f"Wrote {manifest['selected_count']} conversations to {manifest['output_dir']} "
        f"(loaded: {manifest['loaded_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
