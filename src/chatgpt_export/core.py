from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator, Literal


MatchMode = Literal["any", "all"]
MatchScope = Literal["title", "body", "title_or_body"]

DEFAULT_KEYWORDS = ["criticality"]


def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if not text:
        return "conversation"
    return text[:max_len].strip("-")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_conversation_files(root: Path) -> Iterator[Path]:
    yield from sorted(root.glob("conversations-*.json"))


def load_shared_conversation_ids(root: Path) -> set[str]:
    path = root / "shared_conversations.json"
    if not path.exists():
        return set()
    data = load_json(path)
    if not isinstance(data, list):
        return set()
    conversation_ids: set[str] = set()
    for item in data:
        if isinstance(item, dict):
            conversation_id = item.get("conversation_id")
            if conversation_id:
                conversation_ids.add(str(conversation_id))
    return conversation_ids


def extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if "text" in value:
            return extract_text(value["text"])
        if "parts" in value:
            return extract_text(value["parts"])
        if "content" in value:
            return extract_text(value["content"])
        return "\n".join(
            extract_text(v)
            for v in value.values()
            if isinstance(v, (str, list, dict))
        )
    return str(value)


def normalize_keywords(keywords: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for keyword in keywords or []:
        keyword = keyword.strip()
        if keyword and keyword not in normalized:
            normalized.append(keyword)
    return normalized


@dataclass(frozen=True)
class ExtractSpec:
    keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)
    regex_patterns: list[str] = field(default_factory=list)
    match_mode: MatchMode = "any"
    scope: MatchScope = "title_or_body"
    include_project_enabled: bool = False
    include_shared: bool = False
    date_from: datetime | None = None
    date_to: datetime | None = None
    expand_related: bool = False
    related_window_days: int = 14
    min_score: float = 0.0


def build_spec(
    *,
    keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
    regex_patterns: list[str] | None = None,
    match_mode: MatchMode = "any",
    scope: MatchScope = "title_or_body",
    include_project_enabled: bool = False,
    include_shared: bool = False,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    expand_related: bool = False,
    related_window_days: int = 14,
    min_score: float = 0.0,
) -> ExtractSpec:
    return ExtractSpec(
        keywords=normalize_keywords(keywords),
        exclude_keywords=normalize_keywords(exclude_keywords),
        regex_patterns=normalize_keywords(regex_patterns),
        match_mode=match_mode,
        scope=scope,
        include_project_enabled=include_project_enabled,
        include_shared=include_shared,
        date_from=date_from,
        date_to=date_to,
        expand_related=expand_related,
        related_window_days=related_window_days,
        min_score=min_score,
    )


def spec_to_dict(spec: ExtractSpec) -> dict[str, Any]:
    return {
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


def conversation_messages(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = conversation.get("mapping") or {}
    messages: list[dict[str, Any]] = []

    for node_id, node in mapping.items():
        message = (node or {}).get("message")
        if not message:
            continue

        content = message.get("content")
        messages.append(
            {
                "node_id": node_id,
                "parent_id": node.get("parent"),
                "author_role": (message.get("author") or {}).get("role"),
                "author_name": (message.get("author") or {}).get("name"),
                "create_time": message.get("create_time"),
                "status": message.get("status"),
                "content_type": (content or {}).get("content_type"),
                "text": extract_text(content),
                "metadata": message.get("metadata") or {},
            }
        )

    messages.sort(
        key=lambda item: (
            item["create_time"] if item["create_time"] is not None else float("inf"),
            item["node_id"],
        )
    )
    return messages


def search_blob(conversation: dict[str, Any]) -> str:
    title = conversation.get("title") or ""
    message_text = "\n".join(message["text"] for message in conversation_messages(conversation))
    return f"{title}\n{message_text}"


def conversation_timestamp(conversation: dict[str, Any]) -> datetime | None:
    create_time = conversation.get("create_time")
    if create_time is None:
        return None
    try:
        return datetime.fromtimestamp(float(create_time), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def conversation_attachment_keys(conversation: dict[str, Any]) -> set[str]:
    return {record["ref"] for record in conversation_attachment_records(conversation)}


ATTACHMENT_EXTENSIONS = {
    "pdf": "pdf",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "webp": "image",
    "gif": "image",
    "wav": "audio",
    "mp3": "audio",
    "m4a": "audio",
    "mp4": "video",
    "mov": "video",
    "txt": "text",
    "md": "text",
    "json": "text",
    "csv": "text",
}


def _attachment_kind(ref: str) -> str:
    lower = ref.lower()
    for ext, kind in ATTACHMENT_EXTENSIONS.items():
        if lower.endswith(f".{ext}"):
            return kind
    if lower.startswith("file-"):
        return "file"
    return "other"


def _looks_like_attachment_ref(value: str) -> bool:
    lower = value.lower()
    if lower.startswith("file-"):
        return True
    if re.search(r"\.(?:pdf|png|jpg|jpeg|webp|gif|wav|mp3|m4a|mp4|mov|txt|md|json|csv)$", lower):
        return True
    if "/mnt/data/" in lower:
        return True
    return False


def conversation_attachment_records(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(value: Any, source_path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                next_path = f"{source_path}.{key}" if source_path else key
                if isinstance(item, str):
                    if key in {"file_id", "file_name", "filename", "name", "path", "url", "title"} and _looks_like_attachment_ref(item):
                        if item not in seen:
                            seen.add(item)
                            records.append(
                                {
                                    "ref": item,
                                    "kind": _attachment_kind(item),
                                    "source_path": next_path,
                                }
                            )
                    elif _looks_like_attachment_ref(item):
                        if item not in seen:
                            seen.add(item)
                            records.append(
                                {
                                    "ref": item,
                                    "kind": _attachment_kind(item),
                                    "source_path": next_path,
                                }
                            )
                elif isinstance(item, (dict, list)):
                    visit(item, next_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{source_path}[{index}]")
        elif isinstance(value, str) and _looks_like_attachment_ref(value):
            if value not in seen:
                seen.add(value)
                records.append(
                    {
                        "ref": value,
                        "kind": _attachment_kind(value),
                        "source_path": source_path,
                    }
                )

    visit(conversation, "")
    records.sort(key=lambda item: (item["kind"], item["ref"]))
    return records


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def attachment_kind_summary(attachments: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(item["kind"] for item in attachments))


@dataclass(frozen=True)
class MatchResult:
    conversation_id: str
    title: str
    source_file: str
    create_time: float | None
    update_time: float | None
    memory_scope: str | None
    gizmo_type: str | None
    conversation_origin: Any
    is_shared: bool
    project_enabled: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    title_keyword_matches: list[str] = field(default_factory=list)
    body_keyword_matches: list[str] = field(default_factory=list)
    regex_matches: list[str] = field(default_factory=list)
    is_related: bool = False
    related_to: str | None = None


def _scope_text(conversation: dict[str, Any], scope: MatchScope) -> tuple[str, str]:
    title = conversation.get("title") or ""
    body = "\n".join(message["text"] for message in conversation_messages(conversation))
    if scope == "title":
        return title, ""
    if scope == "body":
        return "", body
    return title, body


def _has_keyword(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()


def score_conversation(
    conversation: dict[str, Any],
    spec: ExtractSpec,
    *,
    source_file: str = "",
    shared_ids: set[str] | None = None,
) -> MatchResult | None:
    conversation_id = conversation.get("conversation_id") or conversation.get("id") or ""
    title = conversation.get("title") or ""
    project_enabled = conversation.get("memory_scope") == "project_enabled"
    is_shared = bool(shared_ids and conversation_id in shared_ids)

    timestamp = conversation_timestamp(conversation)
    if spec.date_from and timestamp and timestamp < spec.date_from:
        return None
    if spec.date_to and timestamp and timestamp > spec.date_to:
        return None

    title_text, body_text = _scope_text(conversation, spec.scope)
    matched_keywords: list[str] = []
    title_keyword_matches: list[str] = []
    body_keyword_matches: list[str] = []
    reasons: list[str] = []
    score = 0.0
    regex_matches: list[str] = []

    for keyword in spec.keywords:
        title_hit = bool(title_text and _has_keyword(title_text, keyword))
        body_hit = bool(body_text and _has_keyword(body_text, keyword))
        hit = title_hit or body_hit
        if hit:
            matched_keywords.append(keyword)
        if title_hit:
            title_keyword_matches.append(keyword)
            score += 1.0
            reasons.append(f"title hit: {keyword}")
        if body_hit:
            body_keyword_matches.append(keyword)
            score += 0.5
            reasons.append(f"body hit: {keyword}")

    if spec.regex_patterns:
        haystack = f"{title_text}\n{body_text}".strip()
        for pattern in spec.regex_patterns:
            if re.search(pattern, haystack, flags=re.IGNORECASE | re.MULTILINE):
                regex_matches.append(pattern)
                score += 1.0
                reasons.append(f"regex hit: {pattern}")

    has_keyword_hit = bool(matched_keywords)
    has_regex_hit = bool(regex_matches)
    has_project_or_shared = (spec.include_project_enabled and project_enabled) or (
        spec.include_shared and is_shared
    )

    if spec.keywords:
        if spec.match_mode == "all" and len(matched_keywords) != len(spec.keywords):
            return None
        if not has_keyword_hit and not has_regex_hit and not has_project_or_shared:
            return None
    elif not has_regex_hit and not has_project_or_shared:
        return None

    for keyword in spec.exclude_keywords:
        if _has_keyword(f"{title_text}\n{body_text}", keyword):
            return None

    if project_enabled and spec.include_project_enabled:
        score += 0.25
        reasons.append("project enabled")
    if is_shared and spec.include_shared:
        score += 0.25
        reasons.append("shared conversation")

    if score < spec.min_score:
        return None

    return MatchResult(
        conversation_id=str(conversation_id),
        title=str(title),
        source_file=source_file,
        create_time=conversation.get("create_time"),
        update_time=conversation.get("update_time"),
        memory_scope=conversation.get("memory_scope"),
        gizmo_type=conversation.get("gizmo_type"),
        conversation_origin=conversation.get("conversation_origin"),
        is_shared=is_shared,
        project_enabled=project_enabled,
        score=score,
        reasons=reasons,
        matched_keywords=matched_keywords,
        title_keyword_matches=title_keyword_matches,
        body_keyword_matches=body_keyword_matches,
        regex_matches=regex_matches,
    )


def load_conversations(
    archive_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(archive_root).expanduser().resolve()
    conversations: list[dict[str, Any]] = []
    shared_ids = load_shared_conversation_ids(root)

    for source_file in iter_conversation_files(root):
        data = load_json(source_file)
        if not isinstance(data, list):
            continue
        for conversation in data:
            if not isinstance(conversation, dict):
                continue
            conversations.append(
                {
                    "conversation": conversation,
                    "source_file": source_file.name,
                    "is_shared": str(
                        conversation.get("conversation_id") or conversation.get("id") or ""
                    )
                    in shared_ids,
                }
            )
    return conversations


def preview_matches(
    archive_root: str | Path,
    spec: ExtractSpec,
) -> list[MatchResult]:
    loaded = load_conversations(archive_root)
    shared_ids = load_shared_conversation_ids(Path(archive_root))
    seeds: list[MatchResult] = []
    for item in loaded:
        match = score_conversation(
            item["conversation"],
            spec,
            source_file=item["source_file"],
            shared_ids=shared_ids,
        )
        if match:
            seeds.append(match)

    if spec.expand_related:
        selected = expand_related_matches(seeds, loaded, spec, shared_ids=shared_ids)
    else:
        selected = seeds

    return sorted(selected, key=lambda item: (-item.score, item.create_time or float("inf"), item.title))


def select_conversations(
    archive_root: str | Path,
    spec: ExtractSpec,
) -> list[MatchResult]:
    return preview_matches(archive_root, spec)


def expand_related_matches(
    matches: list[MatchResult],
    conversations: list[dict[str, Any]],
    spec: ExtractSpec,
    *,
    shared_ids: set[str] | None = None,
) -> list[MatchResult]:
    selected_by_id: dict[str, MatchResult] = {match.conversation_id: match for match in matches}
    conversation_lookup = {
        str(item["conversation"].get("conversation_id") or item["conversation"].get("id") or ""): item["conversation"]
        for item in conversations
    }
    seed_conversations = {
        conversation_id: conversation_lookup[conversation_id]
        for conversation_id in selected_by_id
        if conversation_id in conversation_lookup
    }
    seed_attachment_keys = {
        conversation_id: conversation_attachment_keys(conversation)
        for conversation_id, conversation in seed_conversations.items()
    }

    for item in conversations:
        conversation = item["conversation"]
        conversation_id = str(conversation.get("conversation_id") or conversation.get("id") or "")
        if conversation_id in selected_by_id:
            continue

        candidate_timestamp = conversation_timestamp(conversation)
        candidate_attachments = conversation_attachment_keys(conversation)
        candidate_title = conversation.get("title") or ""
        candidate_memory_scope = conversation.get("memory_scope")
        candidate_is_shared = bool(shared_ids and conversation_id in shared_ids)

        best_score = 0.0
        best_reason: str | None = None
        best_seed_id: str | None = None

        for seed_id, seed in seed_conversations.items():
            seed_timestamp = conversation_timestamp(seed)
            score = 0.0
            reason_bits: list[str] = []

            if seed_timestamp and candidate_timestamp:
                delta_days = abs((candidate_timestamp - seed_timestamp).days)
                if delta_days <= spec.related_window_days:
                    score += 0.25
                    reason_bits.append(f"within {delta_days} day(s)")

            similarity = title_similarity(seed.get("title") or "", candidate_title)
            if similarity >= 0.55:
                score += 0.5
                reason_bits.append(f"title similarity {similarity:.2f}")

            if seed_attachment_keys.get(seed_id) and candidate_attachments:
                if seed_attachment_keys[seed_id] & candidate_attachments:
                    score += 0.75
                    reason_bits.append("shared attachment")

            if candidate_memory_scope == seed.get("memory_scope"):
                score += 0.1
                reason_bits.append("same memory scope")

            if candidate_is_shared and spec.include_shared:
                score += 0.1
                reason_bits.append("shared conversation")

            if score > best_score:
                best_score = score
                best_reason = "; ".join(reason_bits)
                best_seed_id = seed_id

        if best_score >= 0.75 and best_reason and best_seed_id:
            selected_by_id[conversation_id] = MatchResult(
                conversation_id=conversation_id,
                title=candidate_title,
                source_file=item["source_file"],
                create_time=conversation.get("create_time"),
                update_time=conversation.get("update_time"),
                memory_scope=candidate_memory_scope,
                gizmo_type=conversation.get("gizmo_type"),
                conversation_origin=conversation.get("conversation_origin"),
                is_shared=candidate_is_shared,
                project_enabled=candidate_memory_scope == "project_enabled",
                score=best_score,
                reasons=[f"related to {best_seed_id}: {best_reason}"],
                matched_keywords=[],
                title_keyword_matches=[],
                body_keyword_matches=[],
                regex_matches=[],
                is_related=True,
                related_to=best_seed_id,
            )

    return list(selected_by_id.values())


def normalize_conversation(
    conversation: dict[str, Any],
    source_file: Path,
    match: MatchResult | None,
) -> dict[str, Any]:
    messages = conversation_messages(conversation)
    user_message_count = sum(1 for msg in messages if msg["author_role"] == "user")
    assistant_message_count = sum(
        1 for msg in messages if msg["author_role"] == "assistant"
    )

    return {
        "conversation": {
            "conversation_id": conversation.get("conversation_id") or conversation.get("id") or "",
            "title": conversation.get("title") or "",
            "source_file": str(source_file.name),
            "create_time": conversation.get("create_time"),
            "update_time": conversation.get("update_time"),
            "memory_scope": conversation.get("memory_scope"),
            "gizmo_type": conversation.get("gizmo_type"),
            "conversation_origin": conversation.get("conversation_origin"),
            "is_shared": match.is_shared if match else False,
            "project_enabled": match.project_enabled if match else conversation.get("memory_scope") == "project_enabled",
            "score": match.score if match else 0.0,
            "reasons": match.reasons if match else [],
            "matched_keywords": match.matched_keywords if match else [],
            "title_keyword_matches": match.title_keyword_matches if match else [],
            "body_keyword_matches": match.body_keyword_matches if match else [],
            "regex_matches": match.regex_matches if match else [],
            "is_related": match.is_related if match else False,
            "related_to": match.related_to if match else None,
            "message_count": len(messages),
            "user_message_count": user_message_count,
            "assistant_message_count": assistant_message_count,
        },
        "messages": messages,
        "raw": {
            key: conversation.get(key)
            for key in [
                "conversation_id",
                "create_time",
                "update_time",
                "title",
                "memory_scope",
                "gizmo_type",
                "conversation_origin",
                "is_archived",
                "is_starred",
                "atlas_mode_enabled",
                "context_scopes",
                "conversation_template_id",
            ]
        },
        "attachments": conversation_attachment_records(conversation),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "conversation_id",
        "title",
        "source_file",
        "create_time",
        "update_time",
        "memory_scope",
        "gizmo_type",
        "project_enabled",
        "is_shared",
        "is_related",
        "related_to",
        "score",
        "matched_keywords",
        "title_keyword_matches",
        "body_keyword_matches",
        "regex_matches",
        "attachment_count",
        "message_count",
        "user_message_count",
        "assistant_message_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_extraction_output(
    output_dir: Path,
    manifest: dict[str, Any],
    selected_records: list[dict[str, Any]],
    normalized_conversations: list[dict[str, Any]],
) -> None:
    write_json(output_dir / "manifest.json", manifest)
    write_csv(output_dir / "selected_conversations.csv", selected_records)
    write_json(output_dir / "selected_conversations.json", selected_records)
    write_json(
        output_dir / "title_keyword_matches.json",
        [row for row in selected_records if row["title_keyword_matches"]],
    )
    write_json(output_dir / "normalized_conversations.json", normalized_conversations)
    write_json(output_dir / "summary.json", build_summary_data(manifest, selected_records))
    (output_dir / "summary.md").write_text(
        build_summary_markdown(manifest, selected_records),
        encoding="utf-8",
    )
    attachment_inventory = build_attachment_inventory_data(normalized_conversations)
    write_json(output_dir / "attachments.json", attachment_inventory)
    (output_dir / "attachments.md").write_text(
        build_attachment_inventory_markdown(attachment_inventory),
        encoding="utf-8",
    )


def build_summary_data(
    manifest: dict[str, Any],
    selected_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "query": manifest["query"],
        "loaded_count": manifest["loaded_count"],
        "seed_count": manifest["seed_count"],
        "selected_count": manifest["selected_count"],
        "selected_titles": manifest["selected_titles"],
        "top_matches": [
            {
                "conversation_id": row["conversation_id"],
                "title": row["title"],
                "score": row["score"],
                "matched_keywords": row["matched_keywords"],
                "is_related": row["is_related"],
                "related_to": row["related_to"],
                "attachment_count": row.get("attachment_count", 0),
            }
            for row in selected_records[:10]
        ],
    }


def build_summary_markdown(
    manifest: dict[str, Any],
    selected_records: list[dict[str, Any]],
) -> str:
    lines = [
        "# ChatGPT Export Summary",
        "",
        f"- Archive: `{manifest['archive_root']}`",
        f"- Output: `{manifest['output_dir']}`",
        f"- Loaded conversations: `{manifest['loaded_count']}`",
        f"- Seed matches: `{manifest['seed_count']}`",
        f"- Selected conversations: `{manifest['selected_count']}`",
        "",
        "## Query",
        "",
        "```json",
        json.dumps(manifest["query"], indent=2, ensure_ascii=True),
        "```",
        "",
        "## Selected Conversations",
        "",
    ]
    if not selected_records:
        lines.append("_No conversations selected._")
        lines.append("")
        return "\n".join(lines)

    for row in selected_records[:20]:
        reasons = row["matched_keywords"] or row["title_keyword_matches"] or row["body_keyword_matches"] or row["regex_matches"]
        lines.append(
            f"- `{row['title']}` ({row['conversation_id']}) score={row['score']:.2f}, attachments={row.get('attachment_count', 0)}"
        )
        if reasons:
            lines.append(f"  - matches: {reasons}")
        if row["is_related"]:
            lines.append(f"  - related to: `{row['related_to']}`")
    lines.append("")
    return "\n".join(lines)


def build_attachment_inventory_data(
    normalized_conversations: list[dict[str, Any]],
) -> dict[str, Any]:
    by_kind = Counter()
    by_ref: dict[str, dict[str, Any]] = {}
    by_conversation: list[dict[str, Any]] = []

    for normalized in normalized_conversations:
        conversation = normalized["conversation"]
        attachments = normalized.get("attachments", [])
        convo_entry = {
            "conversation_id": conversation["conversation_id"],
            "title": conversation["title"],
            "attachment_count": len(attachments),
            "attachment_kinds": attachment_kind_summary(attachments),
            "attachments": attachments,
        }
        by_conversation.append(convo_entry)
        for attachment in attachments:
            by_kind[attachment["kind"]] += 1
            ref = attachment["ref"]
            entry = by_ref.setdefault(
                ref,
                {
                    "ref": ref,
                    "kind": attachment["kind"],
                    "count": 0,
                    "conversation_ids": [],
                    "titles": [],
                },
            )
            entry["count"] += 1
            entry["conversation_ids"].append(conversation["conversation_id"])
            entry["titles"].append(conversation["title"])

    unique_attachments = sorted(by_ref.values(), key=lambda item: (-item["count"], item["ref"]))
    return {
        "by_kind": dict(by_kind),
        "unique_attachments": unique_attachments,
        "by_conversation": by_conversation,
    }


def build_attachment_inventory_markdown(attachment_inventory: dict[str, Any]) -> str:
    lines = [
        "# Attachment Inventory",
        "",
        "## By Type",
        "",
    ]
    by_kind = attachment_inventory.get("by_kind", {})
    if by_kind:
        for kind, count in sorted(by_kind.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {kind}: {count}")
    else:
        lines.append("_No attachments found._")
    lines.extend(["", "## By Conversation", ""])
    by_conversation = attachment_inventory.get("by_conversation", [])
    if not by_conversation:
        lines.append("_No attachments found._")
        lines.append("")
        return "\n".join(lines)
    for convo in by_conversation:
        lines.append(
            f"- `{convo['title']}` ({convo['conversation_id']}): {convo['attachment_count']} attachment(s)"
        )
        for attachment in convo.get("attachments", []):
            lines.append(
                f"  - `{attachment['ref']}` [{attachment['kind']}] at `{attachment['source_path']}`"
            )
    lines.append("")
    return "\n".join(lines)


def extract_archive(
    archive_root: str | Path,
    output_dir: str | Path | None = None,
    spec: ExtractSpec | None = None,
) -> dict[str, Any]:
    spec = spec or build_spec()
    archive_root = Path(archive_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir(archive_root)

    if not archive_root.exists():
        raise FileNotFoundError(f"Archive root does not exist: {archive_root}")

    loaded = load_conversations(archive_root)
    shared_ids = load_shared_conversation_ids(archive_root)
    seed_matches: list[MatchResult] = []
    for item in loaded:
        match = score_conversation(
            item["conversation"],
            spec,
            source_file=item["source_file"],
            shared_ids=shared_ids,
        )
        if match:
            seed_matches.append(match)

    selected_matches = (
        expand_related_matches(seed_matches, loaded, spec, shared_ids=shared_ids)
        if spec.expand_related
        else seed_matches
    )
    selected_matches = sorted(
        selected_matches,
        key=lambda item: (-item.score, item.create_time or float("inf"), item.title),
    )

    selected_records: list[dict[str, Any]] = []
    normalized_conversations: list[dict[str, Any]] = []
    conversation_by_id = {
        str(item["conversation"].get("conversation_id") or item["conversation"].get("id") or ""): item
        for item in loaded
    }

    for match in selected_matches:
        item = conversation_by_id.get(match.conversation_id)
        if not item:
            continue
        conversation = item["conversation"]
        normalized = normalize_conversation(conversation, Path(item["source_file"]), match)
        safe_name = f"{slugify(match.title)}-{match.conversation_id[:8]}"
        out_path = output_dir / "conversations" / f"{safe_name}.json"
        write_json(out_path, normalized)
        normalized_conversations.append(normalized)
        selected_records.append(
            {
                "conversation_id": match.conversation_id,
                "title": match.title,
                "source_file": item["source_file"],
                "create_time": match.create_time,
                "update_time": match.update_time,
                "memory_scope": match.memory_scope,
                "gizmo_type": match.gizmo_type,
                "project_enabled": match.project_enabled,
                "is_shared": match.is_shared,
                "is_related": match.is_related,
                "related_to": match.related_to,
                "score": match.score,
                "matched_keywords": "|".join(match.matched_keywords),
                "title_keyword_matches": "|".join(match.title_keyword_matches),
                "body_keyword_matches": "|".join(match.body_keyword_matches),
                "regex_matches": "|".join(match.regex_matches),
                "attachment_count": len(normalized.get("attachments", [])),
                "message_count": normalized["conversation"]["message_count"],
                "user_message_count": normalized["conversation"]["user_message_count"],
                "assistant_message_count": normalized["conversation"]["assistant_message_count"],
            }
        )

    manifest = {
        "archive_root": str(archive_root),
        "output_dir": str(output_dir),
        "query": spec_to_dict(spec),
        "conversation_files": [str(path.name) for path in iter_conversation_files(archive_root)],
        "loaded_count": len(loaded),
        "seed_count": len(seed_matches),
        "selected_count": len(selected_matches),
        "selected_ids": [row["conversation_id"] for row in selected_records],
        "selected_titles": [row["title"] for row in selected_records],
    }

    write_extraction_output(output_dir, manifest, selected_records, normalized_conversations)
    return manifest


def _default_output_dir(archive_root: Path) -> Path:
    if archive_root.parent.name == "archive":
        return archive_root.parent.parent / "output" / "chatgpt_export_extract"
    return archive_root / "chatgpt_export_extract"
