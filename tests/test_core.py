from __future__ import annotations

import argparse
import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chatgpt_export.core import (  # noqa: E402
    build_spec,
    extract_archive,
    preview_matches,
    score_conversation,
    slugify,
)
from chatgpt_export.cli import _resolve_command, _save_preset, _spec_from_args, _load_presets, main  # noqa: E402


def make_conversation(
    conversation_id: str,
    title: str,
    *,
    body: str = "",
    memory_scope: str = "global_enabled",
    create_time: float = 1_700_000_000.0,
    attachment_id: str | None = None,
) -> dict:
    metadata = {}
    if attachment_id:
        metadata = {"attachments": [{"file_id": attachment_id}]}
    return {
        "conversation_id": conversation_id,
        "id": conversation_id,
        "title": title,
        "create_time": create_time,
        "update_time": create_time + 10,
        "memory_scope": memory_scope,
        "gizmo_type": "gpt",
        "conversation_origin": None,
        "mapping": {
            "root": {"id": "root", "parent": None, "children": ["user"]},
            "user": {
                "id": "user",
                "parent": "root",
                "children": ["assistant"],
                "message": {
                    "author": {"role": "user", "name": None},
                    "content": {"content_type": "text", "parts": [body]},
                    "create_time": create_time,
                    "status": "finished_successfully",
                    "metadata": metadata,
                },
            },
            "assistant": {
                "id": "assistant",
                "parent": "user",
                "children": [],
                "message": {
                    "author": {"role": "assistant", "name": None},
                    "content": {"content_type": "text", "parts": ["reply"]},
                    "create_time": create_time + 1,
                    "status": "finished_successfully",
                    "metadata": {},
                },
            },
        },
    }


def write_export(root: Path, conversations: list[dict], shared_ids: list[str] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "conversations-000.json").open("w", encoding="utf-8") as handle:
        json.dump(conversations, handle)
    if shared_ids is not None:
        shared = [{"conversation_id": cid, "title": ""} for cid in shared_ids]
        with (root / "shared_conversations.json").open("w", encoding="utf-8") as handle:
            json.dump(shared, handle)


def make_deep_research_conversation(
    conversation_id: str,
    title: str,
    *,
    body: str,
    report_text: str,
    create_time: float = 1_700_000_000.0,
) -> dict:
    conversation = make_conversation(
        conversation_id,
        title,
        body=body,
        create_time=create_time,
    )
    conversation["mapping"]["hidden_system"] = {
        "id": "hidden_system",
        "parent": "assistant",
        "children": ["widget_state"],
        "message": {
            "author": {"role": "system", "name": None},
            "content": {"content_type": "text", "parts": ["hidden system text"]},
            "create_time": create_time + 2,
            "status": "finished_successfully",
            "metadata": {"is_visually_hidden_from_conversation": True},
        },
    }
    widget_state = {
        "status": "completed",
        "report_message": {
            "content": {
                "content_type": "text",
                "parts": [report_text],
            }
        },
    }
    conversation["mapping"]["widget_state"] = {
        "id": "widget_state",
        "parent": "hidden_system",
        "children": [],
        "message": {
            "author": {"role": "tool", "name": None},
            "content": {
                "content_type": "text",
                "parts": [f'The latest state of the widget is: {json.dumps(widget_state)}'],
            },
            "create_time": create_time + 3,
            "status": "finished_successfully",
            "metadata": {"is_visually_hidden_from_conversation": True},
        },
    }
    return conversation


def make_cli_args(**overrides) -> argparse.Namespace:
    defaults = {
        "archive_root": ".",
        "output": None,
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
        "dry_run": False,
        "output_format": None,
        "presets": None,
        "presets_file": None,
        "command": "extract",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class CoreTests(unittest.TestCase):
    def test_title_only_match(self) -> None:
        conversation = make_conversation("c1", "Criticality Metrics Proposal", body="nothing")
        spec = build_spec(keywords=["criticality"], scope="title")
        match = score_conversation(conversation, spec, source_file="conversations-000.json")
        self.assertIsNotNone(match)
        self.assertEqual(match.title_keyword_matches, ["criticality"])
        self.assertEqual(match.body_keyword_matches, [])

    def test_body_only_match(self) -> None:
        conversation = make_conversation("c2", "Unrelated title", body="criticality appears here")
        spec = build_spec(keywords=["criticality"], scope="body")
        match = score_conversation(conversation, spec, source_file="conversations-000.json")
        self.assertIsNotNone(match)
        self.assertEqual(match.title_keyword_matches, [])
        self.assertEqual(match.body_keyword_matches, ["criticality"])

    def test_any_vs_all(self) -> None:
        conversation = make_conversation("c3", "Criticality organoid study", body="body")
        any_spec = build_spec(keywords=["criticality", "organoid"], match_mode="any")
        all_spec = build_spec(keywords=["criticality", "organoid"], match_mode="all")
        self.assertIsNotNone(score_conversation(conversation, any_spec, source_file="conversations-000.json"))
        self.assertIsNotNone(score_conversation(conversation, all_spec, source_file="conversations-000.json"))

        partial = make_conversation("c4", "Criticality only", body="body")
        self.assertIsNotNone(score_conversation(partial, any_spec, source_file="conversations-000.json"))
        self.assertIsNone(score_conversation(partial, all_spec, source_file="conversations-000.json"))

    def test_exclude_keywords(self) -> None:
        conversation = make_conversation("c5", "Criticality Metrics Proposal", body="skip this")
        spec = build_spec(keywords=["criticality"], exclude_keywords=["skip"])
        self.assertIsNone(score_conversation(conversation, spec, source_file="conversations-000.json"))

    def test_project_enabled_inclusion(self) -> None:
        conversation = make_conversation("c6", "No keyword here", memory_scope="project_enabled")
        spec = build_spec(keywords=[], include_project_enabled=True)
        match = score_conversation(conversation, spec, source_file="conversations-000.json")
        self.assertIsNotNone(match)
        self.assertTrue(match.project_enabled)

    def test_regex_only_match(self) -> None:
        conversation = make_conversation("c6r", "Unrelated title", body="alpha 123 beta")
        spec = build_spec(keywords=[], regex_patterns=[r"\d{3}"])
        match = score_conversation(conversation, spec, source_file="conversations-000.json")
        self.assertIsNotNone(match)
        self.assertEqual(match.regex_matches, [r"\d{3}"])

    def test_related_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conversations = [
                make_conversation("seed", "Criticality Metrics Proposal", body="seed text", create_time=1_700_000_000.0),
                make_conversation(
                    "related",
                    "Followup notes",
                    body="unrelated body",
                    create_time=1_700_000_050.0,
                    attachment_id="file-abc123",
                ),
                make_conversation(
                    "other",
                    "Completely different",
                    body="no link",
                    create_time=1_700_200_000.0,
                ),
            ]
            conversations[0]["mapping"]["user"]["message"]["metadata"] = {
                "attachments": [{"file_id": "file-abc123"}]
            }
            write_export(root, conversations)
            spec = build_spec(keywords=["criticality"], expand_related=True, related_window_days=2)
            matches = preview_matches(root, spec)
            ids = {match.conversation_id for match in matches}
            self.assertIn("seed", ids)
            self.assertIn("related", ids)
            self.assertNotIn("other", ids)

    def test_preview_and_extract_return_consistent_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            out = Path(tmp) / "out"
            conversations = [
                make_conversation("seed", "Criticality Metrics Proposal", body="criticality in body"),
                make_conversation("body", "Something else", body="criticality in body"),
                make_conversation("ignore", "No match", body="nothing here"),
            ]
            write_export(root, conversations)
            spec = build_spec(keywords=["criticality"])
            preview = preview_matches(root, spec)
            manifest = extract_archive(root, output_dir=out, spec=spec)

            preview_ids = [match.conversation_id for match in preview]
            self.assertEqual(preview_ids, manifest["selected_ids"])
            with (out / "normalized_conversations.json").open("r", encoding="utf-8") as handle:
                normalized = json.load(handle)
            normalized_ids = [item["conversation"]["conversation_id"] for item in normalized]
            self.assertEqual(preview_ids, normalized_ids)
            self.assertTrue((out / "conversations" / "criticality-metrics-proposal-seed.json").exists())
            self.assertTrue((out / "conversations" / "criticality-metrics-proposal-seed.html").exists())
            self.assertTrue((out / "summary.md").exists())
            self.assertTrue((out / "summary.json").exists())

    def test_html_is_clean_and_includes_deep_research_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            out = Path(tmp) / "out"
            report_text = "# Deep Research Report\n\nThis is the report body."
            conversations = [
                make_deep_research_conversation(
                    "deep",
                    "Magical Properties of Criticality",
                    body="what are the magical properties of complex systems that emerge at critical points?",
                    report_text=report_text,
                )
            ]
            write_export(root, conversations)
            spec = build_spec(keywords=["criticality"], scope="title")
            extract_archive(root, output_dir=out, spec=spec)

            html_path = out / "conversations" / f"{slugify('Magical Properties of Criticality')}-deep.html"
            json_path = out / "conversations" / f"{slugify('Magical Properties of Criticality')}-deep.json"
            self.assertTrue(html_path.exists())
            self.assertTrue(json_path.exists())

            html_text = html_path.read_text(encoding="utf-8")
            normalized = json.loads(json_path.read_text(encoding="utf-8"))

            self.assertIn("<title>Magical Properties of Criticality</title>", html_text)
            self.assertIn("Print / Save as PDF", html_text)
            self.assertIn("what are the magical properties", html_text)
            self.assertIn("Deep Research Report", html_text)
            self.assertIn("This is the report body.", html_text)
            self.assertNotIn("hidden system text", html_text)
            self.assertNotIn("The latest state of the widget is:", html_text)
            self.assertNotIn("is_visually_hidden_from_conversation", html_text)
            self.assertTrue(
                any("hidden system text" in message["text"] for message in normalized["messages"])
            )
            self.assertTrue(
                any(
                    "The latest state of the widget is:" in message["text"]
                    for message in normalized["messages"]
                )
            )

    def test_wizard_command_is_preserved(self) -> None:
        self.assertEqual(_resolve_command(["wizard", "/tmp/export"]), ["wizard", "/tmp/export"])

    def test_saved_preset_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            presets_file = root / ".chatgpt_export_presets.json"
            spec = build_spec(
                keywords=["criticality", "organoid"],
                exclude_keywords=["skip"],
                match_mode="all",
                scope="title",
                include_project_enabled=True,
                expand_related=True,
                related_window_days=7,
                min_score=1.5,
            )
            save_args = make_cli_args(archive_root=str(root), presets_file=str(presets_file))
            _save_preset(save_args, "research", spec)

            load_args = make_cli_args(archive_root=str(root), presets_file=str(presets_file))
            presets = _load_presets(load_args)
            self.assertIn("research", presets)

            merged_args = make_cli_args(
                archive_root=str(root),
                presets_file=str(presets_file),
                presets=["research"],
            )
            merged_spec = _spec_from_args(merged_args)
            self.assertEqual(merged_spec.keywords, ["criticality", "organoid"])
            self.assertEqual(merged_spec.exclude_keywords, ["skip"])
            self.assertEqual(merged_spec.match_mode, "all")
            self.assertEqual(merged_spec.scope, "title")
            self.assertTrue(merged_spec.include_project_enabled)
            self.assertTrue(merged_spec.expand_related)
            self.assertEqual(merged_spec.related_window_days, 7)
            self.assertEqual(merged_spec.min_score, 1.5)

    def test_list_presets_command_outputs_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            presets_file = root / ".chatgpt_export_presets.json"
            presets_file.write_text(
                json.dumps(
                    {
                        "presets": {
                            "criticality": {"keywords": ["criticality"], "scope": "title"},
                            "organoid": {"keywords": ["organoid"], "scope": "body"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["list-presets", "--presets-file", str(presets_file)])
            output = buf.getvalue()
            self.assertIn("criticality", output)
            self.assertIn("organoid", output)

    def test_attachment_inventory_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            out = Path(tmp) / "out"
            conversations = [
                make_conversation(
                    "seed",
                    "Criticality Metrics Proposal",
                    body="criticality in body",
                    attachment_id="file-abc123",
                ),
                make_conversation("body", "Something else", body="criticality in body"),
            ]
            write_export(root, conversations)
            spec = build_spec(keywords=["criticality"])
            extract_archive(root, output_dir=out, spec=spec)
            with (out / "attachments.json").open("r", encoding="utf-8") as handle:
                attachments = json.load(handle)
            self.assertGreaterEqual(attachments["by_kind"].get("file", 0), 1)
            self.assertEqual(len(attachments["by_conversation"]), 2)
            self.assertTrue((out / "conversations" / "criticality-metrics-proposal-seed.html").exists())
            self.assertTrue((out / "attachments.md").exists())


if __name__ == "__main__":
    unittest.main()
