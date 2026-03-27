# ChatGPT Export Parser

This repository contains a parser for ChatGPT export bundles with keyword-based
conversation selection and related-conversation expansion.

This tool is provided as-is, without warranty. Use it at your own discretion.

## Requirements

- Python 3.10 or newer
- A ChatGPT export bundle as input
- No third-party runtime dependencies

Using a virtual environment is recommended, but not required.

Recommended local layout:

```text
chatgpt-export/
  archive/
    chatgpt-export-archive/
  output/
  src/
  tests/
```

Keep the raw export in `archive/chatgpt-export-archive/` if you want it inside the repo.
Generated extraction output goes in `output/chatgpt_export_extract/`. Both `archive/`
and `output/` are ignored by git.

## Development

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the local workflow and review expectations.

## Troubleshooting

If the `chatgpt-export` command is installed but `import chatgpt_export` fails inside your
virtual environment, recreate the venv and reinstall the package:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

This usually fixes a stale editable-install hook.

## Install

```bash
python3 -m pip install -e .
```

This makes the `chatgpt-export` command and `python3 -m chatgpt_export`
available in the repo environment.

## Usage

```bash
chatgpt-export /path/to/chatgpt-export
```

Preview the selected conversations without writing files:

```bash
chatgpt-export preview /path/to/chatgpt-export --keyword criticality
```

Extract matches to an output directory:

```bash
chatgpt-export extract /path/to/chatgpt-export --keyword criticality --expand-related
```

Run a safe preview through the extract command:

```bash
chatgpt-export extract /path/to/chatgpt-export --keyword criticality --dry-run
```

If you want a guided first run, use the wizard:

```bash
chatgpt-export wizard /path/to/chatgpt-export
```

Saved presets live in `.chatgpt_export_presets.json` by default. List them with:

```bash
chatgpt-export list-presets
```

Load one with:

```bash
chatgpt-export extract /path/to/chatgpt-export --preset criticality
```

## Preset File Format

Preset files are JSON. The loader accepts either of these shapes:

```json
{
  "presets": {
    "criticality": {
      "keywords": ["criticality"]
    }
  }
}
```

or a direct mapping without the outer `presets` key:

```json
{
  "criticality": {
    "keywords": ["criticality"]
  }
}
```

The example file [chatgpt_export_presets.example.json](chatgpt_export_presets.example.json)
shows the full supported preset fields. A preset can include:

- `keywords`
- `exclude_keywords`
- `regex_patterns`
- `match_mode`
- `scope`
- `include_project_enabled`
- `include_shared`
- `date_from`
- `date_to`
- `expand_related`
- `related_window_days`
- `min_score`

Put your working file next to the archive as `.chatgpt_export_presets.json`, or pass a custom
path with `--presets-file`.

List the built-in default keyword set:

```bash
chatgpt-export --list-default-keywords
```

You can also run it as a module:

```bash
python3 -m chatgpt_export extract /path/to/chatgpt-export
```

The extractor writes output to `output/chatgpt_export_extract/` by default when
the archive lives under the repo-local `archive/` folder. Use repeated
`--keyword` flags for broader searches, or `--exclude-keyword`, `--scope`,
`--match-mode`, and `--expand-related` to tune the selection. Each extraction
also writes a human-readable `summary.md` and a machine-readable `summary.json`,
plus `attachments.md` and `attachments.json` for a per-conversation attachment
inventory. Each selected conversation is exported as both JSON and a companion
Markdown transcript under `conversations/`. The Markdown view is a clean
conversation transcript, with any Deep Research report appended as its own
readable section, while the JSON keeps the full raw record for debugging.
