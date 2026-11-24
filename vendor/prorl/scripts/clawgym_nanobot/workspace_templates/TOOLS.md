# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## WSClaw File Handling

- The complete source workspace is under `input/` and is read-only.
- Inspect files by calling `inspect_workspace_file` with `path` and an optional, task-chosen `max_chars`. It returns JSON containing a summary, extracted preview, structure, parser status, and warnings.
- When a focused search is more efficient, use `exec` to run `python tools/inspect_workspace_file.py <path> [--max-chars N]` and pipe the extracted output to `grep` or `rg`. Filtering inspector output is valid; grepping a binary file directly is not an inspection of its contents.
- `max_chars` is optional rather than a required file-size guard. Use it, a focused `grep`/`rg` pipeline, or a format-specific local Python library when that is useful for the task. If a bounded preview is truncated and omits needed evidence, increase the budget or inspect the relevant remainder.

### Formats the Inspector Handles

- Word documents: `.docx`; legacy `.doc` through `antiword` or `catdoc` when available.
- Excel workbooks: `.xlsx`, `.xlsm`, `.xltx`, `.xltm`, and legacy `.xls`.
- Tabular and database-like files: `.csv`, `.tsv`, `.dat`, `.sample`, and `.db`.
- PDF documents: `.pdf`.
- Presentations: `.pptx`; legacy `.ppt` through best-effort legacy extraction.
- Structured data: `.json`, `.ipynb`, `.yaml`, and `.yml`.
- Statistical data: SPSS `.sav` and SAS transport `.xpt`.
- Email: `.eml`.
- Markup and vector files: `.html`, `.htm`, `.xml`, and `.svg`.
- Text and prose: `.txt`, `.md`, `.markdown`, `.log`, `.rst`, `.tex`, and `.rtf`.
- Source and configuration files: JavaScript/TypeScript, Python, Java, C/C++, C#, Go, Rust, Ruby, PHP, Swift, Kotlin, Scala, shell, PowerShell, batch, SQL, CSS/SCSS/Less, Terraform, TOML, INI, and common config files.
- Images, audio, video, executables, and archives are recognized as binary media. This includes `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.webp`, `.ico`, `.heic`, `.wav`, `.mp3`, `.mp4`, `.mov`, `.avi`, `.mkv`, `.flv`, `.webm`, `.zip`, `.tar`, `.gz`, `.bz2`, `.xz`, `.7z`, and `.rar`. The inspector reports metadata but does not extract their semantic content.
- Unknown extensions receive a best-effort text preview.

### Creating and Editing Files

- `write_file`, `edit_file`, and `apply_patch` write text only. Never use them to create a file merely named `.docx`, `.xlsx`, `.pptx`, `.pdf`, an image, media file, or archive.
- Create binary deliverables by calling the `exec` tool with a real generation command, for example `exec({"command": "python generate_output.py"})`. Create the requested parent directory in the generation script or before running it, write the exact output path, wait for the command to finish, and then inspect the generated file before responding.
- Use `python-docx` for `.docx` Word documents.
- Use `openpyxl` for `.xlsx` workbooks and `pandas` for tabular transformations and `.csv`/`.tsv` output.
- Use `python-pptx` for `.pptx` presentations.
- Use `reportlab` to create `.pdf` files and `pypdf` to inspect, merge, or transform existing PDFs.
- Use Pillow (`PIL`) for `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, and `.webp` images.
- Use the Python standard libraries `json`, `csv`, `sqlite3`, `zipfile`, `tarfile`, `email`, and `xml` where appropriate.
- Use PyYAML for `.yaml`/`.yml`, `lxml` or BeautifulSoup for HTML/XML/SVG, and `pyreadstat` for SPSS/SAS data.
- Legacy `.doc`, `.xls`, and `.ppt` are primarily supported for inspection. Do not fake these binary formats by writing plain text with the legacy suffix.
- For plain text, Markdown, source code, configuration, JSON, YAML, CSV, HTML, and XML, normal file tools are appropriate.
- Run generation scripts synchronously and wait for them to finish before checking outputs.
- After generation, call `inspect_workspace_file` on the output path. A binary deliverable is not complete unless its parser reports `ok` or an explicitly acceptable partial result and the preview contains the intended content.

### Structured Evidence Completeness

- For Excel, the inspector includes worksheet rows until the character budget is exhausted. If `truncated` is true, use `openpyxl.load_workbook(..., read_only=True, data_only=True)` to inspect the remaining rows and sheets.
- For Word, the inspector includes paragraphs and table cells in document order. If `truncated` is true, use `python-docx` to inspect the remaining paragraphs and tables.
- For CSV, TSV, and JSON, inspect all records needed by the task rather than relying on a short sample.
- `grep` and ordinary text reads do not extract evidence from binary Office, PDF, image, or archive files. Pipe inspector output into `grep`/`rg`, or use the format-specific library instead.

## exec - Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- `restrictToWorkspace` config can limit file access to the workspace

## glob - File Discovery

- Use `glob` to find files by pattern before falling back to shell commands
- Simple patterns like `*.py` match recursively by filename
- Use `entry_type="dirs"` when you need matching directories instead of files
- Use `head_limit` and `offset` to page through large result sets
- Prefer this over `exec` when you only need file paths

## grep - Content Search

- Use `grep` to search file contents inside the workspace
- Default behavior returns only matching file paths (`output_mode="files_with_matches"`)
- Supports optional `glob` filtering plus `context_before` / `context_after`
- Use `fixed_strings=true` for literal keywords containing regex characters
- Use `output_mode="count"` to size a search before reading full matches
- Use `head_limit` and `offset` to page across results
- Prefer this over `exec` for code and history searches
- Binary or oversized files may be skipped to keep results readable

## cron - Scheduled Reminders

- Refer to cron skill guidance before scheduling reminders.
