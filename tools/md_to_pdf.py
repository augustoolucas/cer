#!/usr/bin/env python3
"""Convert a Markdown document to PDF (A4) with tables, code blocks and images.

Pipeline: Markdown -> HTML (python-markdown) -> PDF (WeasyPrint/Cairo).

Requirements (isolated environment recommended):
    python3 -m venv .venv-docs
    ./.venv-docs/bin/pip install weasyprint markdown pypdf

Usage:
    ./.venv-docs/bin/python tools/md_to_pdf.py report.md
    ./.venv-docs/bin/python tools/md_to_pdf.py report.md -o out/report.pdf
    ./.venv-docs/bin/python tools/md_to_pdf.py report.md --title "Titulo"
    ./.venv-docs/bin/python tools/md_to_pdf.py report.md --css extra.css

Notes:
- Images are resolved relative to the Markdown file directory.
- Tables are kept on one page when possible.
- Page numbers and the document title are rendered in the footer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_CSS = """
@page {
  size: A4;
  margin: 20mm 16mm 18mm 16mm;
  @bottom-left {
    content: string(doctitle);
    font-size: 7.5pt;
    color: #666;
  }
  @bottom-right {
    content: counter(page) " / " counter(pages);
    font-size: 7.5pt;
    color: #666;
  }
}

html { font-size: 10pt; }

body {
  font-family: "DejaVu Sans", "Liberation Sans", sans-serif;
  line-height: 1.45;
  color: #1a1a1a;
  hyphens: auto;
  text-align: justify;
}

h1, h2, h3, h4 { text-align: left; line-height: 1.25; color: #111; }
h1 {
  font-size: 19pt;
  string-set: doctitle content();
  margin: 0 0 12pt 0;
  padding-bottom: 4pt;
  border-bottom: 1.5pt solid #333;
}
h1 + p, h1 + ul { margin-top: 0; }
h2 { font-size: 13.5pt; margin: 16pt 0 6pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 10pt 0 3pt; page-break-after: avoid; }

p { margin: 0 0 6pt 0; orphans: 2; widows: 2; }
ul, ol { margin: 0 0 6pt 0; padding-left: 16pt; }
li { margin-bottom: 2pt; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 6pt 0 10pt;
  font-size: 8.6pt;
  page-break-inside: avoid;
}
thead { display: table-header-group; }
th, td {
  border: 0.5pt solid #b8b8b8;
  padding: 3pt 5pt;
  text-align: left;
  vertical-align: top;
}
th { background: #eeeeee; font-weight: bold; }
tbody tr:nth-child(even) { background: #f7f7f7; }

code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.6pt;
  background: #f2f2f2;
  padding: 0 1.5pt;
  border-radius: 2pt;
}
pre {
  background: #f6f6f6;
  border: 0.5pt solid #dddddd;
  border-radius: 3pt;
  padding: 6pt 8pt;
  page-break-inside: avoid;
  overflow-wrap: anywhere;
}
pre code { background: none; padding: 0; font-size: 8.2pt; }

img { max-width: 100%; height: auto; }
figure { margin: 8pt 0; page-break-inside: avoid; }

blockquote {
  margin: 6pt 0;
  padding: 4pt 8pt;
  border-left: 2.5pt solid #999;
  color: #333;
  background: #fafafa;
}

hr { border: none; border-top: 0.6pt solid #ccc; margin: 12pt 0; }
a { color: #14489c; text-decoration: none; }
"""


def build_html(markdown_text: str, title: str) -> str:
    import markdown  # noqa: PLC0415

    body = markdown.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "md_in_html"],
        output_format="html5",
    )
    return (
        "<!DOCTYPE html>\n<html lang=\"pt-BR\">\n<head>\n"
        f"<meta charset=\"utf-8\">\n<title>{title}</title>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>\n"
    )


def convert(
    source: Path,
    output: Path,
    *,
    title: str | None = None,
    extra_css: Path | None = None,
    base_css: str = BASE_CSS,
) -> tuple[Path, int]:
    """Render ``source`` (Markdown) into ``output`` (PDF). Returns (path, pages)."""
    from weasyprint import CSS, HTML  # noqa: PLC0415

    text = source.read_text(encoding="utf-8")
    doc_title = title
    if not doc_title:
        for line in text.splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
                break
    doc_title = doc_title or source.stem

    html = build_html(text, doc_title)
    stylesheets = [CSS(string=base_css)]
    if extra_css is not None:
        stylesheets.append(CSS(filename=str(extra_css)))

    output.parent.mkdir(parents=True, exist_ok=True)
    document = HTML(string=html, base_url=str(source.parent)).render(stylesheets=stylesheets)
    document.write_pdf(str(output))
    return output, len(document.pages)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Markdown to A4 PDF (WeasyPrint).",
        epilog="Example: python tools/md_to_pdf.py reports/report.md -o reports/report.pdf",
    )
    parser.add_argument("source", type=Path, help="Input Markdown file.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PDF (default: same stem).")
    parser.add_argument("--title", default=None, help="PDF metadata/footer title (default: first H1).")
    parser.add_argument("--css", type=Path, default=None, help="Extra CSS file appended after the base style.")
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"error: input not found: {args.source}", file=sys.stderr)
        return 2
    output = args.output or args.source.with_suffix(".pdf")

    path, pages = convert(args.source, output, title=args.title, extra_css=args.css)
    size_kib = path.stat().st_size / 1024
    print(f"PDF: {path} ({pages} pages, {size_kib:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
