"""
Utility functions for converting Markdown lecture notes into Notion blocks
and uploading them to your Notion workspace.

Features:
- Full Markdown → Notion API-compatible conversion
- Support for headers, lists, code, quotes, callouts, and inline math
- Handles nested blocks and long uploads (>100 blocks)
- Loads token securely from .env file

References:
- https://developers.notion.com/reference/block
- https://www.notion.so/help/markdown-and-shortcuts
"""

import os
import re
from dotenv import load_dotenv
from notion_client import Client

# === LOAD NOTION TOKEN ===
load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_TOKEN")

# === LECTURE FOLDER CONFIGURATION ===
NOTION_FOLDERS = {
    "Applied Artificial Intelligence": "27da86b1aa1d8099a501c2aad9afcccb",
    "Analytical Software Technologies": "27fa86b1aa1d80f5b28cd24aef0de8fa",
    "Reinforcement Learning 1": "27fa86b1aa1d80868edef2861b7d743b",
    "Foundational Machine Learning": "27ea86b1aa1d808e8701d035a917016b",
    "Understanding Deep Learning": "27da86b1aa1d80c3a113edec9fd396b7",
}

# === INLINE MARKDOWN PARSER ===
_inline_re = re.compile(r"(\*\*.+?\*\*|\*[^*]+\*|`[^`]+`|\$[^$]+\$|$begin:math:display$[^$end:math:display$]+\]$begin:math:text$[^)]+$end:math:text$)")

def parse_inline_to_rich_text(text: str):
    """Parse inline markdown (bold, italic, code, links, math) into Notion rich_text spans."""
    spans = []
    pos = 0
    for m in _inline_re.finditer(text):
        if m.start() > pos:
            spans.append({"type": "text", "text": {"content": text[pos:m.start()]}})
        tok = m.group(0)

        if tok.startswith("**"):
            spans.append({"type": "text", "text": {"content": tok[2:-2]}, "annotations": {"bold": True}})
        elif tok.startswith("*"):
            spans.append({"type": "text", "text": {"content": tok[1:-1]}, "annotations": {"italic": True}})
        elif tok.startswith("`"):
            spans.append({"type": "text", "text": {"content": tok[1:-1]}, "annotations": {"code": True}})
        elif tok.startswith("$"):
            spans.append({"type": "equation", "equation": {"expression": tok[1:-1]}})
        elif tok.startswith("[") and "](" in tok:
            match = re.match(r"\[([^\]]+)\]\(([^)]+)\)", tok)
            if match:
                text_part, link_url = match.groups()
                spans.append({
                    "type": "text",
                    "text": {"content": text_part, "link": {"url": link_url}}
                })
        pos = m.end()
    if pos < len(text):
        spans.append({"type": "text", "text": {"content": text[pos:]}})
    return spans


# === MARKDOWN → NOTION BLOCK CONVERTER ===
def markdown_to_blocks(md_text: str):
    """Convert Markdown text into Notion API-compatible blocks."""
    lines = md_text.splitlines()
    blocks, stack = [], []
    in_code, code_lang, code_buf = False, None, []

    def add(block, level=0):
        """Attach block to correct parent or top-level list."""
        while len(stack) > level:
            stack.pop()
        if stack:
            parent = stack[-1]
            children = parent[parent["type"]].setdefault("children", [])
            children.append(block)
        else:
            blocks.append(block)
        if block["type"] in {"bulleted_list_item", "numbered_list_item"}:
            stack.append(block)

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        # Code blocks
        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_lang = line[3:].strip() or "plain text"
            else:
                in_code = False
                add({
                    "object": "block",
                    "type": "code",
                    "code": {
                        "language": code_lang,
                        "rich_text": [{"type": "text", "text": {"content": "\n".join(code_buf)}}],
                    },
                })
                code_buf.clear()
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # Indentation level (for nested lists)
        indent = len(line) - len(line.lstrip(" "))
        level = indent // 4

        # Headers
        if line.startswith("### "):
            add({"object": "block", "type": "heading_3", "heading_3": {"rich_text": parse_inline_to_rich_text(line[4:].strip())}})
        elif line.startswith("## "):
            add({"object": "block", "type": "heading_2", "heading_2": {"rich_text": parse_inline_to_rich_text(line[3:].strip())}})
        elif line.startswith("# "):
            add({"object": "block", "type": "heading_1", "heading_1": {"rich_text": parse_inline_to_rich_text(line[2:].strip())}})

        # Lists
        elif re.match(r"^\s*\d+\.\s", line):
            text = re.sub(r"^\s*\d+\.\s", "", line).strip()
            add({"object": "block", "type": "numbered_list_item", "numbered_list_item": {"rich_text": parse_inline_to_rich_text(text)}}, level)
        elif re.match(r"^\s*[-*+]\s", line):
            text = re.sub(r"^\s*[-*+]\s", "", line).strip()
            add({"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": parse_inline_to_rich_text(text)}}, level)

        # Quotes
        elif line.lstrip().startswith(">"):
            quote = line.lstrip("> ").strip()
            add({"object": "block", "type": "quote", "quote": {"rich_text": parse_inline_to_rich_text(quote)}})

        # Divider
        elif line.strip() == "---":
            add({"object": "block", "type": "divider", "divider": {}})

        # Callouts
        elif line.strip().startswith("<aside>"):
            aside_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("</aside>"):
                aside_lines.append(lines[i])
                i += 1
            add({
                "object": "block",
                "type": "callout",
                "callout": {
                    "icon": {"emoji": "💡"},
                    "rich_text": parse_inline_to_rich_text("\n".join(aside_lines).strip()),
                    "color": "gray_background",
                },
            })

        else:
            add({"object": "block", "type": "paragraph", "paragraph": {"rich_text": parse_inline_to_rich_text(line.strip())}})
        i += 1

    return blocks


# === CREATE PAGE IN NOTION ===
def create_page(notion_token, parent_id, title, md_body):
    """
    Create a new Notion page from Markdown.
    Automatically chunks >100 blocks into multiple API calls.
    """
    if not notion_token:
        raise ValueError("❌ Missing NOTION_TOKEN in environment or argument.")
    if not parent_id:
        raise ValueError("❌ Parent page ID is required.")

    notion = Client(auth=notion_token)
    blocks = markdown_to_blocks(md_body)
    if not blocks:
        raise ValueError("⚠️ No valid blocks found in Markdown text.")

    # Split into ≤100-block chunks (API limit)
    chunks = [blocks[i:i+100] for i in range(0, len(blocks), 100)]

    # Create the main page
    page = notion.pages.create(
        parent={"type": "page_id", "page_id": parent_id},
        properties={"title": [{"type": "text", "text": {"content": title}}]},
        children=chunks[0],
    )

    page_id = page["id"]
    print(f"✅ Created Notion page '{title}' ({page_id})")

    # Append remaining chunks
    for idx, chunk in enumerate(chunks[1:], start=2):
        notion.blocks.children.append(page_id, children=chunk)
        print(f"📦 Uploaded chunk {idx}/{len(chunks)} ({len(chunk)} blocks)")

    print("✅ Upload complete.")
    return page