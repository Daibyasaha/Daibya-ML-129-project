"""Converts PROJECT_REPORT.md into a fully styled, publication-grade Word Document (.docx)."""

from pathlib import Path
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, fill_hex):
    """Sets background shading of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def build_docx_report(md_path: Path, docx_path: Path):
    doc = docx.Document()

    # Configure Margins (Normal 1 inch)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    content = md_path.read_text(encoding='utf-8')
    lines = content.split('\n')

    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    def flush_table():
        nonlocal table_rows, in_table
        if not table_rows:
            in_table = False
            return
        
        # Check header separator row
        parsed_rows = []
        for r in table_rows:
            cols = [c.strip() for c in r.strip().strip('|').split('|')]
            if all(set(c).issubset({'-', ':', ' '}) for c in cols if c):
                continue  # Separator row
            parsed_rows.append(cols)

        if not parsed_rows:
            table_rows = []
            in_table = False
            return

        num_cols = max(len(r) for r in parsed_rows)
        tbl = doc.add_table(rows=len(parsed_rows), cols=num_cols)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

        for row_idx, row_data in enumerate(parsed_rows):
            is_header = (row_idx == 0)
            for col_idx in range(num_cols):
                cell = tbl.cell(row_idx, col_idx)
                text = row_data[col_idx] if col_idx < len(row_data) else ""
                clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
                clean_text = clean_text.replace('`', '')
                cell.text = clean_text

                # Formatting
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(3)
                p.paragraph_format.space_after = Pt(3)
                set_cell_margins(cell, top=80, bottom=80, left=120, right=120)

                for run in p.runs:
                    run.font.name = 'Calibri'
                    run.font.size = Pt(9.5 if is_header else 9)
                    if is_header:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    else:
                        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

                if is_header:
                    set_cell_background(cell, "1F497D")  # Deep Navy Blue
                elif row_idx % 2 == 1:
                    set_cell_background(cell, "F2F5F9")  # Soft tint

        doc.add_paragraph()  # Spacing after table
        table_rows = []
        in_table = False

    def flush_code():
        nonlocal code_lines, in_code_block
        if not code_lines:
            in_code_block = False
            return
        
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        set_cell_background(cell, "F4F4F6")
        set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
        
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run('\n'.join(code_lines))
        run.font.name = 'Consolas'
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(0x11, 0x11, 0x11)

        doc.add_paragraph()
        code_lines = []
        in_code_block = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Handle code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                flush_code()
            else:
                if in_table:
                    flush_table()
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Handle tables
        if line.strip().startswith('|') and '|' in line.strip()[1:]:
            in_table = True
            table_rows.append(line)
            i += 1
            continue
        elif in_table:
            flush_table()

        # Handle page breaks & HTML markers
        if 'page-break-after: always' in line:
            doc.add_page_break()
            i += 1
            continue

        if '<div' in line or '</div>' in line or '<br>' in line or '<hr>' in line:
            i += 1
            continue

        stripped = line.strip()
        if not stripped or stripped == '---':
            i += 1
            continue

        # Handle images
        img_match = re.search(r'!\[(.*?)\]\((.*?)\)', stripped)
        if img_match:
            img_rel = img_match.group(2)
            img_file = md_path.parent / img_rel
            if img_file.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(4)
                doc.add_picture(str(img_file), width=Inches(5.5))
            i += 1
            continue

        # Handle figure captions
        if stripped.startswith('*Figure') and stripped.endswith('*'):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(8)
            run = p.add_run(stripped.strip('*'))
            run.font.italic = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            i += 1
            continue

        # Headings
        if stripped.startswith('# '):
            text = stripped[2:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(text)
            run.font.name = 'Calibri'
            run.font.size = Pt(18)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
        elif stripped.startswith('## '):
            text = stripped[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(text)
            run.font.name = 'Calibri'
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)
        elif stripped.startswith('### '):
            text = stripped[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(text)
            run.font.name = 'Calibri'
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
        elif stripped.startswith('- ') or stripped.startswith('* ') or re.match(r'^\d+\.\s', stripped):
            p = doc.add_paragraph(style='List Bullet' if not re.match(r'^\d+\.\s', stripped) else 'List Number')
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            text = re.sub(r'^[-\*]\s+', '', stripped)
            text = re.sub(r'^\d+\.\s+', '', text)
            
            # Parse bold/math formatting simply
            parts = re.split(r'(\*\*.*?\*\*)', text)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    run = p.add_run(part[2:-2])
                    run.bold = True
                else:
                    p.add_run(part.replace('`', '').replace('$', ''))
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15

            parts = re.split(r'(\*\*.*?\*\*)', stripped)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    run = p.add_run(part[2:-2])
                    run.bold = True
                else:
                    p.add_run(part.replace('`', '').replace('$', ''))

        i += 1

    if in_table:
        flush_table()
    if in_code_block:
        flush_code()

    doc.save(str(docx_path))
    print(f"Successfully generated Word report: {docx_path}")


if __name__ == "__main__":
    md_file = Path("PROJECT_REPORT.md")
    docx_file = Path("PROJECT_REPORT.docx")
    build_docx_report(md_file, docx_file)
