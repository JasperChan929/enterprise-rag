"""PDF 加载器 v2:文字与表格分流处理,避免内容重复。

核心策略:
  1. 用 find_tables() 识别表格区域(带 bbox)
  2. 过滤掉"伪表格"(列数太少、无数字内容的列表/术语表)
  3. 用表格 bbox 裁剪页面,在表格区域之外抽取纯文字
  4. 表格转 Markdown,文字保留原文
  5. 两者分别成为不同类型的 Chunk,metadata 统一
"""
import re
from pathlib import Path

import pdfplumber

from src.loaders.base import Chunk


# ============================================================
# 工具函数
# ============================================================

def parse_filename(pdf_path: Path) -> dict:
    """从文件名提取 metadata。

    约定格式: {股票代码}_{公司名}_{年份}年年度报告.pdf
    例: 600519_贵州茅台_2023年年度报告.pdf
    """
    name = pdf_path.stem
    match = re.match(r"(\d{6})_(.+?)_(\d{4})年年度报告", name)
    if not match:
        print(f"⚠️  文件名不符合约定: {name}")
        return {"source": pdf_path.name}

    return {
        "source": pdf_path.name,
        "stock_code": match.group(1),
        "company": match.group(2),
        "year": int(match.group(3)),
    }


def is_real_table(table) -> bool:
    """判断 pdfplumber 识别出的 table 是不是"真表格"。

    过滤掉"伪表格"——项目符号列表、术语解释、目录等被误识别的结构。

    启发式规则:
      1. 至少 3 列才算真表格(2 列大概率是"名称-解释"列表)
      2. 至少 2 行数据(只有表头没有数据行 → 不是有价值的表格)
      3. 数据行里至少有一个数字(纯文字表格往往是术语表,不是财务数据)
    """
    rows = table.extract()
    if not rows:
        return False

    num_cols = max(len(row) for row in rows)
    num_data_rows = len(rows) - 1  # 减去表头

    # 规则 1: 列数太少
    if num_cols < 3:
        return False

    # 规则 2: 数据行太少
    if num_data_rows < 1:
        return False

    # 规则 3: 数据行里是否包含数字
    has_number = False
    for row in rows[1:]:  # 跳过表头
        for cell in row:
            if cell and re.search(r"\d", cell):
                has_number = True
                break
        if has_number:
            break

    return has_number


def table_to_markdown(raw_table: list[list[str]]) -> str:
    """把 pdfplumber 的二维列表转成 Markdown 表格。

    处理细节:
      - None → 空字符串
      - 单元格内换行 → 空格(Markdown 表格不支持单元格内换行)
      - 行长度不一致 → 用空串补齐
    """
    if not raw_table or not raw_table[0]:
        return ""

    cleaned = [
        [(cell or "").replace("\n", " ").strip() for cell in row]
        for row in raw_table
    ]

    header = cleaned[0]
    rows = cleaned[1:]
    num_cols = len(header)

    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * num_cols) + " |",
    ]
    for row in rows:
        padded = row + [""] * (num_cols - len(row))
        md_lines.append("| " + " | ".join(padded[:num_cols]) + " |")

    return "\n".join(md_lines)


def clean_text(text: str) -> str:
    """清理从 PDF 抽出来的文字。

    处理:
      - 去掉页眉(匹配"XXXX年年度报告"这种模式)
      - 去掉页码(匹配"数字 / 数字"模式)
      - 去掉表单标记(□适用 √不适用)
      - 合并被错误换行切断的句子
    """
    # 去页眉: "2023年年度报告" 或 "2023 年年度报告"
    text = re.sub(r"\d{4}\s*年\s*年度报告", "", text)

    # 去页码: "6 / 143" 或 "6/143"
    text = re.sub(r"\d+\s*/\s*\d+", "", text)

    # 去表单标记
    text = re.sub(r"[□√](?:适用|不适用)", "", text)

    # 合并被切断的行: 如果一行末尾不是句号、分号等标点,
    # 且下一行开头不是数字/标题标记,说明是同一句话被换行了
    lines = text.split("\n")
    merged = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if merged and not re.match(r"^[一二三四五六七八九十\d（(]", line):
            prev = merged[-1]
            # 上一行末尾不是终结标点 → 合并
            if prev and prev[-1] not in "。；：！？)）》」、,，":
                merged[-1] = prev + line
                continue

        merged.append(line)

    return "\n".join(merged)


# ============================================================
# 主加载函数
# ============================================================

def load_pdf(pdf_path: Path) -> list[Chunk]:
    """主入口:加载 PDF,返回去重后的 Chunk 列表。

    对每一页:
      1. find_tables() 找到所有表格(带 bbox)
      2. 过滤伪表格
      3. 真表格 → 转 Markdown → 成为 table chunk
      4. 扣除真表格 bbox 区域,在剩余区域抽文字 → 成为 text chunk
    """
    pdf_path = Path(pdf_path)
    file_metadata = parse_filename(pdf_path)
    chunks: list[Chunk] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_meta = {**file_metadata, "page": page_num}

            # ---- 1. 识别表格 ----
            tables = page.find_tables() or []

            # ---- 2. 分离真/伪表格 ----
            real_tables = [t for t in tables if is_real_table(t)]

            # ---- 3. 真表格 → Markdown chunks ----
            for i, table in enumerate(real_tables):
                md = table_to_markdown(table.extract())
                if md.strip():
                    chunks.append(Chunk(
                        content=md,
                        chunk_type="table",
                        metadata={**page_meta, "table_index": i},
                    ))

            # ---- 4. 扣除表格区域,抽剩余文字 ----
            if real_tables:
                # 获取所有真表格的 bbox
                table_bboxes = [t.bbox for t in real_tables]

                # 用 pdfplumber 的 filter 功能:
                # 只保留"不在任何表格 bbox 内"的文字
                def not_within_any_table(obj):
                    """判断一个文字对象是否在任何表格区域之外。"""
                    for bbox in table_bboxes:
                        x0, y0, x1, y1 = bbox
                        # obj 的中心点在 bbox 内 → 属于表格,排除
                        obj_center_x = (obj["x0"] + obj["x1"]) / 2
                        obj_center_y = (obj["top"] + obj["bottom"]) / 2
                        if x0 <= obj_center_x <= x1 and y0 <= obj_center_y <= y1:
                            return False
                    return True

                # 过滤后的页面只包含表格区域之外的文字
                filtered_page = page.filter(not_within_any_table)
                text = filtered_page.extract_text() or ""
            else:
                # 没有真表格,直接抽全部文字
                text = page.extract_text() or ""

            # ---- 5. 清理文字 ----
            text = clean_text(text)
            text = text.strip()

            if text and len(text) > 20:  # 太短的忽略(可能只剩标点/空格)
                chunks.append(Chunk(
                    content=text,
                    chunk_type="text",
                    metadata=page_meta,
                ))

    return chunks