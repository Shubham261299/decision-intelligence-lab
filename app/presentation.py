"""Escaped, wrapping tables for recorded values; no metric transformation."""

from collections.abc import Mapping, Sequence
from html import escape

import streamlit as st

TABLE_STYLE = """<style>
.research-table-region { width: 100%; overflow: auto;
    border: 1px solid #303d50; border-radius: 8px; margin: .6rem 0 1rem; }
.research-table-region.long { max-height: 32rem; }
.research-table { width: 100%; min-width: 560px; table-layout: fixed;
    border-collapse: collapse; font-size: .92rem; }
.research-table.wide { min-width: 1050px; }
.research-table th, .research-table td { text-align: left; vertical-align: top;
    padding: .8rem; white-space: normal; overflow-wrap: anywhere; line-height: 1.5;
    border-bottom: 1px solid #303d50; }
.research-table th { position: sticky; top: 0; background: #1b2433; color: #dbe5f2; }
.research-table td { color: #e6edf6; }
.research-table-region:focus { outline: 2px solid #8ca9c9; outline-offset: 2px; }
@media print { .research-table-region, .research-table-region.long { overflow: visible; max-height: none; }
    .research-table, .research-table.wide { min-width: 0; font-size: .75rem; }
    .research-table th { position: static; } }
</style>"""


def table_html(rows: Sequence[Mapping[str, object]]) -> str:
    """Keep column order and every value; escape even archive-supplied labels."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    headers = "".join(f'<th scope="col">{escape(column)}</th>' for column in columns)
    body = "".join(
        "<tr>" + "".join(
            f'<td>{escape(str(row[column])) if column in row and row[column] is not None else "Unavailable"}</td>'
            for column in columns
        ) + "</tr>" for row in rows
    )
    variant = " wide" if len(columns) > 4 else ""
    region_variant = " long" if len(rows) > 20 else ""
    return (f'<div class="research-table-region{region_variant}" role="region" aria-label="Recorded evidence table; scroll for all columns and rows" tabindex="0">'
            f'<table class="research-table{variant}"><thead><tr>{headers}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def render_table(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        st.caption("No recorded rows available.")
        return
    st.caption("Values wrap. Scroll the table horizontally on narrow screens; all columns are retained.")
    if len(rows) > 20:
        st.caption("Long table: scroll vertically inside the table to inspect every recorded row.")
    st.markdown(table_html(rows), unsafe_allow_html=True)
