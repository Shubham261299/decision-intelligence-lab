"""Wrapping evidence tables preserve source strings, missing states and safety."""

from html import unescape

import pytest

pytest.importorskip("streamlit")

from app.presentation import table_html


def test_table_preserves_all_columns_and_rate_strings() -> None:
    rows = [{"Metric": "Required-abstention correctness", "unconstrained-json": "46/60 (76.7%)",
             "schema-enforced-json": "Not assessable (0/0)", "Missing": None,
             "Long label": "Human Review", "Extra": "Unavailable"}]
    rendered = table_html(rows)
    assert 'class="research-table wide"' in rendered
    assert 'tabindex="0"' in rendered and 'role="region"' in rendered
    for label in rows[0]:
        assert f'<th scope="col">{label}</th>' in rendered
    for value in [v for v in rows[0].values() if v is not None]:
        assert f'<td>{value}</td>' in rendered
    assert '<td>Unavailable</td>' in rendered


def test_table_escapes_artifact_text_and_keeps_zero_values() -> None:
    text = '<script>"secret" & test</script>'
    rendered = table_html([{text: text, "count": 0, "flag": False}])
    assert '<script>' not in rendered
    assert text in unescape(rendered)
    assert '<td>0</td>' in rendered and '<td>False</td>' in rendered


def test_table_keeps_rows_and_union_of_columns() -> None:
    rendered = table_html([{"First": "dev"}, {"First": "held_out", "Second": "10/10"}])
    assert rendered.count('<tbody>') == 1
    assert '<td>dev</td>' in rendered and '<td>held_out</td>' in rendered
    assert '<td>10/10</td>' in rendered


def test_only_long_tables_use_vertical_scroll() -> None:
    assert 'research-table-region long' not in table_html([{"Metric": str(i)} for i in range(11)])
    rendered = table_html([{"Case": str(i)} for i in range(30)])
    assert 'research-table-region long' in rendered
    assert rendered.count('<td>') == 30
