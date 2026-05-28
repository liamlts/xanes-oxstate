import json
from pathlib import Path

from xanes_oxstate.data.clean import clean_element, CleaningReport


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records))


def _rec(mp_id, element, ox, formula, n=100):
    return {
        "mp_id": mp_id,
        "element": element,
        "ox_state": ox,
        "formula": formula,
        "energies": list(range(n)),
        "intensities": [0.1 * i for i in range(n)],
    }


def test_clean_drops_short_spectra(tmp_path):
    src = tmp_path / "Mn.jsonl"
    _write_jsonl(src, [
        _rec("mp-1", "Mn", 2, "MnO", n=49),
        _rec("mp-2", "Mn", 2, "MnO", n=100),
    ])
    out, report = clean_element(src, tmp_path / "out.jsonl", min_class_count=1, min_classes=1)
    kept = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["mp_id"] for r in kept] == ["mp-2"]
    assert report.dropped_short == 1


def test_clean_dedups_by_formula_ox(tmp_path):
    src = tmp_path / "Mn.jsonl"
    _write_jsonl(src, [
        _rec("mp-1", "Mn", 2, "MnO"),
        _rec("mp-2", "Mn", 2, "MnO"),
        _rec("mp-3", "Mn", 3, "Mn2O3"),
    ])
    out, report = clean_element(src, tmp_path / "out.jsonl", min_class_count=1, min_classes=1)
    kept = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(kept) == 2
    assert report.dropped_dup == 1


def test_clean_drops_small_classes(tmp_path):
    src = tmp_path / "Mn.jsonl"
    recs = (
        [_rec(f"mp-{i}", "Mn", 2, f"MnO_{i}") for i in range(60)]
        + [_rec(f"mp-x{i}", "Mn", 7, f"KMnO4_{i}") for i in range(5)]
    )
    _write_jsonl(src, recs)
    out, report = clean_element(
        src, tmp_path / "out.jsonl", min_class_count=50, min_classes=1
    )
    kept = [json.loads(l) for l in out.read_text().splitlines()]
    assert {r["ox_state"] for r in kept} == {2}
    assert report.dropped_rare_class == 5


def test_clean_writes_report(tmp_path):
    src = tmp_path / "Mn.jsonl"
    _write_jsonl(src, [_rec(f"mp-{i}", "Mn", 2, f"MnO_{i}") for i in range(60)])
    out, report = clean_element(
        src, tmp_path / "out.jsonl",
        min_class_count=50, min_classes=1, report_path=tmp_path / "rep.json",
    )
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["element"] == "Mn"
    assert rep["kept"] == 60
