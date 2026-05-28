import json
from unittest.mock import MagicMock, patch

from xanes_oxstate.data.fetch import fetch_element_spectra


def _mock_doc(mp_id, element, ox, formula, energies, intensities):
    doc = MagicMock()
    doc.material_id = mp_id
    doc.formula_pretty = formula
    doc.absorbing_element = element
    doc.spectrum = MagicMock(
        x=list(energies), y=list(intensities), absorbing_index=0
    )
    doc.structure = MagicMock()
    doc.structure.species_and_occu = [{element: 1.0}]
    doc.structure.composition.oxi_state_guesses = MagicMock(
        return_value=[{element: ox}]
    )
    return doc


def test_fetch_writes_jsonl_cache(tmp_path):
    fake_docs = [
        _mock_doc("mp-1", "Mn", 2, "MnO",
                  [6520, 6540, 6560], [0.1, 0.5, 0.9]),
        _mock_doc("mp-2", "Mn", 4, "MnO2",
                  [6520, 6540, 6560], [0.0, 0.4, 1.0]),
    ]
    with patch("xanes_oxstate.data.fetch._mp_client") as mk:
        client = mk.return_value.__enter__.return_value
        client.materials.xas.search.return_value = fake_docs

        out = fetch_element_spectra("Mn", cache_dir=tmp_path, edge="K")

    assert out.exists()
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["element"] == "Mn"
    assert rec["mp_id"] == "mp-1"
    assert rec["ox_state"] == 2
    assert rec["formula"] == "MnO"
    assert rec["energies"] == [6520, 6540, 6560]


def test_fetch_is_resumable(tmp_path):
    cache = tmp_path / "Mn.jsonl"
    cache.write_text(json.dumps({"mp_id": "mp-cached"}) + "\n")

    with patch("xanes_oxstate.data.fetch._mp_client") as mk:
        out = fetch_element_spectra(
            "Mn", cache_dir=tmp_path, edge="K", skip_if_exists=True
        )
        mk.assert_not_called()

    assert out == cache
    assert "mp-cached" in cache.read_text()
