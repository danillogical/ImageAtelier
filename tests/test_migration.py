"""Old independent/Turbo/uncensored settings migrate to an official whole suite."""
import json
import pytest
from imageatelier.settings import SettingsStore

@pytest.mark.parametrize("saved", [
    {"turbo": True, "steps": 6, "text_encoder": "heretic-8bit", "pe_variant": "heretic"},
    {"turbo": True, "steps": 8, "variant": "8bit"},
    {"pe_t2i_variant": "heretic", "variant": "4bit"},
])
def test_old_choices_migrate(tmp_path, saved):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({**saved, "resolution": 768, "pe_thinking": False}))
    store = SettingsStore(path)
    value = store.get()
    assert value["model_suite"] == "q8" and value["turbo"] is False
    assert value["text_encoder"] == value["image_model"] == "official-8bit"
    assert value["pe_variant"] == "official" and value["steps"] == 40
    assert value["resolution"] == 768 and value["pe_thinking"] is False
    assert "text_encoder" not in json.loads(path.read_text())

@pytest.mark.parametrize("name,quant", [("bf16", "bf16"), ("q8", "8bit"), ("q6", "6bit"), ("q4", "4bit")])
def test_suite_sets_every_component(tmp_path, name, quant):
    store = SettingsStore(tmp_path / "settings.json")
    value = store.update({"model_suite": name})
    assert value["image_model"] == value["text_encoder"] == f"official-{quant}"
    assert value["pe_quant"] == quant and value["turbo"] is False
    assert SettingsStore(tmp_path / "settings.json").get() == value

@pytest.mark.parametrize("change", [{"pe_variant": "heretic"}, {"turbo": True}, {"image_model": "official-8bit"},
                                    {"text_encoder": "official-4bit"}, {"pe_quant": "6bit"}, {"model_suite": "q5"}, {"model_suite": []}])
def test_partial_choices_rejected_without_changes(tmp_path, change):
    store = SettingsStore(tmp_path / "settings.json")
    before = store.get()
    with pytest.raises(ValueError): store.update(change)
    assert store.get() == before
