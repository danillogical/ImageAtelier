"""Independent part operations are retired: only whole suites are supported."""
import pytest

@pytest.mark.parametrize("kind,name", [("image", "official-8bit"), ("text_encoder", "official-8bit"), ("pe", "t2i-8bit")])
def test_independent_part_operations_retired(client, kind, name):
    url = f"/api/models/{kind}/{name}"
    assert client.post(url + "/download").status_code == 410
    assert client.delete(url).status_code == 410
