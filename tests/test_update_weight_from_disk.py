from pathlib import Path

import pytest

from slime.backends.megatron_utils.update_weight.update_weight_from_disk import _mkdir_shared


def test_mkdir_shared_retries_eventual_directory_visibility():
    class EventuallyVisibleDirectory:
        mkdir_calls = 0

        def mkdir(self, *args, **kwargs):
            self.mkdir_calls += 1
            raise FileExistsError(17, "File exists", "weight_v000001")

        def is_dir(self):
            return self.mkdir_calls >= 2

    target = EventuallyVisibleDirectory()

    _mkdir_shared(target, retries=2, delay=0)

    assert target.mkdir_calls == 2


def test_mkdir_shared_raises_for_file_collision(tmp_path: Path):
    target = tmp_path / "weight_v000001"
    target.touch()

    with pytest.raises(FileExistsError):
        _mkdir_shared(target, retries=1, delay=0)
