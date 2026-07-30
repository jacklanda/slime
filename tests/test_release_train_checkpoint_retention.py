from types import SimpleNamespace

from slime.ray.actor_group import RayTrainGroup


def test_release_train_removes_only_previous_non_periodic_checkpoint(tmp_path):
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = SimpleNamespace(save=str(tmp_path), save_interval=20)

    temporary = tmp_path / "iter_0000018"
    temporary.mkdir()
    group._remove_previous_temporary_checkpoint(19)
    assert not temporary.exists()

    permanent = tmp_path / "iter_0000019"
    permanent.mkdir()
    group._remove_previous_temporary_checkpoint(20)
    assert permanent.is_dir()

    current = tmp_path / "iter_0000020"
    current.mkdir()
    group._remove_previous_temporary_checkpoint(20)
    assert current.is_dir()
