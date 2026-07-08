import random
import pytest

from rooms import generate_room


def test_generate_room_default_dimensions():
    room = generate_room()

    assert room.width > 0
    assert room.depth > 0
    assert room.height > 0
    assert len(room.walls) > 0
    assert len(room.obstacles) >= 0


@pytest.mark.parametrize("trial", range(3))
def test_generate_room_default_does_not_error(trial):
    room = generate_room()

    assert room.width > 0
    assert room.depth > 0
    assert room.height > 0
    assert len(room.walls) > 0


@pytest.mark.parametrize("trial", range(3))
def test_generate_room_with_explicit_dimensions(trial):
    width = random.uniform(4.0, 15.0)
    depth = random.uniform(4.0, 15.0)
    height = random.uniform(2.5, 6.0)

    room = generate_room(width=width, depth=depth, height=height)

    assert room.width == pytest.approx(width)
    assert room.depth == pytest.approx(depth)
    assert room.height == pytest.approx(height)
    assert len(room.walls) > 0
    assert len(room.obstacles) >= 0
