from dataclasses import dataclass


@dataclass(frozen=True)
class Wall:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Obstacle:
    center_x:   float
    center_y:   float
    width:      float
    depth:      float
    max_height: float


@dataclass
class Room:
    width:      float
    depth:      float
    height:     float

    walls:      set[Wall]
    obstacles:  set[Obstacle]

