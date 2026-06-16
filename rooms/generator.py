import random
from rooms.models import Wall, Obstacle, Room


def create_outer_walls(
    room_width: float,
    room_depth: float
) -> set[Wall]:
    return {
        Wall(0, 0, room_width, 0),
        Wall(0, 0, 0, room_depth),
        Wall(room_width, 0, room_width, room_depth),
        Wall(0, room_depth, room_width, room_depth)
    }


def create_inner_walls(
    room_width: float,
    room_depth: float,
    min_length: float = 1.0
) -> set[Wall]:
    n = random.randint(0, int(min(room_width, room_depth) / 2))
    inner_walls = set()
    for _ in range(n):
        while True:
            x1 = random.uniform(0, room_width)
            y1 = random.uniform(0, room_depth)
            x2 = random.uniform(0, room_width)
            y2 = random.uniform(0, room_depth)
            length = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
            if length >= min_length:
                break

        inner_walls.add(Wall(x1, y1, x2, y2))
    return inner_walls


def create_obstacles(
    room_width:     float,
    room_depth:     float,
    room_height:    float,
    min_length:     float = 0.25,
    min_height:     float = 0.25,
) -> set[Obstacle]:
    n = random.randint(0, int(min(room_width, room_depth)))
    obstacles = set()
    for _ in range(n):
        center_x = random.uniform(0, room_width)
        center_y = random.uniform(0, room_depth)
        width = random.uniform(min_length, room_width / 4)
        depth = random.uniform(min_length, room_depth / 4)
        max_height = random.uniform(min_height, room_height)
        obstacles.add(Obstacle(center_x, center_y, width, depth, max_height))
    return obstacles


def generate_room(
    width:                  float = 8.0,
    depth:                  float = 8.0,
    height:                 float = 4.0,
    min_inner_wall_length:  float = 1,
    min_obstacle_length:    float = 0.25,
    min_obstacle_height:    float = 0.25
) -> Room:

    walls = create_outer_walls(width, depth) | create_inner_walls(width, depth, min_inner_wall_length)
    obstacles = create_obstacles(width, depth, height, min_obstacle_length, min_obstacle_height)

    return Room(
        width=width,
        depth=depth,
        height=height,
        walls=walls,
        obstacles=obstacles
    )
