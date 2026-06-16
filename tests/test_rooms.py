from rooms import generate_room
import random

# Test with defaults
for _ in range(3):
    room = generate_room()
    print(f"Room {room.width:.1f}x{room.depth:.1f}x{room.height:.1f}, "
          f"{len(room.walls)} walls, {len(room.obstacles)} obstacles")

# Test with random dimensions
for _ in range(3):
    room = generate_room(
        width=random.uniform(4.0, 15.0),
        depth=random.uniform(4.0, 15.0),
        height=random.uniform(2.5, 6.0),
    )
    print(f"Room {room.width:.1f}x{room.depth:.1f}x{room.height:.1f}, "
          f"{len(room.walls)} walls, {len(room.obstacles)} obstacles")
