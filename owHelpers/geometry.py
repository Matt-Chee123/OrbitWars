import math

BOARD_SIZE = 100.0
CENTER = (BOARD_SIZE / 2.0, BOARD_SIZE / 2.0)
ROTATION_RADIUS_LIMIT = 50.0
SUN_RADIUS = 10

def fleet_speed(ships, max_speed=6.0):
    ships = max(1, int(ships))
    if ships <= 1:
        return 1.0
    pos_speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(max_speed, pos_speed)

def orbital_radius(x, y):
    return math.hypot(x - CENTER[0], y - CENTER[1])


def is_rotating(x, y, radius):
    return orbital_radius(x, y) + radius < ROTATION_RADIUS_LIMIT


def rotate_about_center(x, y, dphi):
    dx, dy = x - CENTER[0], y - CENTER[1]
    c, s = math.cos(dphi), math.sin(dphi)
    return (CENTER[0] + dx * c - dy * s,
            CENTER[1] + dx * s + dy * c)


def predict_planet_pos(x, y, radius, angular_velocity, dt):
    if is_rotating(x, y, radius):
        return rotate_about_center(x, y, angular_velocity * dt)
    return (x, y)

def point_to_segment_distance(p, v, w):
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return math.hypot(p[0] - v[0], p[1] - v[1])
    t = max(0.0, min(1.0, ((p[0]-v[0])*(w[0]-v[0]) + (p[1]-v[1])*(w[1]-v[1])) / l2))
    proj = (v[0] + t*(w[0]-v[0]), v[1] + t*(w[1]-v[1]))
    return math.hypot(p[0] - proj[0], p[1] - proj[1])


def path_crosses_sun(launch_xy, aim_xy):
    return point_to_segment_distance(CENTER, launch_xy, aim_xy) < SUN_RADIUS

