import math
from owHelpers.geometry import (
    is_rotating, rotate_about_center, path_crosses_sun,
)


def _target_pos(target, rotating, omega, t):
    if rotating:
        return rotate_about_center(target.x, target.y, omega * t)
    return (target.x, target.y)


def _roots(L, target, rotating, omega, speed, max_ticks, scan):
    def h(t):
        ax, ay = _target_pos(target, rotating, omega, t)
        return math.hypot(ax - L[0], ay - L[1]) - speed * t

    prev_t, prev = 0.0, h(0.0)
    t = scan
    while t <= max_ticks:
        cur = h(t)
        if cur <= 0.0 and prev > 0.0:
            lo, hi = prev_t, t
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if h(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            yield hi, _target_pos(target, rotating, omega, hi)
        prev_t, prev = t, cur
        t += scan


def solve_intercept(source, target, angular_velocity, speed,
                    max_ticks=250, scan=0.25, avoid_sun=True):
    rotating = is_rotating(target.x, target.y, target.radius)
    src_c = (source.x, source.y)

    L = src_c
    for _ in range(2):
        first = next(_roots(L, target, rotating, angular_velocity, speed,
                            max_ticks, scan), None)
        if first is None:
            return None
        _, aim = first
        angle = math.atan2(aim[1] - L[1], aim[0] - L[0])
        L = (src_c[0] + math.cos(angle) * (source.radius + 0.1),
             src_c[1] + math.sin(angle) * (source.radius + 0.1))


    best = None
    for eta, aim in _roots(L, target, rotating, angular_velocity, speed,
                           max_ticks, scan):
        angle = math.atan2(aim[1] - L[1], aim[0] - L[0])
        blocked = path_crosses_sun(L, aim)
        result = {"angle": angle, "eta": eta, "aim": aim, "launch": L,
                  "blocked": blocked}
        if best is None:
            best = result
        if not blocked or not avoid_sun:
            return result
    return best