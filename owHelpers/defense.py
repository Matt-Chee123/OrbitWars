from owHelpers.geometry import (
    is_rotating, rotate_about_center, fleet_speed,
)
import math


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _xy(obj):
    return (_get(obj, "x"), _get(obj, "y"))


def _ship_count(fleet):
    n = _get(fleet, "ships", None)
    if n is not None:
        return int(n)
    try:
        return len(fleet)
    except TypeError:
        return 1


def assess_planet(planet, enemy_fleets, friendly_fleets, default_omega=0.0, max_ticks=250):
    prod = _get(planet, "production")

    incoming = []
    for f in enemy_fleets:
        eta = fleet_to_planet_time(f, planet, default_omega)
        if eta is not None:
            incoming.append((eta, _ship_count(f), "enemy"))
    for f in friendly_fleets:
        eta = fleet_to_planet_time(f, planet, default_omega)
        if eta is not None:
            incoming.append((eta, _ship_count(f), "friendly"))
    incoming.sort(key=lambda e: e[0])

    garrison = _get(planet, "ships")
    last_t = 0.0
    for eta, count, type in incoming:
        time_dif = eta - last_t
        produced = prod * time_dif
        garrison += produced
        last_t = eta

        if type == 'friendly':
            garrison += count
            continue

        if count > garrison:
            return {
                "planet_id": _get(planet, "id"),
                "planet": planet,
                "eta": eta,
                "attacking_ships": count,
                "defending_ships": round(garrison, 1)
            }
        garrison -= count
    return None

def fleet_to_planet_time(fleet, planet, default_omega=0.0, max_ticks=250, scan=0.25):
    pos, vel, _ = fleet_kinematics(fleet)
    omega = _planet_omega(planet, default_omega)
    return time_to_capture(pos, vel, planet, omega, None, max_ticks, scan)

def time_to_capture(fleet_pos, fleet_vel, planet, omega, capture_radius=None, max_ticks=250, scan=0.25):
    px0, py0 = _xy(planet)
    pr = _get(planet, 'radius')
    cap_r = pr if capture_radius is None else capture_radius
    rotating = is_rotating(px0, py0, pr)
    def gap(t):
        fx = fleet_pos[0] + fleet_vel[0] * t
        fy = fleet_pos[1] + fleet_vel[1] * t
        if rotating:
            qx, qy = rotate_about_center(px0, py0, omega * t)
        else:
            qx, qy = px0, py0
        return math.hypot(fx - qx, fy - qy) - cap_r

    prev_t, prev = 0.0, gap(0.0)
    if prev <= 0.0:
        return 0.0

    t = scan
    while t <= max_ticks:
        cur = gap(t)
        if cur <= 0.0 and prev > 0.0:
            lo, hi = prev_t, t
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if gap(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            return hi
        prev_t, prev = t, cur
        t += scan
    return None


def fleet_kinematics(fleet):
    n = _ship_count(fleet)
    pos = _xy(fleet)
    angle = _get(fleet, 'angle')
    if angle is not None:
        speed = fleet_speed(n)
        return pos, (math.cos(angle) * speed, math.sin(angle) * speed), n
    return pos, (0.0, 0.0), n

def _planet_omega(planet, default_omega):
    if callable(default_omega):
        return default_omega(planet)
    return default_omega

def detect_threatened_planets(my_planets, enemy_fleets, friendly_fleets=(), horizon=None, default_omega=0.0):
    threats = []
    for p in my_planets:
        t = assess_planet(p, enemy_fleets, friendly_fleets, default_omega)
        if t and (horizon is None or _get(t, "eta") <= horizon):
            threats.append(t)
    threats.sort(key=lambda t: t['eta'])
    return threats

def detect_threats(state, my_id, horizon=None, default_omega=0.0):
    planets = _get(state, 'planets')
    fleets = _get(state, "fleets", [])
    mine = [p for p in planets if _get(p, "owner") == my_id]
    enemy = [f for f in fleets if _get(f, "owner") != my_id]
    friendly = [f for f in fleets if _get(f, "owner") == my_id]
    return detect_threatened_planets(mine, enemy, friendly, horizon, default_omega)