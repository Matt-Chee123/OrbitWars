import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
from owHelpers.geometry import fleet_speed
from owHelpers.solver import solve_intercept
from owHelpers.defense import fleet_to_planet_time, _ship_count, _get

FLEET_FIELDS = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")
RESERVE = 1
EPS = 0.1
NEUTRAL_OWNER = -1


def _as_fleet(f):
    if isinstance(f, dict) or hasattr(f, "x"):
        return f
    return dict(zip(FLEET_FIELDS, f))


def attack_plan(source, target, omega, iters=2):
    growing = target.owner != NEUTRAL_OWNER
    prod = max(0.0, target.production or 0) if growing else 0.0
    needed = int(target.ships) + 1
    sol = solve_intercept(source, target, omega, fleet_speed(needed))
    if sol is None or sol["blocked"]:
        return None
    for _ in range(iters):
        needed = int(math.ceil(target.ships + prod * sol["eta"])) + 1
        sol = solve_intercept(source, target, omega, fleet_speed(needed))
        if sol is None or sol["blocked"]:
            return None
    return needed, sol["eta"], sol


def incoming_mine(my_fleets, all_planets, target_ids, omega):
    inc = {}
    for f in my_fleets:
        dest, dest_eta = None, None
        for p in all_planets:
            eta = fleet_to_planet_time(f, p, omega)
            if eta is not None and (dest_eta is None or eta < dest_eta):
                dest, dest_eta = p, eta
        if dest is not None and dest.id in target_ids:
            inc[dest.id] = inc.get(dest.id, 0) + _ship_count(f)
    return inc


def best_target(source, targets, omega, budget, incoming):
    best = None
    for tgt in targets:
        plan = attack_plan(source, tgt, omega)
        if plan is None:
            continue
        needed, eta, _ = plan
        send = needed - incoming.get(tgt.id, 0)
        if send <= 0 or send > budget:
            continue
        score = (tgt.production + EPS) / eta
        if best is None or score > best[0]:
            best = (score, send, tgt)
    return best


def agent(obs):
    moves = []
    is_dict = isinstance(obs, dict)
    player = obs.get("player", 0) if is_dict else obs.player
    raw_planets = obs.get("planets", []) if is_dict else obs.planets
    raw_fleets = obs.get("fleets", []) if is_dict else obs.fleets
    omega = obs.get("angular_velocity", 0.0) if is_dict else obs.angular_velocity

    planets = [Planet(*p) for p in raw_planets]
    fleets = [_as_fleet(f) for f in raw_fleets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    target_ids = {p.id for p in targets}
    available = {p.id: int(p.ships) for p in my_planets}

    my_fleets = [f for f in fleets if _get(f, "owner") == player]
    incoming = incoming_mine(my_fleets, planets, target_ids, omega)

    if not targets:
        return moves

    for mine in my_planets:
        budget = available[mine.id] - RESERVE
        if budget < 1:
            continue
        pick = best_target(mine, targets, omega, budget, incoming)
        if pick is None:
            continue
        _, send, tgt = pick
        sol = solve_intercept(mine, tgt, omega, fleet_speed(send))
        if sol is None or sol["blocked"]:
            continue
        moves.append([mine.id, sol["angle"], send])
        available[mine.id] -= send
        incoming[tgt.id] = incoming.get(tgt.id, 0) + send

    return moves