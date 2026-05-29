from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
from owHelpers.geometry import fleet_speed
from owHelpers.solver import solve_intercept
from owHelpers.defense import detect_threats
import math

RESERVE = 1
HORIZON = None
def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
    omega = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else obs.angular_velocity

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    state = {"planets": planets, "fleets": fleets}
    threats = {}
    available = {p.id: int(p.ships) for p in my_planets}

    for t in detect_threats(state, player, horizon=HORIZON, default_omega=omega):
        need = max(1, math.ceil(t["attacking_ships"] - t["defending_ships"]))
        threats[t["planet_id"]] = {"planet": t["planet"], "eta": t["eta"], "need": need}

    if not targets:
        return moves
    for mine in my_planets:

        avail = available[mine.id]
        rescue = None
        for tid, info in sorted(threats.items(), key=lambda kv: kv[1]["eta"]):
            if info["need"] <= 0 or tid == mine.id:
                continue
            send = min(avail - RESERVE, info["need"])
            if send < 1:
                continue
            sol = solve_intercept(mine, info["planet"], omega, fleet_speed(send))
            if sol is None or sol["blocked"] or sol["eta"] >= info["eta"]:
                continue
            rescue = (info, send, sol)
            break
        if rescue:
            info, send, sol = rescue
            moves.append([mine.id, sol["angle"], send])
            available[mine.id] -= send
            info["need"] -= send
            continue

        if not targets:
            continue

        nearest = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))

        ships_needed = int(nearest.ships) + 1
        if mine.ships < ships_needed:
            continue

        sol = solve_intercept(mine, nearest, omega, fleet_speed(ships_needed))
        if sol is None or sol["blocked"]:
            continue

        moves.append([mine.id, sol["angle"], ships_needed])
        available[mine.id] -= ships_needed
        
    return moves