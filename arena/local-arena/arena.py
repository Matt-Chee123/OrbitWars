"""
Orbit Wars Local Arena — Test agents before uploading to Kaggle.

Features:
  - Head-to-head matchups between any two agents
  - Round-robin tournaments
  - Win/loss/tie stats with ship-count margins
  - Game replays saved as JSON
  - Alternates starting positions for fairness
  - Accepts both .py and .ipynb files directly

Usage:
  python arena.py submission.py submission_958_raw.py --games 20
  python arena.py orbit-wars-agent-v11.ipynb orbit-wars-agent-958.1.ipynb --games 10
  python arena.py --tournament agent1.ipynb agent2.py agent3.ipynb --games 6
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from kaggle_environments import make


def get_final_ship_counts(result):
    """Extract final ship counts from game result."""
    final = result[-1]
    counts = []
    for i, agent_result in enumerate(final):
        obs = agent_result.get("observation")
        if obs is None:
            counts.append(0)
            continue

        ships = 0
        player = i
        raw_planets = obs.get("planets", []) or []
        raw_fleets = obs.get("fleets", []) or []

        for p in raw_planets:
            if p[1] == player:  # owner == player
                ships += int(p[5])  # ships field
        for f in raw_fleets:
            if f[1] == player:  # owner == player
                ships += int(f[6])  # ships field

        counts.append(ships)
    return counts


def run_game(agent_a, agent_b, game_id, save_replay=False, replay_dir="replays"):
    """Run a single game and return results."""
    env = make("orbit_wars", debug=False)

    # Alternate who goes first
    if game_id % 2 == 0:
        agents = [agent_a, agent_b]
        a_player = 0
    else:
        agents = [agent_b, agent_a]
        a_player = 1

    start = time.time()
    try:
        result = env.run(agents)
    except Exception as e:
        return {
            "game_id": game_id,
            "error": str(e),
            "a_player": a_player,
            "winner": None,
            "steps": 0,
            "elapsed": time.time() - start,
        }
    elapsed = time.time() - start

    final = result[-1]
    p0_reward = final[0].get("reward", 0) or 0
    p0_status = final[0].get("status", "DONE")
    p1_status = final[1].get("status", "DONE")
    steps = len(result)

    # Determine winner from A's perspective
    if a_player == 0:
        a_reward = p0_reward
    else:
        a_reward = -p0_reward

    # Get ship counts for margin analysis
    ship_counts = get_final_ship_counts(result)

    if a_player == 0:
        a_ships, b_ships = ship_counts[0], ship_counts[1]
    else:
        a_ships, b_ships = ship_counts[1], ship_counts[0]

    game_result = {
        "game_id": game_id,
        "a_player": a_player,
        "winner": "A" if a_reward > 0 else ("B" if a_reward < 0 else "TIE"),
        "a_reward": a_reward,
        "a_ships": a_ships,
        "b_ships": b_ships,
        "ship_margin": a_ships - b_ships,
        "steps": steps,
        "elapsed": elapsed,
        "p0_status": p0_status,
        "p1_status": p1_status,
        "error": None,
    }

    if save_replay:
        os.makedirs(replay_dir, exist_ok=True)
        replay_path = os.path.join(replay_dir, f"game_{game_id:03d}.json")
        try:
            replay_data = env.toJSON()
            with open(replay_path, "w") as f:
                json.dump(json.loads(replay_data) if isinstance(replay_data, str) else replay_data, f)
            game_result["replay_path"] = replay_path
        except Exception as e:
            game_result["replay_error"] = str(e)

    return game_result


def run_match(agent_a, agent_b, num_games, save_replays=False, verbose=True, name_a=None, name_b=None):
    """Run a full match between two agents."""
    if name_a is None:
        name_a = os.path.basename(agent_a).replace(".py", "")
    if name_b is None:
        name_b = os.path.basename(agent_b).replace(".py", "")

    if verbose:
        print(f"\n{'='*65}")
        print(f"  MATCH: {name_a} vs {name_b}  ({num_games} games)")
        print(f"{'='*65}")

    results = []
    a_wins = 0
    b_wins = 0
    ties = 0
    errors = 0
    total_a_ships = 0
    total_b_ships = 0

    for i in range(num_games):
        game = run_game(agent_a, agent_b, i, save_replays)
        results.append(game)

        if game["error"]:
            errors += 1
            if verbose:
                print(f"  Game {i+1:2d}: ERROR - {game['error'][:60]}")
            continue

        if game["winner"] == "A":
            a_wins += 1
            tag = f"{name_a} WIN"
        elif game["winner"] == "B":
            b_wins += 1
            tag = f"{name_b} WIN"
        else:
            ties += 1
            tag = "TIE"

        total_a_ships += game["a_ships"]
        total_b_ships += game["b_ships"]

        if verbose:
            margin = f"({game['a_ships']:3d} vs {game['b_ships']:3d})"
            pos = f"[{name_a} as P{game['a_player']}]"
            print(f"  Game {i+1:2d}: {tag:20s} {margin}  {game['steps']:3d} steps  {game['elapsed']:.1f}s  {pos}")

    valid = num_games - errors
    if verbose:
        print(f"\n{'-'*65}")
        print(f"  RESULTS ({valid} valid games):")
        print(f"    {name_a:20s}  Wins: {a_wins:3d}  ({a_wins/max(1,valid)*100:.0f}%)")
        print(f"    {name_b:20s}  Wins: {b_wins:3d}  ({b_wins/max(1,valid)*100:.0f}%)")
        print(f"    {'Ties':20s}       {ties:3d}  ({ties/max(1,valid)*100:.0f}%)")
        if valid > 0:
            avg_a = total_a_ships / valid
            avg_b = total_b_ships / valid
            print(f"\n    Avg ships at end:  {name_a}={avg_a:.0f}  {name_b}={avg_b:.0f}  (margin: {avg_a-avg_b:+.0f})")
        if errors > 0:
            print(f"    Errors: {errors}")
        print(f"{'='*65}\n")

    return {
        "agent_a": name_a,
        "agent_b": name_b,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "errors": errors,
        "results": results,
    }


def run_tournament(agent_paths, games_per_match, save_replays=False):
    """Round-robin tournament between multiple agents."""
    n = len(agent_paths)
    names = [os.path.basename(p).replace(".py", "") for p in agent_paths]

    print(f"\n{'#'*65}")
    print(f"  TOURNAMENT: {n} agents, {games_per_match} games per match")
    print(f"  Agents: {', '.join(names)}")
    print(f"{'#'*65}")

    wins = defaultdict(int)
    losses = defaultdict(int)
    played = defaultdict(int)
    match_results = []

    for i in range(n):
        for j in range(i + 1, n):
            result = run_match(agent_paths[i], agent_paths[j], games_per_match, save_replays)
            match_results.append(result)

            wins[names[i]] += result["a_wins"]
            losses[names[i]] += result["b_wins"]
            played[names[i]] += games_per_match - result["errors"]

            wins[names[j]] += result["b_wins"]
            losses[names[j]] += result["a_wins"]
            played[names[j]] += games_per_match - result["errors"]

    # Print standings
    print(f"\n{'='*65}")
    print(f"  TOURNAMENT STANDINGS")
    print(f"{'-'*65}")
    print(f"  {'Agent':25s} {'W':>4s} {'L':>4s} {'P':>4s} {'Win%':>6s}")
    print(f"{'-'*65}")

    standings = sorted(names, key=lambda n: wins[n] / max(1, played[n]), reverse=True)
    for name in standings:
        w = wins[name]
        l = losses[name]
        p = played[name]
        rate = w / max(1, p) * 100
        print(f"  {name:25s} {w:4d} {l:4d} {p:4d} {rate:5.1f}%")

    print(f"{'='*65}\n")
    return match_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Orbit Wars Local Arena",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python arena.py submission.py submission_958_raw.py --games 10
  python arena.py submission.py submission_958_raw.py --games 5 --save-replays
  python arena.py --tournament submission.py submission_958_raw.py agent3.py --games 4
        """,
    )
    parser.add_argument("agents", nargs="+", help="Agent Python files to test")
    parser.add_argument("--games", type=int, default=10, help="Games per match (default: 10)")
    parser.add_argument("--save-replays", action="store_true", help="Save game replays as JSON")
    parser.add_argument("--tournament", action="store_true", help="Run round-robin tournament")

    args = parser.parse_args()

    # Resolve agents — supports both .py and .ipynb
    from notebook_util import resolve_agent_path
    resolved = []
    names = []
    for a in args.agents:
        path, name = resolve_agent_path(a)
        resolved.append(path)
        names.append(name)

    if args.tournament or len(resolved) > 2:
        run_tournament(resolved, args.games, args.save_replays)
    elif len(resolved) == 2:
        run_match(resolved[0], resolved[1], args.games, args.save_replays,
                  name_a=names[0], name_b=names[1])
    else:
        print("ERROR: Need at least 2 agents for a match")
        sys.exit(1)

