"""
Generate replay data from a game and launch the visual arena viewer.

Usage:
  python watch.py submission.py submission_958_raw.py
  python watch.py submission.py submission_958_raw.py --speed 2
"""

import argparse
import json
import os
import sys
import time
import webbrowser
from kaggle_environments import make


def run_and_capture(agent_a, agent_b):
    """Run a game and capture full step-by-step data for visualization."""
    env = make("orbit_wars", debug=True)
    result = env.run([agent_a, agent_b])

    # Extract configuration
    config = {
        "boardSize": 100,
        "sunX": 50,
        "sunY": 50,
        "sunRadius": 10,
        "totalSteps": 500,
        "shipSpeed": env.configuration.get("shipSpeed", 6.0) if hasattr(env.configuration, "get") else 6.0,
    }

    # Extract per-step state
    frames = []
    for step_idx, step_data in enumerate(result):
        frame = {"step": step_idx, "planets": [], "fleets": [], "scores": []}

        for player_idx in range(len(step_data)):
            agent_data = step_data[player_idx]
            obs = agent_data.get("observation")
            reward = agent_data.get("reward")
            status = agent_data.get("status", "ACTIVE")

            if obs is None:
                frame["scores"].append({"player": player_idx, "ships": 0, "planets": 0, "status": status})
                continue

            raw_planets = obs.get("planets", []) or []
            raw_fleets = obs.get("fleets", []) or []

            # Only add planets/fleets from player 0's view (same data for all)
            if player_idx == 0:
                for p in raw_planets:
                    frame["planets"].append({
                        "id": p[0], "owner": p[1],
                        "x": p[2], "y": p[3],
                        "radius": p[4], "ships": p[5],
                        "production": p[6],
                    })
                for f in raw_fleets:
                    frame["fleets"].append({
                        "id": f[0], "owner": f[1],
                        "x": f[2], "y": f[3],
                        "angle": f[4], "from_planet_id": f[5],
                        "ships": f[6],
                    })

            # Calculate score
            player_ships = 0
            player_planets = 0
            for p in raw_planets:
                if p[1] == player_idx:
                    player_ships += int(p[5])
                    player_planets += 1
            for f in raw_fleets:
                if f[1] == player_idx:
                    player_ships += int(f[6])

            frame["scores"].append({
                "player": player_idx,
                "ships": player_ships,
                "planets": player_planets,
                "status": status,
            })

        frames.append(frame)

    return {"config": config, "frames": frames, "numPlayers": len(result[0])}


def main():
    parser = argparse.ArgumentParser(
        description="Watch Orbit Wars games live",
        epilog="  python watch.py agent1.ipynb agent2.ipynb\n  python watch.py submission.py agent.ipynb --speed 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("agents", nargs=2, help="Two agent files (.py or .ipynb)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--port", type=int, default=8765, help="HTTP server port")
    args = parser.parse_args()

    # Resolve agents (supports .ipynb)
    from notebook_util import resolve_agent_path
    agent_a, name_a = resolve_agent_path(args.agents[0])
    agent_b, name_b = resolve_agent_path(args.agents[1])

    print(f"Running game: {name_a} vs {name_b}...")
    start = time.time()
    replay_data = run_and_capture(agent_a, agent_b)
    elapsed = time.time() - start
    print(f"Game completed in {elapsed:.1f}s ({len(replay_data['frames'])} frames)")

    replay_data["agentNames"] = [name_a, name_b]
    replay_data["speed"] = args.speed

    # Save replay data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    replay_path = os.path.join(base_dir, "replay_data.js")
    with open(replay_path, "w") as f:
        f.write(f"const REPLAY_DATA = {json.dumps(replay_data)};")

    # Open viewer directly via file://
    viewer_path = os.path.join(base_dir, "viewer.html")
    url = "file:///" + viewer_path.replace("\\", "/")
    print(f"Opening viewer: {url}")
    webbrowser.open(url)


if __name__ == "__main__":
    main()

