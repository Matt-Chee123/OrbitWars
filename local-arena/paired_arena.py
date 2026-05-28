import argparse
import csv
import hashlib
import json
import os
import time

from kaggle_environments import make

# Reuse the helper that already works in your arena.py
from arena import get_final_ship_counts
from notebook_util import resolve_agent_path


FIELDNAMES = [
    "timestamp", "run_id", "seed", "a_slot",
    "label_a", "hash_a", "label_b", "hash_b",
    "winner", "a_ships", "b_ships", "ship_margin",
    "steps", "elapsed", "p0_status", "p1_status",
    "error", "replay_path",
]


def file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:8]
    except OSError:
        return "????????"


def run_one_game(agent_a, agent_b, seed, a_slot, save_replay=False, replay_dir="replays"):
    agents = [agent_a, agent_b] if a_slot == 0 else [agent_b, agent_a]

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)

    start = time.time()
    try:
        result = env.run(agents)
    except Exception as e:
        return {
            "seed": seed, "a_slot": a_slot, "winner": None,
            "a_ships": 0, "b_ships": 0, "ship_margin": 0,
            "steps": 0, "elapsed": time.time() - start,
            "p0_status": None, "p1_status": None,
            "error": str(e), "replay_path": "",
        }
    elapsed = time.time() - start

    final = result[-1]
    p0_reward = final[0].get("reward", 0) or 0
    # Zero-sum, two-player: A's reward is p0's reward, negated if A sat in slot 1.
    a_reward = p0_reward if a_slot == 0 else -p0_reward

    counts = get_final_ship_counts(result)        # [slot0_ships, slot1_ships]
    a_ships = counts[a_slot]
    b_ships = counts[1 - a_slot]

    row = {
        "seed": seed,
        "a_slot": a_slot,
        "winner": "A" if a_reward > 0 else ("B" if a_reward < 0 else "TIE"),
        "a_ships": a_ships,
        "b_ships": b_ships,
        "ship_margin": a_ships - b_ships,
        "steps": len(result),
        "elapsed": round(elapsed, 2),
        "p0_status": final[0].get("status", "DONE"),
        "p1_status": final[1].get("status", "DONE"),
        "error": "",
        "replay_path": "",
    }

    if save_replay:
        os.makedirs(replay_dir, exist_ok=True)
        path = os.path.join(replay_dir, f"seed{seed}_aslot{a_slot}.json")
        try:
            data = env.toJSON()
            with open(path, "w") as f:
                json.dump(json.loads(data) if isinstance(data, str) else data, f)
            row["replay_path"] = path
        except Exception as e:
            row["replay_path"] = f"ERROR:{e}"

    return row


def append_rows(csv_path, rows):
    """Append rows to CSV, writing the header only if the file is new."""
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run_paired_match(agent_a, agent_b, seeds, csv_path,
                     label_a, label_b, hash_a, hash_b,
                     save_replays=False):
    run_id = time.strftime("%Y%m%d-%H%M%S")
    print(f"\n{'='*64}")
    print(f"  {label_a} ({hash_a}) vs {label_b} ({hash_b})")
    print(f"  {len(seeds)} seeds x 2 slots = {len(seeds) * 2} games  ->  {csv_path}")
    print(f"  run_id = {run_id}")
    print(f"{'='*64}")

    a_wins = b_wins = ties = errors = 0

    for seed in seeds:
        rows = []
        for a_slot in (0, 1):           # same map, A in each slot
            g = run_one_game(agent_a, agent_b, seed, a_slot, save_replays)
            g.update({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "run_id": run_id,
                "label_a": label_a, "hash_a": hash_a,
                "label_b": label_b, "hash_b": hash_b,
            })
            rows.append(g)

            if g["error"]:
                errors += 1
            elif g["winner"] == "A":
                a_wins += 1
            elif g["winner"] == "B":
                b_wins += 1
            else:
                ties += 1

        append_rows(csv_path, rows)     # flush each pair so a crash loses little

        m0, m1 = rows[0]["ship_margin"], rows[1]["ship_margin"]
        tag0 = rows[0]["winner"] or "ERR"
        tag1 = rows[1]["winner"] or "ERR"
        print(f"  seed {seed:>6}:  A@0 {tag0:>4} ({m0:+4d})   A@1 {tag1:>4} ({m1:+4d})")

    valid = a_wins + b_wins + ties
    print(f"\n  {label_a} wins: {a_wins}   {label_b} wins: {b_wins}   ties: {ties}   errors: {errors}")
    if valid:
        print(f"  raw A win rate (decisive): "
              f"{a_wins / max(1, a_wins + b_wins) * 100:.0f}%  "
              f"(this is the headline number — get its CI from analyze.py)")
    print(f"  Wrote {valid + errors} rows to {csv_path}\n")


def main():
    p = argparse.ArgumentParser(description="Seeded, position-paired Orbit Wars match runner")
    p.add_argument("agent_a", help="Agent A file (.py or .ipynb)")
    p.add_argument("agent_b", help="Agent B file (.py or .ipynb)")
    p.add_argument("--seeds", type=int, default=50,
                   help="Number of distinct maps; each is played twice (default 50)")
    p.add_argument("--seed-start", type=int, default=1000,
                   help="First seed value; seeds are seed-start .. seed-start+N-1 (default 1000)")
    p.add_argument("--csv", default="results/results.csv", help="Output CSV (appended to; default results.csv)")
    p.add_argument("--label-a", default=None, help="Version label for A (default: filename stem)")
    p.add_argument("--label-b", default=None, help="Version label for B (default: filename stem)")
    p.add_argument("--save-replays", action="store_true", help="Save per-game replay JSON")
    args = p.parse_args()

    # Hash the original files the user passed (the artifacts being versioned),
    # then resolve to runnable .py paths (handles .ipynb via notebook_util).
    hash_a = file_hash(args.agent_a)
    hash_b = file_hash(args.agent_b)
    path_a, name_a = resolve_agent_path(args.agent_a)
    path_b, name_b = resolve_agent_path(args.agent_b)

    label_a = args.label_a or name_a
    label_b = args.label_b or name_b

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    run_paired_match(path_a, path_b, seeds, args.csv,
                     label_a, label_b, hash_a, hash_b,
                     save_replays=args.save_replays)


if __name__ == "__main__":
    main()