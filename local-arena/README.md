# Orbit Wars Local Arena

Test your Orbit Wars agents locally before uploading to Kaggle.
Supports both `.py` and `.ipynb` agent files.

## Requirements

```bash
pip install kaggle-environments
```

## Files

| File | Purpose |
|------|---------|
| `arena.py` | CLI match runner with stats |
| `watch.py` | Visual game viewer |
| `viewer.html` | Browser-based replay renderer |
| `notebook_util.py` | Auto-extracts code from .ipynb files |

## Usage

### Watch a game (visual)

```bash
# Two .py agents
python watch.py agent1.py agent2.py

# Two .ipynb notebooks
python watch.py my-agent.ipynb opponent.ipynb

# Mix formats
python watch.py submission.py opponent.ipynb
```

Opens `viewer.html` in your browser with playback controls:
- **Space** — play/pause
- **Arrow left/right** — step back/forward
- **Arrow up/down** — speed up/down
- **R** — restart

### Run stats (CLI)

```bash
# Head-to-head: 10 games
python arena.py agent1.ipynb agent2.ipynb --games 10

# Round-robin tournament
python arena.py agent1.ipynb agent2.ipynb agent3.py --games 6

# Save replay JSON files
python arena.py agent1.py agent2.py --games 5 --save-replays
```

## How to use

1. Copy your agent `.py` or `.ipynb` files into this folder
2. Run `watch.py` or `arena.py` with the filenames
3. Results appear in terminal (arena) or browser (watch)
