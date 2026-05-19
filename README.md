# pokepy

Pure-Python Pokemon battle environment for Generation 9 singles, following
Pokemon Showdown mechanics. Designed for fast, deterministic, scriptable
battles — no game client, no network, no external runtime dependencies
beyond NumPy.

Useful for:

- Training and evaluating Pokemon battle agents in pure Python
- Reproducing Showdown damage / status / item / ability interactions
- Running self-play or scripted matchups without spinning up Showdown

## Install

```bash
pip install git+https://github.com/sethkarten/pokepy-engine.git
```

Or for local development:

```bash
git clone https://github.com/sethkarten/pokepy-engine.git
cd pokepy-engine
pip install -e ".[dev]"
```

Pokemon Showdown data tables ship bundled under `pokepy/data/extracted/`.
Override with the `POKEPY_DATA_PATH` environment variable if you want to
point at a different copy.

## Quickstart

```python
from pokepy.env.battle_env import init_battle_state
from pokepy.data.loader import load_game_data, load_id_mappings
from pokepy.engine.battle_gen9 import step_battle_gen9
from pokepy.utils.gen5_prng import Gen5PRNG

gd = load_game_data()
mappings = load_id_mappings()
# Build two teams (see tests/conftest.py for the team-dict shape), then:
# state = init_battle_state(team0, team1, gd, seed=12345)
# state, ... = step_battle_gen9(state, action0, action1, gd, move_effects, type_chart, prng)
```

See `tests/` for end-to-end examples.

## Tests

```bash
pytest tests
```

## License

MIT — see [LICENSE](LICENSE).
