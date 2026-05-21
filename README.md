# greyhack-utils

A collection of research tools for [Grey Hack](https://store.steampowered.com/app/605230/Grey_Hack/) that exploit the game's deterministic RNG to predict network state before you ever touch a target.

Grey Hack uses `System.Random` (seeded deterministically from world seed + IP) throughout its world-generation code. These tools reverse-engineer those chains to let you derive router credentials, network topology, and physical map positions from information that's observable in the game world.

---

## Tools

### `router_predict.py`

Given a world seed and an IP address, predicts everything the game deterministically generates for that router:

- BSSID and ESSID (WiFi name)
- Router password
- LAN subnet base
- Domain name and TLD
- Handoff RNG state for downstream tools

```
python router_predict.py --seed -1285005987 --ip 99.71.91.182
python router_predict.py --seed -1285005987 --ip 99.71.91.182 --wordlist-dir ./python_tools/wordlists/
```

---

### `ip_predict.py`

The inverse of `router_predict`. Given a world seed and a BSSID you've observed in-game (e.g. from a WiFi scan), brute-forces the full 2³² IP space on the GPU to recover the router's public IP. Usually produces 1 candidate, can produce two because of a very specific quirk of `System.Random`. *Hint, it has to do with `abs()`*

```
python ip_predict.py --seed -1285005987 --bssid AA:BB:CC:DD:EE:FF
python ip_predict.py --seed -1285005987 --bssid AA:BB:CC:DD:EE:FF --domain www.somehost.net
```



Requires a CUDA-capable GPU.

---

### `python_tools/predict_router_position.py`

Given the exact coordinates of RouterA and the signal power percentage it reports for RouterB, brute-forces all ~4.3 billion int32 seeds on the GPU to find every candidate position for RouterB on the map.

The game places routers using `new System.Random(Guid.NewGuid().GetHashCode())`, so positions are not tied to the world seed — this tool uses the power reading as a constraint instead.

```
python python_tools/predict_router_position.py <router_a_x> <router_a_y> <power>
```

Requires a CUDA-capable GPU and `serialCeldas.json` (map cell layout extracted from `MapConfig.xml`).

---

## Dependencies

```
pip install numpy numba sqlalchemy
```

A CUDA-capable GPU is required for `ip_predict.py` and `predict_router_position.py`. CPU fallback is not implemented.

---

## Notes

All RNG logic is implemented in `python_tools/appleseed.py`, which provides a Python/Numba reimplementation of `System.Random` (both the CPU `DotNetRandom` class and CUDA device functions `_dn_init` / `_dn_sample`). The word generation for router passwords and ESSIDs uses a Markov chain over the game's wordlists; supply the wordlist directory with `--wordlist-dir` to enable password prediction.

These tools target a specific game version's world-generation logic. If the game updates its RNG usage significantly, predictions may drift.
