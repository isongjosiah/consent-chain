# ConsentChain

Smart contract-enforced dynamic consent and verifiable machine unlearning for federated IoT learning.

IEEE BCCA 2026 submission (deadline June 15, 2026). The authoritative project brief is in [`CLAUDE.md`](./CLAUDE.md).

## Repository layout

| Path | Contents |
|---|---|
| `contracts/` | Solidity smart contracts (Hardhat + TypeScript) |
| `fl/` | Federated learning, unlearning, and MIA pipeline (Python + Flower + PyTorch) |
| `docs/` | Authoritative design docs (`contracts-spec.md`, `mia-protocol.md`, `eval-protocol.md`, …) |
| `results/` | Experiment outputs (JSON) and generated LaTeX tables |
| `paper/` | IEEEtran source for the BCCA submission |
| `scripts/` | Repo-wide orchestration (setup, full run, asset generation) |

## Quick start

```bash
cp .env.example .env                # then fill in keys
cd contracts && npm install         # smart contract toolchain

# FL toolchain — use Python 3.12 (3.13/3.14 lack ML wheels for our pins)
cd ../fl
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'             # quote in zsh
```

See `CLAUDE.md` §10 for the day-by-day execution plan and `docs/contracts-spec.md` for the on-chain interfaces.

## Status

Day 1 (skeleton, contract spec, dataset download scripts) complete. Active day tracked in `CLAUDE.md` §10.
