# :snake: Snake game

The aim of this project is to implement a snake game using Reinforcement Learning.
It will be a revisited game of Snake, where you have random items that can give you more points or make you invincible for a determined period of time.

## Quick start

```bash
git clone https://github.com/alannagenin/reinforcement-learning-project
cd reinforcement-learning-project
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

If you encounter issues with `cuda` please refer yourself to this [issue](https://github.com/pytorch/pytorch/issues/30664).

```bash
raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled
```

```bash
 python3 main.py --help
#  Usage: main.py [OPTIONS]

# Options:
#   --type [AI|HUMAN]  Type of game to compute.
#   --help             Show this message and exit.
```

## Contributors

- Alanna DEVLIN-GENIN
- Clément DIGOIN
- Camille LE POTIER
- Hélène MOGENET
