# :snake: Snake game

The aim of this project is to implement a snake game using Reinforcement Learning.
It will be a revisited game of Snake, where you have random items that can give you more points.


Two game modes are available:
- The human mode where the user can play with the arrows on the numeric keypad.
- The AI mode where an AI learns by itself to play the snake game

Three types of fruits exist in the game:
- Apples worth 1 point (red fruit)
- Bananas worth 5 points (yellow fruit)
- Kiwis worth 10 points (green fruit)

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
# Usage: main.py [OPTIONS]

# Options:
#   --type [AI|HUMAN]  Type of game to compute.
#   --speed INTEGER    Snake speed
#   --help             Show this message and exit.
```

## Play

```bash
 python3 main.py --type AI
 python3 main.py --type HUMAN
 python3 main.py --type HUMAN --speed 10
```

## Demonstration

![Snake demo](https://github.com/alannagenin/reinforcement-learning-project/blob/main/demo/training.gif)

## Contributors :woman_technologist:

<a href="https://github.com/alannagenin/reinforcement-learning-project/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=alannagenin/reinforcement-learning-project" />
</a>
