import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datapath", type=str, default="./data/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--window_length", type=int, default=10)
    parser.add_argument("--cost", type=float, default=0.0025)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--episode_length", type=int, default=10)

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()