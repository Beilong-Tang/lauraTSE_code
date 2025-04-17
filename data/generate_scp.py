# Path

import argparse



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--folders', nargs="+")
    p.add_argument('--output_path', type='str')
    p.add_argument('')
    return p.parse_args()

def main(args):
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass