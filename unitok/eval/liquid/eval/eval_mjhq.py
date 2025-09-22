import argparse
from cleanfid import fid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str)
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()

    score = fid.compute_fid(args.src_dir, args.result_dir)
    print(score)

# pip install clean-fid
# pip install scipy==1.11.1