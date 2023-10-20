import argparse
import os
from werm import execute_werm
from adv_reg import execute_adv_reg
from mmd import execute_mmd

def run_experiments():
    parser = argparse.ArgumentParser(description="Choose an algorithm to run.")
    parser.add_argument("--algorithm", choices=["werm", "adv_reg", "mmd"], required=True, help="Algorithm to be executed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU number to run the code on.")
    parser.add_argument("--test_one_seed", action='store_true', help="Run only one seed to test the code.")    
    args = parser.parse_args()

    # Set the GPU for TensorFlow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.algorithm == "werm":
        execute_werm(args.test_one_seed)
    elif args.algorithm == "adv_reg":
        execute_adv_reg(args.test_one_seed)
    elif args.algorithm == "mmd":
        execute_mmd(args.test_one_seed)
    else:
        print("Invalid algorithm choice!")

if __name__ == "__main__":
    run_experiments()
