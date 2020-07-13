import argparse
import time




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-files", type=str, nargs="+", help="Paths to dataset files")

    args = parser.parse_args()

    arg_dict = vars(args)

    start_time = time.time()



    end_time = time.time()
    elapsed = end_time - start_time

    print("Task completed in " + str(elapsed) + " seconds")


if __name__ == "__main__":
    main()