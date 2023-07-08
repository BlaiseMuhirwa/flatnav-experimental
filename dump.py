import h5py
import numpy as np
import argparse


def main(args):
    filename = args.filename
    h5_file = h5py.File(filename, "r")

    train_data = h5_file.get("train")[()]
    test_data = h5_file.get("test")[()]
    ground_truth = h5_file.get("neighbors")[()]

    print(f"train_data_shape: {train_data.shape}, train_data_type: {train_data.dtype}")
    print(f"test_data_shape: {test_data.shape}, test_data_type: {test_data.dtype}")
    print(
        f"ground_truth_data_shape: {ground_truth.shape}, ground_truth_data_type: {ground_truth.dtype}"
    )

    if args.normalize:
        print("normalize")
        train_data /= np.linalg.norm(train_data, axis=1, keepdims=True) + 1e-30
        test_data /= np.linalg.norm(test_data, axis=1, keepdims=True) + 1e-30

    fname = filename.replace(".hdf5", "")
    np.save(fname + ".train", train_data.astype(np.float32))
    np.save(fname + ".test", test_data.astype(np.float32))
    np.save(fname + ".gtruth", ground_truth.astype(np.uint32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filename", required=True, help="Name of the file to preprocess"
    )
    parser.add_argument(
        "-n", "--normalize", default=False, help="Normalize if argument set"
    )

    args = parser.parse_args()

    main(args)
