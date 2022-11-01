import os
import yaml
import itertools


def main():
    params = {
        "num_epoch": [100],
        "d_hidden": [2, 3],
        "num_layers": [3],
        "batch_size": [64, 128],
        "num_filters": [16, 32, 64],
        "lr": [0.0001, 0.002],
        "kernel_size_1": [60, 70, 80],
        "kernel_size_2": [3,5,7,9],
        "stride_1": [4,5,6],
        "stride_2": [3,5,7,9],
        "maxpool_1": [4,5,6],
        "maxpool_2": [3,4],
        "weight_decay": [0.0001],
    }

    keys, values = zip(*params.items())
    combs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"GENERATING {len(combs)} NEW CONFIGS ...")

    for comb in combs:
        filename = "{}num_layers_({},{})kernel_size_{}n_filters_({},{})stride_({},{})maxpool_{}lr_{}batch_size".format(
            comb["num_layers"],
            comb["kernel_size_1"],
            comb["kernel_size_2"],
            comb["num_filters"],
            comb["stride_1"],
            comb["stride_2"],
            comb["maxpool_1"],
            comb["maxpool_2"],
            comb['lr'],
            comb['batch_size']
        ).replace(".", "_")
        config_path = os.path.join("configs/", "{}.yml".format(filename))
        config = {
            "data_dir": '/home/quanhhh/Documents/model/pickle/',
            'result_dir': '/home/quanhhh/Documents/model/results/', 
        }
        config.update(comb)
        print(filename)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

    print("DONE.")


if __name__ == "__main__":
    main()
