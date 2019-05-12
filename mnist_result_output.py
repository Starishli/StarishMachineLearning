import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from source import TRAIN_LOGGER, CACHE_DIR, RESULT_DIR, FIG_DIR
from source.helpers import cache_write, cache_load


if __name__ == "__main__":
    plt.figure(figsize=(12, 8))

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # model_name_list = ["RawLeNet", "ImprovedLeNet_0", "ImprovedLeNet_1", "ImprovedLeNet_2", "ImprovedLeNet_3",
    #                    "ImprovedLeNet_4", "ImprovedLeNet_5", "ImprovedLeNet_6", "ImprovedLeNet_7"]

    # model_name_list = ["RawLeNet", "ImprovedLeNet_0"]
    # model_name_list = ["ImprovedLeNet_0", "ImprovedLeNet_1", "ImprovedLeNet_2", "ImprovedLeNet_3"]
    # model_name_list = ["ImprovedLeNet_0", "ImprovedLeNet_4", "ImprovedLeNet_5", "ImprovedLeNet_6"]

    # model_name_list = ["ImprovedLeNet_0", "ImprovedLeNet_1", "ImprovedLeNet_4"]
    # model_name_list = ["ImprovedLeNet_0", "ImprovedLeNet_2", "ImprovedLeNet_5"]

    model_name_list = ["ImprovedLeNet_0", "ImprovedLeNet_3", "ImprovedLeNet_6", "ImprovedLeNet_7"]


    font1 = {'size': 15,
             }

    result_df = pd.DataFrame()
    result_df["Epoch"] = list(range(1, 11, 1))

    file_name = "_".join(model_name_list)

    for model_name_ in model_name_list:
        _, test_acc = cache_load(os.path.join(CACHE_DIR, "{}_result.dat".format(model_name_)))

        test_acc = 100 - np.array(test_acc)

        plt.plot(test_acc)

        result_df[model_name_] = test_acc

    plt.xlabel("Epochs", font1)
    plt.ylabel("Error Rate", font1)
    plt.title("Test Error Rate", fontsize=15)
    plt.legend(model_name_list, fontsize=15)

    plt.savefig(os.path.join(FIG_DIR, file_name))

    result_df.to_csv(os.path.join(RESULT_DIR, file_name + ".csv"), float_format="%.3f%%", index=False)

    plt.show()
