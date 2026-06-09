import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def parity_plot(
    preds,
    targets,
    target_headers,
    path,
):
    fig, ax = plt.subplots()
    ax = plt.gca()
    ax.plot(targets, preds, "b.", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="Experimental " + target_headers, ylabel="Prediction " + target_headers)
    fig.savefig(path)
    plt.close()
