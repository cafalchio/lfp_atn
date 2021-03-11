from time import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np


def plot_small_sq(x, y, wview=None):
    ax = plt.figure(figsize=(3, 3))
    ax = plt.axis("off")
    ax = plt.scatter(x, y, c="black", marker=".")
    ax = plt.xlim(40, 300)  # xmax=int(wview['window_max_x']))
    ax = plt.ylim(40, 300)  # ymax=int(wview['window_max_y']))
    #     ax = plt.plot(x,y, c= 'g')
    ax = plt.xlabel("X pixels")
    ax = plt.ylabel("Y pixels")
    #     ax = plt.title('T maze postition plot')
    plt.tight_layout()
    return plt.show()


def plot_tmaze(x, y, wview, dot=True):
    ax = plt.figure(figsize=(6, 6))
    #     ax = plt.axis('off')
    if dot:
        ax = plt.scatter(x, y, c="black", marker=".")
    else:
        ax = plt.plot(x, y, c="g", linewidth=3)
    ax = plt.xlim(0, xmax=int(wview["window_max_x"]))
    ax = plt.ylim(0, ymax=int(wview["window_max_y"]))
    ax = plt.xlabel("X pixels")
    ax = plt.ylabel("Y pixels")
    ax = plt.title("T maze postition plot")
    plt.tight_layout()
    return plt.show()


def plot_mne(raw_array, base_name):
    raw_array.load_data()
    raw_array.plot(
        n_channels=2,
        block=True,
        duration=30,
        show=True,
        clipping="transparent",
        title="Raw LFP Data from {}".format(base_name),
        remove_dc=False,
        scalings=dict(eeg=250e-5),
    )


def plot_pos_over_time(x, y, rate=2, save=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    xdata = []
    ydata = []
    (scatter,) = ax.plot([], [], "ko")
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        """TODO incorporate win size."""
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 600)
        time_text.set_text("")
        return scatter, time_text

    def update(frame):
        scatter.set_data(x[0:frame], y[0:frame])
        frame_time = frame / 50
        time_text.set_text(f"time = {frame_time}")
        return scatter, time_text

    num_samples = int(len(x) // rate)
    interval = int(20 // rate)

    frames = np.linspace(0, len(x), num=num_samples, dtype=np.uint32)

    ani = FuncAnimation(
        fig, update, frames=frames, interval=interval, init_func=init)

    if save:
        Writer = writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("animated.mp4", writer)

    else:
        plt.show()


if __name__ == '__main__':
    from .data_pos import RecPos
    main_fname = r"D:\SubRet_recordings_imaging\CSR6\+ maze\27032018_t3\S8\27032018_CSR6_+maze_t3_.set"
    rc = RecPos(main_fname)
    x, y = rc.get_position()

    plot_pos_over_time(x, y, rate=10, save=False)