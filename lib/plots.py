import matplotlib.pyplot as plt


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
