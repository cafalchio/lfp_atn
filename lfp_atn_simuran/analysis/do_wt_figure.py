PALETTE = "dark"


def set_p():
    """Set the seaborn palette."""
    import seaborn as sns

    sns.set_style("ticks")
    sns.set_palette(PALETTE)
    sns.set_context(
        "paper",
        font_scale=1.4,
        rc={"lines.linewidth": 2.0},
    )


def do_fig(info, extra_info):
    data, fnames = info
    out_dir, name = extra_info
    plot_all_lfp(data, out_dir, name)


def do_spectrum(info, extra_info):
    data, fnames = info
    out_dir, name = extra_info
    plot_all_spectrum(info, out_dir, name)


def plot_all_spectrum(info, out_dir, name):
    import os

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from neurochat.nc_utils import smooth_1d

    here = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(here, "..", ".."))
    os.makedirs(out_dir, exist_ok=True)

    set_p()

    parsed_info = []
    data, fnames = info
    for item_list, fname_list in zip(data, fnames):
        for item_dict, fname in zip(item_list, fname_list):
            start_bit = fname[len(base_dir) + 1]
            if start_bit.lower() == "c":
                data_set = "Control"
            else:
                data_set = "Lesion"
            for r in ["sub", "rsc"]:
                freqs = item_dict[r + " welch"][0].astype(float)
                powers = smooth_1d(
                    item_dict[r + " welch"][1].astype(float),
                    kernel_type="hg",
                    kernel_size=5,
                )
                for f, p in zip(freqs, powers):
                    parsed_info.append((f, p, data_set, r))

    data = np.array(parsed_info)
    df = pd.DataFrame(data, columns=["frequency", "power", "Group", "region"])
    df.replace("Control", "Control (ATN,   N = 6)", inplace=True)
    df.replace("Lesion", "Lesion  (ATNx, N = 5)", inplace=True)
    df[["frequency", "power"]] = df[["frequency", "power"]].apply(pd.to_numeric)

    for ci, oname in zip([95, None], ["_ci", ""]):
        sns.set_style("ticks")
        sns.set_palette("colorblind")
        sns.lineplot(
            data=df[df["region"] == "sub"],
            x="frequency",
            y="power",
            style="Group",
            hue="Group",
            ci=ci,
            estimator="mean",
        )
        sns.despine(offset=0, trim=True)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (uV^2 / Hz)")

        print("Saving plots to {}".format(out_dir))

        os.makedirs(os.path.join(out_dir, "summary"), exist_ok=True)
        plt.savefig(
            os.path.join(out_dir, "summary", name + "--sub--power{}.pdf".format(oname)),
            dpi=400,
        )

        plt.close("all")

        sns.lineplot(
            data=df[df["region"] == "rsc"],
            x="frequency",
            y="power",
            style="Group",
            hue="Group",
            ci=ci,
            estimator="mean",
        )
        sns.despine(offset=0, trim=True)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (uV^2 / Hz)")

        print("Saving plots to {}".format(out_dir))

        plt.savefig(
            os.path.join(out_dir, "summary", name + "--rsc--power{}.pdf".format(oname)),
            dpi=400,
        )

        plt.close("all")


def plot_all_lfp(info, out_dir, name):
    import os

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from neurochat.nc_utils import smooth_1d

    os.makedirs(out_dir, exist_ok=True)

    sns.set_style("ticks")
    sns.set_palette("colorblind")

    parsed_info = []
    control_data = []
    lesion_data = []
    for item in info:
        for val in item:
            this_item = list(val.values())[0]
            to_use = this_item
            this_item[1][-10:] = this_item[1][-20:-10]
            to_use[1] = smooth_1d(
                this_item[1].astype(float), kernel_type="hg", kernel_size=5
            )
            if this_item[2][0] == "Control":
                control_data.append(to_use[1])
            else:
                lesion_data.append(to_use[1])
            x_data = to_use[0].astype(float)
            parsed_info.append(to_use)

    lesion_arr = np.array(lesion_data).astype(float)
    control_arr = np.array(control_data).astype(float)

    y_lesion = np.average(lesion_arr, axis=0)
    y_control = np.average(control_arr, axis=0)

    difference = y_control - y_lesion

    data = np.concatenate(parsed_info[:-1], axis=1)
    df = pd.DataFrame(data.transpose(), columns=["frequency", "coherence", "Group"])
    df.replace("Control", "Control (ATN,   N = 6)", inplace=True)
    df.replace("Lesion", "Lesion  (ATNx, N = 5)", inplace=True)
    df[["frequency", "coherence"]] = df[["frequency", "coherence"]].apply(pd.to_numeric)

    sns.lineplot(
        data=df, x="frequency", y="coherence", style="Group", hue="Group", ci=95
    )

    sns.despine()

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")

    print("Saving plots to {}".format(out_dir))
    os.makedirs(os.path.join(out_dir, "summary"), exist_ok=True)
    plt.savefig(os.path.join(out_dir, "summary", name + "--coherence_ci.png"), dpi=400)
    plt.close("all")

    sns.lineplot(
        data=df, x="frequency", y="coherence", style="Group", hue="Group", ci=None
    )

    sns.despine()

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")

    print("Saving plots to {}".format(out_dir))
    plt.savefig(os.path.join(out_dir, "summary", name + "--coherence.png"), dpi=400)

    plt.ylim(0, 1)

    plt.savefig(
        os.path.join(out_dir, "summary", name + "--coherence_full.png"), dpi=400
    )
    plt.close("all")

    sns.set_style("ticks")
    sns.set_palette("colorblind")

    sns.lineplot(x=x_data, y=difference)

    sns.despine()

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Difference")

    print("Saving to {}".format((os.path.join(out_dir, name + "--difference.pdf"))))
    plt.savefig(os.path.join(out_dir, "summary", name + "--difference.pdf"), dpi=400)
    plt.close("all")
