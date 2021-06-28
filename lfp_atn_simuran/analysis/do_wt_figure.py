def do_fig(info, extra_info):
    data, fnames = info
    out_dir, name = extra_info
    plot_all_lfp(data, out_dir, name)


def do_spectrum(info, extra_info):
    out_dir, name = extra_info
    plot_all_spectrum(info, out_dir, name)


def plot_all_spectrum(info, out_dir, name):
    import os

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import simuran

    from neurochat.nc_utils import smooth_1d
    from lfp_atn_simuran.analysis.parse_cfg import parse_cfg_info
    from skm_pyutils.py_plot import UnicodeGrabber

    cfg = parse_cfg_info()

    scale = cfg["psd_scale"]

    os.makedirs(out_dir, exist_ok=True)

    simuran.set_plot_style()

    # Control args
    smooth_power = False
    fmax = 40

    parsed_info = []
    data, fnames = info
    n_ctrl_animals = 0
    n_lesion_animals = 0
    for item_list, fname_list in zip(data, fnames):
        r_ctrl = 0
        r_les = 0
        for item_dict, _ in zip(item_list, fname_list):
            item_dict = item_dict["powers"]
            data_set = item_dict["SUB" + " welch"][2][0]
            if data_set == "Control":
                r_ctrl += 1
            else:
                r_les += 1

            for r in ["SUB", "RSC"]:
                id_ = item_dict[r + " welch"]
                powers = id_[1]
                if smooth_power:
                    powers = smooth_1d(
                        id_[1].astype(float),
                        kernel_type="hg",
                        kernel_size=5,
                    )
                for v1, v2, v3, v4 in zip(id_[0], powers, id_[2], id_[3]):
                    if float(v1) < fmax:
                        parsed_info.append([v1, v2, v3, v4])

        n_ctrl_animals += r_ctrl / len(fname_list)
        n_lesion_animals += r_les / len(fname_list)
    print(f"{n_ctrl_animals} CTRL animals, {n_lesion_animals} Lesion animals")

    data = np.array(parsed_info)
    df = pd.DataFrame(data, columns=["frequency", "power", "Group", "region"])
    df.replace("Control", "Control (ATN,   N = 6)", inplace=True)
    df.replace("Lesion", "Lesion  (ATNx, N = 6)", inplace=True)
    df[["frequency", "power"]] = df[["frequency", "power"]].apply(pd.to_numeric)

    print("Saving plots to {}".format(os.path.join(out_dir, "summary")))
    for ci, oname in zip([95, None], ["_ci", ""]):
        sns.lineplot(
            data=df[df["region"] == "SUB"],
            x="frequency",
            y="power",
            style="Group",
            hue="Group",
            ci=ci,
            estimator=np.median,
        )
        sns.despine(offset=0, trim=True)
        plt.xlabel("Frequency (Hz)")
        if scale == "volts":
            micro = UnicodeGrabber.get("micro")
            pow2 = UnicodeGrabber.get("pow2")
            plt.ylabel(f"PSD ({micro}V{pow2} / Hz)")
        elif scale == "decibels":
            plt.ylabel("PSD (dB)")
        else:
            raise ValueError("Unsupported scale {}".format(scale))
        plt.title("Subicular LFP power (median)")
        plt.tight_layout()

        os.makedirs(os.path.join(out_dir, "summary"), exist_ok=True)
        plt.savefig(
            os.path.join(out_dir, "summary", name + "--sub--power{}.png".format(oname)),
            dpi=400,
        )

        plt.close("all")

        sns.lineplot(
            data=df[df["region"] == "RSC"],
            x="frequency",
            y="power",
            style="Group",
            hue="Group",
            ci=ci,
            estimator=np.median,
        )
        sns.despine(offset=0, trim=True)
        plt.xlabel("Frequency (Hz)")
        if scale == "volts":
            micro = UnicodeGrabber.get("micro")
            pow2 = UnicodeGrabber.get("pow2")
            plt.ylabel(f"PSD ({micro}V{pow2} / Hz)")
        elif scale == "decibels":
            plt.ylabel("PSD (dB)")
        else:
            raise ValueError("Unsupported scale {}".format(scale))
        plt.title("Retrosplenial LFP power (median)")
        plt.tight_layout()

        plt.savefig(
            os.path.join(out_dir, "summary", name + "--rsc--power{}.png".format(oname)),
            dpi=400,
        )

        plt.close("all")


def plot_all_lfp(info, out_dir, name):
    import os

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import simuran
    import matplotlib.pyplot as plt

    from neurochat.nc_utils import smooth_1d

    os.makedirs(out_dir, exist_ok=True)

    simuran.set_plot_style()

    parsed_info = []
    control_data = []
    lesion_data = []
    for item in info:
        for val in item:
            # l1 = freq, l2 - coherence, l3 - group
            this_item = list(val.values())[0]
            to_use = this_item
            # to_use[1] = smooth_1d(
            #     this_item[1].astype(float), kernel_type="hg", kernel_size=5
            # )
            if this_item[2][0] == "Control":
                control_data.append(to_use[1])
            else:
                lesion_data.append(to_use[1])
            x_data = to_use[0].astype(float)
            parsed_info.append(np.array(to_use))

    lesion_arr = np.array(lesion_data).astype(float)
    control_arr = np.array(control_data).astype(float)

    y_lesion = np.median(lesion_arr, axis=0)
    y_control = np.median(control_arr, axis=0)

    difference = y_control[:80] - y_lesion[:80]

    data = np.concatenate(parsed_info, axis=1)
    df = pd.DataFrame(data.transpose(), columns=["frequency", "coherence", "Group"])
    df.replace("Control", "Control (ATN,   N = 6)", inplace=True)
    df.replace("Lesion", "Lesion  (ATNx, N = 6)", inplace=True)
    df[["frequency", "coherence"]] = df[["frequency", "coherence"]].apply(pd.to_numeric)

    sns.lineplot(
        data=df[df["frequency"] <= 40],
        x="frequency",
        y="coherence",
        style="Group",
        hue="Group",
        estimator=np.median,
        ci=95,
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.ylim(0, 1)
    simuran.despine()

    print("Saving plots to {}".format(out_dir))
    os.makedirs(os.path.join(out_dir, "summary"), exist_ok=True)
    plt.savefig(os.path.join(out_dir, "summary", name + "--coherence_ci.png"), dpi=400)
    plt.close("all")

    sns.lineplot(
        data=df[df["frequency"] <= 40],
        x="frequency",
        y="coherence",
        style="Group",
        hue="Group",
        estimator=np.median,
        ci=None,
    )

    simuran.despine()

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

    sns.lineplot(x=x_data[:80], y=difference)

    simuran.despine()

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Difference")

    print("Saving to {}".format((os.path.join(out_dir, name + "--difference.pdf"))))
    plt.savefig(os.path.join(out_dir, "summary", name + "--difference.pdf"), dpi=400)
    plt.close("all")
