import os
from shapely.geometry import Point, Polygon
import pandas as pd
import numpy as np
import math


def is_inside(x, y, position):
    points_inside = []
    coord = {
        "down_left": [(205, 100), (290, 175), (180, 235), (100, 140)],
        "top_left": [(100, 250), (205, 390), (300, 340), (180, 200)],
        "top_right": [(360, 215), (250, 290), (350, 380), (450, 310)],
        "down_right": [(300, 90), (400, 150), (310, 220), (225, 150)],
        "impossible to find": [(0, 1), (2, 1), (2, 2), (2, 2)],
    }
    region = Polygon(coord[position])
    for vx, vy in zip(x, y):
        p = Point(vx, vy)
        if p.within(region):
            points_inside.append((vx, vy))
    return points_inside


class RecPos:
    """
    This data class contains information about the recording position.
    Read .pos file
    To dos:
        * read different numbers of LEDs
        * Adapt to NeuroChat
    Attributes
    ----------
    _file_tag : str
        The tag of the pos data.
    """

    def __init__(self, file_name):

        self.bytes_per_sample = 20  # Axona daqUSB manual
        file_directory, file_basename = os.path.split(file_name)
        file_tag, file_extension = os.path.splitext(file_basename)
        file_extension = file_extension[1:]
        self.pos_file = os.path.join(file_directory, file_tag + ".pos")
        if os.path.isfile(self.pos_file):
            with open(self.pos_file, "rb") as f:
                while True:
                    line = f.readline()
                    try:
                        line = line.decode("latin-1")
                    except BaseException:
                        break

                    if line == "":
                        break
                    if line.startswith("trial_date"):
                        # Blank pos file
                        if line.strip() == "trial_date":
                            total_samples = 0
                            print("No position data.")
                            return
                        date = " ".join(line.replace(",", " ").split()[1:])
                    if line.startswith("num_colours"):
                        colors = int(line.split()[1])
                    if line.startswith("min_x"):
                        self.min_x = int(line.split()[1])
                    if line.startswith("max_x"):
                        self.max_x = int(line.split()[1])
                    if line.startswith("min_y"):
                        self.min_y = int(line.split()[1])
                    if line.startswith("max_y"):
                        self.max_y = int(line.split()[1])
                    if line.startswith("window_min_x"):
                        self.window_min_x = int(line.split()[1])
                    if line.startswith("window_max_x"):
                        self.window_max_x = int(line.split()[1])
                    if line.startswith("window_min_y"):
                        self.window_min_y = int(line.split()[1])
                    if line.startswith("window_max_y"):
                        self.window_max_y = int(line.split()[1])
                    if line.startswith("bytes_per_timestamp"):
                        self.bytes_per_tstamp = int(line.split()[1])
                    if line.startswith("bytes_per_coord"):
                        self.bytes_per_coord = int(line.split()[1])
                    if line.startswith("pixels_per_metre"):
                        self.pixels_per_metre = int(line.split()[1])
                    if line.startswith("num_pos_samples"):
                        self.total_samples = int(line.split()[1])
                    if line.startswith("data_start"):
                        break

                f.seek(0, 0)
                header_offset = []
                while True:
                    try:
                        buff = f.read(10).decode("UTF-8")
                    except BaseException:
                        break
                    if buff == "data_start":
                        header_offset = f.tell()
                        break
                    else:
                        f.seek(-9, 1)

                if not header_offset:
                    print("Error: data_start marker not found!")
                else:
                    f.seek(header_offset, 0)
                    byte_buffer = np.fromfile(f, dtype="uint8")
                    len_bytebuffer = len(byte_buffer)
                    end_offset = len("\r\ndata_end\r")
                    num_samples = int(len((byte_buffer) - end_offset) / 20)
                    big_spotx = np.zeros([self.total_samples, 1])
                    big_spoty = np.zeros([self.total_samples, 1])
                    little_spotx = np.zeros([self.total_samples, 1])
                    little_spoty = np.zeros([self.total_samples, 1])
                    # pos format: t,x1,y1,x2,y2,numpix1,numpix2 => 20 bytes
                    for i, k in enumerate(
                        np.arange(0, self.total_samples * 20, 20)
                    ):  # Extract bytes from 20 bytes words
                        byte_offset = k
                        big_spotx[i] = int(
                            256 * byte_buffer[k + 4] + byte_buffer[k + 5]
                        )  # 4,5 bytes for big LED x
                        big_spoty[i] = int(
                            256 * byte_buffer[k + 6] + byte_buffer[k + 7]
                        )  # 6,7 bytes for big LED x
                        little_spotx[i] = int(
                            256 * byte_buffer[k + 4] + byte_buffer[k + 5]
                        )
                        little_spoty[i] = int(
                            256 * byte_buffer[k + 6] + byte_buffer[k + 7]
                        )

                    self.raw_position = {
                        "big_spotx": big_spotx,
                        "big_spoty": big_spoty,
                        "little_spotx": little_spotx,
                        "little_spoty": little_spoty,
                    }

        else:
            print(f"No pos file found for file {self.pos_file}")

    def get_cam_view(self):
        self.cam_view = {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
        }
        return self.cam_view

    def get_tmaze_start(self):
        x, y = self.get_position()
        a = x[100:250]
        b = y[100:250]
        a = pd.Series([n if n != 1023 else np.nan for n in a])
        b = pd.Series([n if n != 1023 else np.nan for n in b])
        a.clip(0, 500, inplace=True)
        b.clip(0, 500, inplace=True)
        a.fillna(method="backfill", inplace=True)
        b.fillna(method="backfill", inplace=True)
        if a.mean() < 200 and b.mean() > 300:
            start = "top left"
        elif a.mean() > 400 and b.mean() > 300:
            start = "top right"
        elif a.mean() < 200 and b.mean() < 200:
            start = "down left"
        elif a.mean() > 300 and b.mean() < 200:
            start = "down right"
        else:
            start = "impossible to find"
        return start

    def get_window_view(self):
        try:
            self.windows_view = {
                "window_min_x": self.window_min_x,
                "window_max_x": self.window_max_x,
                "window_min_y": self.window_min_y,
                "window_max_y": self.window_max_y,
            }
            return self.windows_view
        except:
            print("No window view")

    def get_pixel_per_metre(self):
        return self.pixels_per_metre

    def get_raw_pos(self):
        bigx = [value[0] for value in self.raw_position["big_spotx"]]
        bigy = [value[0] for value in self.raw_position["big_spoty"]]

        return bigx, bigy

    def filter_max_speed(self, x, y, max_speed=4):  # max speed 4m/s ()
        tmp_x = x.copy()
        tmp_y = y.copy()
        for i in range(1, len(tmp_x) - 1):
            if (math.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)) > (
                max_speed * self.pixels_per_metre
            ):
                tmp_x[i] = 1023
                tmp_y[i] = 1023
        return tmp_x, tmp_y

    def get_position(self):
        try:
            count_missing = 0
            bxx, sxx = [], []
            byy, syy = [], []
            bigx = [value[0] for value in self.raw_position["big_spotx"]]
            bigy = [value[0] for value in self.raw_position["big_spoty"]]
            smallx = [value[0] for value in self.raw_position["little_spotx"]]
            smally = [value[0] for value in self.raw_position["little_spoty"]]
            for bx, sx in zip(bigx, smallx):  # Try to clean single blocked LED x
                if bx == 1023 and sx != 1023:
                    bx = sx
                elif bx != 1023 and sx == 1023:
                    sx = bx
                elif bx == 1023 and sx == 1023:
                    count_missing += 1
                    bx = np.nan
                    sx = np.nan

                bxx.append(bx)
                sxx.append(sx)

            for by, sy in zip(bigy, smally):  # Try to clean single blocked LED y
                if by == 1023 and sy != 1023:
                    by = sy
                elif by != 1023 and sy == 1023:
                    sy = by
                elif by == 1023 and sy == 1023:
                    by = np.nan
                    sy = np.nan
                byy.append(by)
                syy.append(sy)

            ### Remove coordinates with max_speed > 4ms
            bxx, byy = self.filter_max_speed(bxx, byy)
            sxx, syy = self.filter_max_speed(sxx, syy)

            ### Interpolate missing values
            bxx = (pd.Series(bxx).astype(float)).interpolate("cubic")
            sxx = (pd.Series(sxx).astype(float)).interpolate("cubic")
            byy = (pd.Series(byy).astype(float)).interpolate("cubic")
            syy = (pd.Series(syy).astype(float)).interpolate("cubic")

            ### Average both LEDs
            x = list((bxx + sxx) / 2)
            y = list((byy + syy) / 2)

            ## Boxcar filter 400 ms (axona tint default)
            # sample rate = 20 ms
            b = int(400 / 20)
            kernel = np.ones(b) / b

            def pad_and_convolve(xx, kernel):
                npad = len(kernel)
                xx = np.pad(xx, (npad, npad), "edge")
                yy = np.convolve(xx, kernel, mode="same")
                return yy[npad:-npad]

            x = pad_and_convolve(x, kernel)
            y = pad_and_convolve(y, kernel)

            return list(x), list(y)
        except:
            print(f"No position information found in {self.pos_file}")

    def get_speed(self):
        print("Not implemented")
        pass

    def get_angular_pos(self):
        print("Not implemented")
        pass
