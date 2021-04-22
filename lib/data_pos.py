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

    # Methods
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

    def filter_max_speed(self, x, y, max_speed = 4):  # max speed 4m/s ()
        tmp_x = x.copy()
        tmp_y = y.copy()
        threshold = max_speed * 300 * 50 # max speed * distance (m) /  50 samples (s)
        for i in range(1, len(tmp_x)):
            distance = math.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
            if distance > threshold:
                tmp_x[i] = 1023
                tmp_y[i] = 1023
                
        return tmp_x, tmp_y

    def get_position(self, raw=False):
        # TODO include the checking for big-small mix ups
        # TODO add verbose reading like TINT
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
            bxx = (pd.Series(bxx).astype(float)).interpolate("linear")
            sxx = (pd.Series(sxx).astype(float)).interpolate("linear")
            byy = (pd.Series(byy).astype(float)).interpolate("linear")
            syy = (pd.Series(syy).astype(float)).interpolate("linear")
            if raw:
                return [(bxx, byy), (sxx, syy)]
            
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
        speed = [0]
        # print('Speed in cm/s')
        [(bxx, byy), (sxx, syy)] = self.get_position(raw=True)
        x = list((bxx + sxx) / 2)
        y = list((byy + syy) / 2)
        for i in range(1, len(x)):
            sp = math.sqrt(
                (x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) / (3 * 0.02)  
                # (pixel/s) - 300 pixels per metre * 100 (cm/s)
            speed.append(sp)

        # Try to find jumpy points
        for i, val in enumerate(np.diff(speed)):
            if val > 4: # filter instantaneous speed > 3.6 m / s
                x[i+1] = np.nan
                y[i+1] = np.nan
        x = (pd.Series(x).astype(float)).interpolate("linear")
        y = (pd.Series(y).astype(float)).interpolate("linear")

        b = int(400 / 20)
        kernel = np.ones(b) / b

        def pad_and_convolve(xx, kernel):
            npad = len(kernel)
            xx = np.pad(xx, (npad, npad), "edge")
            yy = np.convolve(xx, kernel, mode="same")
            return yy[npad:-npad]

        x = pad_and_convolve(x, kernel)
        y = pad_and_convolve(y, kernel)

        speed = [0]
        s_rate = 10 #50 Hz is too fine grained
        # Perhaps this should be average speed over these
        for i in range(s_rate * 3 // 2, len(x), s_rate):
            sp = math.sqrt(
                (x[i] - x[i - s_rate]) ** 2 + (y[i] - y[i - s_rate]) ** 2) / (3 * 0.02 * s_rate)  
                # (pixel/s) - 300 pixels per metre * 100 (cm/s)
            speed.append(sp)
        # Smooth first?
        # interpolate back to 50Hz
        xp = [0.0] + [0.02 * i for i in range(s_rate, len(x), s_rate)]
        for x_, y_ in zip(xp[:20], speed[:20]):
            print("{:.2f}: {:.2f}".format(x_, y_))
        print("-----------")
        kernel_size = 5
        kernel = np.array([1 / kernel_size for _ in range(kernel_size)])
        speed = pad_and_convolve(speed, kernel)
        for x_, y_ in zip(xp[:20], speed[:20]):
            print("{:.2f}: {:.2f}".format(x_, y_))

        xs = [0.02 * i for i in range(len(x))]
        interp_speed = np.interp(xs, xp, speed)
        kernel_size = 6
        kernel = np.array([0, 0.2, 0.2, 0.2, 0.2, 0])
        interp_speed = pad_and_convolve(interp_speed, kernel)        

        # TODO fix this
        kernel_size = 5
        half_width = kernel_size / 2
        xx = np.arange(-half_width, half_width + 1, 1)
        sigma = kernel_size / 2 / np.sqrt(3)
        kernel = (0.5 / (np.sqrt(3) * sigma)) * \
            (np.abs(xx) < np.sqrt(3) * sigma)
        print(kernel)

        # TEMP

        # 1. Calculate raw speed (not smooth)
        # 2. TODO try fix mixups in big and small
        # 3. Calculate instantaneous speed at 50Hz
        # 4. From this, mark positions as np.nan where speed > 360 cm/s (rat can't run that fast)
        # 5. Interpolate these positions and box filter over 400ms
        # 6. Calculate speed at a 10Hz sample rate. Do this calculating the speed at time x by using positions at time x + 0.06, and x - 0.06 (I think?). Want the real time point in the middle.
        # 7. Box filter these speeds with a box of size 5 (5 samples, 0.2 sec smoothing either side of the actual point)
        # 8. Interpolate these values to get speed at every time point if needed (50Hz)
         

        return interp_speed, x, y

    def get_angular_pos(self): # Suposing Big Led in the Right side
        
        # To do: (Not working)
            # Check LEDs sides
            # Fix the function
            # implement Axona's checks
                #  no. pixels in big light = 18.35 +/- 8.48
                #  no. pixels in small light = 10.96 +/- 5.92
                #  1615 points swapped as 2-light confusions
        
        def calc_angle(R, L, d = 12): # d - distance from LEDs in pixels
            if R[1] == L[1]:
                if R[0] > L[0]: 
                    return 0.
                return 180.
            elif R[0] == L[0]:
                if R[1] > L[1]: 
                    return 90.
                return 270.
            elif R[1] > L[1]:
                return math.acos((R[0] - L[0]) / d) * 180 / math.pi
            elif R[0] < L[0]:
                return 90 + math.acos((R[0] - L[0]) / d) * 180 / math.pi
            else:
                return 360 - math.acos((R[0] - L[0]) / d) * 180 / math.pi
        
        L, R = self.get_position(raw=True)
        d = []
        for i in range(0, len(R)-1):
            d.append(((R[i] - R[i+1]) ** 2 + (L[i] - L[i+1])**2)**.5)
        d = np.median(np.asarray(d))
        angles = []
        for r, l in zip(R,L):
            angles.append(calc_angle(r, l, d))
        return angles