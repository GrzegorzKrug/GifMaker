import os
import tkinter as tk
import tkinter.filedialog
from tkinter import Button, Label, LabelFrame

import cv2
import imutils
import numpy as np
from PIL import Image, ImageTk

from hyper_generator import make_hyper, move_pic
from image_processing import export_frames, image_downscale, make_jam, image_scale

import threading

import time
import asyncio


def measure_time(fun):
    def wrapper(*a, **kw):
        fmt = ">4.1f"
        time0 = time.perf_counter()
        res = fun(*a, **kw)
        t_end = time.perf_counter()
        dur = t_end - time0
        if dur < 1e-3:
            timeend = f"{dur * 1000000:{fmt}} us"
        elif dur < 1:
            timeend = f"{dur * 1000:{fmt}} ms"
        else:
            timeend = f"{dur:{fmt}} s"
        print(f"{fun.__name__} exec time: {timeend}: ")

        return res

    return wrapper


class ClipExtractor:
    PAD_LEFT = 5
    MAIN_PREV_SIZE = 700
    SIDE_PREV_SIZE = 400
    GIF_FILETYPE = (("Gif image", "*.gif"), ("Gif image", "*.gif"),)
    MP4_FILETYPE = (("Mp4 Videos", "*.mp4"), ("Mp4 Videos", "*.mp4"),)

    def __init__(self):
        self.loop = asyncio.get_event_loop()
        self.last_path = None
        self.cap = None
        self.clip_spinboxes = None

        self.root = tk.Tk()
        self.var_party = tk.IntVar()
        self.var_blend = tk.IntVar()
        self.var_hyper = tk.IntVar()
        self.var_hyper_oneframe = tk.IntVar()

        path = os.path.abspath(f"../unknown.png")
        self.img_tk = ImageTk.PhotoImage(Image.open(path))

        self.cap = cv2.VideoCapture()
        self.frame1 = None
        self.frame2 = None
        self.frame3 = None

        root = self.root
        img_tk = self.img_tk
        lay_preview = LabelFrame(root)
        lay_3images = LabelFrame(root)
        lay_buttons = LabelFrame(root)
        lay_clips = LabelFrame(root)
        lay_start_end = LabelFrame(lay_preview)

        lay_buttons.pack()

        "Preview images setup"
        self.prev_main_image = Label(lay_3images, image=img_tk, text="Main frame preview")
        self.prev_main_image.pack()
        self.prev_start = Label(lay_preview, text="preview start", pady=20, padx=10)
        self.prev_start.pack()
        self.prev_end = Label(lay_preview, text="preview end")
        self.prev_end.pack()

        "Control buttons setup"
        self.but_load = Button(lay_buttons, text="Load video", command=self.ask_user_for_file)
        self.but_clip = Button(lay_buttons, text="Clip video")
        self.but_save = Button(lay_buttons, text="Save gif", command=self.save)
        self.but_save_as = Button(lay_buttons, text="Save gif as...", command=self.save_as)
        self.but_quit = Button(lay_buttons, text="Quit", command=root.quit)

        self.spin_interval = tk.Spinbox(lay_buttons, text='Interval', from_=2, to=50)
        self.spin_interval.configure(from_=1)
        self.spin_duration = tk.Spinbox(lay_buttons, from_=20, to=1000, increment=5)
        self.spin_duration.setvar("40")

        self.but_load.pack(side='left')
        self.but_clip.pack(side='left')
        self.but_save.pack(side='left')
        self.but_save_as.pack(side='left')

        Label(lay_buttons, text='LoopBack').pack(side='left', padx=[self.PAD_LEFT, 0])
        self.var_loop_back = tk.IntVar()
        tk.Checkbutton(lay_buttons, variable=self.var_loop_back).pack(side='left')

        Label(lay_buttons, text='RGB-Mode').pack(side='left', padx=[self.PAD_LEFT, 0])
        self.var_rgb_mode = tk.IntVar()
        tk.Checkbutton(lay_buttons, variable=self.var_rgb_mode).pack(side='left')

        Label(lay_buttons, text='Interval').pack(side='left', padx=[self.PAD_LEFT, 0])
        self.spin_interval.pack(side='left')
        Label(lay_buttons, text='Duration[ms]').pack(side='left', padx=[self.PAD_LEFT, 0])
        self.spin_duration.pack(side='left')

        self.but_quit.pack(side='right')

        self.but_clip.configure(state='disabled')
        self.but_save.configure(state='disabled')
        self.but_save.configure(background="#FBB")
        self.but_quit.configure(background="#FBB")

        self.clip_spinboxes = {}
        for text in ["L", "R", "Top", "Bottom"]:
            Label(lay_clips, text=f"Clip {text}", padx=6, justify='right').pack(side="left")
            e = tk.Spinbox(lay_clips, from_=0, to=9000, increment=1, )
            e.pack(side="left")
            self.clip_spinboxes[text.lower()] = e
        self.lab_ratio = Label(lay_clips, text=f"Ratio: 100.0")
        self.lab_ratio.pack(side='left')

        self.spin_start = tk.Spinbox(lay_start_end, from_=0, to=100, increment=1,
                                     command=lambda: self.get_photo_from_frame(True))
        self.spin_end = tk.Spinbox(lay_start_end, from_=0, to=100, increment=1,
                                   command=lambda: self.get_photo_from_frame(False))

        self.spin_start.pack(side="left")
        Label(lay_start_end, text="Start frame").pack(side="left")
        self.spin_end.pack(side="left")
        Label(lay_start_end, text="End frame").pack(side="left")

        fr_steps = LabelFrame(lay_clips)
        fr_steps.pack(side='bottom')

        # variab
        b1 = tk.Radiobutton(fr_steps, text="Tick  1", value=1, padx=5, command=lambda: self.set_ticks(1))
        b5 = tk.Radiobutton(fr_steps, text="Tick  5", value=2, padx=5, command=lambda: self.set_ticks(5))
        b10 = tk.Radiobutton(fr_steps, text="Tick 10", value=3, padx=5,
                             command=lambda: self.set_ticks(10))
        b50 = tk.Radiobutton(fr_steps, text="Tick 50", value=4, padx=5,
                             command=lambda: self.set_ticks(50))

        "SPECIALS FRAME"
        lay_special = LabelFrame(root)

        "PARTY FRAME"
        party_frame = LabelFrame(lay_special)
        party_frame.pack(side='left')
        self.chk_party = tk.Checkbutton(party_frame, text="Party", variable=self.var_party, padx=8)
        self.chk_party.pack(side='top')
        party_grid = tk.Frame(party_frame)
        party_grid.pack(side='top')

        self.spin_party_blend = tk.Spinbox(party_grid, from_=0.2, to=1, increment=0.05)
        self.spin_party_blend.configure(from_=0.01)
        self.spin_party_blend.grid(row=1, column=2)
        Label(party_grid, text='Blending').grid(row=1, column=1)

        self.spin_party_rot_period = tk.Spinbox(party_grid, from_=60, to=120, increment=5)
        self.spin_party_rot_period.configure(from_=30)
        self.spin_party_rot_period.grid(row=2, column=2)
        Label(party_grid, text='Rotation period').grid(row=2, column=1)

        "HYPER"
        hyper_frame = tk.Frame(lay_special)
        hyper_frame.pack(side='left')

        self.chk_hyper = tk.Checkbutton(hyper_frame, text="Hyper Gif", variable=self.var_hyper, padx=8)
        self.chk_hyper.pack(side='top')
        self.chk_hyper_oneframe = tk.Checkbutton(
                hyper_frame, text="First Frame", variable=self.var_hyper_oneframe, padx=8)
        self.chk_hyper_oneframe.pack(side='top')

        hyper_grid = tk.Frame(hyper_frame)
        hyper_grid.pack(side='top')

        self.spin_hyper_jit_ang = tk.Spinbox(hyper_grid, from_=0.3, to=1.57, increment=0.01)
        self.spin_hyper_jit_ang.configure(from_=0)
        self.spin_hyper_jit_amp = tk.Spinbox(hyper_grid, from_=0.2, to=2, increment=0.01)
        self.spin_hyper_jit_amp.configure(from_=0)
        self.spin_hyper_vision = tk.Spinbox(hyper_grid, from_=0.22, to=0.5, increment=0.01)
        self.spin_hyper_vision.configure(from_=0)
        self.spin_hyper_amp = tk.Spinbox(hyper_grid, from_=0.1, to=0.5, increment=0.01)
        self.spin_hyper_amp.configure(from_=0)
        self.spin_hyper_freq = tk.Spinbox(hyper_grid, from_=8, to=20, increment=1)
        self.spin_hyper_freq.configure(from_=3)
        self.spin_hyper_N = tk.Spinbox(hyper_grid, from_=40, to=500, increment=1)
        self.spin_hyper_N.configure(from_=10)

        self.spin_hyper_freq.grid(row=1, column=2)
        self.spin_hyper_jit_ang.grid(row=2, column=2)
        self.spin_hyper_jit_amp.grid(row=3, column=2)
        self.spin_hyper_vision.grid(row=4, column=2)
        self.spin_hyper_amp.grid(row=5, column=2)
        self.spin_hyper_N.grid(row=6, column=2)

        for i, txt in enumerate(
                ['Freq', 'Noise angle', "Noise pos", "Vision", "Amplitude", "N single frame"]):
            lb = Label(hyper_grid, text=txt)
            lb.grid(row=i + 1, column=1)

        "BLEND COLORS"
        blend_frame = tk.Frame(lay_special)
        blend_frame.pack(side='left')

        blend_grid = tk.Frame(blend_frame)
        blend_grid.pack(side='bottom')

        # Label(blend_frame, text='Blend Color').pack(side='top')
        self.chk_blend = tk.Checkbutton(blend_frame, text="Blend Color", variable=self.var_blend, padx=8)
        self.chk_blend.pack(side='top')

        self.spin_blend_alpha = tk.Spinbox(blend_grid, from_=0.15, to=1, increment=0.05)
        self.spin_blend_alpha.configure(from_=0.01)
        self.spin_blend_alpha.grid(row=0, column=2)

        self.spin_blend_red = tk.Spinbox(blend_grid, from_=180, to=255, increment=5)
        self.spin_blend_red.configure(from_=0)
        self.spin_blend_green = tk.Spinbox(blend_grid, from_=0, to=255, increment=5)
        self.spin_blend_blue = tk.Spinbox(blend_grid, from_=0, to=255, increment=5)

        self.spin_blend_red.grid(row=1, column=2)
        self.spin_blend_green.grid(row=2, column=2)
        self.spin_blend_blue.grid(row=3, column=2)

        for i, txt in enumerate(["Alpha", 'Red', "Green", "Blue"]):
            lb = Label(blend_grid, text=txt)
            lb.grid(row=i, column=1)

        " = = "

        b1.pack(side='left')
        b5.pack(side='left')
        b10.pack(side='left')
        b50.pack(side='left')
        b1.deselect()
        b5.deselect()
        b10.deselect()
        b50.select()

        self.set_ticks(50)

        lay_clips.pack(side="bottom", fill='x')
        lay_preview.pack(side="right")
        lay_start_end.pack(side="bottom")
        lay_3images.pack(side="top")
        lay_buttons.pack(side="bottom", fill="x")
        lay_special.pack()

    def get_photo_from_frame(self, start=True):
        st = int(self.spin_start.get())
        ed = int(self.spin_end.get())

        if start:
            pos = st
        else:
            pos = ed

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = self.cap.read()

        if start:
            self.frame2 = frame
        else:
            self.frame3 = frame

    @staticmethod
    def get_ratio(h, w):
        ratio = h / w * 100
        return f"{ratio:>5.2f}"

    def export_gif(self, cp, path, st=0, end=1, h1=0, h2=10, w1=0, w2=10,
                   downsize=150, smooth=0,
                   spc_party=False, spc_party_blend=0.2, spc_party_period=60,
                   interval=3, duration=20,
                   loop_back=False, rgb_mode=False,
                   hyper=False, hyper_freq=3,
                   hyper_jit_ang=0, hyper_jit_amp=0, hyper_vis=0, hyper_amp=0,
                   hyper_one_frame=False, hyper_N=None,
                   blend=False, blend_color=None, blend_alpha=0.0,
                   ):
        frames = []

        ret = True

        i = -1
        h = int(cp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cp.get(cv2.CAP_PROP_FRAME_WIDTH))
        v = int(cp.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Loaded h: {h}, w: {w}, frames: {v}")
        print(f"Start: {st}, end: {end}")

        # x1, x2 = round(x1 * w / 100), round(x2 * w / 100)
        # y1, y2 = round(y1 * h / 100), round(y2 * h / 100)

        fw = w - w2 - w1
        fh = h - h2 - h1

        cp.set(cv2.CAP_PROP_POS_FRAMES, st)
        while ret:
            i += 1
            pos = cp.get(cv2.CAP_PROP_POS_FRAMES)
            ret, _full_frame = cp.read()

            if end and pos > end:
                break

            if i % interval:
                continue

            if not ret:
                break

            fr = _full_frame[h1:h - h2, w1:w - w2, :]
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = image_downscale(fr, downsize)
            # if downsize:
            #     if h > w:
            #         fr = imutils.resize(fr, height=downsize)
            #     else:
            #         fr = imutils.resize(fr, width=downsize)
            # fr = cv2.resize(fr, (400, 150))

            # raise "CONVER AT END"
            # im = Image.fromarray(fr, mode='RGB')
            # im = im.convert("RGBA")

            frames.append(fr)

        print(f"All frames: {len(frames)}")

        vals = np.linspace(0, 1, smooth + 2)[1:-1]

        for si in range(smooth):
            print("Doing smooth")
            ind2 = len(frames) - smooth + si
            st_fr = frames[si].copy()
            ed_fr = frames[ind2].copy()

            v, v2 = vals[si], 1 - vals[si]

            orig_start = np.array(st_fr, dtype=np.uint8)
            orig_end = np.array(ed_fr, dtype=np.uint8)
            # print(vals[si])
            # print(si, ind2, f"{v:>1.3f}", f"{v2:>1.3f}")

            same = cv2.addWeighted(orig_start, v, orig_end, v2, 0)
            # same = Image.fromarray(same, mode='RGBA')

            frames[si] = same
            frames[ind2] = same

        if smooth > 0:
            frames = frames[smooth:]
        elif loop_back:
            n = len(frames)
            print(f"Looping back: from: {n}")
            for i in range(n - 2, 0, -1):
                frames.append(frames[i])

        if spc_party:
            print("Adding jam")
            h, w, *c = np.array(frames[0]).shape
            size = max(h, w)

            jam_fr = make_jam((size * 2, size * 2))
            jam_cx = jam_cy = size

            dur = len(frames)
            rotates = 1 if dur < spc_party_period else dur // spc_party_period
            step = rotates / dur * 360

            jam_frames = []
            v = spc_party_blend
            for i, fr in enumerate(frames):
                jm = imutils.rotate(jam_fr, i * step, (jam_cy, jam_cx))

                ind_y1 = jam_cy - h // 2
                ind_y2 = jam_cy + h // 2
                ind_x1 = jam_cx - w // 2
                ind_x2 = jam_cx + w // 2

                ind_y2 += h - (ind_y2 - ind_y1)
                ind_x2 += w - (ind_x2 - ind_x1)

                jm = jm[ind_y1:ind_y2, ind_x1:ind_x2, :]

                fr = np.array(fr, dtype=np.uint8)[:, :, :3]

                # 16 974 17 503
                # 106 1014 27 513
                # print(f"frame: {fr.shape}, jam: {jm.shape}")

                f = cv2.addWeighted(fr, 1 - v, jm, v, 0)
                # fr = Image.fromarray(f)
                jam_frames.append(f)

            frames = jam_frames

        if hyper:
            if hyper_one_frame:
                fr_in = frames[0]
            else:
                hyper_N = len(frames)
                fr_in = frames

            key_pos = make_hyper(hyper_N, hyper_freq, hyper_jit_ang, hyper_jit_amp)
            frames = move_pic(fr_in, key_pos, vision=hyper_vis, amp=hyper_amp)

            # for fr in frames:
            #     print(f"framem (after hyp) shape:{fr.shape}")

        if blend:
            blend_color = np.array(blend_color, dtype=np.uint8)
            fr = frames[0]
            frame_color = np.zeros(fr.shape, dtype=np.uint8) + blend_color
            # print(f"Blend color: {frame_color}")
            v = 1 - blend_alpha
            v2 = blend_alpha
            frames = [cv2.addWeighted(fr, v, frame_color, v2, 0) for fr in frames]

            # same = Image.fromarray(same, mode='RGBA')
            # blend_alpha
            # blend_color

        export_frames(frames, path, duration, rgb_mode)
        print(f"H/W ratio({path:^30}): {(fh / fw) * 100:<3.2f}, frames: {len(frames):>3}")

    def _save(self):
        # th = threading.Thread(target=self._save_thread)
        # th.start()
        # th.join()

        """GET PARAMS AND RUN GIF SCRIPT"""

        # th = threading.Thread(target=self.disable_update)
        # th.start()

        # print("Sleeping for save 2")
        # await asyncio.sleep(2)
        # print("Woke up form sleep")

        "Read start"
        st, end = int(self.spin_start.get()), int(self.spin_end.get())

        "READ CLIP VALUES"
        cl_l = int(self.clip_spinboxes['l'].get())
        cl_r = int(self.clip_spinboxes['r'].get())
        cl_top = int(self.clip_spinboxes['top'].get())
        cl_bot = int(self.clip_spinboxes['bottom'].get())

        interval = int(self.spin_interval.get())
        duration = int(self.spin_duration.get())

        "READ SPECIALS"
        # party = chk_party.get()
        party = bool(self.var_party.get())
        party_blend = float(self.spin_party_blend.get())
        party_rot_period = int(self.spin_party_rot_period.get())

        hyper = bool(self.var_hyper.get())
        hyper_jit_ang = float(self.spin_hyper_jit_ang.get())
        hyper_jit_amp = float(self.spin_hyper_jit_amp.get())
        hyper_vis = float(self.spin_hyper_vision.get())
        hyper_amp = float(self.spin_hyper_amp.get())
        hyper_one_frame = bool(self.var_hyper_oneframe.get())
        hyper_freq = int(self.spin_hyper_freq.get())
        hyper_N = int(self.spin_hyper_N.get())

        "BLEND VARIABLES"
        blend = bool(self.var_blend.get())
        blend_color = int(self.spin_blend_red.get()), int(self.spin_blend_green.get()), int(
                self.spin_blend_blue.get())
        blend_alpha = float(self.spin_blend_alpha.get())

        self.export_gif(self.cap, self.last_path, st, end, cl_top, cl_bot, cl_l, cl_r,
                        interval=interval, duration=duration,
                        spc_party=party, spc_party_blend=party_blend, spc_party_period=party_rot_period,
                        loop_back=bool(self.var_loop_back.get()),
                        rgb_mode=bool(self.var_rgb_mode.get()),
                        # HYPER
                        hyper=hyper, hyper_freq=hyper_freq,
                        hyper_jit_ang=hyper_jit_ang, hyper_jit_amp=hyper_jit_amp,
                        hyper_vis=hyper_vis, hyper_amp=hyper_amp,
                        hyper_one_frame=hyper_one_frame, hyper_N=hyper_N,
                        # BLEND
                        blend=blend, blend_alpha=blend_alpha, blend_color=blend_color,


                        )

    def save(self):
        self._save()

    def save_as(self):
        new_path = tk.filedialog.asksaveasfilename(filetypes=self.GIF_FILETYPE)

        if new_path == "":
            return None

        if not new_path.endswith(".gif"):
            new_path += ".gif"
        # print(f"save path: {new_path}")
        print(f"Changed path to: {new_path}")
        self.last_path = new_path

        self.but_save.configure(state='active')
        self.but_save.configure(state='active')

        self._save()

    @staticmethod
    def update_photo(box, photo):
        if photo is None:
            box.configure(image="")
        else:
            box.configure(image=photo)
            box.image = photo

    def set_ticks(self, n):
        print(f"ticks: {n}")
        for k, v in self.clip_spinboxes.items():
            v.configure(increment=n)

        self.spin_start.configure(increment=n)
        self.spin_end.configure(increment=n)

    def run_thread_update_preview(self):
        print("Updating")
        for i in range(1):
            th = threading.Thread(target=self.update_previews)
            th.start()

        self.root.after(500, self.run_thread_update_preview)

    def start_important_loops(self):
        self.root.after(500, self.run_thread_update_preview)

    @measure_time
    def update_previews(self):
        self._update_previews()

    def _update_previews(self):

        if self.frame1 is None:
            return None

        h, w, *c = self.frame1.shape
        sl_w = slice(
                int(self.clip_spinboxes['l'].get()),
                w - int(self.clip_spinboxes['r'].get()),
                None)
        sl_h = slice(
                int(self.clip_spinboxes['top'].get()),
                h - int(self.clip_spinboxes['bottom'].get()),
                None)

        # roi = frame[clip_spinboxes['bottom']:]
        # st = int(self.spin_start.get())
        # ed = int(self.spin_end.get())

        # spin_start.configure(from_=0, to=ed)
        # spin_end.configure(from_=st, to=fr_count)

        if self.frame1 is not None:
            frame1 = self.frame1[sl_h, sl_w, :]
            h, w, *c = frame1.shape
            ratio = self.get_ratio(h, w)
            self.lab_ratio.configure(text=f"Ratio: {ratio}")

            frame1 = image_scale(frame1, self.MAIN_PREV_SIZE)
            fr_tk1 = self.cv_image_to_tk(frame1)
            self.update_photo(self.prev_main_image, fr_tk1)

        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, st)
        # ret, self.frame2 = self.cap.read()
        if self.frame2 is not None:
            frame2 = self.frame2[sl_h, sl_w, :]
            frame2 = image_scale(frame2, self.SIDE_PREV_SIZE)
            fr_tk2 = self.cv_image_to_tk(frame2)
            self.update_photo(self.prev_start, fr_tk2)

        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, ed)
        # ret, self.frame3 = self.cap.read()
        if self.frame3 is not None:
            frame3 = self.frame3[sl_h, sl_w, :]
            frame3 = image_scale(frame3, self.SIDE_PREV_SIZE)
            fr_tk3 = self.cv_image_to_tk(frame3)
            self.update_photo(self.prev_end, fr_tk3)

    @staticmethod
    def cv_image_to_tk(frame):
        frame = frame[:, :, [2, 1, 0]]
        im_tk = ImageTk.PhotoImage(Image.fromarray(frame))
        return im_tk

    def ask_user_for_file(self):
        ret = tk.filedialog.askopenfilename(filetypes=self.MP4_FILETYPE,
                                            initialdir=os.path.dirname(__file__))
        if ret.lower().endswith(".mp4"):
            print(f"Selected: {ret}")
            cap = self.cap

            cap.open(ret)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            red, frame = cap.read()
            self.frame1 = frame.copy()
            self.frame2 = frame.copy()

            frame = image_scale(frame, self.MAIN_PREV_SIZE)
            small_frame = image_scale(frame, self.SIDE_PREV_SIZE)
            h, w, *c = small_frame.shape

            ratio = self.get_ratio(h, w)
            self.lab_ratio.configure(text=f"Ratio: {ratio}")

            fr_tk = self.cv_image_to_tk(frame)
            self.update_photo(self.prev_main_image, fr_tk)

            fr_tk = self.cv_image_to_tk(small_frame)
            self.update_photo(self.prev_start, fr_tk)
            self.update_photo(self.prev_end, fr_tk)

            cap.set(cv2.CAP_PROP_POS_FRAMES, dur)
            ret, self.frame3 = cap.read()
            # self.frame3 = frame.copy()

            self.spin_start.configure(from_=0, to=dur - 2)
            self.spin_end.configure(from_=dur - 2, to=dur - 1)
            self.spin_end.configure(from_=1, to=dur - 1)

            self.root.geometry(f"1400x900")

        else:
            print("Invalid file")

    def start(self):
        self.root.after(1000, self.run_thread_update_preview)
        self.root.mainloop()


if __name__ == "__main__":
    print("Application start")
    app = ClipExtractor()
    app.start()
