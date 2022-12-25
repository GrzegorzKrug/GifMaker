import tkinter as tk
import tkinter.filedialog
import tkinter.simpledialog

import imutils
import numpy as np
import threading
from collections import deque

from PIL import Image, ImageTk

import cv2
import json
import time
import os
import re

from tkinter import Label, LabelFrame, Button

from gui_builder import GuiBuilder

from modules.hyper_generator import make_hyper, move_pic
from modules.image_processing import export_frames, image_downscale, make_jam

from functools import wraps


def parent_replacer(func):
    """decorator"""

    @wraps(func)
    def wrapper(self, *a, parent=None, **kw):
        if parent is None:
            parent = self.root

        return func(self, *a, parent=parent, **kw)

    return wrapper


def params_replacer(func):
    """decorator"""

    @wraps(func)
    def wrapper(self, *a, params=None, **kw):
        if params is None:
            params = {}
        return func(self, *a, params=params, **kw)

    return wrapper


def packing_replacer(func):
    """decorator"""

    @wraps(func)
    def wrapper(self, *a, packing=None, **kw):
        if packing is None:
            packing = {}

        return func(self, *a, packing=packing, **kw)

    return wrapper


class ChadGui(GuiBuilder):
    update_interval = 150
    picture_display_size = 300
    output_size = 600
    default_durations = [53, 46, 49, 50]

    def __init__(self):
        super().__init__()
        self.pics_windows = []
        self.pics_paths = [None, None, None, None]
        self.clips = [[0 for i in range(4)] for j in range(4)]
        self.durations = [30 for i in range(4)]
        self.playback_pos = [0 for i in range(4)]
        self.zoom = True
        self.blend_color = [255, 225, 200]
        self.alfa = 0.2
        self.beta = 1 - self.alfa

        self.moves = [[0, 100, 500, i] for i in range(4)]
        self.enhancements = [0]

        self.root.after(self.update_interval, self.update)

        self.load_settings()
        self.loaded_pics = [
                self.preprocess_preview(cv2.imread(ph)) if ph else None for ph in
                self.pics_paths
        ]
        self.root.title("GigaEmote")

        "Non persistent"
        self.last_save_path = None
        self.export_duration = 40  # 40 ms

    @property
    def app_params(self):
        return (self.pics_paths, self.update_interval, self.clips, self.durations,
                self.blend_color, self.alfa, self.beta,
                self.moves, self.enhancements
                )

    def preprocess_preview(self, img):
        # img = self.square_crop_center(img, )
        return self.scale_outer(img, self.picture_display_size)

    @app_params.setter
    def app_params(self, new_params):
        (self.pics_paths, self.update_interval, self.clips, self.durations,
         self.blend_color, self.alfa, self.beta,
         self.moves, enhancements
         ) = new_params

        self.alfa = float(self.alfa)
        self.beta = float(self.beta)

        if len(enhancements) == len(self.enhancements):
            self.enhancements = enhancements

    def create_preview_box(self, parent, *a, text=None, **kw):
        frame = LabelFrame(parent)

        lb = Label(frame, text=text)
        lb.pack()
        lb = Label(frame)
        lb.pack(expand=True, fill='both')
        self.pics_windows.append(lb)

        return frame

    def update(self):
        th = threading.Thread(target=self.thread, args=(0,))
        th.start()
        th = threading.Thread(target=self.thread, args=(1,))
        th.start()
        th = threading.Thread(target=self.thread, args=(2,))
        th.start()
        th = threading.Thread(target=self.thread, args=(3,))
        th.start()

        self.root.after(self.update_interval, self.update)

    def thread(self, index):
        """
        0 left, 1 right, 2 up, 3 down
        Args:
            index:

        Returns:

        """
        img = self.loaded_pics[index]
        self.playback_pos[index] = (self.playback_pos[index] + 1) % (self.durations[index] + 1)
        pos = self.playback_pos[index] / self.durations[index]

        if img is not None:
            ret = self.get_frame(index, pos)

            tk = self.numpy_pic_to_tk(ret[:, :, [2, 1, 0]])
            self.update_photo(self.pics_windows[index], tk)

    def get_frame(self, index, pos):
        img = self.loaded_pics[index]
        start_pos, distance, size, direction = self.moves[index]
        "Clipping"
        h, w, c = img.shape
        top, down, left, right = np.array(self.clips[index], dtype=float) / 1000
        top = np.round(top * h).astype(int)
        down = np.round(down * h).astype(int)
        left = np.round(left * w).astype(int)
        right = np.round(right * w).astype(int)
        img = img[top:h - down, left:w - right]
        # img = self.square_crop_center(img)
        "Process"
        ret = self.blend_to_single_color(img, self.blend_color)
        img = cv2.addWeighted(img, self.alfa, ret, self.beta, 0)

        "Playback"
        H, W, _ = img.shape
        color = (np.array(self.blend_color) * self.alfa).astype(np.uint8)
        if direction == 0:
            "Left"
            size = np.round(W * size).astype(int)
            ind1 = np.round((start_pos - distance * pos) * W).astype(int)
            ind2 = ind1 + size
            ind1 = np.clip(ind1, 0, ind2)
            # print(f"i1: {ind1}- i2: {ind2}- size:{size}\n")
            ret = img[:, ind1: ind2]
            _, w, _ = ret.shape
            if w < size and ind1 != 0:
                blank = np.zeros((H, size, 3), dtype=np.uint8) + color
                blank[:, :w, :] = ret
                ret = blank
            elif w < size:
                blank = np.zeros((H, size, 3), dtype=np.uint8) + color
                blank[:, size - w:, :] = ret
                ret = blank
        elif direction == 1:
            "Right"
            size = np.round(W * size).astype(int)
            ind1 = np.round((distance * pos - start_pos) * W).astype(int)
            ind2 = ind1 + size
            ind1 = np.clip(ind1, 0, ind2)
            ret = img[:, ind1: ind2]
            _, w, _ = ret.shape

            if w < size and ind1 != 0:
                blank = np.zeros((H, size, 3), dtype=np.uint8) + color
                blank[:, :w, :] = ret
                ret = blank
            elif w < size:
                blank = np.zeros((H, size, 3), dtype=np.uint8) + color
                blank[:, size - w:, :] = ret
                ret = blank

        elif direction == 2:
            "Top"
            size = np.round(H * size).astype(int)
            ind1 = np.round((start_pos - distance * pos) * H).astype(int)
            ind2 = ind1 + size
            ind1 = np.clip(ind1, 0, ind2)
            ret = img[ind1: ind2, :]
            h, _, _ = ret.shape
            if h < size and ind1 != 0:
                blank = np.zeros((size, W, 3), dtype=np.uint8) + color
                blank[:h, :, :] = ret
                ret = blank
            elif h < size:
                blank = np.zeros((size, W, 3), dtype=np.uint8) + color
                blank[size - h:, :, :] = ret
                ret = blank
        elif direction == 3:
            "Down"
            size = np.round(H * size).astype(int)
            ind1 = np.round((distance * pos - start_pos) * H).astype(int)
            ind2 = ind1 + size
            ind1 = np.clip(ind1, 0, ind2)
            ret = img[ind1: ind2, :]
            h, _, _ = ret.shape

            if h < size and ind1 != 0:
                blank = np.zeros((size, W, 3), dtype=np.uint8) + color
                blank[:h, :, :] = ret
                ret = blank
            elif h < size:
                blank = np.zeros((size, W, 3), dtype=np.uint8) + color
                blank[size - h:, :, :] = ret
                ret = blank
        else:
            print(f"WRONG DIRECTION: {direction}")
            ret = img
        ret = self.square_crop_center(ret)

        "Preview"
        if self.zoom:
            ret = self.scale_outer(ret, self.picture_display_size)
        return ret

    def get_frames(self):
        frames = deque(maxlen=sum(self.durations) + 5)

        for index in range(4):
            if self.loaded_pics[index] is None:
                continue

            for step in range(self.durations[index]):
                pos = step / self.durations[index]
                fr = self.get_frame(index, pos)
                frames.append(fr)

        print("got n frames:", len(frames), f",Duration: {sum(self.durations)}")
        return list(frames)

    def export_gif(self):
        path = tk.filedialog.asksaveasfilename(filetypes=self.GIF_TYPES)
        if not path.endswith(".gif"):
            path += ".gif"
        self.last_save_path = path
        self.save()

    def save(self):
        path = self.last_save_path
        if path is None:
            print("File was never saved.")
            return None

        frames = self.get_frames()
        frames = [fr[:, :, [2, 1, 0]] for fr in frames]
        export_frames(frames, path, duration=self.export_duration, rgb_mode=False)

        print(f"Saving to: {path}")

    def load_new_picture(self, index=0):
        ph = self.ask_user_for_picture()
        if not ph:
            return None

        self.pics_paths[index] = ph
        pic = cv2.imread(ph)
        pic = self.preprocess_preview(pic)
        self.loaded_pics[index] = pic
        self.clips[index] = [0, 0, 0, 0]

    def change_refresh(self):
        ret = self.ask_user_for_integer("Refresh rate",
                                        f"Current refresh is {self.update_interval}. What should be new?")
        self.update_interval = ret

    @staticmethod
    def blend_to_single_color(image, color):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray / 255
        gray = gray[:, :, np.newaxis]
        image = gray * color
        image = np.array(image, dtype=np.uint8)
        return image

    def interactive_clips_edit(self, index=0):
        wn = tk.Toplevel()
        wn.geometry("250x300")
        wn.title(f"Clips of pic: {index + 1}")

        fr_hor = LabelFrame(wn)
        fr_hor.pack()

        lb = Label(fr_hor, text="Left Clip")
        lb.pack()

        bt = tk.Scale(
                fr_hor, from_=0, to=1000, orient='horizontal',
                command=lambda x: self.clips_update(index, 2, x)
        )
        bt.pack()

        lb = Label(fr_hor, text="Right Clip")
        lb.pack()
        bt = tk.Scale(
                fr_hor, from_=0, to=1000, orient='horizontal',
                command=lambda x: self.clips_update(index, 3, x)
        )
        bt.pack()

        frame = LabelFrame(wn)
        frame.pack()

        group1 = LabelFrame(frame)
        group1.pack(side='left')
        lb = Label(group1, text="Top Clip")
        lb.pack(side='top')
        bt = tk.Scale(
                group1, from_=0, to=1000, orient='vertical',
                command=lambda x: self.clips_update(index, 0, x)
        )
        bt.pack()

        group2 = LabelFrame(frame)
        group2.pack(side='left')
        lb = Label(group2, text="Bottom Clip")
        lb.pack()
        bt = tk.Scale(
                group2, from_=0, to=1000, orient='vertical',
                command=lambda x: self.clips_update(index, 1, x)
        )
        bt.pack()

        bt = Button(wn, text="Rest clips", command=lambda: self.reset_clips(index))
        bt.pack()

    def clips_update(self, pic_index, clip_index, new_val):
        self.clips[pic_index][clip_index] = new_val

    def reset_clips(self, index):
        self.clips[index] = [0 for i in range(4)]

    def toggle_zoom(self):
        self.zoom = not self.zoom
        print(f"Zoom is now: {self.zoom}")

    def interactive_blending_edit(self):
        wn = tk.Toplevel()
        wn.title("Edit color blending")
        wn.geometry("300x300")

        lb = Label(wn, text="Alpha")
        lb.pack(side='top')
        sc1 = tk.Scale(
                wn, from_=0, to=1000, command=self.change_alpha,
                orient='horizontal')
        sc1.pack()
        sc1.set(int(round(self.alfa * 1000)))

        lb = Label(wn, text="Red")
        lb.pack()
        sc2 = tk.Scale(
                wn, from_=0, to=255, command=lambda x: self.change_color(x, 2),
                orient='horizontal')
        sc2.pack()
        sc2.set(self.blend_color[2])

        lb = Label(wn, text="Green")
        lb.pack()
        sc3 = tk.Scale(
                wn, from_=0, to=255, command=lambda x: self.change_color(x, 1),
                orient='horizontal')
        sc3.pack()
        sc3.set(self.blend_color[1])

        lb = Label(wn, text="Blue")
        lb.pack()
        sc4 = tk.Scale(
                wn, from_=0, to=255, command=lambda x: self.change_color(x, 0),
                orient='horizontal')
        sc4.set(self.blend_color[0])
        sc4.pack()

        bt = Button(wn, text='Reset', command=lambda: reset_color(self))
        bt.pack()

        def reset_color(self):
            self.blend_color = [255, 225, 200]
            self.alfa = 0.23
            self.beta = 1 - self.alfa

            sc1.set(int(self.alfa * 1000))
            sc2.set(self.blend_color[2])
            sc3.set(self.blend_color[1])
            sc4.set(self.blend_color[0])

    def change_color(self, val, index):
        self.blend_color[index] = int(val)

    def change_alpha(self, new_val):
        self.alfa = int(new_val) / 1000
        self.beta = 1 - self.alfa

    def interactive_edit_animation(self, index=0):
        wn = tk.Toplevel()
        wn.title(f"Edit animation of: {index + 1}")
        wn.geometry("300x450")

        lb = Label(wn, text="Start")
        lb.pack(side='top')
        sc = tk.Scale(
                wn, from_=0, to=1000, command=lambda x: self.change_move(x, index, 0),
                orient='horizontal',
        )
        sc.set(np.round(self.moves[index][0] * 1000).astype(int))
        sc.pack(fill='x', expand=True)

        lb = Label(wn, text="Distance")
        lb.pack(side='top')
        sc = tk.Scale(wn, from_=1, to=1000, command=lambda x: self.change_move(x, index, 1),
                      orient='horizontal',
                      )
        sc.set(np.round(self.moves[index][1] * 1000).astype(int))
        sc.pack(fill='x', expand=True)

        lb = Label(wn, text="Window size")
        lb.pack(side='top')
        sc = tk.Scale(wn, from_=2, to=1000, command=lambda x: self.change_move(x, index, 2),
                      orient='horizontal', )
        sc.set(np.round(self.moves[index][2] * 1000).astype(int))
        sc.pack(fill='x', expand=True)

        lb = Label(wn, text="Direction")
        lb.pack(side='top')

        var = tk.IntVar()
        direction = int(self.moves[index][3])
        var.set(direction)

        self.variables_list.append(var)

        r1 = tk.Radiobutton(
                wn, text="Left", value=0, variable=var,
                command=lambda: self.change_direction(0, index)
        )
        r2 = tk.Radiobutton(
                wn, text="Right", value=1, variable=var,
                command=lambda: self.change_direction(1, index)
        )
        r3 = tk.Radiobutton(
                wn, text="Up", value=2, variable=var,
                command=lambda: self.change_direction(2, index)
        )
        r4 = tk.Radiobutton(
                wn, text="Down", value=3, variable=var,
                command=lambda: self.change_direction(3, index)
        )

        r1.pack()
        r2.pack()
        r3.pack()
        r4.pack()

        lb = Label(wn, text="Duration frames")
        lb.pack(side='top')
        sc_dur = tk.Scale(wn, from_=2, to=100, command=lambda x: self.change_duration(x, index),
                          orient='horizontal',
                          )
        sc_dur.set(np.round(self.durations[index]).astype(int))
        sc_dur.pack(fill='x', expand=True)

        bt = Button(wn, text='Reset duration', command=lambda: reset_duration(self, index))
        bt.pack(pady=10)

        def reset_duration(self, index):
            dur = self.default_durations[index]
            self.durations[index] = dur
            sc_dur.set(dur)

    def change_move(self, val, index, val_ind):
        self.moves[index][val_ind] = int(val) / 1000

    def change_duration(self, val, index):
        self.durations[index] = int(val)

    def change_direction(self, val, index):
        # print(f"Changing direction to: {val}  (ind: {index})")
        self.moves[index][3] = int(val)

    def swap_pictures(self):
        txt = tk.simpledialog.askstring("Swap pictures", "What numbers to swap?")
        if txt:
            res = re.findall(u"\d+", txt)
        else:
            return None

        if len(res) >= 2:
            ind1 = int(res[0]) - 1
            ind2 = int(res[1]) - 1

            if 0 <= ind1 < 4 and 0 <= ind2 < 4 and ind1 != ind2:
                print(f"Swapping: {ind1} <> {ind2}")
                self.pics_paths[ind1], self.pics_paths[ind2] = self.pics_paths[ind2], self.pics_paths[
                    ind1]

                self.loaded_pics[ind1], self.loaded_pics[ind2] = self.loaded_pics[ind2], \
                                                                 self.loaded_pics[ind1]


def kwarg(**kw):
    return kw


if __name__ == "__main__":
    gui = ChadGui()

    gui.add_menu([
            ("Load 1", lambda: gui.load_new_picture(0)),
            ("Load 2", lambda: gui.load_new_picture(1)),
            ("Load 3", lambda: gui.load_new_picture(2)),
            ("Load 4", lambda: gui.load_new_picture(3)),
    ], name="Load pictures")

    gui.add_menu([
            ("Clip 1", lambda: gui.interactive_clips_edit(0)),
            ("Clip 2", lambda: gui.interactive_clips_edit(1)),
            ("Clip 3", lambda: gui.interactive_clips_edit(2)),
            ("Clip 4", lambda: gui.interactive_clips_edit(3)),
    ], name="Edit")

    gui.add_menu([
            ("Adjust Blend", gui.interactive_blending_edit),
    ], name="Colors")

    gui.add_menu([
            ("Adjust gif 1", lambda: gui.interactive_edit_animation(0)),
            ("Adjust gif 2", lambda: gui.interactive_edit_animation(1)),
            ("Adjust gif 3", lambda: gui.interactive_edit_animation(2)),
            ("Adjust gif 4", lambda: gui.interactive_edit_animation(3)),
    ], name="Animation")

    gui.add_menu([
            ("Swap pics", gui.swap_pictures),
    ], name="Pics")

    gui.add_menu([
            ("Set Output size", None),
            ("Change Refresh rate", gui.change_refresh),
            ("Toggle zoom", gui.toggle_zoom),
    ], name="Settings")

    main_frame = gui.add_frame(packing=kwarg(fill='both', expand=True))

    # main_frame.grid_rowconfigure(0, weight=1)
    # main_frame.grid_columnconfigure(0, weight=1)
    main_frame.configure(bg='#200')

    fr = gui.add_frame(packing=kwarg(side='bottom', fill='x'), params=kwarg(pady=2))
    array = [
            [
                    (Button, kwarg(text='Save as', command=gui.export_gif)),
                    (Button, kwarg(text='Save', command=gui.save)),
                    (Button, kwarg(text="Save 4-Split", )),
                    (Button, kwarg(text="Quit", command=gui.quit, bg="#933", fg='#FFF')),
            ]
    ]
    refs = gui.make_grid(
            array, parent=fr,
            packing=kwarg(sticky='ew', padx=5, pady=3)
    )

    screen_frame = gui.add_label_frame(
            params=kwarg(height=20), parent=main_frame,
            packing=kwarg(expand=True, fill='both'),
    )
    screen_frame.configure(bg="#66B")

    refs = gui.make_grid(
            [
                    [
                            (gui.create_preview_box, kwarg(text='1')),
                            (gui.create_preview_box, kwarg(text='2'))
                    ],
                    [
                            (gui.create_preview_box, kwarg(text='3')),
                            (gui.create_preview_box, kwarg(text='4'))
                    ]
            ],
            parent=screen_frame, packing=kwarg(sticky='news', ),
    )

    c = ["1", "4", "8", "b"]
    for i, ref in enumerate(refs):
        ref.configure(bg=f"#f{c[i]}f")
        # img = cv2.imread(f"pic ({i + 1}).png")
        # img = gui.squerify_picture(img, 300)[:, :, [2, 1, 0]]
        # tk_pic = gui.numpy_pic_to_tk(img)
        # gui.update_photo(gui.[i], tk_pic)
        # re.configure(width=40)

    gui.start()
    print("END")
