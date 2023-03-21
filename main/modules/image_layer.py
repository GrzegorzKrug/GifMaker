import cv2
import os
import numpy as np
import traceback
import threading

from tkinter.messagebox import showerror, showwarning

from main.modules.collectors import SequenceModifiers
from .image_readers import read_gif, read_webp
from yasiu_native.time import measure_real_time_decorator


# SequenceModifiers = SequenceModSingleton()


class Layer:
    def __init__(self, path=None, filters_list=None, pipe_update=False):
        self.orig_frames = []
        self.output_frames = []

        self.source_file_path = None
        self.pipeline_updates = pipe_update
        self.has_changes = True

        if path:
            self.is_loading = True
            self.load_file(path)
            # self.loading = False
            # print(self.orig_frames[0].shape)
        else:
            self.is_loading = False

        if filters_list is None:
            self.filters_list = []
        else:
            assert type(filters_list) is list, "Filters list is not list"
            self.filters_list = filters_list

    def __repr__(self):
        return f"{self.__class__}: filters: `{self.source_file_path}`, loading:{self.is_loading}"

    @property
    def serial_form(self):
        return self.source_file_path, self.filters_list, self.pipeline_updates

    def clear_layer(self):
        self.filters_list = []
        self.has_changes = True
        self.output_frames = []

    def load_file(self, path):
        th = threading.Thread(target=self._load_thread, args=(path,))
        th.start()
        # self._load_thread(path)

    @measure_real_time_decorator
    def _load_thread(self, path):
        path = os.path.abspath(path)
        *_, ext = path.split('.')
        ext = ext.lower()
        # print(f"Loaded: {ext} ({path})")

        if ext in ['jpg', 'jpeg', 'bmp', 'png']:
            pic = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if pic.shape[2] == 3:
                print("Picture has no alpha")
                pic = pic[:, :, [2, 1, 0]]
            else:
                print("Picture has alpha channel")
                pic = pic[:, :, [2, 1, 0, 3]]

            self.orig_frames = [pic]
            # self.clip_arr = [0, 0, 0, 0]

        elif ext == 'mp4':
            cap = cv2.VideoCapture(path)
            frames = []
            ret = True
            while ret:
                ret, frame = cap.read()
                # if ret is not None:
                #     frames.append(frame)
                if not ret:
                    break

                frame = frame.astype(np.uint8)
                frames.append(frame[:, :, [2, 1, 0]])

            # print(f"Frames: {len(frames)}")
            self.orig_frames = frames

        elif ext == 'webp':
            sequence = read_webp(path)
            self.orig_frames = sequence

        elif ext == 'gif':
            sequence = read_gif(path)
            # sequence = ndimage.imread(path)
            self.orig_frames = sequence

        else:
            showerror("Error!", f"Unsupported file extension: {ext}")
            return None

        self.source_file_path = path
        self.is_loading = False

    def load_rgb_pic(self, ph=None, new_pr=False):
        raise NotImplemented
        if ph is None:
            ph = self.ask_user_for_picture()
            if not ph:
                return None
        ph = os.path.abspath(ph)

        self.source_file_path = ph
        pic = cv2.imread(ph)
        pic = pic[:, :, [2, 1, 0]]
        self.orig_frames = [pic]

        if new_pr:
            self.clear_layer()

    def apply_mods(self):
        if not self.filters_list:
            self.output_frames = self.orig_frames
            return None

        output_frames = [fr.copy() for fr in self.orig_frames]

        for _, fil_name, arg in self.filters_list.copy():
            fun = SequenceModifiers[fil_name]
            if fun is None:
                # self.handle_missing_mod()
                showwarning("Warning!", f"Not found filter function: {fil_name}")
                print(f"Not found pre filter function: {fil_name}")
                continue

            # for fri, fr in enumerate(self.preprocessed_frames):
            try:
                print(f"Applying {fil_name}")
                output_frames = fun(output_frames, *arg)
                # output_frames[fri] = new_fr
            except Exception as err:
                msg = f"Error in function ({fil_name})"
                showerror(f"{msg}", f"{msg}: {err}")
                print(f"{msg}: {err}")
                traceback.print_tb(err.__traceback__)

        shape = output_frames[0].shape

        for fr in output_frames:
            if shape != fr.shape:
                print(f"Incorrect shape: {fr.shape} != {shape}. Filters: {self.filters_list}")
                showwarning("Error", "Sequence changes image dimension!")
                # return sequence
        self.output_frames = output_frames
