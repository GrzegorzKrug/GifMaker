import os
import traceback
from tkinter.messagebox import showerror, showwarning

import cv2

from .image_modifiers import SequenceModifiers
from .image_readers import read_gif, read_webp


class Layer:
    def __init__(self, path=None, filters_list=None, pipe_update=False):
        self.orig_frames = []
        self.output_frames = []

        self.source_file_path = None
        self.pipeline_updates = pipe_update

        if path:
            self.load_file(path)
            print(self.orig_frames[0].shape)

        if filters_list is None:
            self.filters_list = []
        else:
            assert type(filters_list) is list, "Filters list is not list"
            self.filters_list = filters_list

    @property
    def serial_form(self):
        return self.source_file_path, self.filters_list, self.pipeline_updates

    def load_file(self, path):
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
                # pic = cv2.cvtColor(pic, cv2.COLOR_BGRA2RGBA)
                # cv2.imshow("zd", pic[:, :, 3])
                # cv2.waitKey()

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
                frames.append(frame[:, :, [2, 1, 0]])

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
        self.clip_arr = [0, 0, 0, 0]

        if new_pr:
            self.last_project_name = None
            self.filters_list = []
            self.mod_pre_list = []
            self.mod_post_list = []
            self.run_single_update = True
            self.running_update = False

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
