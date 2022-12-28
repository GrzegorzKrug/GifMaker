import json
import sys
import os
import tkinter as tk
from collections import deque
from tkinter import Button

import cv2
import imutils
from PIL import Image, ImageTk
from functools import wraps
from abc import ABC, abstractmethod, abstractproperty

from tkinter.messagebox import askyesno, askquestion

import tkinter.filedialog

import time


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


class GuiBuilder(ABC):
    conf_path = "cgui.conf"
    cfg_extension = ".settings.cfg"

    GIF_TYPES = ((
            ('Gif', "*.gif"),
    ))
    CONFIG_TYPES = ((
            ('config', "*.settings.cfg"),
    ))
    PIC_TYPES = ((
            ("Png Picture", "*.png"),
            ("JPG Picture", "*.jpg"),
            ("JPEG Picture", "*.jpeg"),
            ("BMP Picture", "*.bmp"),
    ))
    ANY_TYPE = ((
            ("*", "*")
    ))

    quit_delay_sec = 0.25

    def __init__(self, height=800, width=700):
        self.variables_list = []
        self.widgets_list = []

        self.root = tk.Tk()
        self.root.geometry(f"{width}x{height}+{100}+{20}")

        self.menu = tk.Menu(self.root, )
        self.root.configure(menu=self.menu)
        self.root.title("Application")
        self.go_quit = False

    # @property
    # def projects(self):
    #     return self._projects

    # @projects.setter
    # def projects(self, new_val):
    #     if not isinstance(new_val, dict):
    #         raise TypeError(f"Project settings should be dict type, not '{type(new_val)}'")
    #     self._projects = new_val

    @abstractproperty
    def cur_project(self):
        raise NotImplemented("ABC Property")

    @cur_project.setter
    @abstractmethod
    def cur_project(self, new_val):
        raise NotImplemented("ABC Property")

    @parent_replacer
    @params_replacer
    @packing_replacer
    def add_button(self, text=None, parent=None, packing=None):
        but = Button(parent, text=text)
        but.pack(**packing)

    @parent_replacer
    @params_replacer
    @packing_replacer
    def add_frame(self, parent, params=None, packing=None):
        fr = tk.Frame(parent, **params)
        fr.pack(**packing)
        return fr

    @parent_replacer
    @params_replacer
    @packing_replacer
    def add_label_frame(self, parent, params=None, packing=None):
        fr = tk.LabelFrame(parent, **params)
        fr.pack(**packing)
        return fr

    @parent_replacer
    def add_radio_buttons(self):
        raise NotImplemented

    def add_variable(self, var):
        self.variables_list.append(var)

    @parent_replacer
    @params_replacer
    @packing_replacer
    def make_grid_repeat_objects(self, ob, rows, columns, parent=None, params=None, packing=None):
        assert rows >= 1
        assert columns >= 1

        array = []

        for row in range(rows):
            cur_row = []
            for c in range(columns):
                cur_row.append((ob, params))
            array.append(cur_row)

        return self.make_grid(array, parent=parent, packing=packing)

    @parent_replacer
    @packing_replacer
    def make_grid(self, element_array, parent=None, packing=None):
        """

        Args:
            element_array:
            parent:

        Returns:
            2d list : containing ref to each grid frame
            2d list : of function return

        """
        assert len(element_array) > 0, "Must be 1d+"
        assert len(element_array[0]) > 0, "Must be 2d+"
        assert len(element_array[0][0]) > 0, "Must be 3d, 2d position and params"
        frames = []
        refs = []

        for ri, row in enumerate(element_array, 1):
            refs.append([])
            frames.append([])

            for ci, ob_tuple in enumerate(row, 1):
                ob = ob_tuple[0]
                if isinstance(ob_tuple[1], dict):
                    arg = []
                    kw = ob_tuple[1]
                else:
                    arg = ob_tuple[1]
                    if len(ob_tuple) == 3:
                        kw = ob_tuple[2]
                    else:
                        kw = {}

                fr = tk.Frame(parent)
                ref = ob(fr, *arg, **kw)

                if ob in [tk.Button, tk.Frame, tk.Label, tk.LabelFrame, tk.Listbox, tk.Spinbox,tk.Checkbutton]:
                    ref.pack(fill='both')

                # print(f"Packing: {ob}")

                fr.grid(row=ri, column=ci, **packing)
                parent.grid_columnconfigure(ci, weight=1)

                frames[ri - 1].append(fr)
                refs[ri - 1].append(ref)

            parent.grid_rowconfigure(ri, weight=1)

        return frames, refs

    def modify_grid(self, parent, el, row, column, params={}):
        ob = el(parent, **params)
        ob.grid(row=row, column=column)
        return ob

    def start(self):
        self.root.mainloop()

    def update_photo(self, box, photo):
        """

        Args:
            box: - Label widget
            photo: image_tk

        Returns:

        """
        if photo is None:
            box.configure(image="")
        else:
            box.configure(image=photo, width=30, height=30)
            box.image = photo

    @staticmethod
    def pil_img_to_tk(pil_img):
        return ImageTk.PhotoImage(pil_img)

    def numpy_pic_to_tk(self, array):
        pil_img = Image.fromarray(array)
        return self.pil_img_to_tk(pil_img)

    @staticmethod
    def scale_outer(pic, dim):
        h, w, c = pic.shape

        if h > w:
            im = imutils.resize(pic, height=dim)
        else:
            im = imutils.resize(pic, width=dim)
        return im

    @staticmethod
    def stretch_to_square(pic, dim):
        """Squerify pic"""
        im = cv2.resize(pic, (dim, dim))
        return im

    @staticmethod
    def square_crop_center(pic):
        h, w, _ = pic.shape
        half_h = h // 2
        half_w = w // 2

        if h < w:
            ind1 = half_w - half_h
            ind2 = ind1 + h
            return pic[:, ind1:ind2]
        else:
            ind1 = half_h - half_w
            ind2 = ind1 + w
            return pic[ind1:ind2, :]

    def ask_user_for_picture(self):
        ret = tk.filedialog.askopenfilename(
                filetypes=self.PIC_TYPES,
                initialdir=os.path.dirname(__file__)
        )
        name = ret.lower()
        if name.endswith(".png") \
                or name.endswith(".jpg") \
                or name.endswith(".jpeg") \
                or name.endswith('.bmp'):
            return ret
        else:
            print("Invalid file")

    @staticmethod
    def ask_user_for_any_file():
        ret = tk.filedialog.askopenfilename(
                # filetypes=self.ANY_TYPE,
                initialdir=os.path.dirname(__file__)
        )
        return ret

    @staticmethod
    def ask_user_for_integer(title, question):
        ret = tk.simpledialog.askinteger(title, question)
        return ret

    def ask_user_for_config_file(self):
        ret = tk.filedialog.askopenfilename(
                filetypes=self.CONFIG_TYPES,
                initialdir=os.path.dirname(__file__)
        )
        return ret

    def ask_user_for_save_config_file(self):
        ret = tk.filedialog.asksaveasfilename(
                filetypes=self.CONFIG_TYPES,
                initialdir=os.path.dirname(__file__)
        )
        return ret

    @staticmethod
    def ask_user_for_string(title, question):
        ret = tk.simpledialog.askstring(title, question)
        return ret

    @staticmethod
    def ask_user_for_confirm(title, question, extra=None):
        # ret = tk.simpledialog.askquestion(title,question)
        if extra:
            ret = askyesno()
        else:
            ret = askyesno(title, question)
        return ret

    @parent_replacer
    @params_replacer
    @packing_replacer
    def add_listbox(self, parent=None, params=None, packing=None):
        menu = tk.Listbox(parent, **params)
        menu.pack(**packing)
        return menu

    @parent_replacer
    @params_replacer
    def add_menu(self, commands, name=None, parent=None, params=None, ):
        # menu = tk.Menu(parent, **params)
        # menu = self.menu
        menu = tk.Menu(self.menu)

        for cmd in commands:
            if cmd == "separator":
                menu.add_separator()
            else:
                label, act = cmd
                menu.add_command(label=label, command=act)

        # self.root.configure(menu=menu)
        self.menu.add_cascade(label=name, menu=menu)
        return menu

    @property
    def settings_path(self):
        return os.path.abspath(f"{self.name}{self.cfg_extension}")

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def app_params(self):
        raise NotImplemented("Meta param")

    @app_params.setter
    def app_params(self, new_params):
        raise NotImplemented("Meta param")

    def load_settings(self, path=None):
        if path is None:
            path = self.settings_path

        if not os.path.isfile(path):
            print(f"Not found config file: {path}")
            return None
        else:
            with open(path, "rt") as fp:
                try:
                    f = json.load(fp)
                    params, last_project = f
                    print(f"Loaded settings - {path}")

                except json.JSONDecodeError as err:
                    print(f"Not loaded, Json error: {err}")
                    ret = self.ask_user_for_confirm("Loading error", "Reset to default?")
                    if ret:
                        return None
                    sys.exit(-1)

                except ValueError as err:
                    print(f"Not loaded, Valuer error: {err}")
                    ret = self.ask_user_for_confirm("Loading error", "Reset to default?")
                    if ret:
                        return None
                    sys.exit(-1)

                self.app_params = params
                self.cur_project = last_project

    def save_settings(self, path=None):
        if path is None:
            path = self.settings_path

        with open(path, "wt") as fp:
            js = json.dumps((self.app_params, self.cur_project), indent=2)
            fp.write(js)
            print(f"Saved settings - {path}")

    def quit(self):
        self.save_settings()
        self.go_quit = True

        self.root.after(int(self.quit_delay_sec * 1000), self._quit)

    def _quit(self):
        self.root.quit()
