from yasiu_native.time import measure_real_time_decorator
from modules.image_helpers import max_image_size
from modules import modifiers_pipeline
from modules import modifiers_image
from modules import modifiers_sequence
from modules.image_layer import Layer
import tkinter as tk
import tkinter.ttk

import os
import numpy as np
import threading


from tkinter import LabelFrame, Button, Label
from tkinter.messagebox import showwarning, showerror

from modules.gui_builder import GuiBuilder
from tkinter import Frame

from functools import wraps
from typing import Union
import time

from PIL import Image, GifImagePlugin

import multiprocessing as mpc
# print("Importing instances!!!")
from modules.collectors import (
    SequenceModSingleton, PiplineModifiersSingleton, _instances,
    # SequenceModifiers, PipeLineModifiers,
)


# print("Grabbing instances")
SequenceModifiers = SequenceModSingleton()
PipeLineModifiers = PiplineModifiersSingleton()
# print("Grabbed.")

# print("\nImport from: layer")

# print("\nImport from: sequence")

# print("\nImport from: image")

# print("Importing pipeline:")
# print("Imported pipeline")


# PipeLineModifers = SequenceModSingleton()
# SequenceModifiers = PiplineModifiersSingleton()

print("Imported everything.\n" * 5)
print(SequenceModifiers)
print(PipeLineModifiers)
print(_instances.keys())


GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LOADING_STRATEGY.RGB_ALWAYS


def format_time(dur, fmt=">4.1f"):
    """

    Args:
        dur: seconds
        fmt: formatter, default '>4.1f'

    Returns:

    """
    if dur < 1e-3:
        text = f"{dur * 1000000:{fmt}} us"
    elif dur < 1:
        text = f"{dur * 1000:{fmt}} ms"
    else:
        text = f"{dur:{fmt}} s"

    return text


# def measure_time(fun):
#     @wraps(wrapped=fun)
#     def wrapper(*a, **kw):
#         time0 = time.perf_counter()
#         res = fun(*a, **kw)
#         t_end = time.perf_counter()
#         dur = t_end - time0
#         text = format_time(dur)
#         print(f"{fun.__name__} exec time: {text}: ")
#         # f"start sec: {time0 % 60:>2.2f}, "
#         # f"end sec:{t_end % 60:>2.2f}")
#
#         return res
#
#     return wrapper


def windowHashingDecorator(func):
    """
    Hash window and reoopen
    """
    @wraps(func)
    def wrapper(self, layer_key=None, modifierName: str = None):
        if layer_key is None:
            newWindow = func(self)
            cur_hash = self.hashAppWindows("pipe")

        elif modifierName is None:
            newWindow = func(self, layer_key=layer_key)
            cur_hash = self.hashAppWindows(layer_key)

        else:
            newWindow = func(self, layer_key=layer_key,
                             modifierName=modifierName)
            cur_hash = self.hashAppWindows(layer_key, modifierName)

        if cur_hash in self.openWindowsDict:
            curWind = self.openWindowsDict[cur_hash]
            curWind: tk.Toplevel
            if curWind is not None:
                curWind.destroy()
            self.openWindowsDict.pop(cur_hash)

        self.openWindowsDict[cur_hash] = newWindow
    return wrapper


class GifClipApp(GuiBuilder):
    picture_display_size = 300
    # output_size = 600
    # default_durations = [35]
    cycle_steps = 300

    "GUI"
    update_interval = 30

    def __init__(self, height=900, width=1000, allow_load=True):
        super().__init__(height=height, width=width)
        self.root.title("GifClip")

        "Run Flags"
        self.run_single_update = True
        self.running_update = False
        self.running_export = False
        self.running_load = False

        "Project"
        self.pipeline_mods_list = []
        self.layers_dict = {}
        self._next_layer_key = 0
        self.pipeline_steps = []

        "UI Frames"
        self.status_label = LabelFrame()
        self.status_label_time = LabelFrame()
        self.preview_frames_list = []

        "Non persistent"
        self.output_frames = []
        self.last_process_time = 0
        self.active_layer = 0  # "For switching pictures"
        self.display_config = []
        self.playback_position = 0
        self.last_project_path = None
        self.last_export_path = None
        self.exp_settings = {}
        self.last_thread = None
        self.last_folder_used = None
        self.apply_thread = None

        "WindowManager"
        self.openWindowsDict = dict()

        if allow_load:
            self.load_settings()

        self.root.after(1000, self.update_display)

    def _clear_project(self):
        self.last_project_name = None
        self.filters_list = []
        self.mod_pre_list = []
        self.mod_post_list = []
        self.run_single_update = True
        self.running_update = False
        self.exp_settings = {}

    def clear_pipe_mods(self):
        self.pipeline_steps = []
        self.pipeline_mods_list = []
        self.run_single_update = True

    # def clear_layer_mods(self, layer_key):
    #     raise NotImplemented

    def clear_error(self):
        self.run_single_update = True
        self.running_update = False
        self.running_export = False
        self.running_load = False

    @property
    def app_params(self):
        return (
            self.update_interval,
            # self.filters, self.projects,
        )

    @app_params.setter
    def app_params(self, new_params):
        N = len(self.app_params)
        if N == len(new_params):
            (
                self.update_interval,
                # self.filters, self.projects,
            ) = new_params
        else:
            print("Could not load App params, does not match!")

    @property
    def layers_serialize(self):
        return {key: lay.serial_form for key, lay in self.layers_dict.items()}

    @property
    def cur_project(self):
        return self.layers_serialize, self.pipeline_mods_list, self._next_layer_key

    @cur_project.setter
    def cur_project(self, new_val):
        if not isinstance(new_val, list):
            showerror(f"Loaded projects are not list type!")
            ret = self.ask_user_for_confirm(
                "Loading error", "Reset to default?")
            if ret:
                return None
            else:
                raise TypeError(
                    f"Loaded projects are not list type! {type(new_val)}")

        if len(self.cur_project) != len(new_val):
            ret = self.ask_user_for_confirm(
                "Project data invalid", "Reset to default?")
            if ret:
                return None
            else:
                raise ValueError("Invalid project data.")

        else:
            (
                layers_dict,
                self.pipeline_mods_list,
                self._next_layer_key
            ) = new_val

            self._next_layer_key = int(self._next_layer_key)
            # print("Loaded last project:")

            for lay_key_str, lay_args in layers_dict.items():
                lay_key = int(lay_key_str)
                lay = Layer(*lay_args)
                # print(f"Loading layer:{lay}")
                # print(f"ARgs: {lay_args}")
                self.layers_dict[lay_key] = lay

    def create_preview_box(self, parent, *a, text=None, **kw):
        frame = LabelFrame(parent)
        frame.pack(expand=True, fill='both')

        lb = tk.Label(frame, text=text)
        lb.pack()
        lb = tk.Label(frame)
        lb.pack(expand=True, fill='both')
        self.preview_frames_list.append(lb)

        return frame

    def update_display(self):
        # print(f"Single update is: {self.run_single_update}")
        if self.go_quit:
            # self.root.quit()
            print("Last update canceled due to quit command.")
            return None

        self.update_status_box()

        # if self.last_thread:
        #     self.last_thread.close()
        #     self.last_thread.join()

        if self.last_thread is not None and not self.last_thread.is_alive():
            self.last_thread = None

        if self.last_thread is None:
            th = threading.Thread(target=self._update_thread, )
            th.start()
            self.last_thread = th
        else:
            pass
            # print("Waiting for thread")

        self.root.after(self.update_interval, self.update_display)

    def update_status_box(self):
        if self.apply_thread and not self.apply_thread.is_alive():
            print("Clear apply thread -> None")
            self.apply_thread = None

        if self.running_load:
            # print("Setting load")
            self.change_status("loading")
            self.check_if_all_are_loaded()
            return

        elif self.run_single_update and not self.running_update and not self.running_export:
            self.run_single_update = False
            self.running_update = True
            self.change_status("processing")

            if self.apply_thread is None:
                # th = threading.Thread(target=self.apply_all_modifications)
                # th.start()
                # self.apply_thread = th
                pass
            self.apply_all_modifications()
            self.running_update = False
            self.playback_position = 0

        if not self.running_update and not self.running_export:
            self.change_status("ready")

    # @measure_time
    def _update_thread(self, ):
        """
        0 left, 1 right, 2 up, 3 down
        Args:
            index:

        Returns:

        """

        if self.go_quit:
            print("Update canceled (Quit flag).")
            return

        if len(self.display_config) < 1:
            return

        self._read_config_show_pic(
            self.display_config[0], self.preview_frames_list[0])
        self._read_config_show_pic(
            self.display_config[1], self.preview_frames_list[1])

        self.playback_position = (
            self.playback_position + 1)  # % self.cycle_steps

    def _read_config_show_pic(self, config, img_frame):
        var, spin = config
        var_value = int(var.get())
        int_key = int(spin.get())
        """
        1 Pipeline
        2 Layer input
        3 Layer output
        4 Output
        """
        if var_value == 1:
            "Output"
            frames = self.output_frames

        elif var_value == 2:
            "Pipe"

            if 0 <= int_key < len(self.pipeline_steps):
                frames = self.pipeline_steps[int_key]
            else:
                frames = None

        elif var_value == 3:
            "Layer input"
            if int_key not in self.layers_dict:
                frames = None
            else:
                lay = self.layers_dict[int_key]
                frames = lay.orig_frames

        elif var_value == 4:
            "Layer out"
            if int_key not in self.layers_dict:
                frames = None
            else:
                lay = self.layers_dict[int_key]
                frames = lay.output_frames
        else:
            "First and last layer"
            if int_key not in self.layers_dict or len(self.layers_dict[int_key]) < 2:
                return
            else:
                lay = self.layers_dict[int_key]
                frames = lay.output_frames

            stacked = (frames[0] / 2 + frames[-1] / 2).round().astype(np.uint8)
            two_frames = np.concatenate(
                [frames[0], frames[-1], stacked], axis=1)
            # print(two_frames.shape, frames[0].shape)

            out_pic = max_image_size(two_frames, max_width=900, max_height=460)
            out_tk_pic = self.numpy_pic_to_tk(out_pic)
            self.update_photo(img_frame, out_tk_pic)
            return

        if len(frames) > 0:
            # out_pos = int(
            #         np.floor((self.playback_position / self.cycle_steps) * len(frames))
            # )
            out_pos = self.playback_position % len(frames)

            out_pic = frames[out_pos]
            out_pic = max_image_size(out_pic, max_width=700, max_height=460)
            out_tk_pic = self.numpy_pic_to_tk(out_pic)
            self.update_photo(img_frame, out_tk_pic)

        elif frames is None:
            frame = np.zeros((10, 10, 4), dtype=np.uint8)
            out_tk_pic = self.numpy_pic_to_tk(frame)
            self.update_photo(img_frame, out_tk_pic)

    @staticmethod
    def handle_missing_mod():
        pass

    @staticmethod
    def handle_missing_mod():
        pass

    @staticmethod
    def hashAppWindows(layerNum: int, filterName: str = None):
        outHashStr = str(layerNum)
        if str is not None:
            outHashStr += "-" + str(filterName)
        return outHashStr

    @staticmethod
    def handle_error_mod():
        pass

    def check_if_all_are_loaded(self):
        for layer in self.layers_dict.values():
            if layer.is_loading:
                return None
        self.running_load = False
        self.run_single_update = True

    @measure_real_time_decorator
    def apply_all_modifications(self):
        """Applies all mods to all layers and merges sequence"""

        t0 = time.perf_counter()
        for layer in self.layers_dict.values():
            # print(layer, layer.is_loading)
            if layer.is_loading:
                self.running_load = True
                # print(f"Found loading layer: {layer}")
                return None

            if layer.pipeline_updates:
                continue
            # layer_frames = layer.orig_frames
            # layer_frames = self.apply_mods_to_sequence(layer_frames, layer.filters_list)
            # layer.output_frames = layer_frames
            print(f"Layer Application: {layer}")
            layer.apply_mods()

        self.running_load = False

        if self.layers_dict and self.layers_dict[0]:
            # if self.layers_dict[0].output_frames:
            #     output_frames = [fr.copy() for fr in self.layers_dict[0].output_frames]
            # else:
            #     output_frames = [fr.copy() for fr in self.layers_dict[0].orig_frames]

            output_frames = self.layers_dict[0].output_frames

            pipeline_steps = [output_frames]

            for mod_typ, mod, args in self.pipeline_mods_list:
                if mod_typ == "modifier":
                    fn = SequenceModifiers[mod]
                    # print(f"Applying: {fn}, {mod}")
                    output_frames = fn(output_frames, *args)
                    pipeline_steps.append(output_frames)

                elif mod_typ == "pipe":
                    fn = PipeLineModifiers[mod]
                    # print("PipeMod no implement")

                    out = fn(output_frames, self.layers_dict, *args)
                    if out is not None:
                        output_frames = out
                        pipeline_steps.append(output_frames)

                    while self._next_layer_key in self.layers_dict:
                        self._next_layer_key += 1
                else:
                    raise KeyError(f"Invalid mod type: {mod_typ}")

            self.output_frames = output_frames
            self.pipeline_steps = pipeline_steps

        tend = time.perf_counter()
        self.last_process_time = tend - t0
        # self.running_update = False
        # self.playback_position = 0

    def change_picture(self):
        # curr_lay = self.active_layer
        num = self.ask_user_for_integer("Layer", "Which layer to change?")
        print(f"Selected: {num}")
        if num not in self.layers_dict:
            text = f"Selected layer does not exist. Got: {list(self.layers_dict.keys())}"
            showerror("Wrong number", text)
            return None
        else:
            curr_lay = num

        keep_filters = self.layers_dict[curr_lay]
        self.load_any_sequence()
        self.layers_dict[curr_lay].filters_list = keep_filters

    def load_any_sequence(self, path=None, layer_key=None, new_pr=False):
        if path is None:
            last_folder = self.last_folder_used
            path = self.ask_user_for_any_file(last_folder)
            print(f"Loading path: {path}")
            self.last_folder_used = os.path.dirname(path)
            print(f"Set last folder to: {self.last_folder_used}")
            if not path:
                return None

            layer = Layer(path=path)

            if new_pr:
                self._next_layer_key = 0
                self.layers_dict = {}
                self.clear_pipe_mods()

            if layer_key is None:
                layer_key = self._next_layer_key
                self._next_layer_key += 1

            self.run_single_update = True
            self.layers_dict[layer_key] = layer

    def change_refresh(self):
        ret = self.ask_user_for_integer("Refresh rate",
                                        f"Current refresh is {self.update_interval}. What should be new?")
        if isinstance(ret, int):
            if ret < 15:
                showwarning("Warning", "Minimal update time changed to 15ms")
                ret = 15

            self.update_interval = ret

    def load_project(self):

        path = self.ask_user_for_config_file(initdir=self.last_folder_used)
        self.load_settings(path)
        self.last_project_path = None

    def save_project(self):
        if self.last_project_path:
            self.save_settings(self.last_project_path)

        self.save_settings()

    def save_project_as(self):
        path = self.ask_user_for_save_config_file(self.last_folder_used)
        if not path.endswith(self.cfg_extension):
            path += self.cfg_extension

        self.last_project_path = path
        self.save_project()

    @windowHashingDecorator
    def modifier_add_menu(self, layer_key=None):
        """Create windows to spawn modifires"""

        if layer_key is not None:
            lay = self.layers_dict[layer_key]
            modifiers_list = lay.filters_list
            allow_pipe = False
            title = f"Layer {layer_key}"
        else:
            modifiers_list = self.pipeline_mods_list
            allow_pipe = True
            title = "Pipeline"

        window = self._modifier_add(
            modifiers_list, allow_pipe_mods=allow_pipe, title=title)
        return window

    def _modifier_add(self,
                      all_mods_union: Union[list, dict],
                      allow_pipe_mods=False,
                      title=""):
        top = tk.Toplevel()
        top.geometry(f"400x600")
        top.title(f"Adding to: {title}")

        tab_control = tk.ttk.Notebook(top)
        tab_control.pack()

        # filter_list = list(Filters.keys.keys())

        def add_tab(collector_instance, tabname=None):
            tab1 = tk.Frame(tab_control)
            tab_control.add(tab1, text=tabname)
            lb = tk.Listbox(tab1, height=30)
            lb.pack()

            for i, k in enumerate(collector_instance.keys):
                lb.insert(i, k)

            lb.select_set(0)

            def add_selection(is_first=False):
                """Add modifier to active list"""
                mod_name = lb.selection_get()
                def_params = collector_instance.get_default(mod_name)

                mod_data = [collector_instance.type_, mod_name, def_params]

                if is_first:
                    all_mods_union.insert(0, mod_data)
                    self.run_single_update = True
                    return 0
                else:
                    all_mods_union.append(mod_data)
                    self.run_single_update = True
                    return len(all_mods_union) - 1

                # "Depracated mods dict"
                # all_mods_union[mod_name] = def_params
                # self.run_single_update = True
                # return mod_name

            def add_and_config(is_first=False):
                indkey = add_selection(is_first=is_first)

                mod_type, mod_name, params_list = all_mods_union[indkey]
                if mod_type == "pipe":
                    collector_instance = PipeLineModifiers
                else:
                    collector_instance = SequenceModifiers

                self._modifier_config(
                    params_list, mod_name,
                    collector_instance, title)

            def remove_all():
                all_mods_union.clear()
                self.run_single_update = True

            group = Frame(tab1)
            group.pack(side='top')

            bt = tk.Button(group, text="Add first",
                           command=lambda: add_and_config(is_first=True))
            bt.pack(side='left')
            bt = tk.Button(group, text="Add last",
                           command=lambda: add_and_config())
            bt.pack(side='left')
            bt = tk.Button(
                group, text="Clear modifiers in this step", command=remove_all)
            bt.pack(side='left')

        if allow_pipe_mods:
            add_tab(SequenceModifiers, "Filters")
            add_tab(PipeLineModifiers, "Pipe mods")
        else:
            add_tab(SequenceModifiers, "Filters")

        return top

    @windowHashingDecorator
    def modifier_select_menu(self, layer_key=None):
        if layer_key is None:
            return self._modifier_select(self.pipeline_mods_list, "Pipeline")
        else:
            if layer_key not in self.layers_dict:
                showwarning(
                    "Wrong key", f"Wrong layer key to edit: {layer_key}")
                return None
            return self._modifier_select(self.layers_dict[layer_key].filters_list, f"Layer {layer_key}")

    def _modifier_select(self, all_mods_union: Union[list, dict], title=""):
        if len(all_mods_union) <= 0:
            showwarning("Warning", f"No active `{title}` filters.")
            return None

        # if isinstance(storage_list, dict):
        #     was_dict = True
        #     storage_list = [(k, v) for k, v in storage_list.items()]
        # else:
        #     was_dict = False

        window = tk.Toplevel()
        window.geometry(f"400x600")
        window.title(f"Editing: {title}")

        if isinstance(all_mods_union, dict):
            fl_names = [fl[1] for fl in all_mods_union.items()]
        else:
            fl_names = [fl[1] for fl in all_mods_union]

        list_box_group = Frame(window)
        list_box_group.pack(side='top')

        lb = tk.Listbox(list_box_group, height=30)
        lb.pack(side='left')
        button_group = Frame(list_box_group)
        button_group.pack(side='left')

        def change_order(move_up=True):
            if len(all_mods_union) <= 1:
                return None

            ind = lb.curselection()[0]
            dest_ind = ind - 1 if move_up else ind + 1

            higher = max([ind, dest_ind])
            print(f"Selected index: {ind}, up:{move_up}")
            # print(f"Higher: {higher}")
            if not (1 <= higher < len(all_mods_union)):
                return None

            temp_low = all_mods_union[higher - 1]
            temp_hi = all_mods_union[higher]
            all_mods_union[higher] = temp_low
            all_mods_union[higher - 1] = temp_hi

            lb.delete(0, "end")

            for i, (_, name, _) in enumerate(all_mods_union):
                # print(f"Inserting: {name}")
                lb.insert(i, name)
            # print(f"{temp_low[0]} and {temp_hi[0]}")
            # print(f"Temp mod: {temp}")
            # lb.delete(higher - 1, higher - 1)
            # lb.delete(higher - 1, higher - 1)
            # lb.insert(higher - 1, temp_hi[0])
            # lb.insert(higher, temp_low[0])

            lb.select_set(dest_ind)
            self.run_single_update = True

        but_up = Button(button_group, text="Move up",
                        command=lambda: change_order())
        but_up.pack(side='top')
        but_low = Button(button_group, text="Move down",
                         command=lambda: change_order(False))
        but_low.pack(side='bottom')

        for i, k in enumerate(fl_names):
            lb.insert(i, k)

        lb.select_set(0)

        def remove_selection():
            ind = lb.curselection()
            key = lb.selection_get()

            if len(ind) > 0:
                ind = ind[0]

                print(f"Removing({title}) index: {ind}")
                lb.delete(ind, ind)
                if isinstance(all_mods_union, dict):
                    all_mods_union.pop(key)
                else:
                    all_mods_union.pop(ind)
                new_sel = min(int(ind), (len(all_mods_union) - 1))
                self.run_single_update = True
                if new_sel >= 0:
                    lb.select_set(new_sel)
                else:
                    # showwarning("Warning")
                    window.destroy()

            windowTitle = f"Config({title}) : {key}"
            if windowTitle in self.openWindowsDict:
                tempWind = self.openWindowsDict[windowTitle]
                if tempWind is not None:
                    tempWind.destroy()
                    print(f"Closing window '{windowTitle}'")
                self.openWindowsDict.pop(windowTitle)

        def edit_selected_filter():
            ind = lb.curselection()
            name = lb.selection_get()
            if len(ind) > 0:
                ind = ind[0]

                mod_type, name, storage_list = all_mods_union[ind]
                if mod_type == "pipe":
                    collector_instance = PipeLineModifiers
                else:
                    collector_instance = SequenceModifiers
                self._modifier_config(storage_list, name,
                                      collector_instance, title)

        last = Frame(window)
        last.pack()

        bt2 = Button(last, text="Edit this", command=edit_selected_filter)
        bt2.pack(side='left')

        bt2 = Button(last, text="Remove this", command=remove_selection)
        bt2.pack(side='left')

        return window

    def _modifier_config(
            self, current_mod_param_list: list, modifierName: str,
            collector_instance,
            title: str
    ):
        args_description = collector_instance.arguments[modifierName]
        # print(f"Current params: {current_mod_param_list}")

        vars = []
        variables_to_check = []
        window = tk.Toplevel()
        windowTitle = f"Config({title}) : {modifierName}"
        window.title(windowTitle)
        height = 50  # + 30 * len(args_description)

        "Close existing config window"
        if windowTitle in self.openWindowsDict:
            prevWindow = self.openWindowsDict[windowTitle]
            if prevWindow is not None:
                prevWindow.destroy()
        self.openWindowsDict[windowTitle] = window

        for ind, ((ar_type, mmin, mmax, nvars, labels), cur_val) in enumerate(
                zip(args_description, current_mod_param_list)):
            if ar_type is int:
                varObClass = tk.IntVar
                height += 50
            elif ar_type is float:
                varObClass = tk.Variable
                height += 50
            elif ar_type is str:
                varObClass = tk.StringVar
                height += 250
            elif ar_type is bool:
                varObClass = tk.BooleanVar
                height += 50
            else:
                raise ValueError(f"Unsupported config type: {ar_type}")

            inner_vars = []
            box = Frame(window)
            box.pack(side="top")

            if not isinstance(cur_val, (set, list)):
                cur_val = [cur_val]

            if len(labels) != len(cur_val):
                showerror(f"{title} error", "Missing value or label")
                print(cur_val)
                print(labels)
                return None

            for inner_ind, (cur_inner_label, cur_inner_val) in enumerate(zip(labels, cur_val)):
                var_instance = varObClass()
                var_instance.set(cur_inner_val)
                inner_vars.append(var_instance)

                group = tk.Frame(box)
                group.pack(side='left')

                if ar_type is float:
                    ent = tk.Entry(group, textvariable=var_instance)
                    ent.pack(side='bottom')
                    variables_to_check.append(
                        (float, ent, cur_inner_val, mmin, mmax))

                elif ar_type is int:
                    spin = tk.Spinbox(group, from_=mmin,
                                      to=mmax, textvariable=var_instance)
                    spin.pack(side='bottom')
                    variables_to_check.append(
                        (int, spin, cur_inner_val, mmin, mmax))

                elif ar_type is str:
                    curVal = var_instance.get().lower()
                    print(f"VarOb Value before set: {var_instance.get()}")
                    for ind, txt in enumerate(mmax):
                        print(f"Adding radio variable: {txt}")
                        radio_box = tk.Radiobutton(
                            group, text=txt, variable=var_instance, value=txt)
                        print(f"Check: mmin: {mmin}, txt: {txt}, ind: {ind}")
                        print(type(mmin), type(txt))

                        if curVal == txt.lower():
                            radio_box.select()
                        elif ind == 0:
                            radio_box.select()

                        radio_box.pack(side='bottom')

                elif ar_type is bool:
                    bt = tk.Checkbutton(
                        window, text=cur_inner_label, variable=var_instance)
                    bt.pack()

                if ar_type is not bool:
                    label = tk.Label(group, text=cur_inner_label)
                    label.pack()
            vars.append(inner_vars)

        def check_values():
            # print("Checking")
            self.run_single_update = True
            for dtype, widget, df, mmin, mmax in variables_to_check:
                if dtype is float:
                    try:
                        value = float(widget.get())
                        if value < mmin:
                            widget.delete(0, len(widget.get()))
                            widget.insert(0, mmin)
                        elif value > mmax:
                            widget.delete(0, len(widget.get()))
                            widget.insert(0, mmax)

                    except ValueError:
                        # showwarning("Wrong Float", "Invalid float value")
                        widget.delete(0, len(widget.get()))
                        widget.insert(0, df)
                elif dtype is int:
                    try:
                        value = int(widget.get())
                        if value < mmin:
                            widget.delete(0, len(widget.get()))
                            widget.insert(0, mmin)
                        elif value > mmax:
                            widget.delete(0, len(widget.get()))
                            widget.insert(0, mmax)

                    except ValueError:
                        # showwarning("Wrong Float", "Invalid float value")
                        widget.delete(0, len(widget.get()))
                        widget.insert(0, df)

        def confirm():
            check_values()
            for ind, cur_descrip in enumerate(args_description):
                ar_type, mmin, mmax, nvars, labels = cur_descrip
                variable = vars[ind]

                if ar_type is int:
                    conv_fun = int
                elif ar_type is float:
                    conv_fun = float
                elif ar_type is str:
                    def conv_fun(x): return x
                elif ar_type is bool:
                    conv_fun = bool
                else:
                    raise TypeError(
                        f"Types is unsupported now (797): {ar_type}")

                if nvars == 1:
                    val = variable[0].get()
                    # print(f"Converting: {val}")
                    new_value = conv_fun(val)
                else:
                    new_value = [0] * nvars
                    for inner_ind, var in enumerate(variable):
                        val = var.get()
                        val = conv_fun(val)
                        new_value[inner_ind] = val

                # print(f"Assign new value: {new_value}")

                current_mod_param_list[ind] = new_value

        def confirm_close():
            confirm()
            window.destroy()

        last = Frame(window)
        last.pack()
        # bt1 = Button(last, text="Check values", command=check_values)
        # bt1.pack(side='left')
        bt2 = Button(last, text="Apply", command=confirm)
        bt2.pack(side='left')
        bt2 = Button(last, text="Save & Close", command=confirm_close)
        bt2.pack()

        window.geometry(f"400x{height}")

    def quit(self):
        if not self.running_load:
            self.save_settings()

        print("Setting quit flag.")
        self.go_quit = True

        if self.last_thread:
            print("Waiting for thread to join")
            self.last_thread.join()
            # self.last_thread.kil()
            print("Thread joined")
            # print("Thread ignored")

        time.sleep(1)
        print("Sleep ended. root.quit()")

        self.root.quit()
        print("Root has quit.")
        self.root.destroy()
        print("Root has been destroyed.")

    def make_preview_switch(self, parent, label=None):
        """
        Preview Layer input
        Preview Layer output
        Preview Pipe step
        Preview Output
        """

        fr = tk.Frame(parent, bg="#0F0")
        fr.pack()

        lab1 = tk.Label(fr, text=label)
        lab1.pack()

        varIntMode = tk.IntVar()
        varIntMode.set(1)

        # self.variables_list.append(var)

        # def toggle_spin():
        #     val = int(var.get())
        #     if val == 4:
        #         "Gray out"
        #         # sp.disable()
        #         # sp.state="readonly"
        #         print("Disabling")
        #     else:
        #         "Enable"
        #         print("Enabling")
        #         # sp.state

        for num, text in enumerate(
                ["Output", "Pipeline step", 'Layer input',
                    'Layer output', 'First/Last frame'],
                1
        ):
            rad1 = tk.Radiobutton(
                fr, variable=varIntMode, text=text, value=num,
                # validatecommand=validate_spin,
                # command=toggle_spin,
            )
            rad1.config()
            rad1.pack(side='top', anchor='c')

        spinLayerNum = tk.Spinbox(
            fr, from_=0, to=100,
            # validate='all',
            # validatecommand=validate_spin,
            # invalidcommand=invalid_cmd,
        )
        spinLayerNum.pack()
        # sp.config(state="readonly")
        # sp.hide()
        # sp.setButtonSymbols

        self.display_config.append([varIntMode, spinLayerNum])

        group_but = Frame(fr)
        group_but.pack()

        def check_preview_mode():
            mode = int(varIntMode.get())
            if mode in [1, 2, ]:
                print("Selected mode is pipeline.")
                return None
            else:
                key = int(spinLayerNum.get())
                print(f"Selected mode is layer: {key}")
                return key

        bt_add = Button(
            group_but, text="Add",
            command=lambda: self.modifier_add_menu(check_preview_mode())
        )
        bt_add.pack(side='left')

        bt_edit = Button(
            group_but, text="Edit",
            command=lambda: self.modifier_select_menu(check_preview_mode())
        )
        bt_edit.pack()

        # rad1.select()

        return fr

    def change_status(self, up_type="processing"):
        up_type = up_type.lower()
        if up_type == "processing":
            self.status_label.config(text="Processing", bg="#811", fg="#DDD")

        elif up_type == "ready":
            self.status_label.config(text="Ready", bg="#AFA", fg="#000")
            self.status_label_time.config(
                text=f"Last process time: {format_time(self.last_process_time)}")

        elif up_type == "exporting":
            print("status export:")
            self.status_label.config(text="Exporting...", bg="#36C", fg="#DDD")

        elif up_type == "loading":
            self.status_label.config(
                text="Some layers are loading...", bg="#36C", fg="#DDD")
            # print("status export:")

        else:
            print(f"VALUE ERRRO : {up_type}")
            raise ValueError(f"Key does not match update option: {up_type}")

    def export_as(self):
        # print(f"Export path: folder: {self.last_folder_used}")
        new_path = tk.filedialog.asksaveasfilename(
            filetypes=self.GIF_TYPES,
            initialdir=self.last_folder_used,
        )
        if not (new_path.endswith("gif") or new_path.endswith("GIF")):
            new_path += ".gif"

        print(f"Updated export path: {new_path}")
        self.last_export_path = new_path

        top = tk.Toplevel()
        top.geometry("250x100")
        top.title("Export settings")

        layerIndex = tk.IntVar()
        layerIndex.set(max(self.layers_dict.keys()))
        layerSpinBox = tk.Spinbox(top, from_=0, to=15, textvariable=layerIndex, increment=1)
        layerSpinBox.pack()

        duration_var = tk.IntVar()
        duration_var.set(45)
        rgba_mode_var = tk.BooleanVar()
        rgba_mode_var.set(False)

        settings_grid = tk.Frame(top)
        settings_grid.pack(side='top')

        array = (
            [
                (tk.Label, dict(text="Duration [ms]")),
                (tk.Spinbox, dict(from_=10, to=150, textvariable=duration_var, increment=5)),
            ],
            [
                (tk.Label, dict(text="Transparent")),
                (tk.Checkbutton, dict(variable=rgba_mode_var)),
            ],
        )
        self.make_grid(array, parent=settings_grid)

        # spin = tk.Spinbox(settings_grid, from_=10, to=150, textvariable=duration_var)
        # spin.grid(row=1, column=2)
        # lab = tk.Label(set)

        def set_and_export():
            dec = {}
            dec['use_rgba'] = bool(int(rgba_mode_var.get()))
            dec['duration'] = int(duration_var.get())
            dec['layer'] = int(layerIndex.get())
            self.exp_settings = dec
            self.export()

        but = tk.Button(top, text="Export", command=set_and_export)
        but.pack()

    def export(self):
        if self.running_update or self.run_single_update:
            showerror("Update is running",
                      "Export or filters are running. Wait for end.")
            return

        if self.last_export_path:
            self.running_export = True
            self.change_status("exporting")
            th = threading.Thread(target=self._export_thread)
            th.start()

    @measure_real_time_decorator
    def _export_thread(self):
        t0 = time.time()
        path = self.last_export_path

        exp_settings = self.exp_settings
        print(f"Export settings: {exp_settings}")

        # loop = exp_settings['loop']
        duration = exp_settings['duration']
        layerInd = exp_settings['layer']
        # disposal = exp_settings['disposal']
        use_rgba = exp_settings['use_rgba']

        # frames = self.output_frames.copy()
        print(f"Layers ({layerInd}): {self.layers_dict.keys()}")
        layer = self.layers_dict[layerInd]
        print(layer)
        layer: Layer
        frames = layer.output_frames

        if use_rgba:
            pil_frames = [Image.fromarray(fr).convert("RGBA") for fr in frames]

            for pil_fr, fr in zip(pil_frames, frames):
                # fr = fr * 0 + 255
                alpha_pil = Image.fromarray(fr[:, :, 3])
                pil_fr.putalpha(alpha_pil)

        else:
            pil_frames = [Image.fromarray(fr).convert("RGB") for fr in frames]

        pil_frames[0].save(
            path, save_all=True, append_images=pil_frames[1:],
            optimize=False, loop=0,
            # background=(0, 0, 0, 255),
            quality=100, duration=duration,
            disposal=2,
        )
        print(f"Saved gif to: {path}. Frames: {len(pil_frames)}")
        tend = time.time() - t0
        self.last_process_time = tend
        self.running_export = False


# def kwargify(**kw):
#     return kw


@measure_real_time_decorator
def build_GifGui():
    gui = GifClipApp(allow_load=True)
    # gui = GifClipApp(allow_load=False)

    gui.add_menu([
        ("New project", lambda: gui.load_any_sequence(new_pr=True)),
        ("Load project", gui.load_project),
        ("Save project", gui.save_project),
        ("Save project as", gui.save_project_as),
    ], name="Project")
    gui.add_menu([
        ("Export Gif", gui.export),
        ("Export Gif as", gui.export_as),
    ], name="Export")

    gui.add_menu([
        ("New layer", gui.load_any_sequence),
        ("Change pic in layer", gui.change_picture),
        # ("Layers", None),
    ], name="Layer")

    gui.add_menu([
        ("Force new render", gui.clear_error),
    ], name="Error")

    gui.add_menu([
        # ("Add sequence mod", gui.modifier_add_menu),
        # ("Edit sequence mods", lambda: gui.modifier_select("sequence")),
        # "separator",
        # ("Add layer filter", lambda: gui.modifier_add_menu("0")),
        # ("Edit layer filter", lambda: gui.modifier_select("filter")),
        "separator",
        # ("Remove layer mods", gui.clear_layer_mods),
        ("Remove pipeline mods", gui.clear_pipe_mods),
    ], name="Filters")
    #
    # gui.add_menu([
    #         ("Add new modifier", gui.add_filter),
    #         ("Edit modifier", gui.select_filter),
    #         ("Remove all except sampler", gui.filters_list.clear),
    # ], name="Sequence")

    gui.add_menu([
        # ("Set Output size", None),
        ("Change Refresh rate", gui.change_refresh),
        # ("Toggle zoom", gui.toggle_zoom),
    ], name="Settings")

    main_frame = gui.add_frame(packing=dict(fill='both', expand=True))

    main_frame.configure(bg='#200')

    fr = gui.add_frame(packing=dict(
        side='bottom', fill='x'), params=dict(pady=2))
    array = [
            [
                # (Button, kwargify(text='Export', command=gui.export_gif)),
                (Label, dict(text="Label", bg="#222")),
                (Label, dict(text="ProcessTime", bg="#252", fg="#EFD")),
                (Button, dict(text="Quit", command=gui.quit, bg="#933", fg='#FFF')),
            ]
    ]

    _, refs = gui.make_grid(
        array, parent=fr,
        packing=dict(sticky='ew', padx=5, pady=3)
    )
    gui.status_label = refs[0][0]
    gui.status_label_time = refs[0][1]

    screen_frame = gui.add_label_frame(
        params=dict(height=20), parent=main_frame,
        packing=dict(expand=True, fill='both'),
    )
    screen_frame.configure(bg="#66B")

    frames, refs = gui.make_grid(
        [
            [
                (gui.make_preview_switch, dict(label='Edit Top')),
                (gui.create_preview_box, dict(text='Edit View')),
            ],
            [
                (gui.make_preview_switch, dict(label='Edit Lower')),
                (gui.create_preview_box, dict(text='Preview'))
            ]
        ],
        parent=screen_frame, packing=dict(sticky='news', ),
    )
    screen_frame.columnconfigure(2, weight=5)

    c = ["1", "4", "8", "b"]
    frames = frames[0] + frames[1]
    for i, ref in enumerate(frames):
        ref.configure(bg=f"#f{c[i]}f")
        # ref.configure(width=40)

    return gui


if __name__ == "__main__":
    gui = build_GifGui()
    gui.start()

    print("END")
