import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import glob
import pickle
import numpy as np
import cv2
import os
from paprle.utils.config_utils import change_working_directory
change_working_directory()

class DataProcessGUI:
    def __init__(self, root, data_list):
        self.root = root
        self.root.title("Data Processing")
        # size
        width, height = 1280, 480
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(width, height)

        self.episodes = data_list
        self.episode_dict = {episode.split("/")[-1]: episode for episode in self.episodes}
        self.data_dict, self.episode_info_dict = {}, {}
        self.changed_status = {}
        for name, dir_path in self.episode_dict.items():
            data_list = glob.glob(dir_path + "/data*.pkl")
            if len(data_list) > 0:
                self.data_dict[name] = data_list
                episode_info_file = dir_path + '/episode_info.pkl'
                if os.path.exists(episode_info_file):
                    self.episode_info_dict[name] = pickle.load(open(episode_info_file, 'rb'))
                self.changed_status[name] = False

        self.current_episode = None
        self.current_index = 0
        self.start_timestep = None
        self.end_timestep = None

        episode_button_frame = tk.Frame(root)
        episode_button_frame.grid(row=0, column=0, padx=5, pady=10, sticky="w")
        self.episode_var = tk.StringVar()
        self.episode_dropdown = ttk.Combobox(episode_button_frame, textvariable=self.episode_var, state="readonly")
        self.episode_dropdown.bind("<<ComboboxSelected>>", self.load_episode)
        self.episode_dropdown.pack(side="left", padx=2)
        self.episode_dropdown['values'] = list(self.data_dict.keys())
        self.episode_var.set("Select Episode")

        self.prev_button = tk.Button(episode_button_frame, text="Previous", command=self.load_prev_episode)
        self.prev_button.pack(side="left", padx=2)
        self.next_button = tk.Button(episode_button_frame, text="Next", command=self.load_next_episode)
        self.next_button.pack(side="left", padx=2)
        # self.next_unprocessed_button = tk.Button(episode_button_frame, text="Next Unprocessed", command=self.load_next_unprocessed_episode)
        # self.next_unprocessed_button.pack(side="left", padx=2)



        button_frame = tk.Frame(root)
        button_frame.grid(row=0, column=1, padx=5, pady=10, sticky="w")


        self.success_button = tk.Button(button_frame, text="Success", command=self.set_success)
        self.success_button.pack(side="left", padx=2)
        self.success_button.bind("<space>", lambda event: "break")

        self.fail_button = tk.Button(button_frame, text="Fail", command=self.set_fail)
        self.fail_button.pack(side="left", padx=2)
        self.fail_button.bind("<space>", lambda event: "break")

        self.invalid_button = tk.Button(button_frame, text="Invalid", command=self.set_invalid)
        self.invalid_button.pack(side="left", padx=2)
        self.invalid_button.bind("<space>", lambda event: "break")

        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset_all)
        self.reset_button.pack(side="left", padx=2)
        self.reset_button.bind("<space>", lambda event: "break")



        button_frame2 = tk.Frame(root)
        button_frame2.grid(row=0, column=2, padx=5, pady=10, sticky="e")

        self.play_button = tk.Button(button_frame2, text="Play", command=self.play_episode)
        self.play_button.pack(side="right", padx=2)
        self.play_button.bind("<space>", lambda event: "break")

        self.stop_button = tk.Button(button_frame2, text="Stop", command=self.stop_episode)
        self.stop_button.pack(side="right", padx=2)
        self.stop_button.bind("<space>", lambda event: "break")


        self.save_button = tk.Button(button_frame2, text="Save", command=self.save_episode)
        self.save_button.pack(side="right", padx=2)
        self.save_button.bind("<space>", lambda event: "break")


        # Image display
        self.image_label = tk.Label(root)
        self.image_label.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

        # Slider for image navigation
        new_length = self.root.winfo_width() - 40
        self.slider = tk.Scale(root, from_=0, to=0, orient="horizontal", command=self.update_image, length=new_length)
        self.slider.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="w")

        self.cut_slider = tk.Canvas(self.root, width=new_length, height=10, background='lavender', highlightthickness=0)
        self.cut_slider.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        self.color_region = self.cut_slider.create_rectangle(50, 5, 70, 25, fill="red", outline="")
        self.start_timestep_text = self.cut_slider.create_text(50, 30, text="Start")
        self.end_timestep_text = self.cut_slider.create_text(70, 30, text="End")


        # Buttons for trimming
        self.start_button = tk.Button(root, text="Set Start Timestep", command=self.set_start_timestep)
        self.start_button.grid(row=4, column=0, padx=10, pady=10)
        self.start_button.bind("<space>", lambda event: "break")

        self.end_button = tk.Button(root, text="Set End Timestep", command=self.set_end_timestep)
        self.end_button.grid(row=4, column=2, padx=10, pady=10)
        self.end_button.bind("<space>", lambda event: "break")


        # Text display for information
        self.info_label = tk.Label(root, text="Select an episode to begin", font=("Arial", 12), wraplength=400)
        self.info_label.grid(row=4, column=1, padx=10, pady=10)

        self.root.bind("<Configure>", self.resize)
        self.root.bind("<KeyPress>", self.on_key_press)

        self.playing = False




    @property
    def window_width(self):
        return max(self.root.winfo_width(), 1280)

    @property
    def window_height(self):
        return max(self.root.winfo_height(), 720)

    def resize(self, event):

        if self.current_episode:
            self.curr_color_im = self.resize_image(self.orig_color_im)
            img_tk = ImageTk.PhotoImage(Image.fromarray(self.curr_color_im))
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk

        new_length = self.root.winfo_width() - 40  # Subtract some padding
        self.slider.configure(length=new_length)  # Ensure minimum length

        if self.current_episode:
            self.cut_slider.configure(width=new_length, height=10, background='lavender', highlightthickness=0)
            #self.cut_slider.place(x=self.slider.winfo_x(), y=self.slider.winfo_y() + 25)
            length_ppx = self.slider.winfo_width()
            start_x = self.curr_episode_info['trim_info'][0]/len(self.data_list) *  length_ppx #+ self.slider.winfo_x()
            end_x = self.curr_episode_info['trim_info'][1]/len(self.data_list) * length_ppx #+ self.slider.winfo_x()
            self.cut_slider.coords(self.color_region,start_x, 5,end_x, 25)

    def load_episode(self, event=None):
        self.current_episode = self.episode_dict[self.episode_var.get()]
        if self.current_episode:

            self.video_ims = []
            self.data_list = sorted(self.data_dict[self.episode_var.get()])
            self.curr_episode_info_file = self.episode_dict[self.episode_var.get()] + '/episode_info.pkl'
            self.curr_episode_info = pickle.load(open(self.curr_episode_info_file, 'rb'))
            self.episode_info_dict[self.episode_var.get()] = self.curr_episode_info
            if 'success' not in self.curr_episode_info:
                self.curr_episode_info['success'] = None
                self.change_to_none()
            else:
                success = self.curr_episode_info['success']
                if success == True : self.change_to_success()
                elif success == False : self.change_to_fail()
                elif success == 'invalid': self.change_to_invalid()
                else: self.change_to_none()
            if 'trim_info' not in self.curr_episode_info:
                self.curr_episode_info['trim_info'] = (0, len(self.data_list))

            self.update_info()
            self.slider.configure(to=len(self.data_list) - 1)
            self.current_index = 0
            self.slider.set(0)
            self.update_image(self.current_index)

    def resize_image(self, orig_color_im):
        # TODO: Lazy implementation, resize every time
        color_im = orig_color_im
        if color_im.shape[0] > self.window_height:
            target_height = self.window_height - 20
            color_im = cv2.resize(color_im,
                                  (int(color_im.shape[1] / color_im.shape[0] * target_height), target_height))
        if color_im.shape[1] > self.window_width:
            target_width = self.window_width - 20
            color_im = cv2.resize(color_im,
                                  (target_width, int(color_im.shape[0] / color_im.shape[1] * target_width)))
        return color_im



    def update_image(self, index):
        if self.current_episode:
            self.current_index = int(self.slider.get()) if index is None else int(index)
            data_path = self.data_list[self.current_index]
            data = pickle.load(open(data_path, 'rb'))

            color_ims, depth_ims = [], []
            H = None
            for key in data['obs']['camera'].keys():
                color_im = data['obs']['camera'][key]['color']
                md = 3.0#20.0 if key != 'top' else 3.0
                depth_im = self.make_depth_color(data['obs']['camera'][key]['depth'], md)
                if H is None:
                    H = color_im.shape[0]
                else:
                    color_im = cv2.resize(color_im, (int(color_im.shape[1] * H / color_im.shape[0]), H))
                    depth_im = cv2.resize(depth_im, (int(depth_im.shape[1] * H / depth_im.shape[0]), H))
                color_ims.append(color_im)
                depth_ims.append(depth_im)
            color_im = np.concatenate(color_ims, axis=1)
            depth_im = np.concatenate(depth_ims, axis=1)
            depth_im = cv2.resize(depth_im, (color_im.shape[1], color_im.shape[0]))
            color_im = np.concatenate([color_im, depth_im], axis=0)
            self.orig_color_im = color_im
            self.curr_color_im = self.resize_image(self.orig_color_im.copy())
            img_tk = ImageTk.PhotoImage(Image.fromarray(self.curr_color_im))
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk
            self.video_ims.append(self.curr_color_im.copy())

    def set_start_timestep(self):
        if self.current_episode:
            curr_idx = self.slider.get()
            self.curr_episode_info['trim_info'] = (curr_idx, self.curr_episode_info['trim_info'][1])
            self.changed_status[self.episode_var.get()] = True
            self.info_label.configure(text=f"Success: {self.curr_episode_info['success']} Trim: {self.curr_episode_info['trim_info']}")
            self.update_trim_info_vis()


    def update_trim_info_vis(self):
        length_ppx = self.slider.winfo_width() - 20
        start_x = self.curr_episode_info['trim_info'][0] / len(self.data_list) * length_ppx + 5#+ self.slider.winfo_x()
        end_x = self.curr_episode_info['trim_info'][1] / len(self.data_list) * length_ppx  #+ self.slider.winfo_x()
        self.cut_slider.coords(self.color_region, start_x, 5, end_x, 25)
        self.cut_slider.coords(self.start_timestep_text, start_x, 5)
        self.cut_slider.itemconfig(self.start_timestep_text, text=f"{self.curr_episode_info['trim_info'][0]}")
        self.cut_slider.coords(self.end_timestep_text, end_x,5)
        self.cut_slider.itemconfig(self.end_timestep_text, text=f"{self.curr_episode_info['trim_info'][1]}")

    def set_end_timestep(self):
        if self.current_episode:
            curr_idx = self.slider.get()
            self.curr_episode_info['trim_info'] = (self.curr_episode_info['trim_info'][0], curr_idx)
            self.changed_status[self.episode_var.get()] = True
            self.update_info()
            self.update_trim_info_vis()

    def make_depth_color(self, depth_im, max_d=3.0):
        if depth_im.dtype == np.uint16:
            depth_im = (depth_im / 1000.)
        depth_im = np.clip(depth_im / max_d, 0.0, 1.0)
        depth_im = cv2.applyColorMap((depth_im * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., [2, 1, 0]]
        return depth_im

    def change_to_success(self):
        if self.current_episode:
            self.success_button.configure(bg="green", fg="white")
            self.fail_button.configure(bg="lightgray", fg="black")
            self.invalid_button.configure(bg="lightgray", fg="black")
    def change_to_fail(self):
        if self.current_episode:
            self.success_button.configure(bg="lightgray", fg="black")
            self.fail_button.configure(bg="red", fg="white")
            self.invalid_button.configure(bg="lightgray", fg="black")
    def change_to_none(self):
        if self.current_episode:
            self.success_button.configure(bg="lightgray", fg="black")
            self.fail_button.configure(bg="lightgray", fg="black")
            self.invalid_button.configure(bg="lightgray", fg="black")

    def change_to_invalid(self):
        if self.current_episode:
            self.invalid_button.configure(bg="MediumPurple3", fg="white")
            self.success_button.configure(bg="lightgray", fg="black")
            self.fail_button.configure(bg="lightgray", fg="black")

    def set_success(self):
        if self.current_episode:
            self.changed_status[self.episode_var.get()] = True
            if self.curr_episode_info['success'] == True:
                self.curr_episode_info['success'] = None
                self.change_to_none()
            else:
                self.curr_episode_info['success'] = True
                self.change_to_success()
            self.update_info()

    def set_fail(self):
        if self.current_episode:
            self.changed_status[self.episode_var.get()] = True
            if self.curr_episode_info['success'] == False:
                self.curr_episode_info['success'] = None
                self.change_to_none()
            else:
                self.curr_episode_info['success'] = False
                self.change_to_fail()
            self.update_info()


    def set_invalid(self):
        if self.current_episode:
            self.changed_status[self.episode_var.get()] = True
            if self.curr_episode_info['success'] == 'invalid':
                self.curr_episode_info['success'] = None
                self.change_to_none()
            else:
                self.curr_episode_info['success'] = 'invalid'
                self.change_to_invalid()
            self.update_info()

    def reset_all(self):
        self.curr_episode_info['success'] = None
        self.curr_episode_info['trim_info'] = (0, len(self.data_list))
        self.update_trim_info_vis()
        self.change_to_none()
        self.update_info()



    def update_info(self):
        if self.current_episode:
            self.info_label.configure(text=f"Success: {self.curr_episode_info['success']} Trim: {self.curr_episode_info['trim_info']}")

    def save_episode(self):
        if self.current_episode:
            pickle.dump(self.curr_episode_info, open(self.curr_episode_info_file, 'wb'))
            self.info_label.configure(
                text=f"Success: {self.curr_episode_info['success']} Trim: {self.curr_episode_info['trim_info']} [Saved]")

            self.episode_info_dict[self.episode_var.get()] = self.curr_episode_info

            if len(self.video_ims) > 0:
                import imageio
                save_file = f"video_{self.episode_var.get()}.mp4"
                imageio.mimsave(save_file, self.video_ims, fps=60)

    def play_a_step(self):
        if self.current_episode:
            if self.playing and self.curr_play_idx < self.curr_episode_info['trim_info'][1]:
                self.slider.set(self.curr_play_idx)
                self.update_image(self.curr_play_idx)
                self.root.update()
                self.curr_play_idx += 10
                self.root.after(1, self.play_a_step)
            else:
                self.playing = False
                self.info_label.configure(text="Stopped playing")
    def play_episode(self, idx=None):
        # play episode from start_timestep to end_timestep
        if self.current_episode:
            self.playing = True
            start_idx, end_idx = self.curr_episode_info['trim_info']
            self.curr_play_idx = start_idx if idx is None else idx
            self.play_a_step()
            self.info_label.configure(text="Playing episode")

    def stop_episode(self):
        if self.current_episode:
            self.playing = False
            self.info_label.configure(text="Stopped playing")

    def load_next_episode(self):
        if self.current_episode:
            if self.changed_status[self.episode_var.get()]:
                self.save_episode()
            keys = list(self.data_dict.keys())
            curr_idx = keys.index(self.episode_var.get())
            next_idx = (curr_idx + 1) % len(keys)
            self.episode_var.set(keys[next_idx])
            self.load_episode()

    def load_prev_episode(self):
        if self.current_episode:
            if self.changed_status[self.episode_var.get()]:
                self.save_episode()
            keys = list(self.data_dict.keys())
            curr_idx = keys.index(self.episode_var.get())
            next_idx = (curr_idx - 1) % len(keys)
            self.episode_var.set(keys[next_idx])
            self.load_episode()

    def check_all_episode_info(self):
        self.process_status = {}
        for ep_name in self.episode_info_dict:
            processed = False
            if 'success' in self.episode_info_dict[ep_name] and self.episode_info_dict[ep_name]['success'] in [True, False]:
                processed = True
            if 'trim_info' not in self.episode_info_dict[ep_name]:
                processed = False
            self.process_status[ep_name] = processed

    #def load_next_unprocessed_episode(self):



    def on_key_press(self, event):
        if event.keysym == "space":
            if self.playing:
                self.stop_episode()
            else:
                self.play_episode()
        elif event.keysym == "a":
            if self.playing:
                self.stop_episode()
            else:
                self.play_episode(self.slider.get())
        elif event.keysym == 's':
            self.save_episode()

        elif event.keysym == '1' or event.keysym == 'KP_1':
            self.set_success()
        elif event.keysym == '2' or event.keysym == 'KP_2':
            self.set_fail()
        elif event.keysym == '3' or event.keysym == 'KP_3':
            self.set_invalid()
        elif event.keysym == 'Escape':
            self.reset_all()
        elif event.keysym == 'q':
            self.load_prev_episode()
        elif event.keysym == 'w':
            self.load_next_episode()


root = tk.Tk()
data_list = sorted(glob.glob("demo_data/*"))
app = DataProcessGUI(root, data_list)
root.mainloop()