import sys
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread

try:
    import tkinter.ttk as ttk
    py3 = False
except ImportError:
    import ttk
    py3 = True
# import mainGUI_support
import os.path
global program_location, algorithm


def start_gui(algorithm_passed):
    '''Starting point when the module is the main routine.'''
    global val, root, top
    global program_location, algorithm
    algorithm = algorithm_passed
    program_call = sys.argv[0]
    program_location = os.path.split(program_call)[0]
    root = tk.Tk()
    top = Toplevel1(root)
    # mainGUI_support.init(root, top)
    root.mainloop()


w = None


def create_toplevel1(root, *args, **kwargs):
    '''Starting point when the module is imported by another program.'''
    global w, w_win, rt
    global program_location
    program_call = sys.argv[0]
    program_location = os.path.split(program_call)[0]
    rt = root
    w = tk.Toplevel(root)
    top = Toplevel1(w)
    # mainGUI_support.init(w, top, *args, **kwargs)
    return (w, top)


def destroy_toplevel1():
    global w
    w.destroy()
    w = None


class Toplevel1:
    def __init__(self, top=None):
        def image_module_gui(event):        
            print("\n\nPlease wait Loading models .....")

            if algorithm=="tiny":
                os.system("python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video "+"0"+" --tiny")
            else:
                os.system("python object_tracker.py  --model yolov4 --video 0")


        def video_module(event):
            global program_location, algorithm
            video_file_name = filedialog.askopenfilename(initialdir="/", title="Select Video file", filetypes=(
            ("MP4 files", "*.mp4"), ("avi files", "*.avi"), ("all files", "*.*")))

        

            print("\n\nPlease wait Loading models .....")
    
            if algorithm=="tiny":
                os.system('python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video "'+video_file_name+'" --tiny')
            else:
                os.system('python object_tracker.py  --model yolov4 --video "'+video_file_name+'"')
         

        def exit_application(event):
            top.destroy()
            import os
            # os._exit(0)
            os.system('python310 start.py')

        background_color = '#d9d9d9'
        foreground_color = '#000000'
        component_color = '#d9d9d9'
        ana_1_color = '#d9d9d9'
        ana_2_color = '#ececec'
        font_16 = "-family Constantia -size 34 -weight bold -slant " \
                 "roman -underline 0 -overstrike 0"
        font_18 = "-family {Sitka Small} -size 15 -weight bold -slant " \
                 "roman -underline 0 -overstrike 0"

        window_width = 1000
        window_height = 650
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_coordinate = (screen_width / 2) - (window_width / 2)
        y_coordinate = (screen_height / 2) - (window_height / 2)
        top.geometry('%dx%d+%d+%d' % (window_width, window_height, x_coordinate, y_coordinate))
        top.title("Smart Loitering Detection System")
        top.configure(background="#ffffff")

        self.label_1 = tk.Label(top)
        self.label_1.place(relx=0.3, rely=0.01, height=250, width=350)
        self.label_1.configure(background="#ffffff")
        self.label_1.configure(disabledforeground="#a3a3a3")
        self.label_1.configure(foreground="#000000")
        photo_location = os.path.join(program_location, "Images/yologo_2.png")
        self.image_0 = tk.PhotoImage(file=photo_location)
        self.label_1.configure(image=self.image_0)
        self.label_1.configure(text='''Label''')

        self.label_2 = tk.Label(top)
        self.label_2.place(relx=0.0, rely=0.35, height=88, width=1000)
        self.label_2.configure(background="#ffffff")
        self.label_2.configure(disabledforeground="#a3a3a3")
        self.label_2.configure(font=font_16)
        self.label_2.configure(foreground="#2365e8")
        self.label_2.configure(text='''Smart Loitering Detection System''')
        self.label_2.configure(width=659)

        self.frame_1 = tk.Frame(top)
        self.frame_1.place(relx=0.03, rely=0.535, relheight=0.402, relwidth=0.94)
        self.frame_1.configure(relief='groove')
        self.frame_1.configure(borderwidth="7")
        self.frame_1.configure(relief="groove")
        self.frame_1.configure(background="#ffffff")
        self.frame_1.configure(width=955)

        self.btn_image = tk.Label(self.frame_1)
        self.btn_image.place(relx=0.410, rely=0.110, height=176, width=172)
        self.btn_image.configure(activebackground="#f9f9f9")
        self.btn_image.configure(activeforeground="black")
        self.btn_image.configure(background="#ffffff")
        self.btn_image.configure(disabledforeground="#a3a3a3")
        self.btn_image.configure(foreground="#000000")
        self.btn_image.configure(highlightbackground="#d9d9d9")
        self.btn_image.configure(highlightcolor="black")
        photo_location = os.path.join(program_location, "Images/images icon bkp.png")
        self.image_2 = tk.PhotoImage(file=photo_location)
        self.btn_image.configure(image=self.image_2)
        self.btn_image.configure(text='''Label''')
        self.btn_image.configure(width=172)
        self.btn_image.bind('<Button-1>', image_module_gui)

        self.btn_video = tk.Label(self.frame_1)
        self.btn_video.place(relx=0.042, rely=0.090, height=186, width=162)
        self.btn_video.configure(activebackground="#f9f9f9")
        self.btn_video.configure(activeforeground="black")
        self.btn_video.configure(background="#ffffff")
        self.btn_video.configure(disabledforeground="#a3a3a3")
        self.btn_video.configure(foreground="#000000")
        self.btn_video.configure(highlightbackground="#d9d9d9")
        self.btn_video.configure(highlightcolor="black")
        photo_location = os.path.join(program_location, "Images/video-camera-png-icon-5.png")
        self.image_3 = tk.PhotoImage(file=photo_location)
        self.btn_video.configure(image=self.image_3)
        self.btn_video.configure(text='''Label''')
        self.btn_video.configure(width=162)
        self.btn_video.bind('<Button-1>', video_module)

        self.label_3_6 = tk.Label(self.frame_1)
        self.label_3_6.place(relx=0.420, rely=0.784, height=36, width=142)
        self.label_3_6.configure(activebackground="#f9f9f9")
        self.label_3_6.configure(activeforeground="black")
        self.label_3_6.configure(background="#ffffff")
        self.label_3_6.configure(disabledforeground="#a3a3a3")
        self.label_3_6.configure(font=font_18)
        self.label_3_6.configure(foreground="#061104")
        self.label_3_6.configure(highlightbackground="#d9d9d9")
        self.label_3_6.configure(highlightcolor="#000000")
        self.label_3_6.configure(text='''Camera''')
        self.label_3_6.configure(width=142)

        self.label_3_6 = tk.Label(self.frame_1)
        self.label_3_6.place(relx=0.047, rely=0.784, height=36, width=142)
        self.label_3_6.configure(activebackground="#f9f9f9")
        self.label_3_6.configure(activeforeground="black")
        self.label_3_6.configure(background="#ffffff")
        self.label_3_6.configure(disabledforeground="#a3a3a3")
        self.label_3_6.configure(font=font_18)
        self.label_3_6.configure(foreground="#061104")
        self.label_3_6.configure(highlightbackground="#d9d9d9")
        self.label_3_6.configure(highlightcolor="#000000")
        self.label_3_6.configure(text='''Video''')
        self.label_3_6.configure(width=142)

        self.btn_exit = tk.Label(self.frame_1)
        self.btn_exit.place(relx=0.822, rely=0.100, height=186, width=150)
        self.btn_exit.configure(activebackground="#f9f9f9")
        self.btn_exit.configure(activeforeground="black")
        self.btn_exit.configure(background="#ffffff")
        self.btn_exit.configure(disabledforeground="#a3a3a3")
        self.btn_exit.configure(foreground="#000000")
        self.btn_exit.configure(highlightbackground="#d9d9d9")
        self.btn_exit.configure(highlightcolor="black")
        photo_location = os.path.join(program_location, "Images/ExitIcon.png")
        self.image_4 = tk.PhotoImage(file=photo_location)
        self.btn_exit.configure(image=self.image_4)
        self.btn_exit.configure(text='''Label''')
        self.btn_exit.configure(width=162)
        self.btn_exit.bind('<Button-1>', exit_application)

        self.label_3_6 = tk.Label(self.frame_1)
        self.label_3_6.place(relx=0.832, rely=0.784, height=26, width=130)
        self.label_3_6.configure(activebackground="#f9f9f9")
        self.label_3_6.configure(activeforeground="black")
        self.label_3_6.configure(background="#ffffff")
        self.label_3_6.configure(disabledforeground="#a3a3a3")
        self.label_3_6.configure(font=font_18)
        self.label_3_6.configure(foreground="#061104")
        self.label_3_6.configure(highlightbackground="#d9d9d9")
        self.label_3_6.configure(highlightcolor="#000000")
        self.label_3_6.configure(text='''Exit''')
        self.label_3_6.configure(width=142)


if __name__ == '__main__':
    start_gui("yolo")
