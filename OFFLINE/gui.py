from tkinter import Tk, Button, Label
import subprocess
import webbrowser
import time
from threading import Thread

# Function to show the splash screen
def show_splash_screen(launch_callback):
    splash = Tk()
    splash.overrideredirect(True)
    splash.configure(bg="#2c3e50")

    w, h = 500, 300
    ws = splash.winfo_screenwidth()
    hs = splash.winfo_screenheight()
    splash.geometry(f"{w}x{h}+{(ws-w)//2}+{(hs-h)//2}")

    Label(
        splash,
        text="Multimodel Landmine Pattern Recognition System",
        font=("Segoe UI", 16, "bold"),
        bg="#2c3e50",
        fg="white"
    ).pack(expand=True)

    Label(
        splash,
        text="Loading...",
        font=("Segoe UI", 12, "italic"),
        bg="#2c3e50",
        fg="white"
    ).pack(pady=20)

    def close_splash():
        splash.destroy()
        launch_callback()

    splash.after(3000, close_splash)
    splash.mainloop()

# Function to run tilesapp.py
def run_tilesapp():
    status_label.config(text="Multimodel Landmine Pattern Recognition System is working: Running tilesapp.py")
    subprocess.Popen(['python', 'tilesapp.py'])

# Function to run server3.py
def run_server3():
    status_label.config(text="Multimodel Landmine Pattern Recognition System is working: Running server3.py")
    subprocess.Popen(['python', 'server.py'])
    webbrowser.open('https://127.0.0.1:5000')  # Adjust the URL as needed

# Function to run app.py
def run_app():
    status_label.config(text="Multimodel Landmine Pattern Recognition System is working: Running app.py")
    subprocess.Popen(['python', 'app.py'])
    webbrowser.open('https://127.0.0.1:5001')  # Adjust the URL as needed

# Function to create the main GUI
def create_gui():
    global status_label

    root = Tk()
    root.title("Multimodel Landmine Pattern Recognition System")
    root.configure(bg="#34495e")

    # Buttons
    button1 = Button(
        root,
        text="Run tilesapp.py",
        command=run_tilesapp,
        font=("Segoe UI", 12, "bold"),
        bg="#1abc9c",
        fg="white",
        width=25,
        height=2
    )
    button1.pack(pady=10)

    button2 = Button(
        root,
        text="Run server3.py",
        command=run_server3,
        font=("Segoe UI", 12, "bold"),
        bg="#3498db",
        fg="white",
        width=25,
        height=2
    )
    button2.pack(pady=10)

    button3 = Button(
        root,
        text="Run app.py",
        command=run_app,
        font=("Segoe UI", 12, "bold"),
        bg="#e74c3c",
        fg="white",
        width=25,
        height=2
    )
    button3.pack(pady=10)

    # Status Label
    status_label = Label(
        root,
        text="Welcome to the Multimodel Landmine Pattern Recognition System",
        font=("Segoe UI", 12),
        bg="#34495e",
        fg="white"
    )
    status_label.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    # Show splash screen and then launch the main GUI
    show_splash_screen(create_gui)