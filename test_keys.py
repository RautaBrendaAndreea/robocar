from pynput import keyboard

def on_press(key):
    print(f"Pressed: {key}")

listener = keyboard.Listener(on_press=on_press)
listener.start()
input("Press some arrow keys, then Enter to quit...\n")
