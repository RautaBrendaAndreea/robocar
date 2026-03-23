# capture les touches du clavier pour piloter la voiture
# utilise pynput pour capter les touches meme si la fenetre du simu est au premier plan

import numpy as np
from pynput import keyboard


class InputController:
    def __init__(self):
        # on track les toucehs
        self._keys = {"up": False, "down": False, "left": False, "right": False}
        self._quit_requested = False

        # listener qui tourne en arriere plan et capte toutes les touches
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        print("Input ready. Use arrow keys to drive, Q to quit.")

    def _on_press(self, key):
        # quand on appuie sur une touche, on met le flag a true
        if key == keyboard.Key.up: self._keys["up"] = True
        elif key == keyboard.Key.down: self._keys["down"] = True
        elif key == keyboard.Key.left: self._keys["left"] = True
        elif key == keyboard.Key.right: self._keys["right"] = True
        elif hasattr(key, 'char') and key.char == 'q': self._quit_requested = True

    def _on_release(self, key):
        # quand on relache, on remet a false
        if key == keyboard.Key.up: self._keys["up"] = False
        elif key == keyboard.Key.down: self._keys["down"] = False
        elif key == keyboard.Key.left: self._keys["left"] = False
        elif key == keyboard.Key.right: self._keys["right"] = False

    def get_action(self):
        # transforme les touches en valeurs: haut-bas = throttle, droite-gauche = steering
        # throttle: 1.0 = avancer, -1.0 = reculer, 0 = rien
        # steering: 1.0 = droite, -1.0 = gauche, 0 = tout droit
        throttle = float(self._keys["up"]) - float(self._keys["down"])
        steering = float(self._keys["right"]) - float(self._keys["left"])
        return np.array([[throttle, steering]], dtype=np.float32)

    def should_quit(self):
        return self._quit_requested

    def close(self):
        self._listener.stop()
