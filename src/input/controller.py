# capture les inputs pour piloter la voiture
# supporte manette (pygame) et clavier (pynput) en meme temps
# la manette est prioritaire si elle est branchee

import numpy as np
import pygame
from pynput import keyboard


class InputController:
    def __init__(self):
        # init pygame juste pour le joystick (pas de fenetre)
        pygame.init()
        pygame.joystick.init()

        # detecte la manette
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Manette detectee: {self.joystick.get_name()}")
        else:
            print("Pas de manette, utilisation du clavier.")

        # clavier en fallback avec pynput
        self._keys = {"up": False, "down": False, "left": False, "right": False}
        self._quit_requested = False
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        print("Input ready. Ctrl+C to stop.")

    def _on_press(self, key):
        if key == keyboard.Key.up: self._keys["up"] = True
        elif key == keyboard.Key.down: self._keys["down"] = True
        elif key == keyboard.Key.left: self._keys["left"] = True
        elif key == keyboard.Key.right: self._keys["right"] = True
        elif hasattr(key, 'char') and key.char == 'q': self._quit_requested = True

    def _on_release(self, key):
        if key == keyboard.Key.up: self._keys["up"] = False
        elif key == keyboard.Key.down: self._keys["down"] = False
        elif key == keyboard.Key.left: self._keys["left"] = False
        elif key == keyboard.Key.right: self._keys["right"] = False

    def get_action(self):
        pygame.event.pump()
        if self.joystick:
            return self._joystick_action()
        return self._keyboard_action()

    def _joystick_action(self):
        # joystick gauche: axe 0 = steering, axe 1 = rien
        # gachettes: R2 (axe 5) = accelerer, L2 (axe 4) = freiner
        # sur PS5: R2 = axe 5 (-1 = relache, 1 = appuye), L2 = axe 4
        # sensibilite du steering: 0.5 = doux, 1.0 = normal, 2.0 = nerveux
        steering_sensitivity = 0.4
        raw_steering = self.joystick.get_axis(0)

        # deadzone pour eviter les micro-mouvements
        if abs(raw_steering) < 0.05:
            steering = 0.0
        else:
            steering = raw_steering * steering_sensitivity

        # gachettes PS5: valeur de -1 (relache) a 1 (appuye)
        r2 = (self.joystick.get_axis(5) + 1) / 2  # accelerer: 0 a 1
        l2 = (self.joystick.get_axis(4) + 1) / 2  # freiner: 0 a 1
        throttle = r2 - l2
        if abs(throttle) < 0.05:
            throttle = 0.0

        return np.array([[throttle, steering]], dtype=np.float32)

    def _keyboard_action(self):
        throttle = float(self._keys["up"]) - float(self._keys["down"])
        steering = float(self._keys["right"]) - float(self._keys["left"])
        return np.array([[throttle, steering]], dtype=np.float32)

    def should_quit(self):
        return self._quit_requested

    def close(self):
        self._listener.stop()
        pygame.quit()
