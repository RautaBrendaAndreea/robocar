import pygame, time

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("Aucune manette detectee!")
    exit()

js = pygame.joystick.Joystick(0)
js.init()
print(f"Manette: {js.get_name()}")
print(f"Axes: {js.get_numaxes()}, Boutons: {js.get_numbuttons()}")
print("Bouge le joystick et appuie sur les gachettes... (Ctrl+C pour quitter)")

try:
    while True:
        pygame.event.pump()
        axes = [f"axe{i}={js.get_axis(i):+.2f}" for i in range(js.get_numaxes())]
        print("  ".join(axes), end="\r")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nDone!")
    pygame.quit()
