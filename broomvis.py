from src.environment import CleaningEnv

env = CleaningEnv(35, True, 8)

for _ in range(10):
    env.reset()
    env.room_mechanics.show_room()

    proceed = input("Proceed? ")
    if proceed == 'no': continue

    x1 = float(input('x1:'))
    y1 = float(input('y1:'))
    x2 = float(input('x2:'))
    y2 = float(input('y2:'))

    env.room_mechanics.move_broom((x1, y1), (x2, y2))
    env.room_mechanics.show_room()