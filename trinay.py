import pygame
from pygame.math import Vector2
import numpy as np

# for supressing the hello message from pygame
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class Car:
    def __init__(self, x, y, angle=0.0, length=3, max_steering=15, max_acceleration=1.8):
        self.position = Vector2(x, y)
        self.velocity = 0.0
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering

        self.max_velocity = 15
        self.brake_deacceleration = 2.5
        self.free_deacceleration = 0.8  # stimulate friction
        self.steering_speed = 0.5  # 15 deg per dt(1 sec)
        self.acceleration_speed = 0.5

        self.acceleration = 0.0
        self.steering = 0.0

    def update(self, action, dt):

        # taking the input
        self.input(action, dt)

        # negative sign because the y-axis in inverted
        self.position.x += self.velocity * np.cos(np.deg2rad(-self.angle)) * dt
        self.position.y += self.velocity * np.sin(np.deg2rad(-self.angle)) * dt
        self.velocity += self.acceleration * dt
        # limiting the velocity
        self.velocity = np.clip(
            self.velocity, -self.max_velocity, self.max_velocity)

        if self.steering:
            turning_radius = self.length / np.sin(np.radians(self.steering))
            angular_velocity = self.velocity / turning_radius
        else:
            angular_velocity = 0

        self.angle += np.degrees(angular_velocity) * dt

    def input(self, action, dt):
        '''
        actions :   pedal_gas,
                    pedal_brake,
                    pedal_none,
                    pedal_reverse,
                    steer_right,
                    steer_left,
                    steer_none,
        '''
        if action == 'pedal_gas':
            if self.velocity < 0:
                self.acceleration = self.brake_deacceleration
            else:
                self.acceleration += self.acceleration_speed

        elif action == 'pedal_reverse':
            if self.velocity > 0:
                self.acceleration = -self.brake_deacceleration
            else:
                self.acceleration -= self.acceleration_speed
        elif action == 'pedal_brake':
            if abs(self.velocity) > dt*self.brake_deacceleration:
                # abs(brake_deacceleration) * sign(velocity)
                self.acceleration = - \
                    np.copysign(self.brake_deacceleration, self.velocity)
            else:
                # small acceleration to make velocity zero
                self.acceleration = -self.velocity/dt
        elif action == 'pedal_none':
            if abs(self.velocity) > dt*self.free_deacceleration:
                self.acceleration = - \
                    np.copysign(self.free_deacceleration, self.velocity)
            else:
                if dt != 0:
                    self.acceleration = -self.velocity/dt

        elif action == 'steer_right':
            self.steering -= self.steering_speed
        elif action == 'steer_left':
            self.steering += self.steering_speed
        elif action == 'steer_none':
            self.steering = 0
        else:
            # error
            print("No action is given")

        # limiting the acceleration
        self.acceleration = np.clip(
            self.acceleration, -self.max_acceleration, self.max_acceleration)

        # limiting the steering angle
        self.steering = np.clip(
            self.steering, -self.max_steering, self.max_steering)


def game():

    pygame.init()
    pygame.display.set_caption("car testing")
    width, height = (1280, 720)
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    FPS = 60
    exit = False

    car_image = pygame.image.load('car.png')
    # resizing the car image
    new_car_size = (round(car_image.get_width() * 0.7),
                    round(car_image.get_height() * 0.7))
    car_image = pygame.transform.scale(car_image, new_car_size)
    car = Car(10, 30)
    ppu = 5  # pixels per unit = (76*0.7) pixels/ 3 meters(length)

    while not exit:
        dt = clock.get_time() / 1000  # return time in milliseconds betwwen two ticks

        # Event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit = True

        # User input
        pressed = pygame.key.get_pressed()

        if pressed[pygame.K_UP]:
            car.update("pedal_gas", dt)
        elif pressed[pygame.K_DOWN]:
            car.update("pedal_reverse", dt)
        elif pressed[pygame.K_SPACE]:
            car.update("pedal_brake", dt)
        else:
            car.update("pedal_none", dt)

        if pressed[pygame.K_RIGHT]:
            car.update("steer_right", dt)
        elif pressed[pygame.K_LEFT]:
            car.update("steer_left", dt)
        else:
            car.update("steer_none", dt)

        # Drawing
        screen.fill((0, 0, 0))
        rotated = pygame.transform.rotate(car_image, car.angle)
        rect = rotated.get_rect()
        screen.blit(rotated, car.position * ppu -
                    (rect.width / 2, rect.height / 2))
        pygame.display.flip()

        clock.tick(FPS)
    pygame.quit()


if __name__ == '__main__':
    print("-- For testing the car properties --")
    print(" Redering at 60 FPS ")
    print("<< Controls >>")
    print("<< up arrow >>    - gas pedal ")
    print("<< down arrow >>  - reverse ")
    print("<< Space bar >>   - brake ")
    print("<< right arrow >> - steer right ")
    print("<< left arrow >>  - steer left ")
    game()
