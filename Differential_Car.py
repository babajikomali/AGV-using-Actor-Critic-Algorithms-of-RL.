import pygame
from pygame.math import Vector2, Vector3
import math
import time

'''
PSEUDOCODE:
vl                  ----->   left wheels velocity
vr                  ----->   right wheels velocity
position            ----->   car's position
maximum_velocity    ----->   maximum velocity of wheels
minimum_velocity    ----->   minimum velocity of wheels
theta               ----->   heading angle
m2p = 3779.52       ----->   meters to pixel converter
Action Space: Discrete - 5 actions
    KEY 4 - right wheel velocity increase
    KEY 1 - left wheel velocity decrease
    KEY 6 - left wheel velocity increase
    KEY 3 - right wheel velocity decrease       
    None  - No change

Update equations:
    vl = min(maximum_velocity, vl+0.001*m2p); vr = vr    ---->left_increase
    vl = max(minimum_velocity, vl-0.001*m2p); vr = vr    ---->left_decrease
    vr = min(maximum_velocity, vr+0.001*m2p); vl = vl    ---->right_increase
    vr = max(minimum_velocity, vr-0.001*m2p); vl = vl    ---->right_decrease
    vl = vl ; vr = vr                                    ----> None

    x_position += ((vl+vr)/2)*cos(theta)*dt
    y_position -= ((vl+vr)/2)*sin(theta)*dt
    theta+= ((vl-vr)/car_width)*dt
'''

class Car:
    def __init__(self,x,y,width):
        self.m2p = 3779.52          #meters to pixel 
        self.width = width          #width of image
        self.position = Vector2(x,y)
        self.theta = 0.0
        self.vl = 0.005*self.m2p    #velocity of left wheels
        self.vr = 0.005*self.m2p    #velocity of right wheels
        self.max_velocity = 0.015*self.m2p
        self.min_velocity = -0.005*self.m2p
        
    def update(self,action,dt):
        if action == 'left_increase':
            self.vl = min(self.max_velocity, self.vl+0.001*self.m2p)
        elif action == 'left_decrease':
            self.vl = max(self.min_velocity, self.vl-0.001*self.m2p)
        elif action == 'right_increase':
            self.vr = min(self.max_velocity, self.vr+0.001*self.m2p)
        elif action == 'right_decrease':
            self.vr = max(self.min_velocity, self.vr-0.001*self.m2p)
        elif action == 'none':
            self.vl = self.vl
            self.vr = self.vr
        
        self.position.x += ((self.vl+self.vr)/2)*math.cos(self.theta)*dt
        self.position.y -= ((self.vl+self.vr)/2)*math.sin(self.theta)*dt
        self.theta += ((self.vr-self.vl)/self.width)*dt

         
def game():
    pygame.init()
    FPS = 60    #frame rate
    clock = pygame.time.Clock()

    #setting up window
    pygame.display.set_caption('RL based Path Following')
    screen = pygame.display.set_mode((800,600))
    
    #scaling the image
    carimg = pygame.image.load('car.png')
    size = round(carimg.get_width()*0.37), round(carimg.get_height()*0.37)
    carimg = pygame.transform.scale(carimg, size)

    car = Car(200,200,carimg.get_width())   #car object
    run = True
    while run:
        dt = clock.get_time() / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        #action trigger - key press
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_KP6]:
            car.update('left_increase', dt)
        elif pressed[pygame.K_KP1]:
            car.update('left_decrease', dt)
        elif pressed[pygame.K_KP4]:
            car.update('right_increase', dt)
        elif pressed[pygame.K_KP3]:
            car.update('right_decrease', dt)
        else:
            car.update('none',dt)
        
        #updating the screen
        screen.fill((255, 255, 255))
        rotated = pygame.transform.rotozoom(carimg, math.degrees(car.theta),1)
        rect = rotated.get_rect(center=(car.position.x, car.position.y))
        screen.blit(rotated, car.position - (rect.width / 2, rect.height / 2))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == '__main__':
    game()
