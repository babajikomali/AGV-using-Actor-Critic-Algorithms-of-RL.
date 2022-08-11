import pygame
from Differential_Car import Car
import numpy as np
import math
import time
from scipy import spatial
import cv2

pygame.init()
FONT = pygame.font.SysFont('assets/ComicNeue-Regular.ttf', 20)

'''text_surface = FONT.render(f"FPS : {int(self._clock.get_fps())}", True, color)
self.screen.blit(text_surface, (self.path.map_size[0]*self.ppu-100, 10))'''

action_space = {
    0: 'left_increase',
    1: 'left_decrease',
    2: 'right_increase',
    3: 'right_decrease',
    4: 'none'
}

class Path():
    def __init__(self):
        self.x = [122.0,123.17,124.35,125.54,126.73,127.94,129.15,130.36,131.59,132.82,134.07,135.31,136.57,137.83,139.1,140.37,141.65,142.94,144.24,145.53,146.84,148.15,149.47,150.79,152.12,153.45,154.79,156.13,157.47,158.83,160.18,161.54,162.9,164.27,165.64,167.02,168.4,169.78,171.16,172.55,173.94,175.34,176.73,178.13,179.54,180.94,182.34,183.75,185.16,186.57,187.99,189.4,190.81,192.23,193.65,195.07,196.48,197.9,199.32,200.74,202.16,203.58,205.0,206.42,207.84,209.25,210.67,212.09,213.5,214.91,216.32,217.74,219.14,220.55,221.96,223.36,224.76,226.16,227.55,228.94,230.33,231.72,233.11,234.49,235.86,237.24,238.61,239.97,241.33,242.69,244.05,245.39,246.74,248.08,249.41,250.74,252.07,253.39,254.7,256.01,257.31,258.61,259.9,261.19,262.46,263.74,265.0,266.26,267.51,268.76,269.99,271.22,272.45,273.66,274.87,276.07,277.26,278.44,279.62,280.79,281.94,283.09,284.24,285.37,286.49,287.61,288.71,289.81,290.9,291.98,293.05,294.11,295.16,296.2,297.24,298.26,299.28,300.28,301.28,302.26,303.24,304.2,305.16,306.11,307.04,307.97,308.89,309.79,310.69,311.57,312.45,313.31,314.17,315.01,315.84,316.67,317.48,318.28,319.07,319.85,320.61,321.37,322.11,322.85,323.57,324.28,324.98,325.67,326.35,327.01,327.67,328.31,328.94,329.56,330.16,330.76,331.34,331.91,332.47,333.02,333.55,334.07,334.58,335.07,335.56,336.03,336.5,336.95,337.39,337.82,338.24,338.65,339.05,339.44,339.83,340.2,340.57,340.93,341.28,341.62,341.95,342.28,342.6,342.92,343.23,343.53,343.83,344.12,344.41,344.69,344.97,345.24,345.51,345.78,346.04,346.3,346.56,346.81,347.06,347.31,347.56,347.81,348.05,348.3,348.54,348.78,349.03,349.27,349.51,349.76,350.0,350.25,350.5,350.75,351.0,351.25,351.51,351.77,352.03,352.3,352.57,352.85,353.13,353.41,353.7,353.99,354.29,354.6,354.91,355.22,355.55,355.88,356.21,356.56,356.91,357.27,357.64,358.02,358.4,358.8,359.2,359.61,360.03,360.46,360.9,361.35,361.81,362.27,362.75,363.23,363.72,364.22,364.73,365.25,365.78,366.32,366.87,367.42,367.99,368.56,369.15,369.74,370.34,370.95,371.57,372.2,372.84,373.49,374.14,374.81,375.48,376.17,376.86,377.57,378.28,379.0,379.73,380.47,381.22,381.98,382.75,383.53,384.32,385.12,385.92,386.74,387.57,388.4,389.25,390.1,390.97,391.84,392.72,393.62,394.52,395.43,396.36,397.29,398.23,399.18,400.14,401.11,402.09,403.08,404.08,405.09,406.11,407.14,408.18,409.23,410.29,411.36,412.44,413.53,414.63,415.74,416.85,417.98,419.12,420.27,421.43,422.6,423.78,424.96,426.16,427.36,428.58,429.8,431.03,432.27,433.51,434.77,436.03,437.3,438.57,439.85,441.14,442.44,443.74,445.05,446.36,447.68,449.01,450.34,451.67,453.01,454.36,455.71,457.06,458.42,459.78,461.15,462.52,463.89,465.27,466.65,468.03,469.42,470.8,472.19,473.58,474.98,476.37,477.77,479.16,480.56,481.96,483.36,484.76,486.16,487.56,488.96,490.36,491.76,493.16,494.56,495.95,497.35,498.74,500.14,501.53,502.91,504.3,505.68,507.06,508.44,509.82,511.19,512.57,513.94,515.3,516.67,518.03,519.4,520.76,522.11,523.47,524.82,526.18,527.53,528.88,530.23,531.57,532.92,534.26,535.6,536.94,538.28,539.62,540.96,542.29,543.62,544.96,546.29,547.62,548.95,550.28,551.61,552.93,554.26,555.58,556.91,558.23,559.56,560.88,562.2,563.52,564.84,566.16,567.48,568.8,570.12,571.44,572.76,574.08,575.4,576.72,578.04,579.36,580.67,581.99,583.31,584.63,585.95,587.27,588.59,589.91,591.23,592.55,593.87,595.19,596.51,597.84,599.16,600.49,601.81,603.14,604.46,605.79,607.12,608.45,609.78,611.11,612.44,613.77,615.11,616.44,617.78,619.12,620.46,621.8,623.14,624.49,625.83,627.18,628.53,629.88,631.23,632.58,633.94]
        self.y = [96.11,96.55,96.97,97.37,97.74,98.08,98.4,98.7,98.98,99.24,99.47,99.69,99.88,100.05,100.21,100.34,100.46,100.55,100.63,100.69,100.74,100.76,100.78,100.77,100.75,100.72,100.67,100.6,100.53,100.44,100.33,100.22,100.09,99.95,99.8,99.64,99.47,99.29,99.1,98.9,98.69,98.48,98.26,98.03,97.79,97.54,97.3,97.04,96.78,96.52,96.25,95.97,95.7,95.42,95.14,94.85,94.57,94.28,93.99,93.71,93.42,93.13,92.84,92.56,92.28,92.0,91.72,91.44,91.17,90.9,90.64,90.38,90.13,89.88,89.64,89.4,89.17,88.95,88.74,88.54,88.34,88.15,87.97,87.81,87.65,87.5,87.37,87.24,87.13,87.03,86.95,86.87,86.81,86.77,86.74,86.72,86.72,86.74,86.77,86.82,86.89,86.97,87.08,87.2,87.34,87.5,87.68,87.87,88.09,88.34,88.6,88.88,89.19,89.52,89.87,90.25,90.65,91.07,91.52,92.0,92.5,93.02,93.56,94.13,94.72,95.34,95.97,96.63,97.31,98.0,98.72,99.46,100.22,101.0,101.79,102.61,103.44,104.29,105.16,106.05,106.95,107.87,108.8,109.75,110.72,111.7,112.69,113.7,114.72,115.76,116.81,117.87,118.95,120.03,121.13,122.24,123.36,124.49,125.63,126.78,127.94,129.11,130.29,131.47,132.67,133.87,135.08,136.29,137.52,138.74,139.98,141.22,142.46,143.71,144.96,146.22,147.48,148.74,150.01,151.28,152.55,153.82,155.1,156.37,157.65,158.93,160.21,161.49,162.78,164.06,165.35,166.64,167.93,169.22,170.51,171.8,173.1,174.4,175.7,176.99,178.3,179.6,180.9,182.21,183.51,184.82,186.13,187.44,188.75,190.07,191.38,192.7,194.01,195.33,196.65,197.97,199.29,200.61,201.94,203.26,204.59,205.92,207.25,208.58,209.91,211.24,212.57,213.91,215.24,216.58,217.92,219.26,220.6,221.94,223.28,224.62,225.96,227.31,228.65,230.0,231.35,232.7,234.05,235.4,236.75,238.1,239.45,240.81,242.16,243.52,244.88,246.23,247.59,248.95,250.31,251.67,253.03,254.4,255.76,257.12,258.49,259.85,261.21,262.58,263.94,265.3,266.66,268.02,269.37,270.73,272.08,273.43,274.77,276.11,277.45,278.78,280.11,281.44,282.76,284.07,285.38,286.68,287.98,289.27,290.56,291.83,293.1,294.37,295.62,296.87,298.11,299.34,300.56,301.77,302.97,304.17,305.35,306.52,307.68,308.83,309.97,311.1,312.22,313.32,314.41,315.49,316.56,317.61,318.65,319.68,320.69,321.69,322.67,323.64,324.59,325.53,326.45,327.35,328.24,329.11,329.97,330.81,331.63,332.43,333.21,333.98,334.73,335.46,336.17,336.86,337.53,338.18,338.81,339.41,340.0,340.57,341.11,341.64,342.14,342.62,343.07,343.51,343.92,344.32,344.69,345.04,345.37,345.69,345.98,346.25,346.51,346.75,346.97,347.18,347.36,347.53,347.69,347.83,347.95,348.06,348.15,348.23,348.3,348.35,348.39,348.42,348.43,348.43,348.42,348.4,348.37,348.32,348.27,348.21,348.13,348.05,347.96,347.86,347.76,347.64,347.52,347.39,347.26,347.12,346.97,346.82,346.66,346.5,346.33,346.16,345.98,345.81,345.63,345.44,345.26,345.07,344.88,344.7,344.51,344.32,344.13,343.94,343.75,343.56,343.38,343.19,343.01,342.82,342.64,342.46,342.28,342.11,341.93,341.76,341.59,341.42,341.25,341.08,340.91,340.75,340.59,340.43,340.27,340.12,339.96,339.81,339.66,339.52,339.37,339.23,339.09,338.95,338.82,338.69,338.56,338.43,338.31,338.19,338.07,337.95,337.84,337.73,337.62,337.52,337.42,337.32,337.23,337.13,337.05,336.96,336.88,336.8,336.73,336.66,336.59,336.53,336.47,336.41,336.36,336.31,336.27,336.23,336.19,336.16,336.13,336.11,336.09,336.07,336.06,336.06,336.05,336.05,336.06,336.07,336.09,336.11,336.13,336.16,336.2,336.24,336.28,336.33,336.38,336.44,336.51,336.57,336.65,336.73,336.81,336.9,337.0,337.1,337.21,337.32,337.44,337.56,337.69,337.82,337.96,338.11]
        self.path_width = 20
        self.start = (self.x[0],self.y[0])
        self.end_x = np.full((20,), self.x[-1])
        self.end_y = np.linspace(
            self.y[-1]-self.path_width/2, self.y[-1]+self.path_width/2, 20)
        self.initial_theta = 0.0
        self.map_boundaries = (800,600)    
        self.checkpoint = {
            (186.57, 96.52) : 0,
            (256.01, 86.82) : 0,
            (311.57, 115.76) : 0,
            (341.62, 176.99) : 0,
            (355.22, 243.52) : 0,
            (381.98, 308.83) : 0,
            (432.27, 346.25) : 0,
            (500.14, 344.32) : 0,
            (567.48, 336.96) : 0,
            (633.94, 338.11) : 0
        }

class Env:
    def __init__(self,path):
        self.path = path
        self.carimg = pygame.image.load('car.png')
        self.size = round(self.carimg.get_width()*0.37), round(self.carimg.get_height()*0.37)
        self.carimg = pygame.transform.scale(self.carimg, self.size)
        self.car = Car(self.path.start[0],self.path.start[1],self.carimg.get_width())
        self.screen = pygame.display.set_mode((800, 600))
        self.car = Car(
            self.path.start[0], self.path.start[1], 3)
        self.done = False     #tells whether car reached the end
        self.dt = 1/60
        self.trajectory = np.vstack((self.path.x, self.path.y)).T
        self.end_flag = np.vstack((self.path.end_x,self.path.end_y)).T
        self.m2p = 3779.52          #meters to pixel 
        self.clock = pygame.time.Clock()
        self.record = False
        self.recorder = None

    def reset(self):
        self.car = Car(self.path.start[0], self.path.start[1],self.carimg.get_width())
        self.done = False
        self.dt = 1/60
        self.record = False
        self.recorder = None
        self.check = {
            50: 0,
            100: 0,
            150: 0,
            200: 0,
            250: 0,
            300: 0,
            350: 0,
            400: 0,
            450: 0
        }
        return np.array([self.car.position.x, self.car.position.y, self.car.theta, 0.0])

    def render_path(self):
        for i in self.check.keys():
            pygame.draw.circle(self.screen, (0, 0, 0), self.trajectory[i,:], 3, 0)

        pygame.draw.lines(self.screen, (255, 0, 0),closed=0, points=self.trajectory)
        array = np.full(self.end_flag.shape, self.path.path_width/10)
        array[:, 1] = 0
        pygame.draw.lines(self.screen, (255, 0, 0),closed=0, points=self.end_flag+array)
    
    def render_env(self, FPS_lock=60):
        #print(self.clock.get_fps())
        self.screen = pygame.display.set_mode((800, 600))
        carimg = pygame.image.load('car.png')
        size = round(carimg.get_width()*0.37), round(carimg.get_height()*0.37)
        carimg = pygame.transform.scale(carimg, size)
        #updating the screen
        self.screen.fill((255, 255, 255))
        rotated = pygame.transform.rotozoom(carimg, math.degrees(self.car.theta), 1)
        rect = rotated.get_rect(center=(self.car.position.x, self.car.position.y))
        self.render_path()
        self.screen.blit(rotated, self.car.position - (rect.width / 2, rect.height / 2))

        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.text = self.font.render('default', True, (255,0,0), (255,255,255))
        self.textRect = self.text.get_rect()
        self.textRect.center = (self.path.map_boundaries[0]-650, 10)
        #text_surface = FONT.render(3
        #    f"FPS : {int(self.clock.get_fps())}", True, (255,0,0))
        #self.screen.blit(text_surface, (self.path.map_boundaries[0]-100, 10))
        txt = f"position={self.car.position} vl={self.car.vl} vr={self.car.vr} theta={int(math.degrees(self.car.heading))}"
        self.text = self.font.render(txt, True, (255,0,0), (255,255,255))
        self.screen.blit(self.text, self.textRect)
        pygame.display.flip()
        #self.clock.tick(FPS_lock)
        if FPS_lock:
            self.clock.tick(FPS_lock)
        else:
            self.clock.tick()
    
    def reward(self,cross_track_error):
        if 0<=cross_track_error<=self.path.path_width/2:
            return -0.5
        else:
            return -1

    def step(self,action,count=0,MAX_STEPS_PER_EPISODE=0):
        if count > MAX_STEPS_PER_EPISODE : 
            done = True
            return np.array([self.car.position.x, self.car.position.y, self.car.theta, max(self.path.map_boundaries[0], self.path.map_boundaries[1])]), -200.0, self.done
        
        self.car.update(action,self.dt)
        car_position = np.array([[self.car.position.x, self.car.position.y]])
        distance_from_end = np.min(spatial.distance.cdist(
            self.end_flag, car_position)) 

        cross_track_error = np.min(spatial.distance.cdist(self.trajectory, car_position))

        index = np.argmin(spatial.distance.cdist(self.trajectory, car_position))
        forward = index + 10
        if forward > 499:
            forward = 499
        
        slope = (self.path.y[forward]-self.sey[index])/(self.x[forward]-self.x[index])

        inclination = math.degrees(math.atan(slope))










        if count > MAX_STEPS_PER_EPISODE : 
            done = True
            return np.array([self.car.position.x, self.car.position.y, self.car.theta, cross_track_error]), -200.0, self.done
        if cross_track_error > self.path.path_width:
            self.done = True
            state_arr = np.array(
                [self.car.position.x, self.car.position.y, self.car.theta, cross_track_error])
            reward_float = -200.0
            return state_arr, reward_float, self.done
        for i in self.check.keys():
            if not self.check[i] and abs(np.linalg.norm(self.trajectory[i,:]-car_position)-np.linalg.norm(self.trajectory[i+4, :]-car_position)) < 0.5 : 
                self.check[i] = 1
                state_arr = np.array(
                    [self.car.position.x, self.car.position.y, self.car.theta, cross_track_error]
                )
                reward_float = 50.0
                return state_arr, reward_float, self.done

            elif self.check[i] and abs(np.linalg.norm(self.trajectory[i,:]- car_position)-np.linalg.norm(self.trajectory[i+4, :]-car_position)) < 0.5 : 
                state_arr = np.array(
                    [self.car.position.x, self.car.position.y, self.car.theta, cross_track_error]
                )
                reward_float = -10.0
                return state_arr, reward_float, self.done

        if abs(int(math.degrees(self.car.theta))) > (360.0*3.0):
            self.done = True
            state_arr = np.array(
                    [self.car.position.x, self.car.position.y, self.car.theta, cross_track_error]
                )
            reward_float = -200.0
            return state_arr, reward_float, self.done

        if distance_from_end < self.path.path_width/10:
            self.done = True
            state_arr = np.array(
                [self.car.position.x, self.car.position.y, self.car.theta, max(self.path.map_boundaries[0], self.path.map_boundaries[1])])
            reward_float = 200.0
            return state_arr, reward_float, self.done
        if not self.done:
            if cross_track_error<=self.path.path_width:
                state_arr = np.array(
                    [self.car.position.x, self.car.position.y, self.car.theta, cross_track_error])
                return state_arr, self.reward(cross_track_error), self.done
            else:
                self.done = True
                state_arr = np.array(
                    [self.car.position.x, self.car.position.y, self.car.theta, cross_track_error])
                reward_float = -200.0
                return state_arr, reward_float, self.done
        
    def close_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.record == True:  self.recorder.release()
                pygame.display.quit()
                pygame.quit()
                # sys.exit()
                self.done = True

    def record_env(self, filename):
        if self.record == False and self.done == False:
            self.recorder = cv2.VideoWriter(f'{filename}.mp4', 0x7634706d, 60.0, (
                self.path.map_boundaries[0], self.path.map_boundaries[1]))
            print(f'Environment recording will be saved to {filename}.mp4')
            self.record = True

        pixels = cv2.rotate(pygame.surfarray.pixels3d(
            self.screen), cv2.ROTATE_90_CLOCKWISE)
        pixels = cv2.flip(pixels, 1)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        self.recorder.write(pixels)

        if self.record == True and self.done == True:
            self.recorder.release()


if __name__ == '__main__':
    pygame.init()
    import random
    print("<< Random Agent >>")
    path1 = Path()
    env = Env(path1)
    #env.record('recorded')
    done = False
    env.reset()
    while not done:
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_KP6]:
            action = 0
        elif pressed[pygame.K_KP1]:
            action = 1
        elif pressed[pygame.K_KP4]:
            action = 2
        elif pressed[pygame.K_KP3]:
            action = 3
        else:
            action = 4
        cords, reward, done =  env.step(action_space[action])
        #print(reward, end='')
        env.render_env()
        #env.record_env('output')
        env.close_quit()
