import pygame
import os
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Initialize pygame
pygame.init()

# Base folder for assets relative to the project
ASSET_BASE = os.path.join(os.path.dirname(__file__), "..", "Assets")

# Subfolders
DINO_DIR = os.path.join(ASSET_BASE, "Dino")
CACTUS_DIR = os.path.join(ASSET_BASE, "Cactus")
BIRD_DIR = os.path.join(ASSET_BASE, "Bird")
OTHER_DIR = os.path.join(ASSET_BASE, "Other")

# Global constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join(DINO_DIR, "DinoRun1.png")),
           pygame.image.load(os.path.join(DINO_DIR, "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join(DINO_DIR, "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join(DINO_DIR, "DinoDuck1.png")),
           pygame.image.load(os.path.join(DINO_DIR, "DinoDuck2.png"))]
SMALL_CACTUS = [pygame.image.load(os.path.join(CACTUS_DIR, "SmallCactus1.png")),
                pygame.image.load(os.path.join(CACTUS_DIR, "SmallCactus2.png")),
                pygame.image.load(os.path.join(CACTUS_DIR, "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join(CACTUS_DIR, "LargeCactus1.png")),
                pygame.image.load(os.path.join(CACTUS_DIR, "LargeCactus2.png")),
                pygame.image.load(os.path.join(CACTUS_DIR, "LargeCactus3.png"))]

BIRD = [pygame.image.load(os.path.join(BIRD_DIR, "Bird1.png")),
        pygame.image.load(os.path.join(BIRD_DIR, "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join(OTHER_DIR, "Cloud.png"))
BG = pygame.image.load(os.path.join(OTHER_DIR, "Track.png"))


# ------------------------
# Game classes (Dino, Obstacles, etc.)
# ------------------------
class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, action):
        # action: 0 = do nothing, 1 = jump, 2 = duck
        if action == 1 and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif action == 2 and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif not self.dino_jump:
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

        if self.dino_duck:
            self.duck()
        elif self.dino_run:
            self.run()
        elif self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel < - self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL

    def draw(self, screen):
        screen.blit(self.image, (self.dino_rect.x, self.dino_rect.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self, game_speed, obstacles):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.remove(self)

    def draw(self, screen):
        screen.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 300


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = 250
        self.index = 0

    def draw(self, screen):
        if self.index >= 9:
            self.index = 0
        screen.blit(self.image[self.index//5], self.rect)
        self.index += 1


# ------------------------
# Entorno Gymnasium
# ------------------------
class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(DinoEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: do nothing, 1: jump, 2: duck
        # Observation: dino position + first obstacle
        self.observation_space = spaces.Box(
            low=0, high=SCREEN_WIDTH,
            shape=(4,), dtype=np.float32
        )
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self, seed=None, options=None):
        self.player = Dinosaur()
        self.obstacles = []
        self.game_speed = 20
        self.points = 0

        # ground positions
        self.x_pos_bg = 0
        self.y_pos_bg = 380

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
            self.player.update(action)

            # Generate obstacles
            if len(self.obstacles) == 0:
                choice = random.randint(0, 2)
                if choice == 0:
                    self.obstacles.append(SmallCactus(SMALL_CACTUS))
                elif choice == 1:
                    self.obstacles.append(LargeCactus(LARGE_CACTUS))
                else:
                    self.obstacles.append(Bird(BIRD))

            # Base reward for surviving one more frame
            reward = 1 
            done = False

            for obstacle in list(self.obstacles):
                obstacle.update(self.game_speed, self.obstacles)
                if self.player.dino_rect.colliderect(obstacle.rect):
                    reward = -100
                    done = True
            
            self.points += 1
            if self.points % 100 == 0:
                self.game_speed += 1

            obs = self._get_obs()
            return obs, reward, done, False, {}

    def _get_obs(self):
        # Dino Y, Dino JumpVel, first obstacle X, obstacle type
        if len(self.obstacles) > 0:
            obstacle = self.obstacles[0]
            return np.array([self.player.dino_rect.y,
                             self.player.jump_vel,
                             obstacle.rect.x,
                             obstacle.rect.y], dtype=np.float32)
        else:
            return np.array([self.player.dino_rect.y,
                             self.player.jump_vel,
                             SCREEN_WIDTH,
                             0], dtype=np.float32)


    def render(self, mode="human"):
        SCREEN.fill((255, 255, 255))

        image_width = BG.get_width()
        SCREEN.blit(BG, (self.x_pos_bg, self.y_pos_bg))
        SCREEN.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
        if self.x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
            self.x_pos_bg = 0
        self.x_pos_bg -= self.game_speed

        self.player.draw(SCREEN)
        for obstacle in self.obstacles:
            obstacle.draw(SCREEN)

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.quit()