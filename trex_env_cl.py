import pygame
import os
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Inicializar pygame
pygame.init()

# Constantes globales
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))
BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


# ------------------------
# Clases del juego (Dino, Obstáculos, etc.)
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
        # acción: 0 = nada, 1 = saltar, 2 = agacharse
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
    def __init__(self, image, difficulty=1):
        self.type = 0
        if difficulty == 1:
            scale = 1.0  
            min_y, max_y = 250, 300
        elif difficulty == 2:
            scale = 1.3  
            min_y, max_y = 200, 300
        else:
            scale = 1.6  
            min_y, max_y = 150, 300
        scaled_images = [pygame.transform.scale(img,
                         (int(img.get_width() * scale),
                          int(img.get_height() * scale))) for img in image]
        super().__init__(scaled_images, self.type)

        self.index = 0
        self.rect.y = random.randint(min_y, max_y)

    def draw(self, screen):
        if self.index >= 9:
            self.index = 0
        screen.blit(self.image[self.index // 5], self.rect)
        self.index += 1



# ------------------------
# Entorno Gymnasium
# ------------------------
class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=SCREEN_WIDTH, shape=(4,), dtype=np.float32)
        self.clock = pygame.time.Clock()

        # Inicializamos dificultad
        self.difficulty = 1

        self.reset()

    def reset(self, seed=None, options=None):
        self.player = Dinosaur()
        self.obstacles = []
        self.game_speed = 20
        self.points = 0
        self.x_pos_bg = 0
        self.y_pos_bg = 380

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.player.update(action)

        # Actualizar dificultad según puntos (curriculum learning)
        if self.points < 500:
            self.difficulty = 1
        elif self.points < 1500:
            self.difficulty = 2
        else:
            self.difficulty = 3

        # Generar obstáculos si no hay ninguno
        if len(self.obstacles) == 0:
            choice = random.randint(0, 2)
            if choice == 0:
                self.obstacles.append(SmallCactus(SMALL_CACTUS))
            elif choice == 1:
                self.obstacles.append(LargeCactus(LARGE_CACTUS))
            else:
                # Pasar dificultad actual al pájaro
                self.obstacles.append(Bird(BIRD, difficulty=self.difficulty))

        reward = 1
        done = False
        truncated = False

        # Actualizar obstáculos y comprobar colisiones
        for obstacle in list(self.obstacles):
            obstacle.update(self.game_speed, self.obstacles)
            if self.player.dino_rect.colliderect(obstacle.rect):
                reward = -100
                done = True

        self.points += 1
        if self.points % 100 == 0:
            self.game_speed += 1

        obs = self._get_obs()
        return obs, reward, done, truncated, {}

    def _get_obs(self):
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
