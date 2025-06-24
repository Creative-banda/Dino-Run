# 3.1 Creating Ground. Creating a class named Ground that use pygame's sprite system to manage the ground layer.

import pygame, sys

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Run")

# Variables
bg = []  # List to hold parallax background images
game_speed = 5  # Speed of the game, can be adjusted for difficulty

# Load image
ground_image = pygame.image.load("assets/images/background/ground.png")
ground_image = pygame.transform.scale(ground_image, (WIDTH, 60))

# ----------------   Parallax Background/ Speed Increment --------------- #
def create_parallax_background():
    for i in range(5):
        bg_image = pygame.image.load(f"assets/images/background/bg_{i}.png").convert_alpha()
        bg_image = pygame.transform.scale(bg_image, (WIDTH, HEIGHT))
        bg.append({"x": 0, "img": bg_image, "scroll": i * 0.25})  # Adjust scroll speed for parallax effect

def update_parallax_background():
    for i in bg:
        i["x"] -= game_speed * i["scroll"]
        screen.blit(i["img"], (i["x"], 0))
        screen.blit(i["img"], (i["x"] + WIDTH, 0))

        # Reset position if it goes off screen
        if i["x"] <= -WIDTH:
            i["x"] = 0

create_parallax_background()

# ----------------   Ground --------------- #
class Ground(pygame.sprite.Sprite):
    def __init__(self, x):
        super().__init__()
        self.image = ground_image
        self.rect = self.image.get_rect()
        self.rect.left = x
        self.rect.y = HEIGHT - 60

    def update(self):
        self.rect.x -= game_speed


# Main loop
clock = pygame.time.Clock()

running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen with black
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update and draw the parallax background
    update_parallax_background()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
