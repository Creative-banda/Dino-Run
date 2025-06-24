# 2.3 Creating Background, Drawing all the backgrounds

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


# ----------------   Parallax Background/ Speed Increment --------------- #
def create_parallax_background():
    for i in range(5):
        bg_image = pygame.image.load(f"assets/images/background/bg_{i}.png").convert_alpha()
        bg_image = pygame.transform.scale(bg_image, (WIDTH, HEIGHT))
        bg.append({"x": 0, "img": bg_image}) 

def update_parallax_background():
    for i in bg:
        i["x"] -= game_speed
        screen.blit(i["img"], (i["x"], 0))
        screen.blit(i["img"], (i["x"] + WIDTH, 0))

        # Reset position if it goes off screen
        if i["x"] <= -WIDTH:
            i["x"] = 0

create_parallax_background()

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


# Right now, all our background layers are zooming by at the same speed!
# Let's make it more fun by giving each layer its own speed.
# That way, the faraway trees move slowly, and the close-up trees move fast—just like real life!