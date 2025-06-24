# Basics of Pygame, Using font to display text

# Task for you:
# - Load a new custom font from the assets folder name `Chewy.ttf` and display text using it.

import pygame, sys

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pygame Basics")


# Main loop
clock = pygame.time.Clock()

# Load Default font
font = pygame.font.Font(None, 40)

# Load custom font
custom_font = pygame.font.Font("assets/retro.ttf", 40)

running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen with black
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Render text
    text = font.render("Pygame is Awesome", True, (255, 255, 255))
    screen.blit(text, (100, 50))

    # Render text with custom font
    custom_text = custom_font.render("Pygame is Awesome", True, (255, 255, 255))
    screen.blit(custom_text, (100, 150))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()