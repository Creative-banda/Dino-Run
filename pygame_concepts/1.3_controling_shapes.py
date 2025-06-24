# Basics of Pygame, Move shapes around the screen

# Task for you:
# - Comment the screen.fill line to see the difference.
# - Increase the speed of the rectangle when you press the arrow keys.

import pygame, sys

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Run")

# Main loop
clock = pygame.time.Clock()

rect_x = 10
rect_y = 100

running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen with black
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Handle key presses to move the rectangle
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        rect_x -= 5
    if keys[pygame.K_RIGHT]:
        rect_x += 5
    if keys[pygame.K_UP]:
        rect_y -= 5
    if keys[pygame.K_DOWN]:
        rect_y += 5
        
    # Draw rect
    pygame.draw.rect(screen, (255, 0, 0), (rect_x, rect_y, 200, 150), 5)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
