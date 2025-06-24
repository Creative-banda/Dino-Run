# Basics of Pygame, Check collision between shapes

# Task for you:
# - Change the color of the rectangle when it collides with another rectangle.
# Hint : Store color in a variable and change it when collision is detected.

import pygame, sys

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pygame Basics")

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
    
    rect_1 = pygame.draw.rect(screen, (255, 0, 0), (rect_x, rect_y, 200, 150), 5)
    rect_2 = pygame.draw.rect(screen, (0, 255, 0), (400, 300, 200, 150), 5)

    # Check for collision
    if rect_1.colliderect(rect_2):
        print("Collision detected!")


    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
