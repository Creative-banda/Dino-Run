# Basics of Pygame, Use Images instead of shapes

# Task for you:
# - Rectrangle size is not same as the image size, so if image touches the rectangle, it will not be detected as a collision.
#   Your Task is to fix this by using the image size for the rectangle.

# - Image size is so big, search for how you can scale the image up/down in Pygame.


import pygame, sys

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Run")


# Load images
player_image = pygame.image.load("assets/player.png")

# Main loop
clock = pygame.time.Clock()

rect_1 = pygame.Rect(10, 100, 50, 50)
rect_2 = pygame.Rect(300, 200, 50, 50)

running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen with black
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Handle key presses to move the rectangle
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        rect_1.x -= 5
    if keys[pygame.K_RIGHT]:
        rect_1.x += 5
    if keys[pygame.K_UP]:
        rect_1.y -= 5
    if keys[pygame.K_DOWN]:
        rect_1.y += 5

    # Draw the player image
    screen.blit(player_image, rect_1)

    # Draw the first rectangle
    pygame.draw.rect(screen, (0, 255, 0), rect_1)

    # Draw the second rectangle
    pygame.draw.rect(screen, (255, 0, 0), rect_2)

    # Check for collision
    if rect_1.colliderect(rect_2):
        print("Collision detected!")


    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()


# Note: Make sure to have the image "assets/player.png" in the correct path.
# You can replace the image with any other image file you have.