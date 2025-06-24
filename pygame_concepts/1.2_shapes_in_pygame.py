# Basics of Pygame, Create shapes and display them

# Try this task:
# 1. Take away the last number (the 5) from the end of the `pygame.draw.rect`, `pygame.draw.circle` and `pygame.draw.line` lines.
# - Watch what happens to the shapes on the screen!
# - Check Line 34, 35, 36

# 2. Create new shapes using the `pygame.draw.rect`, `pygame.draw.circle` and `pygame.draw.line` functions.
# - Use different colors, positions, and sizes.


import pygame, sys

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Run")

# Main loop
clock = pygame.time.Clock()

running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen with black
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw shapes
    pygame.draw.rect(screen, (255, 0, 0), (10, 100, 200, 150), 5)
    pygame.draw.circle(screen, (0, 255, 0), (400, 300), 75, 5)
    pygame.draw.line(screen, (0, 0, 255), (600, 100), (700, 200), 5)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()

# Parameters for creating rectangles : 
# - pygame.draw.rect(surface, color, rect, width=0)
# - Here , `surface` is the surface to draw on like `screen`
# - `color` is the color of the rectangle
# - `rect` is a tuple (x, y, width, height) defining the rectangle's position and size
# - `width` is the width of the rectangle's border width;


# Parameters for creating circles :
# - pygame.draw.circle(surface, color, center, radius, width=0)
# - Here, `surface` is the surface to draw on like `screen`
# - `color` is the color of the circle
# - `center` is a tuple (x, y) defining the center of the circle
# - `radius` is the radius of the circle
# - `width` is the width of the circle's border width;


# Parameters for creating lines :
# - pygame.draw.line(surface, color, start_pos, end_pos, width=1)
# - Here, `surface` is the surface to draw on like `screen`
# - `color` is the color of the line
# - `start_pos` is a tuple (x, y) defining the starting position of the line
# - `end_pos` is a tuple (x, y) defining the ending position of the line
# - `width` is the width of the line's border width;