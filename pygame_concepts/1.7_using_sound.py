# Basics of Pygame, Loading and playing sounds

# Task for you:
# - Load a new music file from the assets folder and play with different keys.
# - Use a image and put on the background.


import pygame, sys

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pygame Basics")

is_music_playing = False

# Main loop
clock = pygame.time.Clock()

# Load custom font
custom_font = pygame.font.Font("assets/retro.ttf", 20)

# Load and set up music
music =  pygame.mixer.Sound("assets/music.mp3")

running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen with black
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if not is_music_playing:
                    music.play(-1)
                    is_music_playing = True
                else:
                    music.stop()
                    is_music_playing = False
    # Render Title 
    custom_text = custom_font.render("Press SPACE BAR To Play Music", True, (255, 255, 255))
    screen.blit(custom_text, ((WIDTH- custom_text.get_width()) // 2, 150))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()