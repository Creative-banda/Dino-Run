# 4.6 Creating Player, We will update the player animation to play animation of running


import pygame, sys, os

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

# ----------------   Ground --------------- #
class Ground(pygame.sprite.Sprite):
    def __init__(self, x):
        super().__init__()
        self.image = ground_image
        self.rect = self.image.get_rect()
        self.rect.left = x
        self.rect.y = HEIGHT - 60

    def move(self):
        self.rect.x -= game_speed


# ----------------   Player --------------- #
class Player():
    def __init__(self):
        self.gravity = 0.5
        self.velocity_y = 0
        self.Inair = False
        self.animation_cooldown = 80
        self.last_update = pygame.time.get_ticks()

        # Loading animations
        self.animation_list = []
        self.animation_names = ["running", "jump", "duck"]
        self.current_action = 0  # Default to running
        self.frame_index = 0
        self.load_animation()

        self.image = self.animation_list[self.current_action][self.frame_index]
        self.rect = self.image.get_rect()
        self.rect.topleft = (50,  HEIGHT - 150)

    def load_animation(self):
        """Loads all animations from assets folder."""
        for action in self.animation_names:  
            temp_list = []
            action_path = f"assets/images/dino/{action}"
                   
            num_of_frames = len(os.listdir(action_path))  # Count files

            for i in range(num_of_frames):  # Fix range (use +1)
                img_path = f"assets/images/dino/{action}/{i}.png"

                img = pygame.image.load(img_path).convert_alpha()
                img = pygame.transform.scale(img, (img.get_width() * 4, img.get_height() * 4))

                temp_list.append(img)

            self.animation_list.append(temp_list)

    def update(self):
        """Update the player animation frame."""
        if pygame.time.get_ticks() - self.last_update > self.animation_cooldown:
            self.last_update = pygame.time.get_ticks()
            self.frame_index += 1

        if self.frame_index >= len(self.animation_list[self.current_action]):
            self.frame_index = 0
        
        self.image = self.animation_list[self.current_action][self.frame_index]

    def move(self):
        dy = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and not self.Inair:
            self.velocity_y = -15   # Jumping effect
            self.Inair = True

        # Apply gravity
        self.velocity_y += self.gravity
        dy = self.velocity_y

        # Check for ground collision
        for ground in ground_group:
            if self.rect.colliderect(ground.rect) and dy >= 0:
                self.rect.bottom = ground.rect.top
                self.velocity_y = 0
                self.Inair = False

        self.rect.y += dy

    def draw(self):
        screen.blit(self.image, self.rect)


# Groups
ground_group = pygame.sprite.Group()


# Instance of classes
player = Player()
ground_1 = Ground(0)
ground_2 = Ground(WIDTH)
ground_group.add(ground_1, ground_2)

create_parallax_background()

# Main loop
clock = pygame.time.Clock()

running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen with black
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ground_1.move()
    ground_2.move()
    if ground_1.rect.right <= 0:
        ground_1.rect.left = ground_2.rect.right
    if ground_2.rect.right <= 0:
        ground_2.rect.left = ground_1.rect.right

    # Update and draw the parallax background
    update_parallax_background()

    # Update and draw the ground
    ground_group.draw(screen)

    # Update and draw the player
    player.update()
    player.move()
    player.draw()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
