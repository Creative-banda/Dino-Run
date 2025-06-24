# 9.1 Sound Effects, We will load all the sounds and music we need for the game.

import pygame, sys, os, random

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Run")

# Variables
bg = []  # List to hold parallax background images
game_speed = 5  # Speed of the game, can be adjusted for difficulty
last_increment_time = pygame.time.get_ticks()  # Track last increment time

last_obstacle_spawn_time = pygame.time.get_ticks()
last_bird_spawn_time = pygame.time.get_ticks()
isAlive = False


# Score variables
score = 0
high_score = 0

# Load image
ground_image = pygame.image.load("assets/images/background/ground.png")
ground_image = pygame.transform.scale(ground_image, (WIDTH, 60))
title_img = pygame.image.load("assets/images/title.png").convert_alpha()
title_img = pygame.transform.scale(title_img, (int(title_img.get_width()), int(title_img.get_height())))


# Load Sounds
jump_sound = pygame.mixer.Sound("assets/sounds/jump.mp3")
background_music = pygame.mixer.Sound("assets/sounds/background_music.mp3")
loose_sound = pygame.mixer.Sound("assets/sounds/loose.wav")
game_start_sound = pygame.mixer.Sound("assets/sounds/game_start.mp3")
main_menu_sound = pygame.mixer.Sound("assets/sounds/starting_background.mp3")

# Font setup
score_font = pygame.font.Font("assets/font/Retro.ttf", 25)
big_font = pygame.font.Font("assets/font/Retro.ttf", 30)


# -----------------   High Score/ Score --------------- #
def load_high_score():
    """Load high score from file."""
    global high_score
    try:
        with open("high_score.txt", "r") as file:
            high_score = int(file.read())
    except FileNotFoundError:
        high_score = 0  # Default value if file doesn't exist


def save_high_score(score):
    """Save high score to file."""
    global high_score
    with open("high_score.txt", "w") as file:
        file.write(str(score))

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

def increment_game_speed():
    global game_speed, last_increment_time
    current_time = pygame.time.get_ticks()
    if current_time - last_increment_time > 7000:  # Increment speed every 7 seconds
        game_speed += 1 
        last_increment_time = current_time


#  ----------------   Score Display --------------- #
def draw_scores():
    global score, high_score
    """Draw the current score and high score on the screen."""
    score_text = score_font.render(f"Score: {score}", True, (255, 255, 255))
    high_score_text = score_font.render(f"High Score: {high_score}", True, (255, 255, 255))
    
    screen.blit(score_text, (10, 10))
    screen.blit(high_score_text, (WIDTH - high_score_text.get_width() - 10, 10))

def main_menu():
    """Display the main menu with parallax background and title image."""



    # Font setup for start text
    start_text = big_font.render("Press SPACE to Start", True, (255, 255, 255))
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                    reset_level()
        
        # Draw parallax background
        update_parallax_background()
        
        # Draw title image if loaded
        if title_img:
            title_rect = title_img.get_rect(center=(WIDTH // 2, HEIGHT // 3))
            screen.blit(title_img, title_rect)
        
        # Draw start text
        start_rect = start_text.get_rect(center=(WIDTH // 2, HEIGHT - 100))
        screen.blit(start_text, start_rect)
        
        pygame.display.flip()
        clock.tick(60)  # Limit frame rate

def reset_level():
    global player, ground_1, ground_2
    """Reset the game state to start a new game."""
    global score, isAlive, game_speed
    score = 0
    isAlive = True
    game_speed = 5   # Reset game speed
    ground_group.empty()  # Clear ground group
    obstacle_group.empty()  # Clear obstacle group

    # Recreate ground and player instances
    ground_1 = Ground(0)
    ground_2 = Ground(WIDTH)
    player = Player()

    ground_group.add(ground_1, ground_2)  # Add ground sprites to group

    load_high_score()  # Load high score at the start of the game
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
        self.ducking = False 
        self.alive = True  # New attribute to track if player is alive

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

    def change_animation(self, action):
        """Change the current animation action."""
        if self.current_action != action:
            self.current_action = action
            self.frame_index = 0
            prev_midbottom = self.rect.midbottom
            self.image = self.animation_list[self.current_action][self.frame_index]
            self.rect = self.image.get_rect()
            self.rect.midbottom = prev_midbottom  # Lock to ground level

    def move(self):
        dy = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and not self.Inair:
            self.velocity_y = -15   # Jumping effect
            self.Inair = True
        if keys[pygame.K_DOWN]:
            if self.Inair:
                self.gravity = 5  # Increase gravity when ducking in air
            else:
                self.ducking = True
        else:
            self.ducking = False


        # Apply gravity
        self.velocity_y += self.gravity
        dy = self.velocity_y

        # Check for ground collision
        for ground in ground_group:
            if self.rect.colliderect(ground.rect) and dy >= 0:
                self.rect.bottom = ground.rect.top
                self.velocity_y = 0
                self.Inair = False
                self.gravity = 0.5  # Reset gravity when on ground
        

        # Check for obstacle collision
        for obstacle in obstacle_group:
            if self.rect.colliderect(obstacle.rect):
                self.alive = False  # Set alive to False on collision

        # Changing animation based on state
        if self.Inair:
            self.change_animation(1) # Jumping animation
        elif self.ducking:
            self.change_animation(2) # Ducking animation
        else:
            self.change_animation(0)  # Running animation

        self.rect.y += dy

        return self.alive

    def draw(self):
        screen.blit(self.image, self.rect)

# ----------------   Bird --------------- #
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.x = WIDTH
        self.y = random.choice([HEIGHT - 180, HEIGHT - 200, HEIGHT - 120])  # Randomize bird height
        self.animation_list = []
        self.load_animation()
        self.frame_index = 0
        self.image = self.animation_list[self.frame_index]
        self.rect = self.image.get_rect()
        self.rect.topleft = (self.x, self.y)
        self.animation_cooldown = 100  # Milliseconds between frame updates
        self.last_update = pygame.time.get_ticks()
        self.speed = 5 

    def load_animation(self):
        """Loads all animations from assets folder."""
        num_of_frames = len(os.listdir("assets/images/bird")) 
        for i in range(num_of_frames):  # Fix range (use +1)
            img_path = f"assets/images/bird/{i}_BirdSprite.png"
            img = pygame.image.load(img_path).convert_alpha()
            img = pygame.transform.scale(img, (img.get_width() * 4, img.get_height() * 4))
            self.animation_list.append(img)

    def update(self):
        """Update bird animation frame."""
        if pygame.time.get_ticks() - self.last_update > self.animation_cooldown:
            self.last_update = pygame.time.get_ticks()
            self.frame_index += 1

        if self.frame_index >= len(self.animation_list):
            self.frame_index = 0        
        self.image = self.animation_list[self.frame_index]
    
        self.rect.x -= game_speed + self.speed
        if self.rect.right < 0:
            self.kill()

# ---------------   Obstacle --------------- #
class Obstacle(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.x, self.y = WIDTH, HEIGHT - 60  # Position the obstacle at the bottom of the screen 
        image_name = random.randint(1, 6)  # Randomly select an obstacle image
        self.image = pygame.image.load(f"assets/images/boxes/{image_name}.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (self.image.get_width() * 3, self.image.get_height() * 3))
        self.rect = self.image.get_rect()
        self.rect.bottomleft = (self.x, self.y)

    def update(self):
        """Update obstacle position."""
        self.rect.x -= game_speed
        if self.rect.right < 0:
            self.kill()

# Groups
ground_group = pygame.sprite.Group()
obstacle_group = pygame.sprite.Group()


# Instance of classes
create_parallax_background()

# Main loop
clock = pygame.time.Clock()

running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen with black
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if not isAlive:
        main_menu()
        continue
    
    score += 1  # Increment score over time

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

    # Draw scores
    draw_scores()

    # Create new obstacles
    if random.randint(1, 150) == 1 and pygame.time.get_ticks() - last_obstacle_spawn_time > 500:  # Adjust frequency of obstacles
        obstacle = Obstacle()
        obstacle_group.add(obstacle)
        last_obstacle_spawn_time = pygame.time.get_ticks()

    # Create new birds
    if random.randint(1, 400) == 1 and pygame.time.get_ticks() - last_bird_spawn_time > 2000:  # Adjust frequency of birds
        bird = Bird()
        obstacle_group.add(bird)
        last_bird_spawn_time = pygame.time.get_ticks()

    # Update and draw obstacles
    obstacle_group.update()
    obstacle_group.draw(screen)

    # Update and draw the player
    player.update()
    isAlive = player.move()
    if not isAlive:
        if score > high_score:
            background_music.stop()  # Stop background music on game over
            save_high_score(score)  # Save high score if current score is higher
    player.draw()

    increment_game_speed()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
