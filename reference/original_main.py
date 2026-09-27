import pygame, sys, os, random

# Initialize Pygame
pygame.init()

BASE_WIDTH = 800
BASE_HEIGHT = 600

# get actual screen size
WIDTH, HEIGHT = pygame.display.get_desktop_sizes()[0]
# WIDTH, HEIGHT = 480, 720

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Run")

# Get Ratio of Base to Actual Screen Size
ratio_x = WIDTH / BASE_WIDTH
ratio_y = HEIGHT / BASE_HEIGHT
ratio = min(ratio_x, ratio_y)

# Variables
bg = []  # List to hold parallax background images
game_speed = 4 * ratio_x  # Speed of the game
last_increment_time = pygame.time.get_ticks()  # Track last increment time
score = 0  # Initialize score
high_score = 0  # Initialize high score
isAlive = False
last_bird_spawn_time = pygame.time.get_ticks()
last_obstacle_spawn_time = pygame.time.get_ticks()

next_target_arrival = 0
next_entity_is_bird = False

# Font setup
score_font = pygame.font.Font("assets/font/Retro.ttf", int(25 * ratio_x))
big_font = pygame.font.Font("assets/font/Retro.ttf", int(30 * ratio_x))

# Load image
ground_image = pygame.image.load("assets/images/background/ground.png")
ground_image = pygame.transform.scale(ground_image, (WIDTH, int(40 * ratio_y)))

# Load Sounds
jump_sound = pygame.mixer.Sound("assets/sounds/jump.mp3")
background_music = pygame.mixer.Sound("assets/sounds/background_music.mp3")
loose_sound = pygame.mixer.Sound("assets/sounds/loose.wav")
game_start_sound = pygame.mixer.Sound("assets/sounds/game_start.mp3")
main_menu_sound = pygame.mixer.Sound("assets/sounds/starting_background.mp3")


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
    if current_time - last_increment_time > 10000:  # Increment speed every 10 seconds
        game_speed += 0.5 * ratio_x  # Increment speed based on ratio
        last_increment_time = current_time

#  ----------------   Score Display/ Main Menu --------------- #
def draw_scores():
    global score, high_score
    """Draw the current score and high score on the screen."""
    score_text = score_font.render(f"Score: {score}", True, (255, 255, 255))
    high_score_text = score_font.render(f"High Score: {high_score}", True, (255, 255, 255))
    
    screen.blit(score_text, (10, 10))
    screen.blit(high_score_text, (WIDTH - high_score_text.get_width() - 10, 10))

def main_menu():
    """Display the main menu with parallax background and title image."""

    main_menu_sound.play(-1)  # Play main menu background music

    # Load title image
    try:
        title_img = pygame.image.load("assets/images/title.png").convert_alpha()
        title_img = pygame.transform.scale(title_img, (int(title_img.get_width() * ratio_x), int(title_img.get_height() * ratio_y)))
    except:
        print("Warning: Could not load title image")
        title_img = None
    
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
                    game_start_sound.play()
                    main_menu_sound.stop()  # Stop main menu music
                    background_music.play(-1)
                    reset_level()  # Reset game state to start the game
        
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
    global score, isAlive, game_speed, next_target_arrival, next_entity_is_bird
    score = 0
    next_target_arrival = 0
    next_entity_is_bird = False
    isAlive = True
    game_speed = 4 * ratio_x  # Reset game speed
    ground_group.empty()  # Clear ground group
    enemy_group.empty()  # Clear enemy group
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
        self.rect.y = HEIGHT - 40 * ratio_y

    def move(self):
        self.rect.x -= game_speed

# ----------------   Player --------------- #
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.animation_names = ["running", "jump", "duck"]
        self.animation_list = []
        self.load_animation()
        self.current_action = 0
        self.frame_index = 0
        self.image = self.animation_list[self.current_action][self.frame_index]
        self.rect = self.image.get_rect()
        
        self.rect.topleft = (50, HEIGHT // 2)
        self.animation_cooldown = 80  # Milliseconds between frame updates
        self.last_update = pygame.time.get_ticks()  # Track last update time
        self.gravity = 0.8 * ratio_y  # Gravity effect
        self.velocity_y = 0  # Vertical velocity for jumping
        self.Inair = False  # Check if player is in the air
        self.ducking = False  # Check if player is ducking

    def load_animation(self):
        """Loads all animations from assets folder."""
        for action in self.animation_names:  
            temp_list = []
            action_path = f"assets/images/dino/{action}"
                   
            num_of_frames = len(os.listdir(action_path))  # Count files

            for i in range(num_of_frames):  # Fix range (use +1)
                img_path = f"assets/images/dino/{action}/{i}.png"

                img = pygame.image.load(img_path).convert_alpha()
                img = pygame.transform.scale(img, (int(img.get_width() * 2 * ratio), int(img.get_height() * 2 * ratio)))

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
        dy, alive = 0, True
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and not self.Inair:
            self.velocity_y = -14 * ratio_y  # Jumping effect
            self.Inair = True
            jump_sound.play()  # Play jump sound
        if keys[pygame.K_DOWN]:
            if self.Inair:
                self.gravity = 5 * ratio_y
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
                self.gravity = 0.8 * ratio_y

        # Check for boxes collision
        for obstacle in obstacle_group:
            if self.rect.colliderect(obstacle.rect):
                alive = False
        
        # Check for bird collision
        for bird in enemy_group:
            if self.rect.colliderect(bird.rect):
                alive = False
        
        if self.ducking:
            self.change_animation(2)
        elif self.Inair:
            self.change_animation(1)
        else:
            self.change_animation(0)

        self.rect.y += dy
        return alive

    def draw(self):
        screen.blit(self.image, self.rect)

# ----------------   Bird --------------- #
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.x = WIDTH
        self.y = random.choice([HEIGHT - 40 * ratio_y - 32 * ratio, HEIGHT - 40 * ratio_y - 60 * ratio, HEIGHT - 40 * ratio_y - 80 * ratio])  # Randomize bird height
        self.animation_list = []
        self.load_animation()
        self.frame_index = 0
        self.image = self.animation_list[self.frame_index]
        self.rect = self.image.get_rect()
        self.rect.topleft = (self.x, self.y)
        self.animation_cooldown = 100  # Milliseconds between frame updates
        self.last_update = pygame.time.get_ticks()
        self.speed = 5 * ratio_x  # Speed of the bird

    def load_animation(self):
        """Loads all animations from assets folder."""
        num_of_frames = len(os.listdir("assets/images/bird")) 
        for i in range(num_of_frames):  # Fix range (use +1)
            img_path = f"assets/images/bird/{i}_BirdSprite.png"
            img = pygame.image.load(img_path).convert_alpha()
            img = pygame.transform.scale(img, (int(img.get_width() * 2 * ratio), int(img.get_height() * 2 * ratio)))
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
        self.x, self.y = WIDTH, HEIGHT - 40 * ratio_y  # Position the obstacle at the bottom of the screen
        image_name = random.randint(1, 6)  # Randomly select an obstacle image
        self.image = pygame.image.load(f"assets/images/boxes/{image_name}.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (int(self.image.get_width() * 1.5 * ratio), int(self.image.get_height() * 1.5 * ratio)))
        self.rect = self.image.get_rect()
        self.rect.bottomleft = (self.x, self.y)

    def update(self):
        """Update obstacle position."""
        self.rect.x -= game_speed
        if self.rect.right < 0:
            self.kill()

# groups
ground_group = pygame.sprite.Group()
enemy_group = pygame.sprite.Group()
obstacle_group = pygame.sprite.Group()

create_parallax_background()

# Main loop
clock = pygame.time.Clock()

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    if not isAlive:
        main_menu()
        continue

    score += 1
    
    update_parallax_background()
    ground_1.move()
    ground_2.move()
    if ground_1.rect.right <= 0:
        ground_1.rect.left = ground_2.rect.right
    if ground_2.rect.right <= 0:
        ground_2.rect.left = ground_1.rect.right

    ground_group.draw(screen)
    
    draw_scores()  # Draw scores on the screen

    # Spawning logic based on arrival gaps to avoid impossible combinations

    # Speed of next entity
    entity_speed = game_speed + (5 * ratio_x if next_entity_is_bird else 0)
    dist_to_player = WIDTH - (50 * ratio_x)
    frames_to_arrive = dist_to_player / entity_speed
    expected_arrival_tick = score + frames_to_arrive

    if expected_arrival_tick >= next_target_arrival:
        if next_entity_is_bird:
            bird = Bird()
            enemy_group.add(bird)
        else:
            obstacle = Obstacle()
            obstacle_group.add(obstacle)

        # Decide what to spawn next and when
        gap = 40 + random.randint(0, 50)
        next_entity_is_bird = score > 200 and random.randint(1, 100) <= 40
        if next_entity_is_bird:
            gap += 15 # Birds need more gap because they move faster, give player more time

        next_target_arrival = expected_arrival_tick + gap

    # Draw obstacles
    obstacle_group.update()
    obstacle_group.draw(screen)

    # Update and draw birds
    enemy_group.update()
    enemy_group.draw(screen)

    # Player update and draw
    player.update()
    isAlive = player.move()
    if not isAlive:
        background_music.stop()
        loose_sound.play()
        game_speed = 4 * ratio_x  # Reset game speed on death
        if score > high_score:
            
            save_high_score(score)  
    player.draw()

    increment_game_speed()  # Increment game speed periodically

    pygame.display.flip()
    clock.tick(60 * ratio_y)  # Limit frame rate based on ratio

pygame.quit()
sys.exit()