
import random
import sys
import cv2
import mediapipe as mp
import pygame

pygame.init()

WIDTH, HEIGHT = 900, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Runner - Hand Gesture")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 25)
big_font = pygame.font.SysFont("arial", 42, bold=True)

# Load gambar dari folder assets
RUN_FILE = "assets/lari.jpg"
JUMP_FILE = "assets/lompat2.jpg"


def load_sheet(path, boundaries):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")

    frames = []
    height, width = image.shape[:2]

    for i in range(len(boundaries) - 1):
        x1 = boundaries[i]
        x2 = boundaries[i + 1]
        frame = image[:, x1:x2].copy()

        # Menghapus sebagian besar warna abu-abu latar kotak-kotak.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray_background = (
            (hsv[:, :, 1] < 28) &
            (hsv[:, :, 2] > 125)
        )
        frame[gray_background] = (0, 255, 0)

        # Green colorkey akan dibuat transparan oleh Pygame.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surface = pygame.image.frombuffer(
            rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB"
        ).convert()
        surface.set_colorkey((0, 255, 0))

        # Ukuran karakter disamakan.
        target_h = 115
        ratio = target_h / surface.get_height()
        target_w = max(1, int(surface.get_width() * ratio))
        surface = pygame.transform.scale(surface, (target_w, target_h))
        frames.append(surface)

    return frames


try:
    # Batas potongan perkiraan untuk kedua sprite sheet.
    run_frames = load_sheet(RUN_FILE, [0, 210, 430, 620, 830, 1024])
    jump_frames = load_sheet(JUMP_FILE, [0, 280, 530, 750, 1024])
except Exception as error:
    print("Gagal memuat gambar:", error)
    pygame.quit()
    sys.exit()

# Webcam dan deteksi tangan
camera = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.6
)

def open_palm(hand):
    # Periksa apakah empat jari terangkat.
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    extended = sum(
        hand.landmark[tip].y < hand.landmark[pip].y
        for tip, pip in zip(tips, pips)
    )
    thumb_open = abs(
        hand.landmark[4].x - hand.landmark[2].x
    ) > 0.07
    return extended >= 3 and thumb_open


player_x = 100
ground_y = 390
player_y = ground_y
velocity_y = 0
jumping = False
score = 0
obstacles = []
spawn_timer = 0
frame_index = 0
game_over = False
previous_palm = False
camera_ok = True

running = True
while running:
    dt = clock.tick(60) / 1000
    jump_pressed = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE and not game_over:
                jump_pressed = True
            if event.key == pygame.K_r and game_over:
                score = 0
                obstacles.clear()
                player_y = ground_y
                velocity_y = 0
                jumping = False
                game_over = False

    # Baca webcam
    ret, camera_frame = camera.read()
    palm_detected = False

    if ret:
        camera_frame = cv2.flip(camera_frame, 1)
        result = hands.process(
            cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        )
        if result.multi_hand_landmarks:
            palm_detected = open_palm(
                result.multi_hand_landmarks[0]
            )

        cv2.putText(
            camera_frame,
            "OPEN PALM: JUMP" if palm_detected else "Hand: neutral",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 220, 0) if palm_detected else (255, 255, 255), 2
        )
        cv2.imshow("Webcam - Hand Gesture (Q to close)", camera_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
    else:
        camera_ok = False

    # Lompatan hanya dipicu saat telapak baru terdeteksi.
    if palm_detected and not previous_palm and not game_over:
        jump_pressed = True
    previous_palm = palm_detected

    if jump_pressed and not jumping and not game_over:
        velocity_y = -650
        jumping = True

    if not game_over:
        velocity_y += 1500 * dt
        player_y += velocity_y * dt

        if player_y >= ground_y:
            player_y = ground_y
            velocity_y = 0
            jumping = False

        spawn_timer += dt
        if spawn_timer >= 1.5:
            spawn_timer = 0
            obstacles.append(pygame.Rect(WIDTH, ground_y - 38, 32, 38))

        for obstacle in obstacles:
            obstacle.x -= int(300 * dt)

        obstacles = [o for o in obstacles if o.right > 0]
        score += dt * 10

    # Gambar game
    screen.fill((190, 225, 245))
    pygame.draw.rect(screen, (90, 170, 100), (0, 410, WIDTH, 90))
    pygame.draw.line(screen, (50, 120, 60), (0, 410), (WIDTH, 410), 4)

    if jumping:
        frame = jump_frames[min(2, int(frame_index / 8) % len(jump_frames))]
    else:
        frame = run_frames[int(frame_index / 6) % len(run_frames)]
    frame_index += 1

    player_rect = frame.get_rect(midbottom=(player_x, int(player_y + 20)))
    screen.blit(frame, player_rect)

    for obstacle in obstacles:
        pygame.draw.rect(screen, (180, 65, 55), obstacle, border_radius=5)
        pygame.draw.rect(screen, (100, 35, 35), obstacle, 3, border_radius=5)

    # Hitbox karakter sedikit diperkecil.
    hitbox = player_rect.inflate(-30, -18)
    for obstacle in obstacles:
        if hitbox.colliderect(obstacle) and not game_over:
            game_over = True

    screen.blit(font.render(f"Skor: {int(score)}", True, (20, 35, 55)), (20, 20))
    screen.blit(font.render("Buka telapak tangan = lompat | SPACE = lompat", True,
                            (20, 35, 55)), (20, 55))

    if not camera_ok:
        screen.blit(font.render("Webcam tidak terbaca; gunakan SPACE", True,
                                (160, 40, 40)), (20, 90))

    if game_over:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))
        title = big_font.render("GAME OVER", True, (255, 255, 255))
        retry = font.render("Tekan R untuk bermain lagi | ESC untuk keluar",
                            True, (255, 255, 255))
        screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 25)))
        screen.blit(retry, retry.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 25)))

    pygame.display.flip()

camera.release()
hands.close()
cv2.destroyAllWindows()
pygame.quit()