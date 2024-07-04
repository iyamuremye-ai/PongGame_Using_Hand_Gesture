import mediapipe as mp
import cv2
import pygame
import numpy as np


pygame.init()
WIDTH, HEIGHT = int(800 * 2 / 3), int(600 * 2 / 3)
WHITE = (144, 238, 144)
BLACK = (0, 100, 0)
SCORE_FONT = pygame.font.Font(None, 36)


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ID:2120236025_My final Project 2024 Web Application Dev for PONG GAME USING HAND GESTURE")


player_paddle = pygame.Rect(60, HEIGHT // 2 - 50, 10, 100)
ai_paddle = pygame.Rect(WIDTH - 60, HEIGHT // 2 - 50, 10, 100)
ball = pygame.Rect(WIDTH // 2 - 15, HEIGHT // 2 - 15, 30, 30)
ball_speed_x = 7
ball_speed_y = 7
player_score = 0
ai_score = 0


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


running = True
while running:
    ret, frame = cap.read()  # Read frame from camera
    if not ret:
        break


    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                if cx < WIDTH // 2:
                    player_paddle.y = int(np.interp(cy, [0, HEIGHT], [0, HEIGHT - player_paddle.height]))
                else:
                    ai_paddle.y = int(np.interp(cy, [0, HEIGHT], [0, HEIGHT - ai_paddle.height]))


    cv2.imshow('ID:2120236025_My final Project 2024 Web Application Dev for PONG GAME USING HAND GESTURE', frame)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Ball collision with walls
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y = -ball_speed_y
    if ball.left <= 0:
        ai_score += 1
        ball_speed_x = 7
        ball_speed_y = 7
        ball.x, ball.y = WIDTH // 2 - 15, HEIGHT // 2 - 15
    if ball.right >= WIDTH:
        player_score += 1
        ball_speed_x = -7
        ball_speed_y = -7
        ball.x, ball.y = WIDTH // 2 - 15, HEIGHT // 2 - 15


    if ball.colliderect(player_paddle) or ball.colliderect(ai_paddle):
        ball_speed_x = -ball_speed_x


    if ai_paddle.top < ball.y:
        ai_paddle.top += 7
    if ai_paddle.bottom > ball.y:
        ai_paddle.bottom -= 7


    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, player_paddle)
    pygame.draw.rect(screen, WHITE, ai_paddle)
    pygame.draw.ellipse(screen, WHITE, ball)


    score_text = f"Player: {player_score}  AI: {ai_score}"
    text_surface = SCORE_FONT.render(score_text, True, WHITE)
    screen.blit(text_surface, ((WIDTH - text_surface.get_width()) // 2, 20))


    pygame.display.flip()


    pygame.time.Clock().tick(60)


cap.release()
cv2.destroyAllWindows()
pygame.quit()