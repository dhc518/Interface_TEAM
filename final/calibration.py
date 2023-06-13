import cv2 as cv
import numpy as np
import mediapipe as mp
import pygame
import random
import math
import copy
import sys

running = True
calib_run = False
game_run = False

# pygame.init()
# screen = pygame.display.set_mode((1920, 1080))

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 128), (0, 128, 128)
]

q = [0, 1, 2, 3, 4, 5, 6, 7]
random.shuffle(q)
print(q)

posNum = 0

def test_init():
    global q
    global posNum

    q = [0, 1, 2, 3, 4, 5, 6, 7]
    random.shuffle(q)
    q.append(q[7])

    posNum = 0


def create_points():
    global q

    center_color = random.choice(colors)
    other_colors = copy.deepcopy(colors)
    other_colors.remove(center_color)
    target_colors = other_colors
    random.shuffle(target_colors)
    target_colors.insert(q.pop(0),center_color)
    print(target_colors)

    return center_color, target_colors

def draw_points(center_color, target_colors):
    pygame.draw.circle(screen, center_color, (960, 540), 50)

    positions = [
        (25, 25), (960, 25),(1895, 25),
        (25, 540), (1895, 540),
        (25, 1055), (960, 1055), (1895, 1055)
    ]


    for i, position in enumerate(positions):
        pygame.draw.circle(screen, target_colors[i], position, 25)


def draw_Ipoint(p2):
    print(p2)
    pygame.draw.circle(screen, (255,0,0), p2, 10)


def check_clicked_color(mouse_pos, center_color, target_colors):
    global posNum

    positions = [
        (25, 25), (960, 25),(1895, 25),
        (25, 540), (1895, 540),
        (25, 1055), (960, 1055), (1895, 1055)
    ]

    for i, position in enumerate(positions):
        dx, dy = mouse_pos[0] - position[0], mouse_pos[1] - position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance <= 25:
            if i>=4:
                posNum = i+1
            else:
                posNum = i
            return target_colors[i] == center_color

    return False




def draw_timer(time_elapsed):
    font = pygame.font.Font(None, 36)
    timer_text = font.render(f"{time_elapsed:.1f}s", True, (0, 0, 0))
    screen.blit(timer_text, (935, 400))

def draw_complite():
    font = pygame.font.Font(None, 36)
    complite_text = font.render("Complite & Ready", True, (0, 0, 0))
    screen.blit(complite_text, (885, 400))


#카메라 회전 함수
def Rotate(src, degrees):
    if degrees == 90:
        dst = cv.transpose(src)
        dst = cv.flip(dst, 1)

    elif degrees == 180:
        dst = cv.flip(src, -1)

    elif degrees == 270:
        dst = cv.transpose(src)
        dst = cv.flip(dst, 0)
    else:
        dst = null
    return dst

def Flip(src, direction):
    # 0: 상하 반전, 1: 좌우반전
    dst = cv.flip(src, direction)
    return dst


cam = cv.VideoCapture(0)

#카메라 이미지 해상도 얻기
width = cam.get(cv.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv.CAP_PROP_FRAME_HEIGHT)
print ('size = [%f, %f]\n' % (width, height))

#윈도우 생성 및 사이즈 변경
#cv.namedWindow('Main')
#cv.resizeWindow('Main', 1280, 720)


#회전 윈도우 생성
#cv.namedWindow('CAM_RotateWindow')
'''
                if i == 9 or i == 164: # 얼굴의 중심(상, 하)
                    cv.circle(frame, pt, 1, (255, 0, 0), -1, cv.LINE_AA)
                elif i == 468 or i == 473 : # 눈동자(좌,우) 중심
                    cv.circle(frame, pt, 1, (255, 0, 0), -1, cv.LINE_AA)
                elif i == 130 or i == 243: # 왼 눈 양끝
                    cv.circle(frame, pt, 1, (255, 0, 0), -1, cv.LINE_AA)
                elif i == 359 or i == 463: # 오른 눈 양끝
                    cv.circle(frame, pt, 1, (255, 0, 0), -1, cv.LINE_AA)
                elif i == 6:  # 양눈 중심
                    cv.circle(frame, pt, 1, (255, 0, 0), -1, cv.LINE_AA)
'''

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE = [ 33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246 ]
LEFT_IRIS = [474,475,476,477 ]
RIGHT_IRIS = [469,470,471,472 ]
# left eye center 468, right eye center 473
FACE_HEAD_POSE_LACNMARKS = [1, 33, 61, 199, 291, 263, 6, 473, 468, 9, 164, 468, 473, 130, 243, 359, 463]

'''
화면 비율 = 16 : 9 = 1920 : 1080

A = 1920 / 2 = 960
B = 1080 / 2 = 540

I = 9개의 점에 대한 눈동자의 좌표

a = ( I3(x)-I1(x)+I6(x)-I4(x)+I9(x)-I7(x) ) / 6 
b = ( I7(y)-I1(y)+I8(y)-I2(y)+I9(y)-I3(y) ) / 6

현재 눈동자 좌표 = Now_I
눈동자 좌표 
(Now_I(x) - I5) * A/a
(Now_I(y) - I5) * B/b
'''
setting = 0
Ix = [''] * 9
Iy = [''] * 9
IRx = [''] * 9
IRy = [''] * 9

def set_left_eye(left_center, LL,LR, high_middle, low_middle):
    return (left_center[0]-LL) / (LR-LL), (left_center[1]-high_middle) / (low_middle-high_middle)

def set_right_eye(right_center, RL, RR, high_middle, low_middle):
    return (right_center[0] - RL) / (RR - RL), (right_center[1] - high_middle) / (low_middle - high_middle)


def show_window(frame):
    # 원래 이미지 표시
    cv.imshow('Main', frame)

    # 회전된 이미지 표시
    # cv.imshow('CAM_RotateWindow', img)

    # 반전된 이미지 표시
    # cv.imshow('CAM_FlipWindow', img2)

    # 윈도우 크기 늘리기
    resolutuon = 1080
    # dst2 = cv.resize(img, dsize=(resolution/9*16, resolution), interpolation=cv.INTER_AREA)
    # cv.imshow('CAM_RotateWindow2', dst2)


def calibration_test():
    global calib_run
    global game_started
    global center_color, target_colors
    global clock
    global start_ticks
    global q
    global posNum

    click = False
    if calib_run == False:return click



    if len(q) == 0:
        q = [0, 1, 2, 3, 4, 5, 6, 7]
        random.shuffle(q)
        print(q)
        #q.append(q[7])
        game_started = False

    if len(q)==9:
        del q[-1]



    screen.fill((255, 255, 255))
    time_elapsed = (pygame.time.get_ticks() - start_ticks) / 1000

    for event in pygame.event.get():
        keys = pygame.key.get_pressed()

        if keys[pygame.K_q]:
            #running = False
            pygame.quit()
            calib_run = False
            return click


        elif event.type == pygame.MOUSEBUTTONDOWN:
            print(q)
            if not game_started:
                game_started = True
                start_ticks = pygame.time.get_ticks()
                posNum = 4

                return True


            elif check_clicked_color(pygame.mouse.get_pos(), center_color, target_colors):
                correct_click_position = pygame.mouse.get_pos()
                print("좌표:", correct_click_position)
                center_color, target_colors = create_points()
                start_ticks = pygame.time.get_ticks()
                click = True

    draw_points(center_color, target_colors)

    if game_started:
        draw_timer(time_elapsed)
    else:
        draw_complite()

    pygame.display.flip()
    clock.tick(60)
    return click

###################################pacman#####################################

def draw_cirlce(image, x, y):
    circle_color = (0, 0, 0, 0)  # 투명 (RGBA 형식)

    # 원의 중심 좌표 무작위 생성
    center_x = x
    center_y = y

    # 원의 반지름 무작위 생성
    radius = 100

    # 원 그리기
    cv.circle(image, (center_x, center_y), radius, circle_color, -1)


def save_image(width, height, x1, y1, x2, y2):
    # 투명한 이미지 생성 (RGBA 형식)
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :, 3] = 255  # 전체 이미지를 완전히 투명하게 설정

    # 원의 개수와 색상 설정
    num_circles = 2

    draw_cirlce(image, x1, y1)
    draw_cirlce(image, x2, y2)

    # 이미지 저장 (PNG 형식으로 저장해야 투명한 영역이 보존됩니다)
    cv.imwrite('light.png', image)


class GameMap:
    def __init__(self):
        self.map = [
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2],
            [2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2],
            [2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2],
            [2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2],
            [2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2],
            [2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2],
            [2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2],
            [1, 1, 2, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1],
            [1, 1, 2, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1],
            [2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2],
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
        ]

    def draw(self, screen):
        for y in range(29):
            for x in range(26):
                if self.map[y][x] == 1:  # 벽이 있다면
                    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(x * 30 + 500, y * 30 + 100, 30, 30))
                elif self.map[y][x] == 0:  # 벽이 없다면
                    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(x * 30 + 500, y * 30 + 100, 30, 30))
                elif self.map[y][x] == 2:  # 점수 포인트가 있다면
                    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(x * 30 + 500, y * 30 + 100, 30, 30))
                    pygame.draw.circle(screen, (255, 255, 255), (x * 30 + 515, y * 30 + 115), 1)

    def draw_border(self, screen, border_width):
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(500, 100, 26 * 30, border_width))  # 위쪽
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(500, 100, border_width, 29 * 30))  # 왼쪽
        pygame.draw.rect(screen, (255, 0, 0),
                         pygame.Rect(500, 29 * 30 - border_width + 100, 26 * 30, border_width))  # 아래쪽
        pygame.draw.rect(screen, (255, 0, 0),
                         pygame.Rect(26 * 30 + 500 - border_width, 100, border_width, 29 * 30))  # 오른쪽

    def count_score_points(self):
        count = 0
        for row in self.map:
            count += row.count(2)
        return count


class Pacman:
    def __init__(self):
        ######################게임맵의 좌표는(500,100)왼쪽 위, (1280,100)오른쪽 위, (500, 970)왼쪽 아래, (1280,970)오른쪽 아래############################
        self.screen = pygame.display.set_mode((1920, 1080))
        self.map = GameMap()
        self.start_ticks = pygame.time.get_ticks()
        self.time_limit = 0

        self.ctime = 150
        self.pacman_pos = [25.5 * 30 + 500, 28.5 * 30 + 100]  # 팩맨의 시작 위치를 오른쪽 아래로 설정

        self.pacman_pos_enemy = [12 * 1 + 500, 12 * 1 + 100]
        self.pacman_pos_enemy2 = [12 * 1 + 500, 850 + 100]
        self.pacman_pos_enemy3 = [1260, 12 * 1 + 100]
        self.current_direction = None
        self.current_direction2 = None
        self.current_direction3 = None
        self.pacman_radius = 12  # 팩맨의 반지름 설정

        self.running = True

    # def update(self):
    #     # 팩맨 이동
    #     #self.pacman_pos += self.pacman_vel
    #
    #     # 팩맨과 점수 포인트의 중심 사이의 거리 계산
    #     for y in range(self.map_height):
    #         for x in range(self.map_width):
    #             if self.map[y][x] == 2:  # 점수 포인트
    #                 point_center = Vector2(x * self.tile_size + self.tile_size // 2,
    #                                        y * self.tile_size + self.tile_size // 2)
    #                 if (self.pacman_pos - point_center).length() < self.tile_size // 2:
    #                     # 팩맨이 점수 포인트를 먹음
    #                     self.map[y][x] = 0
    #                     self.score += 1

    def check_wall_collision(self, future_pos):
        radius = 12
        future_grid_pos = [int(((future_pos[0] - radius) - 500) // 30),
                           int(((future_pos[1] - radius) - 100) // 30)]  # top-left
        future_grid_pos2 = [int(((future_pos[0] + radius) - 500) // 30),
                            int(((future_pos[1] - radius) - 100) // 30)]  # top-right
        future_grid_pos3 = [int(((future_pos[0] - radius) - 500) // 30),
                            int(((future_pos[1] + radius) - 100) // 30)]  # bottom-left
        future_grid_pos4 = [int(((future_pos[0] + radius) - 500) // 30),
                            int(((future_pos[1] + radius) - 100) // 30)]  # bottom-right

        if min(future_grid_pos[0], future_grid_pos2[0], future_grid_pos3[0], future_grid_pos4[0]) < 0 or max(
                future_grid_pos[0], future_grid_pos2[0], future_grid_pos3[0], future_grid_pos4[0]) >= len(
                self.map.map[0]) or min(future_grid_pos[1], future_grid_pos2[1], future_grid_pos3[1],
                                        future_grid_pos4[1]) < 0 or max(future_grid_pos[1], future_grid_pos2[1],
                                                                        future_grid_pos3[1],
                                                                        future_grid_pos4[1]) >= len(self.map.map):
            return True

        if self.map.map[future_grid_pos[1]][future_grid_pos[0]] == 1 or self.map.map[future_grid_pos2[1]][
            future_grid_pos2[0]] == 1 or self.map.map[future_grid_pos3[1]][future_grid_pos3[0]] == 1 or \
                self.map.map[future_grid_pos4[1]][future_grid_pos4[0]] == 1:
            return True

        return False

    def check_collision(self, pos1, pos2, pos3):
        return pos1[0] == pos2[0] and pos1[1] == pos2[1] or pos1[0] == pos3[0] and pos1[1] == pos3[1]

    def change_direction(self):
        directions = [(0, -3), (0, 3), (-3, 0), (3, 0)]
        return random.choice(directions)

    def ghost_update(self):
        directions = [(0, -3), (0, 3), (-3, 0), (3, 0)]
        # 첫 번째 업데이트 또는 벽에 부딪힌 후라면 새로운 방향을 선택합니다.
        while True:
            if self.current_direction is None or self.check_wall_collision(
                    [sum(x) for x in zip(self.pacman_pos_enemy, self.current_direction)]):
                valid_directions = [dir for dir in directions if
                                    not self.check_wall_collision([sum(x) for x in zip(self.pacman_pos_enemy, dir)])]
                self.current_direction = random.choice(valid_directions)  # 가능한 방향 중에서 하나를 선택

            # 새로운 위치를 계산합니다.
            future_pos = [self.pacman_pos_enemy[i] + self.current_direction[i] for i in range(2)]

            # 벽에 부딪치지 않는다면, 이동합니다.
            if not self.check_wall_collision(future_pos):
                self.pacman_pos_enemy = future_pos
                break
            else:
                self.current_direction = None

    def ghost_update2(self):
        directions = [(0, -3), (0, 3), (-3, 0), (3, 0)]
        # 첫 번째 업데이트 또는 벽에 부딪힌 후라면 새로운 방향을 선택합니다.
        while True:
            if self.current_direction2 is None or self.check_wall_collision(
                    [sum(x) for x in zip(self.pacman_pos_enemy2, self.current_direction2)]):
                valid_directions = [dir for dir in directions if
                                    not self.check_wall_collision([sum(x) for x in zip(self.pacman_pos_enemy2, dir)])]
                self.current_direction2 = random.choice(valid_directions)  # 가능한 방향 중에서 하나를 선택

            # 새로운 위치를 계산합니다.
            future_pos = [self.pacman_pos_enemy2[i] + self.current_direction2[i] for i in range(2)]

            # 벽에 부딪치지 않는다면, 이동합니다.
            if not self.check_wall_collision(future_pos):
                self.pacman_pos_enemy2 = future_pos
                break
            else:
                self.current_direction2 = None

    def ghost_update3(self):
        directions = [(0, -3), (0, 3), (-3, 0), (3, 0)]
        # 첫 번째 업데이트 또는 벽에 부딪힌 후라면 새로운 방향을 선택합니다.
        while True:
            if self.current_direction3 is None or self.check_wall_collision(
                    [sum(x) for x in zip(self.pacman_pos_enemy3, self.current_direction3)]):
                valid_directions = [dir for dir in directions if
                                    not self.check_wall_collision([sum(x) for x in zip(self.pacman_pos_enemy3, dir)])]
                self.current_direction3 = random.choice(valid_directions)  # 가능한 방향 중에서 하나를 선택

            # 새로운 위치를 계산합니다.
            future_pos = [self.pacman_pos_enemy3[i] + self.current_direction3[i] for i in range(2)]

            # 벽에 부딪치지 않는다면, 이동합니다.
            if not self.check_wall_collision(future_pos):
                self.pacman_pos_enemy3 = future_pos
                break
            else:
                self.current_direction3 = None

    def move_pacman(self):
        key = pygame.key.get_pressed()
        dist = 2  # 이동 거리
        future_pos = list(self.pacman_pos)  # 이동 후의 잠재적 위치

        next_pos = self.pacman_pos.copy()
        for _ in range(1):  # 이동 거리만큼 반복
            if key[pygame.K_DOWN]:  # 아래 방향키
                future_pos[1] += dist
                if not self.check_wall_collision(future_pos):
                    self.pacman_pos[1] += dist
            elif key[pygame.K_UP]:  # 위 방향키
                future_pos[1] -= dist
                if not self.check_wall_collision(future_pos):
                    self.pacman_pos[1] -= dist
            elif key[pygame.K_RIGHT]:  # 오른쪽 방향키
                future_pos[0] += dist
                if not self.check_wall_collision(future_pos):
                    self.pacman_pos[0] += dist
            elif key[pygame.K_LEFT]:  # 왼쪽 방향키
                future_pos[0] -= dist
                if not self.check_wall_collision(future_pos):
                    self.pacman_pos[0] -= dist

        next_grid_pos = [int((next_pos[0] - 500) // 30), int((next_pos[1] - 100) // 30)]
        if self.map.map[next_grid_pos[1]][next_grid_pos[0]] == 2:
            self.map.map[next_grid_pos[1]][next_grid_pos[0]] = 0  # 점수 포인트를 먹으면 그 위치는 빈 칸이 됩니다.

    def score_map(self):
        self.score_image = pygame.image.load('UI/SCORE.png')
        self.score_pos = (1500, 10)
        self.screen.blit(self.score_image, self.score_pos)
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(1600, 100, 1620, 140))
        self.font = pygame.font.Font(None, 50)
        self.score_text = self.font.render(f"{self.map.count_score_points()}", True, (255, 255, 255))
        self.screen.blit(self.score_text, (1610, 130))

    def clear_map(self):
        self.score_image = pygame.image.load('UI/SUCCESS.png')
        self.score_pos = (720, 450)
        self.screen.blit(self.score_image, self.score_pos)

    def level_map(self):
        self.score_image = pygame.image.load('UI/LEVEL.png')
        self.score_pos = (120, 10)
        self.screen.blit(self.score_image, self.score_pos)
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(150, 100, 170, 140))
        self.font = pygame.font.Font(None, 50)
        self.score_text = self.font.render(f"{1}", True, (255, 255, 255))
        self.screen.blit(self.score_text, (240, 130))

    def check_collision(self):
        if self.check_collision_with_ghost(self.pacman_pos_enemy) or self.check_collision_with_ghost(
                self.pacman_pos_enemy2) or self.check_collision_with_ghost(self.pacman_pos_enemy3):
            return True
        return False

    def check_collision_with_ghost(self, ghost_pos):
        distance = math.sqrt((self.pacman_pos[0] - ghost_pos[0]) ** 2 + (self.pacman_pos[1] - ghost_pos[1]) ** 2)
        if distance < self.pacman_radius + self.pacman_radius:
            return True
        return False

    def game_over(self):
        self.score_image = pygame.image.load('UI/FAIL.png')
        self.score_pos = (780, 450)
        self.screen.blit(self.score_image, self.score_pos)

    def update_time(self):
        self.elapsed_ticks = pygame.time.get_ticks() - self.start_ticks
        self.elapsed_seconds = self.elapsed_ticks // 1000  # 밀리초를 초로 변환
        self.current_time = self.time_limit + self.elapsed_seconds
        self.ttime = self.ctime - self.current_time

    def time_limit_l(self):
        self.score_image = pygame.image.load('UI/LIMIT.png')
        self.score_pos = (1500, 300)
        self.screen.blit(self.score_image, self.score_pos)
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(1600, 400, 1670, 400))
        self.font = pygame.font.Font(None, 50)
        self.score_text = self.font.render(f"{self.ttime}", True, (255, 255, 255))
        self.screen.blit(self.score_text, (1610, 400))

    def check_time_limit(self):
        if self.ttime <= self.time_limit:
            return True
        return False

    def run(self):
        clock = pygame.time.Clock()
        self.screen.fill((0, 0, 0))
        running = True
        self.update_game()

    def update_game(self):
        global game_run
        dt = clock.tick(60) / 600
        self.update_time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                #sys.exit()
                self.running = False
                game_run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    #sys.exit()
                    self.running = False
                    game_run = False
        if self.running == True:
            if self.map.count_score_points() == 0:
                self.clear_map()

            # 이미지 크기 설정 (선택 사항)
            # image_width = 780
            # image_height = 870
            # image = pygame.transform.scale(image, (image_width, image_height))

            if self.map.count_score_points() != 0 and self.check_collision() == False and self.ttime >= 0:
                self.move_pacman()
                self.ghost_update()
                self.ghost_update2()
                self.ghost_update3()
                self.map.draw(self.screen)
                self.score_map()
                self.level_map()
                self.time_limit_l()
                self.map.draw_border(self.screen, 1)  # 테두리 두께를 1로 설정
                pygame.draw.circle(self.screen, (255, 255, 0), self.pacman_pos, self.pacman_radius)  # 팩맨을 원으로 그림
                pygame.draw.circle(self.screen, (255, 0, 0), self.pacman_pos_enemy, self.pacman_radius)
                pygame.draw.circle(self.screen, (255, 0, 0), self.pacman_pos_enemy2, self.pacman_radius)
                pygame.draw.circle(self.screen, (255, 0, 0), self.pacman_pos_enemy3, self.pacman_radius)

                # 게임맵의 좌표는(500,100)왼쪽 위, (1280,100)오른쪽 위, (500, 970)왼쪽 아래, (1280,970)오른쪽 아래
                global p2
                sight_x, sight_y = 500, 100
                try:
                    print(p2)
                    if p2[0] < 500: sight_x = 500
                    elif p2[0] > 1280: sight_x = 1280
                    else: sight_x = int(p2[0])
                    if p2[1] < 100: sight_y = 100
                    elif p2[1] > 970: sight_y = 970
                    else: sight_y = int(p2[1])
                except NameError:
                    pass

                save_image(780, 870, sight_x- 500, sight_y- 100, int(self.pacman_pos[0]) - 500, int(self.pacman_pos[1]) - 100)
                image_path = "light.png"  # 이미지 파일 경로 설정
                image = pygame.image.load(image_path)

                self.screen.blit(image, (500, 100))

            if self.check_collision():
                self.game_over()
            if self.ttime <= 0:
                self.game_over()
            pygame.display.flip()
            clock.tick(60)  # 60 FPS

def pacman_init():
    global game
    pygame.init()
    game = Pacman()
    game.run()

def pacman():
    game.update_game()

###################################pacman#####################################

def main():
    global q
    global posNum
    global running
    global calib_run
    global screen
    global game_run
    global game_started
    game_started = False
    global center_color, target_colors
    center_color, target_colors = create_points()
    global clock
    clock = pygame.time.Clock()
    global start_ticks
    start_ticks = 0
    global p2

    while running:

        with mp_face_mesh.FaceMesh(max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.8,  # 높이면 정확성 up, 속도 down
                                   min_tracking_confidence=0.5  # 내리면 속도 up, 정학성 down
                                   ) as face_mesh:
            while True:
                if cam.get(cv.CAP_PROP_POS_FRAMES) == cam.get(cv.CAP_PROP_FRAME_COUNT):
                    cam.set(cv.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cam.read()
                if not ret:
                    break
                img_h, img_w = frame.shape[:2]
                results = face_mesh.process(frame)
                if results.multi_face_landmarks:
                    #print("Recognized Face")
                    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                            for p in results.multi_face_landmarks[0].landmark])

                    # face_mesh 전부 표시하기 색(B,G,R)
                    i = 0
                    for pt in mesh_points:

                        if i == 9 or i == 164:  # 얼굴의 중심(상, 하)
                            cv.circle(frame, pt, 1, (255, 0, 0), -1, cv.LINE_AA)
                        elif i == 468 or i == 473:  # 눈동자(좌,우) 중심
                            cv.circle(frame, pt, 1, (255, 0, 0), -1, cv.LINE_AA)
                        elif i == 130 or i == 243:  # 왼 눈 양끝
                            cv.circle(frame, pt, 1, (255, 0, 0), -1, cv.LINE_AA)
                        elif i == 359 or i == 463:  # 오른 눈 양끝
                            cv.circle(frame, pt, 1, (255, 0, 0), -1, cv.LINE_AA)
                        elif i == 6:  # 양눈 중심
                            cv.circle(frame, pt, 1, (255, 0, 0), -1, cv.LINE_AA)
                        else:
                            # cv.circle(img, center,radius,color,thicknex,lineType)
                            cv.circle(frame, pt, 1, (255, 255, 255), -1, cv.LINE_AA)
                        i += 1

                    # print(mesh_points)
                    cv.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 2, cv.LINE_AA)
                    cv.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 2, cv.LINE_AA)
                    cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0, 0, 255), 2, cv.LINE_AA)
                    cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0, 0, 255), 2, cv.LINE_AA)
                    (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                    (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)
                    cv.circle(frame, center_left, int(l_radius), (0, 0, 255), 2, cv.LINE_AA)
                    cv.circle(frame, center_right, int(r_radius), (0, 0, 255), 2, cv.LINE_AA)
                    # print(center_left)
                    # face_direction
                    face_2d = []
                    face_3d = []

                    for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
                        if idx in FACE_HEAD_POSE_LACNMARKS:
                            # LL, LR, RR, RL, high_middle, low_middle
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                            if idx == 6:
                                middle = (lm.x * img_w, lm.y * img_h)
                            if idx == 9:
                                high_middle = lm.y
                            if idx == 164:
                                low_middle = lm.y
                            if idx == 468:
                                left_center = (lm.x, lm.y)
                            if idx == 130:
                                LL = lm.x
                            if idx == 243:
                                LR = lm.x
                            if idx == 473:
                                right_center = (lm.x, lm.y)
                            if idx == 463:
                                RL = lm.x
                            if idx == 359:
                                RR = lm.x

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_len = 1 * img_w

                    camera_mat = np.array([[focal_len, 0, img_h / 2],
                                           [0, focal_len, img_w / 2],
                                           [0, 0, 1]])

                    dist_mat = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, camera_mat, dist_mat)

                    rot_mat, jac = cv.Rodrigues(rot_vec)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rot_mat)
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360

                    # nose+3d_projection, jacobian = cv.projectionPoints(nose_3d, rot_vec, trans_vec, camera_mat, dist_mat)
                    # print(Ix, Iy)
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    # print(f'눈 사이 : {p1}')
                    # print(f'왼눈, 오른눈 : {center_left}, {center_right} ')
                    if Iy[0] and Iy[1] and Iy[2] and Iy[3] and Iy[8] and Iy[5] and Iy[6] and Iy[7]:
                        #print('complete')
                        X = (Ix[2] - Ix[0] + Ix[5] - Ix[3] + Ix[8] - Ix[6]) / 6
                        Y = (Iy[6] - Iy[0] + Iy[7] - Iy[1] + Iy[8] - Iy[2]) / 6

                        half_w = img_w / 2
                        half_h = img_h / 2

                        set_width = 935
                        set_height = 515



                        p2 = (int((nose_2d[0]-Ix[4])/X*935),int((nose_2d[1]-Iy[4])/Y*515))
                        cv.line(frame, p1, p2, (255, 255, 0), 3)
                        #print(p2)

                else:
                    #print("Not Recognized Face")
                    pass

                # 이미지를 회전시켜서 img로 돌려받음
                img = Rotate(frame, 90)  # 뒷면90 or 180 or 앞면270

                # 이미지를 반전시켜 img2로 돌려받음
                img2 = Flip(frame, 1)

                show_window(frame)
                cv.imshow('Main', frame)

                click = calibration_test() #캘리브레이션

                if game_run == True: pacman()

                #print(click)
                key = cv.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('!'):
                    calib_run = True
                    pygame.init()
                    test_init()
                    screen = pygame.display.set_mode((1920, 1080))
                elif key == ord('@'):
                    game_run = True
                    pacman_init()
                    #pygame.init()
                    #screen = pygame.display.set_mode((1920, 1080))
                elif click and results.multi_face_landmarks:
                    print(posNum)
                    setting = posNum
                    Ix[setting], Iy[setting] = nose_2d[0], nose_2d[1]
                    print(Ix, Iy)

                elif key == ord('\\'):
                    setting = setting % 9
                    print(setting)

                    Ix[setting], Iy[setting] = nose_2d[0], nose_2d[1]
                    setting += 1
                elif key == ord('u'):
                    Ix[0], Iy[0] = nose_2d[0], nose_2d[1]
                elif key == ord('i'):
                    Ix[1], Iy[1] = nose_2d[0], nose_2d[1]
                elif key == ord('o'):
                    Ix[2], Iy[2] = nose_2d[0], nose_2d[1]
                elif key == ord('j'):
                    Ix[3], Iy[3] = nose_2d[0], nose_2d[1]
                    print(Ix[3], Iy[3])
                elif key == ord('k'):
                    Ix[4], Iy[4] = nose_2d[0], nose_2d[1]
                    print(Ix[4], Iy[4])
                elif key == ord('l'):
                    Ix[5], Iy[5] = nose_2d[0], nose_2d[1]
                    print(Ix[5], Iy[5])
                elif key == ord('n'):
                    Ix[6], Iy[6] = nose_2d[0], nose_2d[1]
                    print(Ix[6], Iy[6])
                elif key == ord('m'):
                    Ix[7], Iy[7] = nose_2d[0], nose_2d[1]
                    print(Ix[7], Iy[7])
                elif key == ord(','):
                    Ix[8], Iy[8] = nose_2d[0], nose_2d[1]
                    print(Ix[8], Iy[8])



        frame.release()
        # 메인 윈도우 제거
        cv.destroyAllWindows()
        # 회전 원도우 제거
        # cv.destroyWindow('CAM_RotateWindow')


main()