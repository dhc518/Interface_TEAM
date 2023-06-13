import pygame
import sys
import random
import math

import cv2
import numpy as np

def draw_cirlce(image, x, y):
    circle_color = (0, 0, 0, 0)  # 투명 (RGBA 형식)

    # 원의 중심 좌표 무작위 생성
    center_x = x
    center_y = y

    # 원의 반지름 무작위 생성
    radius = 100

    # 원 그리기
    cv2.circle(image, (center_x, center_y), radius, circle_color, -1)


def save_image(width, height, x1,y1, x2,y2):

    # 투명한 이미지 생성 (RGBA 형식)
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :, 3] = 255  # 전체 이미지를 완전히 투명하게 설정

    # 원의 개수와 색상 설정
    num_circles = 2

    draw_cirlce(image, x1, y1)
    draw_cirlce(image, x2, y2)



    # 이미지 저장 (PNG 형식으로 저장해야 투명한 영역이 보존됩니다)
    cv2.imwrite('light.png', image)



class GameMap:
    def __init__(self):
        self.map = [
                    [2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2],
                    [2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2],
                    [2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2],
                    [2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2],
                    [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                    [2,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2],
                    [2,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2],
                    [2,2,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,2,2],
                    [1,1,1,1,1,2,1,1,1,1,1,0,1,1,0,1,1,1,1,1,2,1,1,1,1,1],
                    [1,1,1,1,1,2,1,1,1,1,1,0,1,1,0,1,1,1,1,1,2,1,1,1,1,1],
                    [1,1,1,1,1,2,1,1,0,0,0,0,0,0,0,0,0,0,1,1,2,1,1,1,1,1],
                    [1,1,1,1,1,2,1,1,0,0,0,0,0,0,0,0,0,0,1,1,2,1,1,1,1,1],
                    [1,1,1,1,1,2,1,1,0,0,0,0,0,0,0,0,0,0,1,1,2,1,1,1,1,1],
                    [2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2],
                    [1,1,1,1,1,2,1,1,0,0,0,0,0,0,0,0,0,0,1,1,2,1,1,1,1,1],
                    [1,1,1,1,1,2,1,1,0,0,0,0,0,0,0,0,0,0,1,1,2,1,1,1,1,1],
                    [1,1,1,1,1,2,1,1,0,0,0,0,0,0,0,0,0,0,1,1,2,1,1,1,1,1],
                    [1,1,1,1,1,2,1,1,0,1,1,1,1,1,1,1,1,0,1,1,2,1,1,1,1,1],
                    [1,1,1,1,1,2,1,1,0,1,1,1,1,1,1,1,1,0,1,1,2,1,1,1,1,1],
                    [2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2],
                    [2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2],
                    [2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2],
                    [2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2],
                    [1,1,2,1,1,2,1,1,0,1,1,1,1,1,1,1,1,2,1,1,2,1,1,2,1,1],
                    [1,1,2,1,1,2,1,1,0,1,1,1,1,1,1,1,1,2,1,1,2,1,1,2,1,1],
                    [2,2,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,2,2],
                    [2,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,2],
                    [2,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,2],
                    [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0],
                    ]

    def draw(self, screen):
        for y in range(29):
            for x in range(26):
                if self.map[y][x] == 1:  # 벽이 있다면
                    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(x*30+500, y*30+100, 30, 30))
                elif self.map[y][x] == 0:  # 벽이 없다면
                    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(x*30+500, y*30+100, 30, 30))
                elif self.map[y][x] == 2:  # 점수 포인트가 있다면
                    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(x*30+500, y*30+100, 30, 30))
                    pygame.draw.circle(screen, (255, 255, 255), (x*30+515, y*30+115), 1)
        

    def draw_border(self, screen, border_width):
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(500, 100, 26*30, border_width))  # 위쪽
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(500, 100, border_width, 29*30))  # 왼쪽
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(500, 29*30 - border_width+100, 26*30, border_width))  # 아래쪽
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(26*30+500 - border_width, 100, border_width, 29*30))  # 오른쪽

    
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
        self.pacman_pos = [25.5*30+500, 28.5*30+100]  # 팩맨의 시작 위치를 오른쪽 아래로 설정
       
        self.pacman_pos_enemy = [12*1+500, 12*1+100]
        self.pacman_pos_enemy2 = [12*1+500, 850+100]
        self.pacman_pos_enemy3 = [1260, 12*1+100]
        self.current_direction = None
        self.current_direction2 = None
        self.current_direction3 = None
        self.pacman_radius = 12  # 팩맨의 반지름 설정

    def update(self):
        # 팩맨 이동
        self.pacman_pos += self.pacman_vel

        # 팩맨과 점수 포인트의 중심 사이의 거리 계산
        for y in range(self.map_height):
            for x in range(self.map_width):
                if self.map[y][x] == 2:  # 점수 포인트
                    point_center = Vector2(x*self.tile_size+self.tile_size//2, y*self.tile_size+self.tile_size//2)
                    if (self.pacman_pos - point_center).length() < self.tile_size//2:
                        # 팩맨이 점수 포인트를 먹음
                        self.map[y][x] = 0
                        self.score += 1

    def check_wall_collision(self, future_pos):
        radius = 12
        future_grid_pos = [int(((future_pos[0] - radius)-500)//30), int(((future_pos[1] - radius)-100)//30)]  # top-left
        future_grid_pos2 = [int(((future_pos[0] + radius)-500)//30), int(((future_pos[1] - radius)-100)//30)]  # top-right
        future_grid_pos3 = [int(((future_pos[0] - radius)-500)//30), int(((future_pos[1] + radius)-100)//30)]  # bottom-left
        future_grid_pos4 = [int(((future_pos[0] + radius)-500)//30), int(((future_pos[1] + radius)-100)//30)]  # bottom-right

        if min(future_grid_pos[0], future_grid_pos2[0], future_grid_pos3[0], future_grid_pos4[0]) < 0 or max(future_grid_pos[0], future_grid_pos2[0], future_grid_pos3[0], future_grid_pos4[0]) >= len(self.map.map[0]) or min(future_grid_pos[1], future_grid_pos2[1], future_grid_pos3[1], future_grid_pos4[1]) < 0 or max(future_grid_pos[1], future_grid_pos2[1], future_grid_pos3[1], future_grid_pos4[1]) >= len(self.map.map):
            return True

        if self.map.map[future_grid_pos[1]][future_grid_pos[0]] == 1 or self.map.map[future_grid_pos2[1]][future_grid_pos2[0]] == 1 or self.map.map[future_grid_pos3[1]][future_grid_pos3[0]] == 1 or self.map.map[future_grid_pos4[1]][future_grid_pos4[0]] == 1:
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
            if self.current_direction is None or self.check_wall_collision([sum(x) for x in zip(self.pacman_pos_enemy, self.current_direction)]):
                valid_directions = [dir for dir in directions if not self.check_wall_collision([sum(x) for x in zip(self.pacman_pos_enemy, dir)])]
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
            if self.current_direction2 is None or self.check_wall_collision([sum(x) for x in zip(self.pacman_pos_enemy2, self.current_direction2)]):
                valid_directions = [dir for dir in directions if not self.check_wall_collision([sum(x) for x in zip(self.pacman_pos_enemy2, dir)])]
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
            if self.current_direction3 is None or self.check_wall_collision([sum(x) for x in zip(self.pacman_pos_enemy3, self.current_direction3)]):
                valid_directions = [dir for dir in directions if not self.check_wall_collision([sum(x) for x in zip(self.pacman_pos_enemy3, dir)])]
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

        next_grid_pos = [int((next_pos[0]-500)//30), int((next_pos[1]-100)//30)]
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
        if self.check_collision_with_ghost(self.pacman_pos_enemy) or self.check_collision_with_ghost(self.pacman_pos_enemy2) or self.check_collision_with_ghost(self.pacman_pos_enemy3):
            return True
        return False

    def check_collision_with_ghost(self, ghost_pos):
        distance = math.sqrt((self.pacman_pos[0] - ghost_pos[0])**2 + (self.pacman_pos[1] - ghost_pos[1])**2)
        if distance < self.pacman_radius +self.pacman_radius :
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
        while running:
            dt = clock.tick(60) / 600
            self.update_time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                
            if self.map.count_score_points() == 0 :
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

                save_image(780, 870, 0, 0, int(self.pacman_pos[0])-500, int(self.pacman_pos[1])-100)
                image_path = "light.png"  # 이미지 파일 경로 설정
                image = pygame.image.load(image_path)


                self.screen.blit(image, (500,100))

            if self.check_collision():
                self.game_over()
            if self.ttime <= 0:
                self.game_over()
            pygame.display.flip()
            clock.tick(60)  # 60 FPS

        
        

if __name__ == '__main__':
    pygame.init()
    game = Pacman()
    game.run()
