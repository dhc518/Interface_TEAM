import time
import random
import pygame

######배열을 생성하여 맵을 만듬 (테스트용)#########
# 10x10 배열 생성
array = [['empty' for _ in range(10)] for _ in range(10)]
# 35개의 block을 랜덤하게 넣기
for _ in range(35):
    while True:
        x = random.randint(0, 9)
        y = random.randint(0, 9)
        if array[x][y] == 'empty':
            array[x][y] = 'block'
            break
# 결과 출력
for row in array:
    print(row)
######################테스트용##################

class Enemy:
    def __init__(self, array_x, array_y, window_x, window_y, velocity):
        self.array_x = array_x #배열 상의 위치
        self.array_y = array_y
        self.window_x = window_x # 화면 상의 위치
        self.window_y = window_y
        #self.velocity = velocity 1초에 배열 1칸
        self.alive = True

    def draw(self, window):
        # 적을 임의로 빨간 원으로 그림
        # 배열 위치당 화면의 비율 만큼 이동 시켜서 그림
        pygame.draw.circle(window, (255, 0, 0), (self.array_x * self.window_x / 10, self.array_y * self.window_y / 10), 20)

    def check_collision(self, damage_x, dagmage_y): #공격에 대한 충돌 처리
        if damage_x == self.array_x and dagmage_y == self.array_y:
            self.alive = False


    def update(self):

        #이동 방향 설정
        direct_list = [0, 1, 2, 3] #방향
        random.shuffle(direct_list) #방향 랜덤 설정
        while True:
            direction = direct_list.pop()
            temp_x, temp_y = self.array_x, self.array_y
            if direction == 0:
                temp_y += 1
            elif direction == 1:
                temp_y -= 1
            elif direction == 2:
                temp_x += 1
            elif direction == 3:
                temp_x -= 1

            if array[temp_x][temp_y] == 'empty':# 이동한 장소가 비어있다면 중단
                self.array_x , self.array_y = temp_x, temp_y
                break
                
        # 적이 살아 있는 지에 대한 리턴값
        return self.alive


