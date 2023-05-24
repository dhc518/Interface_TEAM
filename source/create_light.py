import cv2
import numpy as np

def draw_cirlce(image, x, y):
    circle_color = (0, 0, 0, 0)  # 투명 (RGBA 형식)

    # 원의 중심 좌표 무작위 생성
    center_x = x
    center_y = y

    # 원의 반지름 무작위 생성
    radius = 50

    # 원 그리기
    cv2.circle(image, (center_x, center_y), radius, circle_color, -1)


def save_image(width, height, x1,y1, x2,y2):
    print(width, height)

    # 투명한 이미지 생성 (RGBA 형식)
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :, 3] = 255  # 전체 이미지를 완전히 투명하게 설정

    # 원의 개수와 색상 설정
    num_circles = 2

    draw_cirlce(image, x1, y1)
    draw_cirlce(image, x2, y2)



    # 이미지 저장 (PNG 형식으로 저장해야 투명한 영역이 보존됩니다)
    cv2.imwrite('light.png', image)
    /*
    image_path = "light.png"  # 이미지 파일 경로 설정
    image = pygame.image.load(image_path)

    # 이미지 크기 설정 (선택 사항)
    image_width = width
    image_height = height
    image = pygame.transform.scale(image, (image_width, image_height))
    window.blit(image, (0, 0))   */ 
