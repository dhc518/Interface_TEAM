import cv2 as cv
import numpy as np
import mediapipe as mp
import pygame
import random
import math
import copy


pygame.init()
screen = pygame.display.set_mode((1920, 1080))

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 128), (0, 128, 128)
]

q = [0,1,2,3,4,5,6]
random.shuffle(q)

posNum = 0

def create_points():
    global q



    center_color = random.choice(colors)
    other_colors = copy.deepcopy(colors)
    other_colors.remove(center_color)
    target_colors = other_colors
    random.shuffle(target_colors)
    target_colors.insert(q.pop(0),center_color)

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

def set_left_eye():
    global left_center, LL,LR, high_middle, low_middle
    return (left_center[0]-LL) / (LR-LL), (left_center[1]-high_middle) / (low_middle-high_middle)

def set_right_eye():
    global right_center, RL, RR, high_middle, low_middle
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
    global running
    global game_started
    global center_color, target_colors
    global clock
    global start_ticks
    global q
    global posNum

    click = False

    if len(q) == 0:
        q = [0, 1, 2, 3, 4, 5, 6]
        random.shuffle(q)
        game_started = False

    screen.fill((255, 255, 255))
    time_elapsed = (pygame.time.get_ticks() - start_ticks) / 1000

    for event in pygame.event.get():
        keys = pygame.key.get_pressed()

        if keys[pygame.K_q]:
            running = False
            pygame.quit()


        elif event.type == pygame.MOUSEBUTTONDOWN:
            if not game_started:
                game_started = True
                start_ticks = pygame.time.get_ticks()
                posNum = 4


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

def main():
    global q
    global posNum
    global running
    running = True
    global game_started
    game_started = False
    global center_color, target_colors
    center_color, target_colors = create_points()
    global clock
    clock = pygame.time.Clock()
    global start_ticks
    start_ticks = 0

    while running:

        with mp_face_mesh.FaceMesh(max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,  # 높이면 정확성 up, 속도 down
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
                    print("Recognized Face")
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
                    p1 = (int(middle[0]), int(middle[1]))
                    # print(f'눈 사이 : {p1}')
                    # print(f'왼눈, 오른눈 : {center_left}, {center_right} ')
                    if Iy[8]:
                        X = (Ix[2] - Ix[0] + Ix[5] - Ix[3] + Ix[8] - Ix[6]) / 6
                        Y = (Iy[6] - Iy[0] + Iy[7] - Iy[1] + Iy[8] - Iy[2]) / 6
                        # print(X, Y)

                        RX = (IRx[2] - IRx[0] + IRx[5] - IRx[3] + IRx[8] - IRx[6]) / 6
                        RY = (IRy[6] - IRy[0] + IRy[7] - IRy[1] + IRy[8] - IRy[2]) / 6

                        half_w = img_w / 2
                        half_h = img_h / 2

                        set_width = 935
                        set_height = 515

                        # L_eye = (half_w + (((left_center[0]-LL) / (LR-LL)) -Ix[4]) * half_w / X, half_h + (((left_center[1] - high_middle) / (low_middle - high_middle)) -Iy[4]) * half_h / Y)
                        # R_eye = (half_w + (((right_center[0]-RL) / (RR-RL)) -IRx[4]) * half_w / RX, half_h + (((right_center[1] - high_middle) / (low_middle - high_middle)) -IRy[4]) * half_h / RY)

                        L_eye = (half_w + (((left_center[0] - LL) / (LR - LL)) - Ix[4]) * set_width / X, half_h + (
                                    ((left_center[1] - high_middle) / (low_middle - high_middle)) - Iy[
                                4]) * set_height / Y)
                        R_eye = (half_w + (((right_center[0] - RL) / (RR - RL)) - IRx[4]) * set_width / RX, half_h + (
                                    ((right_center[1] - high_middle) / (low_middle - high_middle)) - IRy[
                                4]) * set_height / RY)

                        # print(L_eye)

                        # p2 = (int(L_eye[0]), int(L_eye[1]))
                        p2 = (int((L_eye[0] + R_eye[0]) / 2), int((L_eye[1] + R_eye[1]) / 2))
                        cv.line(frame, p1, p2, (255, 255, 0), 3)

                else:
                    print("Not Recognized Face")

                # 이미지를 회전시켜서 img로 돌려받음
                img = Rotate(frame, 90)  # 뒷면90 or 180 or 앞면270

                # 이미지를 반전시켜 img2로 돌려받음
                img2 = Flip(frame, 1)

                # show_window(frame)
                #cv.imshow('Main', frame)

                click = calibration_test()
                print(click)
                key = cv.waitKey(1)
                if key == ord('q'):
                    break
                elif click and results.multi_face_landmarks:
                    setting = posNum
                    print(setting)

                    Ix[setting], Iy[setting] = set_left_eye()
                    IRx[setting], IRy[setting] = set_right_eye()
                    # print(Ix[0], Iy[0])q
                    setting += 1



        frame.release()
        # 메인 윈도우 제거
        cv.destroyAllWindows()
        # 회전 원도우 제거
        # cv.destroyWindow('CAM_RotateWindow')

    pygame.quit()


main()