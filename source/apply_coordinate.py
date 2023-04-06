import cv2 as cv
import numpy as np
import mediapipe as mp

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


mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE = [ 33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246 ]
LEFT_IRIS = [474,475,476,477 ]
RIGHT_IRIS = [469,470,471,472 ]
# left eye center 473, right eye center 468
FACE_HEAD_POSE_LACNMARKS = [1, 33, 61, 199, 291, 263, 6, 473, 468]

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
Ix = [''] * 9
Iy = [''] * 9
IRx = [''] * 9
IRy = [''] * 9







with mp_face_mesh.FaceMesh(max_num_faces =1,
                           refine_landmarks =True,
                           min_detection_confidence =0.5, # 높이면 정확성 up, 속도 down
                           min_tracking_confidence =0.5   # 내리면 속도 up, 정학성 down
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
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                 for p in results.multi_face_landmarks[0].landmark])

            #face_mesh 전부 표시하기 색(B,G,R)
            i = 0
            for pt in mesh_points:
                if i == 6:
                    cv.circle(frame, pt, 1, (255, 0, 0), -1, cv.LINE_AA)
                elif i == 468 :
                    cv.circle(frame, pt, 1, (255, 0, 0), -1, cv.LINE_AA)
                else:
                    #cv.circle(img, center,radius,color,thicknex,lineType)
                    cv.circle(frame, pt, 1, (255,255,255), -1, cv.LINE_AA)
                i += 1

            #print(mesh_points)
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
            #print(center_left)
            # face_direction
            face_2d = []
            face_3d = []

            for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
                if idx in FACE_HEAD_POSE_LACNMARKS:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    if idx == 6:
                        I_middle = (lm.x, lm.y)
                    if idx == 473:
                        left_center = (lm.x, lm.y)
                        pass

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            print(I_middle)
            print(left_center)

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
            #print(Ix, Iy)
            p1 = (int(I_middle[0]), int(I_middle[1]))
            #print(f'눈 사이 : {p1}')
            #print(f'왼눈, 오른눈 : {center_left}, {center_right} ')
            if Iy[8]:
                X = (Ix[2] - Ix[0] + Ix[5] - Ix[3] + Ix[8] - Ix[6]) / 6
                Y = (Iy[6] - Iy[0] + Iy[7] - Iy[1] + Iy[8] - Iy[2]) / 6
                print(X, Y)

                RX = (IRx[2] - IRx[0] + IRx[5] - IRx[3] + IRx[8] - IRx[6]) / 6
                RY = (IRy[6] - IRy[0] + IRy[7] - IRy[1] + IRy[8] - IRy[2]) / 6

                half_w = img_w / 2
                half_h = img_h / 2

                L_eye = (half_w - (Ix[4]-(center_left[0] - I_middle[0])) * half_w / X, half_h + (Iy[4] + (center_left[1] - I_middle[1])) * half_h / Y)

                R_eye = (half_w + (center_right[0] - IRx[5]) * half_w / RX, half_h + (center_left[0] - IRy[5]) * half_h / RY)

                print(center_left[0] - I_middle[0])
                print(L_eye)

                p2 = (int(L_eye[0]), int(L_eye[1]))
                #p2 = ((L_eye[0]+R_eye[0]) / 2, (L_eye[1]+R_eye[1]) / 2)

                cv.line(frame, p1, p2, (255, 255, 0), 3)



        # 이미지를 회전시켜서 img로 돌려받음
        img = Rotate(frame, 90)  # 뒷면90 or 180 or 앞면270

        # 이미지를 반전시켜 img2로 돌려받음
        img2 = Flip(frame, 1)

        #원래 이미지 표시
        cv.imshow('Main', frame)

        # 회전된 이미지 표시
        #cv.imshow('CAM_RotateWindow', img)

        #반전된 이미지 표시
        #cv.imshow('CAM_FlipWindow', img2)

        #윈도우 크기 늘리기
        resolutuon = 1080
        #dst2 = cv.resize(img, dsize=(resolution/9*16, resolution), interpolation=cv.INTER_AREA)
        #cv.imshow('CAM_RotateWindow2', dst2)

        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('u'):
            Ix[0], Iy[0] = center_left[0] - I_middle[0], center_left[1] - I_middle[1]
            IRx[0], IRy[0] = center_right[0], center_right[1]
            print(Ix[0], Iy[0])
        elif key == ord('i'):
            Ix[1], Iy[1] = center_left[0] - I_middle[0], center_left[1] - I_middle[1]
            IRx[1], IRy[1] = center_right[0], center_right[1]
            print(Ix[1], Iy[1])
        elif key == ord('o'):
            Ix[2], Iy[2] = center_left[0] - I_middle[0], center_left[1] - I_middle[1]
            IRx[2], IRy[2] = center_right[0], center_right[1]
            print(Ix[2], Iy[2])
        elif key == ord('j'):
            Ix[3], Iy[3] = center_left[0] - I_middle[0], center_left[1] - I_middle[1]
            IRx[3], IRy[3] = center_right[0], center_right[1]
            print(Ix[3], Iy[3])
        elif key == ord('k'):
            Ix[4], Iy[4] = center_left[0] - I_middle[0], center_left[1] - I_middle[1]
            IRx[4], IRy[4] = center_right[0], center_right[1]
            print(Ix[4], Iy[4])
        elif key == ord('l'):
            Ix[5], Iy[5] = center_left[0] - I_middle[0], center_left[1] - I_middle[1]
            IRx[5], IRy[5] = center_right[0], center_right[1]
            print(Ix[5], Iy[5])
        elif key == ord('n'):
            Ix[6], Iy[6] = center_left[0] - I_middle[0], center_left[1] - I_middle[1]
            IRx[6], IRy[6] = center_right[0], center_right[1]
            print(Ix[6], Iy[6])
        elif key == ord('m'):
            Ix[7], Iy[7] = center_left[0] - I_middle[0], center_left[1] - I_middle[1]
            IRx[7], IRy[7] = center_right[0], center_right[1]
            print(Ix[7], Iy[7])
        elif key == ord(','):
            Ix[8], Iy[8] = center_left[0] - I_middle[0], center_left[1] - I_middle[1]
            IRx[8], IRy[8] = center_right[0], center_right[1]
            print(Ix[8], Iy[8])


frame.release()
#메인 윈도우 제거
cv.destroyAllWindows()
#회전 원도우 제거
#cv.destroyWindow('CAM_RotateWindow')
