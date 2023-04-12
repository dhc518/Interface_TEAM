import multiprocessing
import time

import apply_coordinate
import pygame
import test







if __name__ == '__main__':
    pygame.init()
    queue1 = multiprocessing.Queue()
    queue2 = multiprocessing.Queue()

    p1 = multiprocessing.Process(target=test.main, args=(queue1, queue2))
    p2 = multiprocessing.Process(target=apply_coordinate.check_face, args=(queue1, queue2))


    p1.start()
    p2.start()

    p1.join() #test 프로그램 종료 대기
    p2.join() #mediapipe 프로그램 종료 대기