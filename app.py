from tkinter import Label
import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
MODEL = load_model("mymodel.h5")

pygame.init()
BOUNDARYINC = 5
WINDOWSIZEX = 640
WINDOWSIZEY = 480
IMAGESAVE = False
DISPLAYSURFACE = pygame.display.set_mode((640,480))
# WHITE_INT = DISPLAYSURFACE.map_rgb(WHITE)
pygame.display.set_caption('Digit Recognizer')
PREDICT = True

iswriting = False

number_xcord = []
number_ycord = []
inmg_cnt = 1
font = pygame.font.Font('freesansbold.ttf', 32)
while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURFACE, WHITE, (xcord, ycord), 4,0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0]-BOUNDARYINC,0), min(WINDOWSIZEX, number_xcord[-1]+BOUNDARYINC)
            rect_min_y, rect_max_y = max(number_ycord[0]-BOUNDARYINC,0), min(WINDOWSIZEY, number_ycord[-1]+BOUNDARYINC)

            number_xcord = []
            number_ycord = []
            img_arr = np.array(pygame.PixelArray(DISPLAYSURFACE))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype('float32')
            if IMAGESAVE:
                cv2.imwrite('image.png')
                inmg_cnt +=1

            if PREDICT:
                image = cv2.resize(img_arr, (28,28))
                image = np.pad(image, (10,10), 'constant', constant_values = 0)
                print(image)
                image = cv2.resize(image, (28,28))/255
                # label = str(np.argmax(MODEL.predict(image.reshape((-1,28,28,1)))))
                prediction = np.argmax(MODEL.predict(image.reshape(-1,28,28,1)))
                print(prediction)
                label = str(prediction)
                text = font.render(label, True, WHITE, RED)
                DISPLAYSURFACE.blit(text, [rect_min_x+1, rect_max_y+1])
                pygame.draw.rect(DISPLAYSURFACE, RED, (rect_min_x, rect_min_y, rect_max_x-rect_min_x, rect_max_y-rect_min_y), 3)

            if event.type == KEYDOWN:
                if event.unicode == 'N':
                    DISPLAYSURFACE.fill(BLACK)

        pygame.display.update()