import os

from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from django.shortcuts import render
import mediapipe as mp
from keras.models import load_model
import webbrowser
import cvzone
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import random
from datetime import time
import math
import time


def index(request):
    return render(request,'index.html')

def music1(request):

    model = load_model("model.h5")
    label = np.load("labels.npy")

    holistic = mp.solutions.holistic
    hands = mp.solutions.hands
    holis = holistic.Holistic()
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    last_open_time = 0
    time_gap = 5  # Gap of 5 seconds

    while True:
        lst = []

        _, frm = cap.read()

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]

            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            # Open YouTube playlist based on the predicted emotion
            if pred and time.time() - last_open_time > time_gap:
                last_open_time = time.time()
                lang = ""  # Replace with your preferred language
                singer = ""  # Replace with your preferred singer
                webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{pred}+song+{singer}")

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        cv2.imshow("window", frm)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
    return render(request,'index.html')

def doctstr(request):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    video = cv2.VideoCapture(0)

    video.set(3, 1000)
    video.set(4, 780)

    img_1 = cv2.imread('magic_circles/magic_circle_ccw.png', -1)
    img_2 = cv2.imread('magic_circles/magic_circle_cw.png', -1)

    deg = 0

    def position_data(lmlist):
        global wrist, thumb_tip, index_mcp, index_tip, midle_mcp, midle_tip, ring_tip, pinky_tip
        wrist = (lmlist[0][0], lmlist[0][1])
        thumb_tip = (lmlist[4][0], lmlist[4][1])
        index_mcp = (lmlist[5][0], lmlist[5][1])
        index_tip = (lmlist[8][0], lmlist[8][1])
        midle_mcp = (lmlist[9][0], lmlist[9][1])
        midle_tip = (lmlist[12][0], lmlist[12][1])
        ring_tip = (lmlist[16][0], lmlist[16][1])
        pinky_tip = (lmlist[20][0], lmlist[20][1])

    def draw_line(p1, p2, size=5):
        cv2.line(img, p1, p2, (50, 50, 255), size)
        cv2.line(img, p1, p2, (255, 255, 255), round(size / 2))

    def calculate_distance(p1, p2):
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        lenght = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)
        return lenght

    def transparent(targetImg, x, y, size=None):
        if size is not None:
            targetImg = cv2.resize(targetImg, size)

        newFrame = img.copy()
        b, g, r, a = cv2.split(targetImg)
        overlay_color = cv2.merge((b, g, r))
        mask = cv2.medianBlur(a, 1)
        h, w, _ = overlay_color.shape
        roi = newFrame[y:y + h, x:x + w]

        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
        newFrame[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

        return newFrame

    while True:
        ret, img = video.read()
        img = cv2.flip(img, 1)
        rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgbimg)
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(hand.landmark):
                    h, w, c = img.shape
                    coorx, coory = int(lm.x * w), int(lm.y * h)
                    lmList.append([coorx, coory])
                    # cv2.circle(img, (coorx, coory),6,(50,50,255), -1)
                # mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
                position_data(lmList)
                palm = calculate_distance(wrist, index_mcp)
                distance = calculate_distance(index_tip, pinky_tip)
                ratio = distance / palm
                print(ratio)
                if (1.3 > ratio > 0.5):
                    draw_line(wrist, thumb_tip)
                    draw_line(wrist, index_tip)
                    draw_line(wrist, midle_tip)
                    draw_line(wrist, ring_tip)
                    draw_line(wrist, pinky_tip)
                    draw_line(thumb_tip, index_tip)
                    draw_line(thumb_tip, midle_tip)
                    draw_line(thumb_tip, ring_tip)
                    draw_line(thumb_tip, pinky_tip)
                if (ratio > 1.3):
                    centerx = midle_mcp[0]
                    centery = midle_mcp[1]
                    shield_size = 3.0
                    diameter = round(palm * shield_size)
                    x1 = round(centerx - (diameter / 2))
                    y1 = round(centery - (diameter / 2))
                    h, w, c = img.shape
                    if x1 < 0:
                        x1 = 0
                    elif x1 > w:
                        x1 = w
                    if y1 < 0:
                        y1 = 0
                    elif y1 > h:
                        y1 = h
                    if x1 + diameter > w:
                        diameter = w - x1
                    if y1 + diameter > h:
                        diameter = h - y1
                    shield_size = diameter, diameter
                    ang_vel = 2.0
                    deg = deg + ang_vel
                    if deg > 360:
                        deg = 0
                    hei, wid, col = img_1.shape
                    cen = (wid // 2, hei // 2)
                    M1 = cv2.getRotationMatrix2D(cen, round(deg), 1.0)
                    M2 = cv2.getRotationMatrix2D(cen, round(360 - deg), 1.0)
                    rotated1 = cv2.warpAffine(img_1, M1, (wid, hei))
                    rotated2 = cv2.warpAffine(img_2, M2, (wid, hei))
                    if (diameter != 0):
                        img = transparent(rotated1, x1, y1, shield_size)
                        img = transparent(rotated2, x1, y1, shield_size)

        # print(result)
        cv2.imshow("Image", img)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    return render(request,'index.html')

def handSnake(request):
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    class SnakeGameCLass:
        def _init_(self, pathFood):
            self.points = []  # all points of the snake
            self.lengths = []  # distance between each point
            self.currentLength = 0  # total langth of snake
            self.allowedLength = 150  # total allowed length
            self.previousHead = 0, 0  # previous head

            self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
            self.hFood, self.wFood, _ = self.imgFood.shape
            self.foodPoint = 0, 0
            self.randomFoodLocation()
            self.score = 0
            self.gameOver = False

        def randomFoodLocation(self):
            self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

        def resetScore(self):
            self.score = 0

        def update(self, imgMain, currentHead):

            if self.gameOver:
                cvzone.putTextRect(imgMain, "GameOver", [300, 400], scale=7, thickness=5, offset=20)
                cvzone.putTextRect(imgMain, f'Your Score : {self.score}', [300, 550], scale=7, thickness=5, offset=20)

            else:
                px, py = self.previousHead
                cx, cy = currentHead

                self.points.append([cx, cy])
                distance = math.hypot(cx - px, cy - py)
                self.lengths.append(distance)
                self.currentLength += distance
                self.previousHead = cx, cy

                # Length Reduction
                if self.currentLength > self.allowedLength:
                    for i, length in enumerate(self.lengths):
                        self.currentLength -= length
                        self.lengths.pop(i)
                        self.points.pop(i)
                        if self.currentLength < self.allowedLength:
                            break

                # Check if snake ate the food
                rx, ry = self.foodPoint
                if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                    self.randomFoodLocation()
                    self.allowedLength += 50
                    self.score += 1
                    print(self.score)

                # Draw Snake
                if self.points:
                    for i, point in enumerate(self.points):
                        if i != 0:
                            cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)
                    cv2.circle(img, self.points[-1], 16, (200, 100, 200), cv2.FILLED)

                # Check for Collision
                pts = np.array(self.points[:-2], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(imgMain, [pts], False, (0, 200, 0), 3)
                minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

                if -1 <= minDist <= 1:
                    self.gameOver = True
                    self.points = []  # all points of the snake
                    self.lengths = []  # disntance between each point
                    self.currentLength = 0  # total langth of snake
                    self.allowedLength = 150  # total allowed length
                    self.previousHead = 0, 0  # previous head

                # Draw Food
                rx, ry = self.foodPoint
                imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

            return imgMain

    game = SnakeGameCLass("Donut.png")

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False)

        if hands:
            lmList = hands[0]['lmList']
            pointIndex = lmList[8][0:2]
            img = game.update(img, pointIndex)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('r'):
            game.gameOver = False
            game.resetScore()
    cv2.destroyAllWindows()
    return render(request,'index.html')

def bottSnake(request):
    import time
    # Initializing font for puttext
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    # Loading apple image and making its mask to overlay on the video feed
    apple = cv2.imread("Apple-Fruit-Download-PNG.png", -1)
    apple_mask = apple[:, :, 3]
    apple_mask_inv = cv2.bitwise_not(apple_mask)
    apple = apple[:, :, 0:3]

    # Resizing apple images
    apple = cv2.resize(apple, (40, 40), interpolation=cv2.INTER_AREA)
    apple_mask = cv2.resize(apple_mask, (40, 40), interpolation=cv2.INTER_AREA)
    apple_mask_inv = cv2.resize(apple_mask_inv, (40, 40), interpolation=cv2.INTER_AREA)

    # Initializing a black blank image
    blank_img = np.zeros((480, 640, 3), np.uint8)

    # Capturing video from webcam
    video = cv2.VideoCapture(0)

    # Kernels for morphological operations
    kernel_erode = np.ones((4, 4), np.uint8)
    kernel_close = np.ones((15, 15), np.uint8)

    # Variables for the game
    point_x, point_y = 0, 0
    last_point_x, last_point_y = 0, 0
    dist, length = 0, 0
    snake_len, score, temp, q = 200, 0, 1, 0
    start_time = int(time.time())
    points, list_len = [], []
    random_x = random.randint(10, 550)
    random_y = random.randint(10, 400)

    # Functions for detecting sky blue color and intersection of line segments
    def detect_sky_blue(hsv):
        lower = np.array([100, 100, 100])  # Lower hue, saturation, and value (adjust these values as needed)
        upper = np.array([130, 255, 255])  # Upper hue, saturation, and value (adjust these values as needed)

        mask_sky_blue = cv2.inRange(hsv, lower, upper)
        mask_sky_blue = cv2.erode(mask_sky_blue, kernel_erode, iterations=1)
        mask_sky_blue = cv2.morphologyEx(mask_sky_blue, cv2.MORPH_CLOSE, kernel_close)

        return mask_sky_blue

    def orientation(p, q, r):
        val = int(((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1])))
        if val == 0:
            # Linear
            return 0
        elif val > 0:
            # Clockwise
            return 1
        else:
            # Anti-clockwise
            return 2

    def intersect(p, q, r, s):
        o1 = orientation(p, q, r)
        o2 = orientation(p, q, s)
        o3 = orientation(r, s, p)
        o4 = orientation(r, s, q)
        if o1 != o2 and o3 != o4:
            return True
        return False

    # Main loop
    while True:
        xr, yr, wr, hr = 0, 0, 0, 0
        _, frame = video.read()

        # Flipping the frame horizontally.
        frame = cv2.flip(frame, 1)

        # Initializing the accepted points so that they are not at the top left corner
        if q == 0 and point_x != 0 and point_y != 0:
            last_point_x = point_x
            last_point_y = point_y
            q = 1

        # Converting to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detecting sky blue color
        mask_sky_blue = detect_sky_blue(hsv)

        # Finding contours
        contour_sky_blue, _ = cv2.findContours(mask_sky_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Drawing rectangle around the accepted blob
        try:
            for i in range(0, 10):
                xr, yr, wr, hr = cv2.boundingRect(contour_sky_blue[i])
                if wr * hr > 2000:
                    break
        except:
            pass

        cv2.rectangle(frame, (xr, yr), (xr + wr, yr + hr), (255, 0, 0), 2)

        # Making snake body
        point_x = int(xr + (wr / 2))
        point_y = int(yr + (hr / 2))

        # Finding distance between the last point and the current point
        dist = int(math.sqrt(pow((last_point_x - point_x), 2) + pow((last_point_y - point_y), 2)))

        if point_x != 0 and point_y != 0 and dist > 5:
            # If the point is accepted, it is added to points list and its length added to list_len
            list_len.append(dist)
            length += dist
            last_point_x = point_x
            last_point_y = point_y
            points.append([point_x, point_y])

        # If length becomes greater than the expected length, removing points from the back to decrease length
        if length >= snake_len:
            for i in range(len(list_len)):
                length -= list_len[0]
                list_len.pop(0)
                points.pop(0)
                if length <= snake_len:
                    break

        # Initializing blank black image
        blank_img = np.zeros((480, 640, 3), np.uint8)

        # Drawing the lines between all the points
        for i, j in enumerate(points):
            if i == 0:
                continue
            cv2.line(blank_img, (points[i - 1][0], points[i - 1][1]), (j[0], j[1]), (0, 0, 255), 5)

        cv2.circle(blank_img, (last_point_x, last_point_y), 5, (10, 200, 150), -1)

        # If snake eats apple, increase score and find a new position for apple
        if (last_point_x > random_x and last_point_x < (random_x + 40) and last_point_y > random_y and last_point_y < (
                random_y + 40)):
            score += 1
            random_x = random.randint(10, 550)
            random_y = random.randint(10, 400)

        # Adding blank image to captured frame
        frame = cv2.add(frame, blank_img)

        # Adding apple image to frame
        roi = frame[random_y:random_y + 40, random_x:random_x + 40]
        img_bg = cv2.bitwise_and(roi, roi, mask=apple_mask_inv)
        img_fg = cv2.bitwise_and(apple, apple, mask=apple_mask)
        dst = cv2.add(img_bg, img_fg)
        frame[random_y:random_y + 40, random_x:random_x + 40] = dst

        cv2.putText(frame, str("Score - " + str(score)), (250, 450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Checking for snake hitting itself
        if len(points) > 5:
            b = points[len(points) - 2]
            a = points[len(points) - 1]

            for i in range(len(points) - 3):
                c = points[i]
                d = points[i + 1]
                if intersect(a, b, c, d) and len(c) != 0 and len(d) != 0:
                    temp = 0
                    break

            if temp == 0:
                break

        cv2.imshow("frame", frame)

        # Increasing the length of snake 40px per second
        if (int(time.time()) - start_time) > 1:
            snake_len += 40
            start_time = int(time.time())

        key = cv2.waitKey(1)
        if key == 27:
            break

    video.release()
    cv2.destroyAllWindows()
    cv2.putText(frame, str("Game Over!"), (100, 230), font, 3, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, str("Press any key to Exit."), (180, 260), font, 1, (255, 200, 0), 2, cv2.LINE_AA)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return render(request,'index.html')

def execBackgroundRem(request):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    segmentor = SelfiSegmentation()
    fpsReader = cvzone.FPS()
    indexImg = 0

    listImg = os.listdir("ImagesForBackgroundRemover")
    print(listImg)

    imgList = []

    for imgPath in listImg:
        img = cv2.imread(f'ImagesForBackgroundRemover/{imgPath}')
        imgList.append(img)

    while True:
        success, img = cap.read()
        imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.8)
        _, imgOut = fpsReader.update(imgOut, color=(0, 0, 255))

        cv2.imshow("Image", imgOut)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('a'):
            if indexImg > 0:
                indexImg -= 1
        elif key == ord('d'):
            if indexImg < len(listImg) - 1:
                indexImg += 1

    cv2.destroyAllWindows()
    return render(request,'index.html')


def execInteractiveReading(request):
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)

    textList = ["Welcome To", "my Python Project . ",
                "It is an", "interactive reading Panel .",
                "I hope you like it . "]

    while True:
        success, img = cap.read()
        imgText = np.zeros_like(img)
        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]

            w, _ = detector.findDistance(pointLeft, pointRight)
            W = 6.3

            # Finding the distance or depth
            f = 500
            d = (W * f) / w
            print(d)

            cvzone.putTextRect(img, f'Depth : {int(d)} cm', (face[10][0] - 100, face[10][1] - 50), scale=2)

            for i, text in enumerate(textList):
                singleHeight = 20 + int(d / 5)
                scale = 0.4 + (int(d / 10) * 10) / 80
                cv2.putText(imgText, text, (50, 50 + (i * singleHeight)), cv2.FONT_ITALIC, scale, (255, 255, 255), 2)

        imgStacked = cvzone.stackImages([img, imgText], 2, 1)
        cv2.imshow("Image", imgStacked)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    return render(request,'index.html')