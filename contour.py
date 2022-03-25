import json
import cv2 as cv
import numpy as np
from random import randint


# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super.default(self, obj)


def readContour(imgPath: str) -> None:
    img_color = cv.imread(imgPath)
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv.findContours(
        img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        lineColor = (randint(10, 255), randint(10, 255), randint(10, 255))
        cv.drawContours(img_color, [cnt], 0, lineColor, 3)

    # for cnt in contours:
    #     print(type(cnt))
    #     print(cnt.ndim)
    #     print(cnt.shape)
    #     print("\n------------------------------------------------\n")

    cv.imshow("result", img_color)

    print("\n------------------------------------------------\n")

    k = cv.waitKey(0)

    print(json.dumps(contours, cls=NumpyEncoder))


if __name__ == '__main__':
    readContour('images/contour-test.png')

    # for i in range(10):
    #     readContour(f'images/{i}.png')
