#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#

import cv2
import time
import argparse
from pathlib import Path
import numpy as np


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('--data',
                        help='使用するWebカメラのチャンネル [default: 0]')
    parser.add_argument('-o', '--out_path', default='./capture/',
                        help='画像の保存先 (default: ./capture/)')
    return parser.parse_args()


def get_gray(img, lower, upper):
    b, g, r = cv2.split(img)
    b = cv2.inRange(b, lower, upper)
    g = cv2.inRange(g, lower, upper)
    r = cv2.inRange(r, lower, upper)
    bg = cv2.bitwise_and(b, g)
    bgr = cv2.bitwise_and(bg, r)
    # bgr = cv2.bitwise_not(bgr)
    return bgr


def get_contours(img, src, k_size=7, sum_max=100000, num=5,
                 color=(0, 255, 255), thickness=3):
    kernel = np.ones((k_size, k_size), np.uint8)
    mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )[0]

    cnt_size = [[i, cv2.contourArea(cnt)]
                for i, cnt in enumerate(contours) if cv2.contourArea(cnt) < sum_max]
    cnt_size.sort(key=lambda x: x[-1], reverse=True)
    s_cnt = [contours[i] for i, _ in cnt_size[:int(num)]]
    return cv2.drawContours(src, s_cnt, -1, color, thickness)


def main(args):
    img = cv2.imread(args.data, cv2.IMREAD_COLOR)
    # h,w = img.shape[:2]
    # img = cv2.resize(img,(int(w / 2),int(h/ 2)))
    print(args.data, img.shape)
    cv2.imshow('test1', img)

    mask1 = get_gray(img, 0, 225)

    img2 = cv2.bitwise_and(img, img, mask=mask1)
    cv2.imshow('test2', mask1)

    img3 = get_contours(mask1, img)
    cv2.imshow('test3', img3)

    cv2.waitKey()
    exit()

    mask2 = get_white(img)
    img3 = cv2.bitwise_and(img, img, mask=mask2)
    cv2.imshow('test3', img3)

    mask3 = cv2.bitwise_not(cv2.bitwise_or(mask1, mask2))
    img4 = cv2.merge([mask3, mask2, mask1])
    cv2.imshow('test4', img4)
    cv2.imwrite('rgb.png', img4)

    mask4 = cv2.bitwise_not(cv2.bitwise_or(mask1, mask2))
    kernel = np.ones((7, 7), np.uint8)
    mask4 = cv2.morphologyEx(mask4, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(
        mask4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    s_cnt = [cnt for cnt in contours if cv2.contourArea(cnt) < 4000]
    img = cv2.drawContours(img, s_cnt, -1, (0, 0, 255), 3)
    # img5 = cv2.bitwise_and(img,img,mask=mask4)
    cv2.imshow('test5', img)
    cv2.imwrite('black.png', img)

    cv2.waitKey()


if __name__ == '__main__':
    main(command())
