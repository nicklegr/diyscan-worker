from tesserocr import PyTessBaseAPI, PSM
import cv2 # 4.2.0
import numpy as np

target = cv2.imread('/test/nicklegr_item_1st_row.png', cv2.IMREAD_COLOR)
target_debug = target.copy()
template = cv2.imread('/test/hand_icon.png', cv2.IMREAD_COLOR)
mask = cv2.imread('/test/hand_icon_mask.png', cv2.IMREAD_COLOR)
_, w, h = template.shape[::-1]

# 指カーソルの位置を検出
res = cv2.matchTemplate(target, template, cv2.TM_CCORR_NORMED, mask)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(target_debug, top_left, bottom_right, (255,0,0), 2)

# ふきだしの範囲を推定
# 長い名前のレシピの例: 「キュートなチューリップのリース」
balloon_center = (top_left[0] - 21, top_left[1] - 95)
balloon_top_left = (balloon_center[0] - 220, balloon_center[1] - 25)
balloon_bottom_right = (balloon_center[0] + 220, balloon_center[1] + 30)

cv2.rectangle(target_debug, balloon_top_left, balloon_bottom_right, (0,0,255), 2)

cv2.imwrite('/test/result.png', target_debug)

# OCR
balloon_top_left = (balloon_top_left[0] + 50, balloon_top_left[1] + 10)
balloon_bottom_right = (balloon_bottom_right[0] - 50, balloon_bottom_right[1])
cropped_balloon = target[balloon_top_left[1]:balloon_bottom_right[1], balloon_top_left[0]:balloon_bottom_right[0]].copy()
cv2.imwrite('/test/cropped_balloon.png', cropped_balloon)

api = PyTessBaseAPI(psm=8, lang='jpn') # PSM.AUTO
api.SetImageFile('/test/cropped_balloon.png')

print(api.GetUTF8Text())
