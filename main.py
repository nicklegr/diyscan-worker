from tesserocr import PyTessBaseAPI, PSM
from PIL import Image
import cv2 # 4.2.0
import numpy as np

def cv2pil(image):
  if image.ndim == 2: # モノクロ
    buf = image
  elif image.shape[2] == 3: # カラー
    buf = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  elif image.shape[2] == 4: # 透過
    buf = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
  return Image.fromarray(buf)

cap = cv2.VideoCapture('/mov/nicklegr_item.mp4')
template = cv2.imread('/test/hand_icon.png', cv2.IMREAD_COLOR)
mask = cv2.imread('/test/hand_icon_mask.png', cv2.IMREAD_COLOR)

with PyTessBaseAPI(psm=PSM.AUTO, lang='jpn') as ocr:
  i = 0
  while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
      break

    print(f"frame {i}:")

    # target = cv2.imread('/test/nicklegr_item_1st_row.png', cv2.IMREAD_COLOR)
    target = frame
    # target_debug = target.copy()
    _, w, h = template.shape[::-1]

    # 指カーソルの位置を検出
    # res = cv2.matchTemplate(target, template, cv2.TM_CCORR_NORMED, mask)

    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # print(f"{max_val}, {max_loc}")

    # cv2.rectangle(target_debug, top_left, bottom_right, (255,0,0), 2)

    # 色でふきだしを抽出
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    balloon_color_bgr = np.uint8([[[173,189,78]]])
    balloon_color_hsv = cv2.cvtColor(balloon_color_bgr, cv2.COLOR_BGR2HSV)[0][0]
    balloon_color_lower = np.array([86-10,50,50])
    balloon_color_upper = np.array([86+10,255,255])

    target_mask = cv2.inRange(target_hsv, balloon_color_lower, balloon_color_upper)
    # cv2.imwrite("/test/target_mask.png", target_mask)

    # ふきだしの輪郭抽出
    contours, hierarchy = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETR_TREE
    biggest_contour = max(contours, key=lambda x:cv2.contourArea(x))
    area = cv2.contourArea(biggest_contour)
    print(f"biggest_contour: {area}")

    # ふきだしが出ていないようなら以降の処理をスキップ
    if area < 5000:
      i += 1
      continue

    # target_contours = cv2.drawContours(target, [biggest_contour], 0, (0,255,0), cv2.FILLED)
    # cv2.imwrite("/test/target_contours.png", target_contours)

    # 輪郭内を塗りつぶし
    _, target_w, target_h = target.shape[::-1]
    baloon_mask = np.zeros((target_h, target_w, 1), np.uint8)
    baloon_mask = cv2.drawContours(baloon_mask, [biggest_contour], 0, 255, cv2.FILLED)
    # cv2.imwrite("/test/baloon_mask.png", baloon_mask)

    # 元画像にマスクかけてふきだし部分を抽出
    baloon_only = cv2.bitwise_and(target, target, mask=baloon_mask)
    # cv2.imwrite("/test/baloon_only.png", baloon_only)

    # ふきだし周辺を切り出し
    # 長い名前のレシピの例: 「キュートなチューリップのリース」
    x,y,w,h = cv2.boundingRect(biggest_contour)
    balloon_top_left = (x, y)
    balloon_bottom_right = (x + w, y + h)
    cropped_balloon = baloon_only[balloon_top_left[1]:balloon_bottom_right[1], balloon_top_left[0]:balloon_bottom_right[0]].copy()
    # cv2.imwrite('/test/cropped_balloon.png', cropped_balloon)

    # グレースケール
    gray = cv2.cvtColor(cropped_balloon, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("/test/gray.png", gray)

    # ブラー
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # cv2.imwrite("/test/blur.png", blur)

    # 二値化
    _, thres = cv2.threshold(blur,200,255,cv2.THRESH_BINARY)
    # cv2.imwrite("/test/thres.png", thres)

    # OCR
    # 短い名前
    # あみ
    # みの
    # つき

    # よく似た名前のレシピ
    # アイアンウッドチェア
    # アイアンウッドチェスト
    # ダンボールチェア
    # ダンボールソファ
    # ひっこしダンボールS
    # ひっこしダンボールM
    # ひっこしダンボールL
    # たけのスツール
    # たけのスクリーン
    # バンブーなかべ
    # バンブーなゆか
    # こおりのアーチ
    # こおりのアート
    # アネモネのかんむり・クール
    # アネモネのかんむり・パープル
    # キクのステッキ
    # バラのステッキ
    # イースターなバルーンA
    # イースターなバルーンB
    # じめんのたまごのから
    # じめんのたまごのふく
    # じめんのたまごのくつ
    ocr_input = cv2pil(thres)
    ocr.SetImage(ocr_input)

    print(f"{ocr.GetUTF8Text().rstrip()}", flush=True)

    # cv2.imwrite(f"/test/mov/frame_{i:03d}.png", frame)
    i += 1

  cap.release()
