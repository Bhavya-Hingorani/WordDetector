import argparse
from tokenize import String
from typing import List
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from path import Path
import os
from word_detector import detect, prepare_img, sort_multiline

# def get_img_files(data_dir: Path) -> List[Path]:
#     """Return all image files contained in a folder."""
#     res = []
#     for ext in ['*.png', '*.jpg', '*.bmp', '*.jfif']:
#         res += Path(data_dir).files(ext)
#     return res

def cropImg(xs, ys, img, i):
    #croppedImg = img.crop((xs[0], ys[0], xs[2], ys[1])) # left top right bottom
    croppedImg = img[ys[0]:ys[1], xs[0]:xs[2]]
    # plt.imshow(croppedImg)
    # print("Cropped img: {}".format(croppedImg))
    # image = Image.fromarray(croppedImg)
    # print(str(img[1][1]))
    # print(str(croppedImg[1][1]))
    image = Image.fromarray(croppedImg)
    # image.show()
    im1 = image.save("img.png")
    img_path = 'D:/project/WordDetector/examples/img.png'
    os.system('python D:/project/SimpleHTR/src/main.py --img_file {}'.format(img_path))
    
    return croppedImg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('../data/line'))
    parser.add_argument('--file_name', type=str, default='../data/custom_lines/custom_2.png')
    parsed = parser.parse_args()
    image = cv2.imread(parsed.file_name)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--sigma', type=float, default=11)
    parser.add_argument('--theta', type=float, default=7)
    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--img_height', type=int, default=image.shape[0])
    parsed = parser.parse_args()


    # print(f'Processing file {fn_img}')

    # load image and process it
    img = prepare_img(cv2.imread(parsed.file_name), parsed.img_height)
    detections = detect(img,
                        kernel_size=parsed.kernel_size,
                        sigma=parsed.sigma,
                        theta=parsed.theta,
                        min_area=parsed.min_area)

    # sort detections: cluster into lines, then sort each line
    lines = sort_multiline(detections)

    # plot results
    plt.imshow(img, cmap='gray')
    num_colors = 7
    colors = plt.cm.get_cmap('rainbow', num_colors)
    f = open("D:/project/WordDetector/Output/output.txt", "w")
    f.write("")
    for line_idx, line in enumerate(lines):        
        for word_idx, det in enumerate(line):
            xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
            ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]

            # print("---------------------")
            # print(xs)
            # print(ys)
            # print("---------------------")

            plt.plot(xs, ys, c=colors(line_idx % num_colors))
            plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')

            cropImg(xs, ys, img, word_idx)

    plt.show()


if __name__ == '__main__':
    main()
