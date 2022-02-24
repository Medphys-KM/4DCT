import numpy as np
import random
import cv2


def lungwin_mask(image_org, mask, threshold):

    if mask is not None:
        lung_org = image_org * mask
    else:
        lung_org = image_org

    lung_org_min = lung_org.min()
    lung_org_max = lung_org.max()

    a_org = (threshold[1] - threshold[0]) / (lung_org_max - lung_org_min)
    b_org = (threshold[0] * lung_org_max - lung_org_min * threshold[1]) / (lung_org_max - lung_org_min)

    image_org = image_org * a_org + b_org

    image_org[image_org < threshold[0]] = threshold[0]
    image_org[image_org > threshold[1]] = threshold[1]

    return image_org



def nonlinearImg(src):

    It = random.randint(5, 30)
    alpha_sum = 0
    v = 0.1

    n = random.uniform(v, 1.0)
    if random.uniform(0, 1) > 0.5:
        n = 1.0 / n

    alpha = random.uniform(0.001, 1.0)
    newsrc = alpha * np.power(src, n)
    alpha_sum += alpha
    for i in range(It - 1):
        n = random.uniform(v, 1.0)
        if random.uniform(0, 1) > 0.5:
            n = 1.0 / n

        alpha = random.uniform(0.001, 1.0)
        newsrc += alpha * np.power(src, n)
        alpha_sum += alpha

    newsrc = newsrc / alpha_sum
    return newsrc


# 软组织图灰度增广
def generator(image, image_aug, target):

    ws = random.uniform(0.0, 0.5)
    wb = 1.0 - ws
    newsrc = image * wb + image_aug * ws

    if random.choice([True, False]):
        ws = random.uniform(0.0, 0.5)
        wb = 1.0 - ws
        newsrc = newsrc * wb + target * ws


    return newsrc



def resolution(img):
    if np.random.uniform(0,1) > 0.5 :
        shape = img.shape
        img =cv2.resize(img, (shape[1]//2, shape[0]//2))
        img = cv2.resize(img, (shape[1], shape[0]))

    return img


def generatetag(img, target, alpha):

    ys, xs = np.where(img==0)
    id = random.randint(0, len(xs)-1)
    xp = xs[id]
    yp = ys[id]

    value = np.random.uniform(alpha, 1.0000001)
    img[yp, xp] = value

    return img, target


# 软组织图训练灰度增广
def grayaug(image, target, threshold):

    #  归一化
    image = lungwin_mask(image, None, [0, 1])

    #非线形增广
    image_aug = nonlinearImg(image)

    # 使用软组织图与骨图合成图
    newimg = generator(image, image_aug, target)

    # #归一化合成图
    alpha = np.random.uniform(0.1, 1.0)
    newimg = alpha * newimg

    # 添加组织外标签
    # if np.random.uniform(0, 1) > 0.5:
    #      newimg, target = generatetag(newimg, target, alpha)

    newimg = lungwin_mask(newimg, None, threshold)

    return newimg, target























