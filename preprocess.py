import numpy as np

def pad(sub_img):
    height, width, depth = sub_img.shape

    if depth != 3:
        print("GREYSCALE")

    median_buffer = {}
    mk = 0
    mvec = 0

    if width > height:
        maxp = width
        for j in range(0, width):
            for edge in [0, height - 1]:
                check = 0
                for k in range(0, 3):
                    check += (sub_img[edge][j][k] // 10) * (26 ** k)
                if check in median_buffer:
                    median_buffer[check] += 1
                else:
                    median_buffer[check] = 1
    else:
        maxp = height
        for j in range(0, height):
            for edge in [0, width - 1]:
                check = 0
                for k in range(0, 3):
                    check += (sub_img[j][edge][k] // 10) * (26 ** k)
                if check in median_buffer:
                    median_buffer[check] += 1
                else:
                    median_buffer[check] = 1

    for k in list(median_buffer.keys()):
        if(mk <= median_buffer[k]):
            mk = median_buffer[k]
            mvec = k

    b, g, r = (mvec % 26, (mvec % (26 ** 2) // 26), mvec // (26 ** 2))
    b, g, r = (b * 10, g * 10, r * 10)
    b, g, r = (b + 5, g + 5, r + 5)
    f_img = np.array([b, g, r] * (maxp ** 2),
                     dtype='uint8').reshape([maxp, maxp, 3])
    c_h = (maxp - height)//2
    c_w = (maxp - width)//2

    for j in range(c_h, c_h + height):
        for k in range(c_w, c_w + width):
            f_img[j][k] = sub_img[j - c_h][k - c_w]
    return f_img
