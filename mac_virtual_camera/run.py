"""
Read from the computer's camera and save images to bitmaps
"""

import argparse
import functools
import os
import time
import math

import cv2
import torch
from torch import nn
import numpy as np

WINDOW_NAME = "camera"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def rgb_to_gray(img_nchw):
    r = img_nchw[:, 0, :, :]
    g = img_nchw[:, 1, :, :]
    b = img_nchw[:, 2, :, :]
    # https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).unsqueeze(dim=1)


def gray_to_rgb(img_nchw):
    return torch.cat([img_nchw, img_nchw, img_nchw], dim=1)


def create_gaussian_conv(kernel_size=7, sigma=3):
    # https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_conv = nn.Conv2d(
        in_channels=1, out_channels=1, kernel_size=(kernel_size, kernel_size), bias=False, padding=(kernel_size//2, kernel_size//2)
    )
    gaussian_conv.weight.data.copy_(gaussian_kernel)
    return gaussian_conv


class SobelImageProcessor(nn.Module):
    def __init__(self):
        super().__init__()

        self.gaussian_conv = create_gaussian_conv(kernel_size=7, sigma=1)
        self.vertical_conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=(1, 1)
        )
        self.horizontal_conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=(1, 1)
        )

        sobel_vertical_kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.vertical_conv.weight.data.copy_(sobel_vertical_kernel)
        self.horizontal_conv.weight.data.copy_(sobel_vertical_kernel.transpose(0, 1))

    def forward(self, img):
        gray_img = self.gaussian_conv(rgb_to_gray(img))
        grad_vert = self.vertical_conv(gray_img)
        grad_horiz = self.horizontal_conv(gray_img)
        grad = torch.sqrt(grad_vert**2 + grad_horiz**2)
        normalized_grad = (grad - grad.min()) / (grad.max() - grad.min())
        return gray_to_rgb(normalized_grad)


def process_image_with_model(model, img):
    with torch.no_grad():
        input_hwc = torch.from_numpy(img).to(dtype=torch.float32) / 255
        input_nhwc = input_hwc.unsqueeze(dim=0)
        input_nchw = input_nhwc.permute(0, 3, 1, 2)
        output_nchw = model(input_nchw)
        output_nhwc = output_nchw.permute(0, 2, 3, 1)
        processed_img = (output_nhwc[0] * 255).to(dtype=torch.uint8).numpy()
    return processed_img


def save_image(img, path):
    assert img.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    name, ext = os.path.splitext(path)
    tmp_path = name + "-tmp" + ext
    cv2.imwrite(tmp_path, img)
    os.replace(tmp_path, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", choices=["passthrough", "sobel"], default="passthrough")
    parser.add_argument("--display-window", action="store_true", help="pop up a window showing the output")
    args = parser.parse_args()

    vc = cv2.VideoCapture(0)

    if args.filter == "passthrough":
        processor = lambda x: x
    elif args.filter == "sobel":
        model = SobelImageProcessor()
        processor = functools.partial(process_image_with_model, model)
    else:
        raise ValueError()

    output_path = "/tmp/camera.bmp"
    count = 0
    while vc.isOpened():
        start = time.time()
        success, frame = vc.read()
        if not success:
            break
        # opencv deals in BGR format, convet to normal person colors
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        read_time = time.time() - start
        start = time.time()
        processed_frame = processor(frame)
        process_time = time.time() - start
        start = time.time()
        save_image(processed_frame, output_path)
        write_time = time.time() - start
        if count % 10 == 0:
            print(f"read={int(read_time*1000)}ms process={int(process_time*1000)}ms write={int(write_time*1000)}ms")
        if args.display_window:
            display_frame = np.concatenate([frame,processed_frame], axis=1)
            print(display_frame.shape)
            resized_display_frame = cv2.resize(display_frame, (IMAGE_WIDTH, IMAGE_HEIGHT // 2), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(WINDOW_NAME, cv2.cvtColor(resized_display_frame, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
        count += 1


if __name__ == "__main__":
    main()