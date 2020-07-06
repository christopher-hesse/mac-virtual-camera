"""
Read from the computer's camera and save images to bitmaps
"""

import argparse
import os
import time
import math

import cv2
import torch
from torch import nn
import numpy as np
import urllib.request

from mac_virtual_camera import cyclegan

WINDOW_NAME = "camera"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def torch_rgb_to_gray(img_nchw):
    r = img_nchw[:, 0, :, :]
    g = img_nchw[:, 1, :, :]
    b = img_nchw[:, 2, :, :]
    # https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).unsqueeze(dim=1)


def np_rgb_to_gray(img_hwc):
    r = img_hwc[:, :, 0]
    g = img_hwc[:, :, 1]
    b = img_hwc[:, :, 2]
    # https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    return np.expand_dims((0.2989 * r + 0.5870 * g + 0.1140 * b), axis=2)


def torch_gray_to_rgb(img_nchw):
    return torch.cat([img_nchw, img_nchw, img_nchw], dim=1)


def np_gray_to_rgb(img_hwc):
    return np.concatenate([img_hwc, img_hwc, img_hwc], axis=2)


def create_gaussian_conv(kernel_size=7, sigma=3):
    # https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2.0

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
    )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_conv = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=(kernel_size, kernel_size),
        bias=False,
        padding=(kernel_size // 2, kernel_size // 2),
    )
    gaussian_conv.weight.data.copy_(gaussian_kernel)
    return gaussian_conv


class SobelImageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.gaussian_conv = create_gaussian_conv(kernel_size=7, sigma=1)
        self.vertical_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 3),
            bias=False,
            padding=(1, 1),
        )
        self.horizontal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 3),
            bias=False,
            padding=(1, 1),
        )

        sobel_vertical_kernel = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        )
        self.vertical_conv.weight.data.copy_(sobel_vertical_kernel)
        self.horizontal_conv.weight.data.copy_(sobel_vertical_kernel.transpose(0, 1))

    def forward(self, img):
        gray_img = self.gaussian_conv(torch_rgb_to_gray(img))
        grad_vert = self.vertical_conv(gray_img)
        grad_horiz = self.horizontal_conv(gray_img)
        grad = torch.sqrt(grad_vert ** 2 + grad_horiz ** 2)
        normalized_grad = (grad - grad.min()) / (grad.max() - grad.min())
        return torch_gray_to_rgb(normalized_grad)


class SobelImageProcessor:
    def __init__(self):
        self._model = SobelImageModel()
    
    def process(self, img):
        return process_image_with_model(self._model, img)


def get_model_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_image_with_model(model, img):
    device = get_model_device()
    with torch.no_grad():
        input_hwc = torch.from_numpy(img).to(dtype=torch.float32, device=device) / 255
        input_nhwc = input_hwc.unsqueeze(dim=0)
        input_nchw = input_nhwc.permute(0, 3, 1, 2)
        output_nchw = model(input_nchw)
        output_nhwc = output_nchw.permute(0, 2, 3, 1)
        processed_img = (output_nhwc[0] * 255).to(dtype=torch.uint8, device=torch.device("cpu")).numpy()
    return processed_img


class IdentityProcessor:
    def process(self, img):
        return img


class WatershedProcessor:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

        # https://stackoverflow.com/questions/8076889/how-to-use-opencv-simpleblobdetector
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 200
        params.processorByArea = True
        params.minArea = 10
        params.processorByCircularity = False
        params.minCircularity = 0.1
        params.processorByConvexity = False
        params.minConvexity = 0.87
        params.processorByInertia = False
        params.minInertiaRatio = 0.01
        self.detector = cv2.SimpleBlobDetector_create(params)

    def process(self, img):
        # https://stackoverflow.com/questions/42294109/remove-background-of-the-image-using-opencv-python
        # Create a blank image of zeros (same dimension as img)
        # It should be grayscale (1 color channel)
        marker = np.zeros_like(img[:, :, 0]).astype(np.int32)
        height, width, _ = img.shape

        # Dictate the background and set the markers to 1
        for inset in [1, 50, 100]:
            marker[inset][inset] = 1
            marker[height - inset][inset] = 1
            marker[inset][width - inset] = 1
            marker[height - inset][width - inset] = 1

        # determine area of interest using blob detector on background subtractor
        # fgmask = self.fgbg.apply(img)
        # rgb_mask = np.tile(np.expand_dims(fgmask, axis=2), (1, 1, 3))
        # return rgb_mask
        # keypoints = self.detector.detect(rgb_mask)
        # sorted_keypoints = list(sorted(keypoints, key=lambda kp: kp.size))
        # for kp in sorted_keypoints[:10]:
        #     x, y = kp.pt
        #     marker[int(y)][int(x)] = 255

        for offset in [10, 30]:
            for x_factor, y_factor in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                marker[IMAGE_HEIGHT // 2 + y_factor * offset][IMAGE_WIDTH // 2 + x_factor * offset] = 255

        # Now we have set the markers, we use the watershed
        # algorithm to generate a marked image
        marked = cv2.watershed(img, marker)

        # Make the background black, and what we want to keep white
        marked[marked == 1] = 0
        marked[marked > 1] = 255

        # Use a kernel to dilate the image, to not lose any detail on the outline
        # I used a kernel of 3x3 pixels
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(marked.astype(np.float32), kernel, iterations=1)

        # Now apply the mask we created on the initial image
        final_img = cv2.bitwise_and(img, img, mask=dilation.astype(np.uint8))

        return final_img

VALID_CYCLEGAN_NAMES = "apple2orange, orange2apple, summer2winter_yosemite, winter2summer_yosemite, horse2zebra, zebra2horse, monet2photo, style_monet, style_cezanne, style_ukiyoe, style_vangogh, sat2map, map2sat, cityscapes_photo2label, cityscapes_label2photo, facades_photo2label, facades_label2photo, iphone2dslr_flower".split(", ")

class CycleGANProcessor:
    def __init__(self, name, cache_dir="/tmp/cyclegan-checkpoints"):
        os.makedirs(cache_dir, exist_ok=True)
        checkpoint_url = f"http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/{name}.pth"
        local_path = os.path.join(cache_dir, f"{name}.pth")
        if not os.path.exists(local_path):
            print(f"downloading checkpoint {name}...")
            urllib.request.urlretrieve(checkpoint_url, local_path)
            print("done")
        self._model = cyclegan.load_checkpoint(local_path, get_model_device())

    def _run_model(self, img):
        # cyclegan models expect images in the range [-1, 1]
        img = img * 2 - 1
        out = self._model(img)
        return (out + 1) / 2
        
    def process(self, img):
        height, width, _ = img.shape
        resized_img = cv2.resize(img, (cyclegan.IMAGE_SIZE, cyclegan.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        out = process_image_with_model(self._run_model, resized_img)
        return cv2.resize(out, (width, height), interpolation=cv2.INTER_LINEAR)


def write_rgb_data(path, img):
    """
    CGCreateImage seems to accept only raw RGB images
    """
    assert len(img.shape) == 3 and img.shape[-1] == 3 and img.dtype == np.uint8
    with open(path, "wb") as f:
        for row in reversed(img):
            f.write(row.tobytes())


def save_image(img, path):
    assert img.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    name, ext = os.path.splitext(path)
    tmp_path = name + "-tmp" + ext
    write_rgb_data(tmp_path, img)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--processor",
        choices=["passthrough", "sobel", "watershed", "cyclegan"],
        default="passthrough",
    )
    parser.add_argument(
        "--display-window",
        action="store_true",
        help="pop up a window showing the output",
    )
    args = parser.parse_args()

    vc = cv2.VideoCapture(args.camera_index)

    if args.processor == "passthrough":
        processor = IdentityProcessor()
    elif args.processor == "sobel":
        processor = SobelImageProcessor()
    elif args.processor == "watershed":
        processor = WatershedProcessor()
    elif args.processor == "cyclegan":
        processor = CycleGANProcessor("horse2zebra")
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
        processed_frame = processor.process(frame)
        process_time = time.time() - start
        start = time.time()
        save_image(processed_frame, output_path)
        write_time = time.time() - start
        if count % 10 == 0:
            print(
                f"read={int(read_time*1000)}ms process={int(process_time*1000)}ms write={int(write_time*1000)}ms"
            )
        if args.display_window:
            display_frame = np.concatenate([frame, processed_frame], axis=1)
            resized_display_frame = cv2.resize(
                display_frame,
                (IMAGE_WIDTH, IMAGE_HEIGHT // 2),
                interpolation=cv2.INTER_LINEAR,
            )
            cv2.imshow(
                WINDOW_NAME, cv2.cvtColor(resized_display_frame, cv2.COLOR_BGR2RGB)
            )
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
        count += 1


if __name__ == "__main__":
    main()
