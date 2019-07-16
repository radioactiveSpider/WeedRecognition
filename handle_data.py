import cv2
import csv
import argparse
import numpy as np
from Data import Data
from imutils import paths
from Coordinates import Coordinates


def create_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of images")
    ap.add_argument("-l", "--label", required=True, help="path to input labels")
    args = vars(ap.parse_args())
    return args


def read_image_paths(path):
    return list(paths.list_images(path))


def read_images_into_arr(image_paths):
    raw_data = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        raw_data.append(image)
    return raw_data


def create_mask(image, coordinates):
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[int(coordinates.x_1):int(coordinates.x_2), int(coordinates.y_1):int(coordinates.y_2)] = 255
    return mask


def mask_image(image, coordinates):
    mask = create_mask(image, coordinates)
    return cv2.bitwise_and(image, image, mask=mask)


def cut_image(image, coordinates):
    return image[int(coordinates.x_1):int(coordinates.x_2), int(coordinates.y_1):int(coordinates.y_2)]


def print_image(image):
    cv2.imshow("img", image)
    cv2.waitKey(0)


def generate_columns(hist_size):
    colors = ["blue", "green", "red"]
    color_arr = []
    for color in colors:
        for i in range(hist_size):
            str_ = color + str(i)
            color_arr.append(str_)
    return color_arr


def create_csv(data, labels, filename, hist_size):
    lbl_idx = 0
    colors = generate_columns(hist_size)
    with open(filename, "w", newline="") as file:
        columns = []
        columns = colors.copy()
        columns.extend(["number_of_pic", "x_1", "x_2", "y_1", "y_2", "weed/not weed"])
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for item in data:
            dict = {}
            for i in range(len(colors)):
                dict[colors[i]] = item.data[i]
            dict["number_of_pic"] = item.number
            dict["x_1"] = item.coordinates.x_1
            dict["x_2"] = item.coordinates.x_2
            dict["y_1"] = item.coordinates.y_1
            dict["y_2"] = item.coordinates.y_2
            dict["weed/not weed"] = labels[lbl_idx][0]
            lbl_idx = lbl_idx + 1
            writer.writerow(dict)


def split_image(image, rows, cols, number_of_pic):
    titles = []
    coordinates = Coordinates()
    for i in range(rows):
        coordinates.update_x_coordinates(image, rows)
        for j in range(cols):
            entity = Data()
            copy_coordinates = Coordinates()
            coordinates.update_y_coordinates(image, cols)
            sample = cut_image(image, coordinates)
            copy_coordinates.copy(coordinates)
            entity.coordinates = copy_coordinates
            entity.data = sample
            entity.number = number_of_pic
            titles.append(entity)

        coordinates.y_1 = 0
        coordinates.y_2 = 0
    return titles


def split_images(raw_data, rows, cols):
    data = []
    for i, image in enumerate(raw_data):
        titles = []
        titles = split_image(image, rows, cols, i)
        data.extend(titles)
    return data


def create_labels(annotated_data):
    labels = []
    for entity in annotated_data:
        if is_red_in_image(entity.data):
            labels.append(1)
        else:
            labels.append(0)
    labels = np.asarray(labels).reshape(len(labels), 1)
    return labels


def build_hist(image, hist_size):
    color = ('blue', 'green', 'red')
    sample = []
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [hist_size], [0, 256])
        hist = np.ravel(hist)
        sample.extend(hist)
    return sample


def build_rgb_hist(splited_data, hist_size):
    for j in range(len(splited_data)):
        splited_data[j].data = build_hist(splited_data[j].data, hist_size)


def get_rgb_hist_by_mask(raw_data, rows, cols, hist_size):
    color = ('b', 'g', 'r')
    sample = []
    data = []
    create_flag = 0
    for image in raw_data:
        coordinates = Coordinates()
        for i in range(rows):
            coordinates.update_x_coordinates(image, rows)
            for j in range(cols):
                coordinates.update_y_coordinates(image, cols)
                mask = create_mask(image, coordinates)
                for z, col in enumerate(color):
                    hist = cv2.calcHist([image], [z], mask, [hist_size], [0, 256])
                    hist = np.ravel(hist)
                    sample.extend(hist)
                if not create_flag:
                    data = np.asarray(sample).reshape(1, 63)
                    create_flag = 1
                else:
                    data = np.append(data, np.asarray(sample).reshape(1, 63), axisпше=0)
                sample.clear()
            coordinates.y_1 = 0
            coordinates.y_2 = 0
    return data


def is_red_in_image(input_image):
    hist = cv2.calcHist([input_image], [2], None, [256], [0, 256])
    return hist[hist.shape[0] - 1] != 0


def apply_median_blur(raw_data, median_filter_param):
    for i in range(len(raw_data)):
        raw_data[i] = cv2.medianBlur(raw_data[i], median_filter_param)


def run(filename, image_path, annotation_path, rows, cols, amount_of_bins, median_filter_param):

    image_paths = read_image_paths(image_path)
    raw_data = read_images_into_arr(image_paths)
    apply_median_blur(raw_data, median_filter_param)

    annotation_paths = read_image_paths(annotation_path)
    annotations = read_images_into_arr(annotation_paths)

    splited_data = split_images(raw_data, rows, cols)
    splited_annotated_data = split_images(annotations, rows, cols)

    build_rgb_hist(splited_data, amount_of_bins)

    labels = create_labels(splited_annotated_data)
    create_csv(splited_data, labels, filename, amount_of_bins)
    return filename
