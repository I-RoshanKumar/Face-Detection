import numpy as np
import cv2

def get_images(frame, faces_coord, shape):
    if shape == "rectangle":
        faces_img = cut_face_rectangle(frame, faces_coord)
        frame = draw_face_rectangle(frame, faces_coord)
    elif shape == "ellipse":
        faces_img = cut_face_ellipse(frame, faces_coord)
        frame = draw_face_ellipse(frame, faces_coord)
    faces_img = normalize_intensity(faces_img)
    faces_img = resize(faces_img)
    return (frame, faces_img)

def resize(images, size=(100, 100)):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)
    return images_norm

def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm


def cut_face_rectangle(image, face_coord):
    images_rectangle = []
    for (x, y, w, h) in face_coord:
        images_rectangle.append(image[y: y + h, x: x + w])
    return images_rectangle

def cut_face_ellipse(image, face_coord):
    images_ellipse = []
    for (x, y, w, h) in face_coord:
        center1 =int(round(x + w / 2))
        center2=int(round(y + h / 2))
        center=(center1,center2) 
        axis_major =int(round(h/2))
        axis_minor =int(round(w/2))
        axis=(axis_major,axis_minor)
        angle=(0,0,360)
        mask = np.zeros_like(image)
        box=(center,axis,angle)
        mask1 = cv2.ellipse(mask,center,(axis_major, axis_minor),0,0,360,(255, 255, 255),-1)
        image_ellipse = np.bitwise_and(image, mask1)
        images_ellipse.append(image_ellipse[y: y + h, x: x + w])

    return images_ellipse

def draw_face_rectangle(image, faces_coord):
    for (x, y, w, h) in faces_coord:
        cv2.rectangle(image, (x, y), (x + w, y + h), (206, 0, 209), 2)
    return image

def draw_face_ellipse(image, faces_coord):
    for (x, y, w, h) in faces_coord:
        center1 = int(round(x + w / 2))
        center2 = int(round(y + h / 2))
        center=(center1,center2)
        axis_major =int(round(h / 2))
        axis_minor = int(round(w / 2))
        cv2.ellipse(image,
                    center=center,
                    axes=(axis_major, axis_minor),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=(206, 0, 209),
                    thickness=2)
    return image
