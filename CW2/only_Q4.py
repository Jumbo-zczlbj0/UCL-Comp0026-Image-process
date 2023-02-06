import cv2
import numpy as np
from math import pi
from scipy import fftpack as fft

points = []

image = cv2.imread("./images/illu_apple.jpg")

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        points.append([x, y])
        cv2.circle(image, (x, y), 1, (20, 200, 20),thickness=3)
        cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (20, 200, 20), thickness=2)
        cv2.imshow("image", image)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)


# Extract the region of interest from the original input
def get_possion_inputs(src_image, mask, bounding_rect, dst_image=None, dst_offset=[0, 0], gray_mode=False):
    offset_y, offset_x = dst_offset
    x, y, w, h = bounding_rect
    if gray_mode:
        src_image_patch = src_image[y:y + h, x:x + w]
        mask_patch = mask[y:y + h, x:x + w]
        if dst_image is None:
            return src_image_patch, mask_patch
        dst_image_patch = dst_image[y + offset_y:y + offset_y + h, x + offset_x:x + offset_x + w]
        return src_image_patch, mask_patch, dst_image_patch

    src_image_patch = src_image[y:y + h, x:x + w, :]
    mask_patch = mask[y:y + h, x:x + w]
    if dst_image is None:
        return src_image_patch, mask_patch
    dst_image_patch = dst_image[y + offset_y:y + offset_y + h, x + offset_x:x + offset_x + w, :]
    return src_image_patch, mask_patch, dst_image_patch


# Calculate the first-order gradient of the image
def calulate_grad(image):
    image = image.astype('int')
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)
    grad_x[:, :-1] = image[:, 1:] - image[:, :-1]
    grad_y[:-1, :] = image[1:, :] - image[:-1, :]
    return grad_x, grad_y


# Calculate the second-order gradient of the image
def calulate_laplacian(grad_x, grad_y):
    laplacian_x = np.zeros_like(grad_x)
    laplacian_y = np.zeros_like(grad_y)
    laplacian_x[:, :-1] = grad_x[:, 1:] - grad_x[:, :-1]
    laplacian_y[:-1, :] = grad_y[1:, :] - grad_y[:-1, :]
    return (laplacian_x + laplacian_y).astype('float32')


# Calculate the discrete sinusoidal transform(DST) of possion edited patch from laplacian and target image.
def possion_dst(laplacian, image_patch):
    possion_patch = image_patch.copy().astype('float32')

    # Only the edge information is retained, and the others are set to 0
    possion_patch[1:-1, 1:-1] = 0


    # The Laplacian on the edge was droped
    laplacian = laplacian[1:-1, 1:-1]

    laplacian_x = possion_patch[1:-1, 2:] + possion_patch[1:-1, :-2] - 2 * possion_patch[1:-1, 1:-1]
    laplacian_y = possion_patch[2:, 1:-1] + possion_patch[:-2, 1:-1] - 2 * possion_patch[1:-1, 1:-1]

    laplacian_gap = laplacian - laplacian_x - laplacian_y


    dst = fft.dst(laplacian_gap, axis=0, type=1)
    dst = dst.T
    dst = fft.dst(dst, axis=0, type=1)
    dst = dst.T
    return dst / 4


# Calculate the fusion result by calculating the discrete sinusoidal inverse transform(IDST)
def possion_idst(image_patch, dst):
    possion_patch = image_patch.copy().astype('float32')
    height, width = image_patch.shape

    # Drop edge, only internal results are calculated
    # u = p / 2 * (cos(π*m/Row) + cos(π*n/Col) - 2)
    mesh_x, mesh_y = np.meshgrid(np.arange(1, width - 1), np.arange(1, height - 1))
    down_scale = 2 * (np.cos(pi * mesh_x / (width - 1)) + np.cos(pi * mesh_y / (height - 1)) - 2)
    dst = dst / down_scale

    idst = np.real(fft.idst(dst, axis=0, type=1))
    idst = idst.T
    idst = np.real(fft.idst(idst, axis=0, type=1))
    idst = idst.T
    idst = idst / ((dst.shape[0] + 1) * (dst.shape[1] + 1))

    possion_patch[1:-1, 1:-1] = idst
    return possion_patch


# possion equation solver
def possion_equation_solver(laplacian, image_patch):
    dst = possion_dst(laplacian, image_patch)
    img = possion_idst(image_patch, dst)

    img = np.clip(img, 0, 255)
    img = img.astype('uint8')
    return img

# Paste the poisson editing patch in the target image
def apply_patch(image, patch, position):
    img = image.copy()
    y, x = position
    scale_y, scale_x = patch.shape[:2]
    img[y:y + scale_y, x:x + scale_x] = patch
    return img

while (True):
    cv2.imshow("image", image)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
        break

point=np.array(points)

n=len(point)
max=np.zeros((4,2)).reshape(4,2)
max[0,0]=max[1,0]=max[2,0]=max[3,0]= point[0,0]
max[0,1]= max[1,1]=max[2,1]=max[3,1]=point[0,1]
for i in range(0,n):
    if point[i,0]>max[1,0]:
        max[2,0]=max[1,0]= point[i,0]
    elif point[i,0]<max[0,0]:
        max[3,0]=max[0,0]= point[i,0]

for i in range(0,n):
    if point[i,1]>max[2,1]:
        max[3,1]=max[2,1]= point[i,1]
    elif point[i,1]<max[0,1]:
        max[1,1]= max[0,1]= point[i,1]

mask_points = max.astype(int)
image = cv2.imread("./images/illu_apple.jpg")


h, w, zz = image.shape
mask = np.zeros((h,w), dtype="uint8")
#mask_points = np.array([[228, 213], [336, 231], [326, 269],[277, 277],[223,258]])
mask = cv2.polylines(mask, [mask_points], True, 255)
mask = cv2.fillPoly(mask, [mask_points], 255)
#plt.figure()
#plt.imshow(mask, cmap="gray")
cv2.imshow('mask', mask)
cv2.waitKey(0)

def possion_edit_T4(src_image_patch, mask_patch):
    possion_result = np.zeros_like(src_image_patch)
    for channel in range(3):
        src_grad_x, src_grad_y = calulate_grad(src_image_patch[:,:,channel])

        grad_sum = np.sum(src_grad_x + src_grad_y)
        grad_avg = abs(grad_sum * 0.2) / (src_grad_x.shape[0] * src_grad_x.shape[1])

        grad_x_flag = (src_grad_x != 0)
        grad_x_flag = (mask_patch != 0) * grad_x_flag

        grad_y_flag = (src_grad_y != 0)
        grad_y_flag = (mask_patch != 0) * grad_y_flag

        src_grad_x[grad_x_flag] = (grad_avg ** 0.2) * src_grad_x[grad_x_flag] / (np.abs(src_grad_x[grad_x_flag]) ** 0.2)
        src_grad_y[grad_y_flag] = (grad_avg ** 0.2) * src_grad_y[grad_y_flag] / (np.abs(src_grad_y[grad_y_flag]) ** 0.2)
        laplacian = calulate_laplacian(src_grad_x, src_grad_y )

        possion_patch = possion_equation_solver(laplacian, src_image_patch[:,:,channel])
        possion_result[:,:,channel] = possion_patch

    return possion_result


bounding_rect = cv2.boundingRect(mask_points)
x,y,w,h = bounding_rect
src_image_patch, mask_patch= get_possion_inputs(image,mask,bounding_rect,gray_mode=False)
patch = possion_edit_T4(src_image_patch, mask_patch)
result_image = apply_patch(image, patch, [y,x])
cv2.imshow('apple', result_image)
cv2.waitKey(0)



