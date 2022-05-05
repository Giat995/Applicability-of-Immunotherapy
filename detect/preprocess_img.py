import os
import numpy as np
from scipy.ndimage.interpolation import zoom
from skimage import measure
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image
from PIL import Image
import cv2 as cv

def convertToGrayImg(imagePath):
    img = Image.open(imagePath)
    # Img = img.convert('L')
    np_img = np.array(img)
    return np_img

def origin_generate():
    origin = [-210.000, -210.000, -267.750]
    return origin

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
#    print image[50]
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample():
    spacing = np.array([1, 1, 1], dtype=np.float32)
    return spacing

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            # print(mask1)
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 1.5 * np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask

def lumTrans(img):
    lungwin = np.array([-1200., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg

def resampledeeplung(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')



def savenpy(path, name, savepath1, file):

    # image = cv.imread(path, 0)
    image = file
    fimage = np.array(image).astype(float)
    fimage = (fimage * (1636 + 2048)) / 255 - 2048
    image = np.array(fimage, dtype=int)

    # image = np.load(r'G:\immunotherapy\simgleimage\image.npy', allow_pickle=True)
    # print(image)
    # print(image.shape)

    origin = origin_generate()
    sliceim = np.stack(image for i in range(10))
    spacing = resample()
    Mask = segment_lung_mask(sliceim, True)
    # print(Mask)
    xx, yy, zz = np.where(Mask)
    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    resolution = np.array([1, 1, 1])
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    box = np.floor(box).astype('int')
    margin = 5
    newshape = np.round(np.array(Mask.shape) * spacing / resolution).astype('int')
    extendbox = np.vstack([np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T
#    print Mask.shape
    dilatedMask = process_mask(Mask)
    extramask = dilatedMask ^ Mask
    bone_thresh = 210
    pad_value = 170
#    print dilatedMask.shape
#    print sliceim.shape
    sliceim = lumTrans(sliceim)
#    print sliceim.shape
    sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype('uint8')
    bones = (sliceim * extramask) > bone_thresh
    sliceim[bones] = pad_value
    sliceim1, _ = resampledeeplung(sliceim, spacing, resolution, order=1)
    sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1], extendbox[1, 0]:extendbox[1, 1], extendbox[2, 0]:extendbox[2, 1]]
    sliceim = sliceim2[np.newaxis, ...]
    label = []
    np.save(os.path.join(savepath1, name + '_clean.npy'), sliceim)
    np.save(os.path.join(savepath1, name + '_spacing.npy'), spacing)
    np.save(os.path.join(savepath1, name + '_extendbox.npy'), extendbox)
    np.save(os.path.join(savepath1, name + '_mask.npy'), Mask)
    np.save(os.path.join(savepath1, name + '_label.npy'), label)
    np.save(os.path.join(savepath1, name + '_origin.npy'), origin)
