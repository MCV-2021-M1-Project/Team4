import cv2
import matplotlib.pyplot as plt
# Scikit-image
from skimage.metrics import structural_similarity as ssim


def remove_noise(img, type='median', kernel_size=3, plot_results=False, img_gt=0):
    """
    This function applies denoising to the input image

    Parameters
    ----------
    img (required): Image to apply denoising (rgb image)
    type: Type of filter/algorithm to denoise: median, gaussian, non local means algorithm, bilateral (string)
    kernel_size: Size of the kernel for cases median and gaussian (int)
    plot_results: if true, plots results. (bool)
    img_gt: Ground truth image, only used in case plot_results = True (rgb image)

    Returns
    -------
    dst: image denoised (rgb image)
    """
    kernel = kernel_size
    # Applying median filter
    if type == 'median':
        dst = cv2.medianBlur(img, kernel)
    # Applying gaussian filter
    elif type == 'gaussian':
        dst = cv2.GaussianBlur(img, (kernel, kernel), 0)
    # Applying non_local_means algorithm (used to remove gaussian noise)
    elif type == 'non_local_means':
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # Applying bilateral filter
    elif type == 'bilateral':
        dst = cv2.bilateralFilter(img, 15, 75, 75)

    if plot_results:
        psnr = cv2.PSNR(img_gt, dst)
        mssim = ssim(img_gt, dst, multichannel=True)

        psnr_original = cv2.PSNR(img_gt, img)
        mssim_original = ssim(img_gt, img, multichannel=True)

        plt.subplot(131), plt.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
        plt.title('Ground truth image')
        plt.subplot(132), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.xlabel(f'PSNR[dB]: {psnr_original:.2f}, SSIM: {mssim_original:.2f}')
        plt.subplot(133), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        plt.xlabel(f'PSNR[dB]: {psnr:.2f}, SSIM: {mssim:.2f}')
        plt.title('Img w/ noise processed: ' + type)
        plt.show()

    return dst


if __name__ == "__main__":
    path_image = 'C:/Users/Joan/Desktop/Master_Computer_Vision_2021/M1/data/qsd1_w3/00000.jpg'
    path_image_gt = 'C:/Users/Joan/Desktop/Master_Computer_Vision_2021/M1/data/qsd1_w3/non_augmented/00000.jpg'

    p = 8
    r = 1

    img = cv2.imread(path_image)
    img_gt = cv2.imread(path_image_gt)
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))

    img_processed = remove_noise(img, type='median', plot_results=True, img_gt=img_gt)
    plt.subplot(133)
    plt.imshow(img_processed)
    plt.show()


# This lines are just to test the function
'''
t = 5
for j in range(t):
    img_file = '/Users/Cesc47/Documents/CesC_47/MCV/M1/data/qsd1_w3' + '/00' + ('00' if j < 10 else '0') + str(j) + '.jpg'
    
    img = cv2.imread(img_file)
    img_gt = cv2.imread(img_file_gt)
    img_processed = remove_noise(img, type='median', plot_results=True, img_gt=img_gt)
'''




















'''
# ANALISIS DFTS
def detect_noise(img_to_detect_noise):
    img_bw = cv2.cvtColor(np.float32(img_to_detect_noise), cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(img_bw, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    plt.subplot(121), plt.imshow(img_bw, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
'''

'''
# ANALISIS HISTOGRAMAS 
def detect_noise(image):

    # Convert image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    bins = 256
    s = cv2.calcHist([image], [1], None, [bins], [0, 256])
    v = cv2.calcHist([image], [2], None, [bins], [0, 256])


    ##### Just for visualization and debug; remove in final
    plt.subplot(221), plt.imshow(image[:, :, 2], cmap='gray')
    plt.title('Img - HSV(Value)')
    plt.subplot(223), plt.imshow(image[:, :, 1], cmap='gray')
    plt.title('Img - HSV(saturation)')
    x = np.arange(bins)
    plt.subplot(224), plt.plot(x[:, np.newaxis], s)
    plt.title('Histogram - Saturation')
    plt.subplot(222), plt.plot(x[:, np.newaxis], v)
    plt.title('Histogram - Value')
    plt.show()
    ##### Just for visualization and debug; remove in final
'''











