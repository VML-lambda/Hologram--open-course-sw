import rawpy
import cv2
from PIL import Image
import numpy as np
import torch

def nef_to_png(input_path, output_path):
    # NEF 파일을 읽어들입니다.
    with rawpy.imread(input_path) as raw:
        # RAW 데이터를 이미지로 변환 (손실 없이)
        rgb_image = raw.postprocess(output_bps=16)  # 16-bit output
        rgb_image_8bit = (rgb_image / 256).astype(np.uint8)

        # numpy 배열을 Pillow 이미지로 변환
        image = Image.fromarray(rgb_image_8bit)

        # PNG 형식으로 저장 (손실 없이)
        image.save(output_path, format="PNG")
    print(f"Converted {input_path} to {output_path}")
    

def rotate_image(input_path,output_path, angle, keep_size=True):
    """
    Rotates an image by the specified angle.

    Parameters:
    ----------
    image : numpy.ndarray
        Input image to rotate.
    angle : float
        Rotation angle in degrees. Positive values mean counter-clockwise rotation.
    keep_size : bool, optional
        If True, the output image will have the same size as the input by cropping.
        If False, the output image will expand to include all rotated pixels.

    Returns:
    -------
    numpy.ndarray
        Rotated image.
    """
    image = cv2.imread(input_path)

    # Get the image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    if keep_size:
        # Rotate without changing the image size (may crop the rotated image)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    else:
        # Rotate with expanding the output image size to fit the entire rotated image
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix to account for the new size
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    cv2.imwrite(output_path, rotated_image)

    return rotated_image

def flip_image(input_path, output_path, flip_code):
    """
    Flips an image horizontally, vertically, or both.

    Parameters:
    ----------
    input_path : str
        Path to the input image.
    output_path : str
        Path to save the flipped image.
    flip_code : int
        Code specifying how to flip the image:
        - 0: Flip vertically
        - 1: Flip horizontally
        - -1: Flip both horizontally and vertically

    Returns:
    -------
    numpy.ndarray
        Flipped image.
    """
    # Read the input image
    image = cv2.imread(input_path)
    
    if image is None:
        raise FileNotFoundError(f"Image not found at {input_path}")

    # Flip the image
    flipped_image = cv2.flip(image, flip_code)

    # Save the flipped image
    cv2.imwrite(output_path, flipped_image)

    return flipped_image


def fft(image, device):
    image_tensor = torch.from_numpy(np.float32(image)).to(device)
    dft_shift = torch.fft.fftshift(torch.fft.fft2(image_tensor))
    return dft_shift


def ifft(fshift):
    f_ishift = torch.fft.ifftshift(fshift)
    img_back = torch.fft.ifft2(f_ishift).abs()
    img_back = torch.clamp(img_back, 0, 255).cpu().numpy().astype(np.uint8)
    return img_back


def apply_low_pass_filter(fshift, cutoff=5):
    """
    Applies a low-pass filter in the frequency domain.
    
    Parameters:
    ----------
    fshift : torch.Tensor
        The input tensor in the frequency domain.
    cutoff : int
        The cutoff radius for the low-pass filter.
    
    Returns:
    -------
    torch.Tensor
        The low-pass filtered frequency domain tensor.
    """
    rows, cols = fshift.shape[-2:]
    crow, ccol = rows // 2, cols // 2

    # Create a mask with ones in the center and zeros elsewhere
    mask = torch.zeros_like(fshift).to(fshift.device)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff, :] = 1
    fshift_filtered = fshift * mask
    return fshift_filtered

def apply_high_pass_filter(fshift, cutoff=5):
    """
    Applies a high-pass filter in the frequency domain.
    
    Parameters:
    ----------
    fshift : torch.Tensor
        The input tensor in the frequency domain.
    cutoff : int
        The cutoff radius for the high-pass filter.
    
    Returns:
    -------
    torch.Tensor
        The high-pass filtered frequency domain tensor.
    """
    rows, cols = fshift.shape[-2:]
    crow, ccol = rows // 2, cols // 2

    # Create a mask with zeros in the center and ones elsewhere
    mask = torch.ones_like(fshift).to(fshift.device)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff, :] = 0
    fshift_filtered = fshift * mask
    return fshift_filtered
