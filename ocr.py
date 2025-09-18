from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import pytesseract
import cv2
import numpy as np
import os

# ---------------------------
# Load TrOCR model (HuggingFace)
# ---------------------------
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


# ---------------------------
# Image Preprocessing
# ---------------------------
def preprocess_image(pil_img: Image.Image, save_debug: bool = False, debug_path: str = "processed_image.jpeg") -> Image.Image:
    """
    Convert to grayscale, apply thresholding, and denoise for better OCR.
    Optionally save the preprocessed image for debugging.
    """
    img = np.array(pil_img.convert("L"))  # grayscale

    # Apply thresholding (binary black/white)
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    # Median blur for denoising
    denoised = cv2.medianBlur(thresh, 3)

    processed = Image.fromarray(denoised)

    # Save debug output if requested
    if save_debug:
        processed.save(debug_path)
        print(f"[OCR] Preprocessed image saved to {debug_path}")

    return processed


# ---------------------------
# OCR Methods
# ---------------------------
def trocr_ocr(image: Image.Image) -> str:
    """Extract text using Microsoft TrOCR"""
    try:
        pixel_values = processor(image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    except Exception as e:
        print(f"[OCR] TrOCR failed: {e}")
        return ""


def tesseract_ocr(image: Image.Image) -> str:
    """Extract text using Tesseract"""
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"[OCR] Tesseract failed: {e}")
        return ""


# ---------------------------
# Smart OCR (Combined)
# ---------------------------
def smart_ocr(image: Image.Image, debug: bool = True) -> str:
    """
    Try TrOCR first, then Tesseract, and merge results.
    Always returns a non-empty string.
    """
    # Preprocess first
    processed_img = preprocess_image(image, save_debug=debug)

    text_trocr = trocr_ocr(processed_img)
    text_tess = tesseract_ocr(processed_img)

    if text_trocr and text_tess:
        text = text_trocr + "\n" + text_tess
    elif text_trocr:
        text = text_trocr
    elif text_tess:
        text = text_tess
    else:
        text = "[OCR produced no text]"

    print(f"[OCR] Extracted text preview: {text[:200]}")
    return text
