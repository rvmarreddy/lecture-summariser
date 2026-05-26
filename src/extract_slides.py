import fitz  # PyMuPDF
import os
from typing import List, Dict

def extract_slide_content(pdf_path: str, output_folder: str = "outputs/slides_output") -> List[Dict]:
    """Extract text and images from PDF slides.
    Returns: list of {"slide": int, "text": str, "images": [str]}
    """
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    results = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        images = page.get_images(full=True)
        image_refs = []

        for img_index, img in enumerate(images, start=1):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            try:
                if pix.n - pix.alpha < 4:  # RGB or Gray
                    img_filename = f"slide_{page_num}_img_{img_index}.png"
                    pix.save(os.path.join(output_folder, img_filename))
                    image_refs.append(img_filename)
            finally:
                pix = None

        results.append({
            "slide": page_num,
            "text": text.strip(),
            "images": image_refs
        })

    doc.close()
    return results