import cv2

def classify_grade_by_aspect_ratio(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return "Unknown"

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    if min(w, h) == 0:
        return "Unknown"

    ratio = max(w, h) / min(w, h)

    if ratio >= 3.0:
        return "Slender"
    elif ratio >= 2.1:
        return "Medium"
    elif ratio >= 1.1:
        return "Bold"
    else:
        return "Round"
