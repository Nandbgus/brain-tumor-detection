import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# =======================
# Load Model
# =======================
@st.cache_resource
def load_model():
    model = YOLO("best_model.pt")  # pastikan file best.pt ada di folder yang sama
    return model

model = load_model()

# =======================
# Utility Functions
# =======================
def read_image(uploaded_file):
    """Convert uploaded file to OpenCV image"""
    bytes_data = uploaded_file.read()
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    return np.array(img)


def get_segmentation(model, img):
    """Run YOLO segmentation and return results"""
    results = model.predict(img, conf=0.25, verbose=False)
    return results[0]


def extract_mask(result):
    """Combine YOLO masks + generate ultra-smooth contour mask"""
    import cv2
    import numpy as np
    from scipy.signal import savgol_filter

    if result.masks is None:
        return None

    # YOLO masks: (N, H, W)
    masks = result.masks.data.cpu().numpy()

    # Combine masks
    combined = np.zeros_like(masks[0])
    for m in masks:
        combined = np.maximum(combined, m)

    combined = (combined * 255).astype(np.uint8)

    # Step 1: Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return combined

    cnt = contours[0].reshape(-1, 2)

    # Step 2: Interpolate contour points (upsampling)
    num_points = 1500  # semakin besar semakin halus
    t = np.linspace(0, 1, len(cnt))
    t_new = np.linspace(0, 1, num_points)

    x = np.interp(t_new, t, cnt[:, 0])
    y = np.interp(t_new, t, cnt[:, 1])

    # Step 3: Smooth curve (Savitzky-Golay smoothing)
    x_smooth = savgol_filter(x, 41, 3)
    y_smooth = savgol_filter(y, 41, 3)

    smooth_contour = np.vstack([x_smooth, y_smooth]).T.astype(np.int32)

    # Step 4: Create empty mask and draw smooth contour
    smooth_mask = np.zeros_like(combined)
    cv2.fillPoly(smooth_mask, [smooth_contour], 255)

    return smooth_mask


def overlay_mask(img, mask):
    """Highlight tumor only. Background completely dark."""
    import cv2
    import numpy as np

    # Resize mask to match image
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Convert grayscale RAW to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Normalize mask (0 or 255 → 0 or 1)
    mask_bin = (mask_resized > 0).astype(np.uint8)
    mask_3ch = np.repeat(mask_bin[:, :, None], 3, axis=2)

    # 1. Create pure black background
    background = np.zeros_like(img)

    # 2. Create bright, colorful tumor visualization
    tumor_color = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
    tumor_color = cv2.convertScaleAbs(tumor_color, alpha=1.8, beta=60)  # super bright

    # 3. Composite image:
    # tumor where mask == 1
    # black where mask == 0
    output = background * (1 - mask_3ch) + tumor_color * mask_3ch
    output = output.astype(np.uint8)

    return output


# =======================
# Streamlit UI
# =======================
st.title("🧠 Brain Tumor Segmentation System")
st.write("Sistem prediksi segmentasi tumor menggunakan YOLO. Pilih mode di bawah:")

mode = st.sidebar.selectbox("Pilih Mode", ["Single Image", "Multiple Images"])

# =======================
# MODE SINGLE IMAGE
# =======================
if mode == "Single Image":
    uploaded = st.file_uploader("Upload 1 gambar MRI", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = read_image(uploaded)
        result = get_segmentation(model, img)

        # Extract mask
        mask = extract_mask(result)
        if mask is None:
            st.error("Tidak ada tumor terdeteksi dalam gambar ini.")
            st.stop()

        # Overlay
        overlay = overlay_mask(img, mask)

        # Bounding box + label
        annotated = result.plot()

        # Confidence
        conf = float(result.boxes.conf.cpu().numpy()[0]) * 100

        # DISPLAY in 3 columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("RAW Image")
            st.image(img, use_container_width=True)


        with col2:
            st.subheader("Mask")
            st.image(mask, use_container_width=True, clamp=True)

        with col3:
            st.subheader("Mask + Bounding Box")
            st.image(annotated, use_container_width=True)
            st.write(f"Confidence: **{conf:.2f}%**")


# =======================
# MODE MULTIPLE IMAGES
# =======================
if mode == "Multiple Images":
    uploaded_files = st.file_uploader("Upload banyak gambar MRI", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"Total gambar diupload: **{len(uploaded_files)}**")

        for i, file in enumerate(uploaded_files):
            st.write(f"### Gambar {i+1}: {file.name}")

            img = read_image(file)
            result = get_segmentation(model, img)

            mask = extract_mask(result)
            if mask is None:
                st.warning(f"Tidak ada tumor pada gambar: {file.name}")
                continue

            annotated = result.plot()
            conf = float(result.boxes.conf.cpu().numpy()[0]) * 100

            # Display grid
            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(img, caption="RAW", use_container_width=True)

            with col2:
                st.image(mask, caption="MASK", use_container_width=True)

            with col3:
                st.image(annotated, caption="Segmentation + Box", use_container_width=True)
                st.write(f"Confidence: **{conf:.2f}%**")

            st.markdown("---")
