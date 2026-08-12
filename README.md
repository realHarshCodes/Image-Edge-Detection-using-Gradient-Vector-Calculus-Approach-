# 🌀 Image Edge Detection Using Gradient — Vector Calculus Approach

A compact Python computer vision project that detects image edges using the **gradient vector** from multivariable calculus.

The image is treated as a scalar intensity field, where strong intensity changes correspond to edges. **Sobel operators** are used to approximate the partial derivatives, calculate the gradient magnitude, and generate a binary edge map.

---

## 📐 Mathematical Foundation

An image can be represented as a scalar function:

```text
I(x, y)
```

Its gradient is:

```text
∇I(x, y) = (∂I/∂x, ∂I/∂y) = (Gx, Gy)
```

The gradient magnitude represents the strength of the intensity change:

```text
|∇I| = √(Gx² + Gy²)
```

A large gradient magnitude generally indicates an edge.

The gradient direction can be calculated using:

```text
θ = atan2(Gy, Gx)
```

The project uses the gradient magnitude for edge detection.

---

## ⚙️ Pipeline

```text
Input Image
     ↓
Grayscale Conversion
     ↓
Sobel Gx & Gy
     ↓
Gradient Magnitude
     ↓
Normalize to 0–255
     ↓
Threshold
     ↓
Binary Edge Map
```

---

## 🗂️ Repository Structure

```text
Image-Edge-Detection-using-Gradient-Vector-Calculus-Approach-/
├── vector_calculus.ipynb
├── image/
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/realHarshCodes/Image-Edge-Detection-using-Gradient-Vector-Calculus-Approach-.git

cd Image-Edge-Detection-using-Gradient-Vector-Calculus-Approach-
```

### 2. Install Dependencies

```bash
pip install opencv-python numpy matplotlib notebook
```

### 3. Add an Input Image

Place an image named `input.jpg` in the project root, or change the image path in the notebook.

### 4. Run the Notebook

```bash
jupyter notebook vector_calculus.ipynb
```

---

## 🧠 Core Implementation

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and convert to grayscale
image = cv2.imread("input.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute gradients using Sobel operators
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Gradient magnitude
magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Normalize to 0–255
magnitude = (magnitude / magnitude.max()) * 255
magnitude = magnitude.astype(np.uint8)

# Threshold to obtain binary edges
_, edges = cv2.threshold(
    magnitude, 50, 255, cv2.THRESH_BINARY
)
```

---

## 📊 Output

The notebook visualizes:

1. **Original Image**
2. **Gradient Magnitude** — shows edge strength
3. **Thresholded Edge Map** — final detected edges

---

## 🎛️ Tuning

### Sobel Kernel

```python
ksize=3
```

* Smaller kernel → finer local changes
* Larger kernel → smoother response

### Threshold

```python
50
```

* Lower → more edges detected
* Higher → only stronger edges remain

---

## ⚠️ Limitations

* **Sensitive to noise** — noise can create false edges.
* **Manual threshold** — a fixed threshold may not work equally well for every image.
* **Weak edges may be missed** — especially with a high threshold.
* **Thicker edges** — Sobel does not guarantee one-pixel-wide edges.
* **Lighting and shadows** — can produce unwanted gradients.
* **No semantic understanding** — detects intensity changes, not objects.
* **Grayscale processing** — color information is discarded.

For more robust edge detection, methods such as **Canny** can be used.

---

## 💡 Why This Approach?

This project connects **vector calculus** with **classical computer vision** in a simple and interpretable way.

| Concept                   | Application     |
| ------------------------- | --------------- |
| Scalar field              | Image intensity |
| Partial derivatives       | `Gx`, `Gy`      |
| Gradient vector           | `∇I`            |
| Vector magnitude          | Edge strength   |
| Numerical differentiation | Sobel operator  |

---

## 🛠️ Tech Stack

* **Python 3**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **Jupyter Notebook**

---

## 🚀 Future Improvements

* [ ] Add gradient direction visualization
* [ ] Implement Prewitt, Scharr, and Roberts operators
* [ ] Compare Sobel with Canny
* [ ] Add automatic threshold selection
* [ ] Add noise-reduction experiments

---

## 👤 Author

**Harsh** — [@realHarshCodes](https://github.com/realHarshCodes)

## 🔗 Repository

[Image Edge Detection using Gradient — Vector Calculus Approach](https://github.com/realHarshCodes/Image-Edge-Detection-using-Gradient-Vector-Calculus-Approach-)
