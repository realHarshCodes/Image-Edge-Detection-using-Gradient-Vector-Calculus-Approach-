# 🌀 Image Edge Detection using Gradient (Vector Calculus Approach)

A compact Python project that detects edges in images by treating pixel intensity as a scalar field and computing its **gradient vector** — the same core idea from multivariable/vector calculus, applied to computer vision.

Edges correspond to points of steep intensity change, i.e. points where the **gradient magnitude |∇I|** is large. This project computes that gradient using **Sobel operators**, derives the magnitude, and thresholds it to produce a clean binary edge map.

---

## 📐 The Math Behind It

An image can be modeled as a scalar intensity function:

```
I(x, y)
```

Its gradient is a 2D vector field:

```
∇I(x, y) = ( ∂I/∂x , ∂I/∂y ) = ( Gx , Gy )
```

- **Gx** — rate of change of intensity in the horizontal direction
- **Gy** — rate of change of intensity in the vertical direction

The **gradient magnitude** tells us *how strong* an edge is at each pixel:

```
|∇I(x, y)| = √(Gx² + Gy²)
```

The **gradient direction** (not visualized in this project, but implicit in Gx/Gy) tells us the orientation of the edge:

```
θ(x, y) = atan2(Gy, Gx)
```

In practice, the continuous partial derivatives ∂I/∂x and ∂I/∂y are approximated on a discrete pixel grid using the **Sobel operator** — a pair of 3×3 convolution kernels that estimate horizontal and vertical derivatives while smoothing out noise.

Once the magnitude is computed, a simple **threshold** separates strong intensity transitions (edges) from flat, low-gradient regions (background), producing a binary edge map.

---

## ⚙️ Pipeline

```
Input Image (BGR)
      │
      ▼
Grayscale Conversion
      │
      ▼
Sobel Gx  &  Sobel Gy   (∂I/∂x, ∂I/∂y)
      │
      ▼
Gradient Magnitude  |∇I| = √(Gx² + Gy²)
      │
      ▼
Normalize to 0–255
      │
      ▼
Threshold  →  Binary Edge Map
      │
      ▼
Side-by-side Visualization
```

---

## 🗂️ Repository Structure

```
Image-Edge-Detection-using-Gradient-Vector-Calculus-Approach-/
├── vector_calculus.ipynb   # Main notebook — full implementation
├── image/                  # Sample input/output images
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/realHarshCodes/Image-Edge-Detection-using-Gradient-Vector-Calculus-Approach-.git
cd Image-Edge-Detection-using-Gradient-Vector-Calculus-Approach-
```

### 2. Install dependencies

```bash
pip install opencv-python numpy matplotlib
```

### 3. Add an input image

Place an image named `input.jpg` in the project root (or update the filename in the notebook to point to your own image).

### 4. Run the notebook

```bash
jupyter notebook vector_calculus.ipynb
```

Run all cells to see the original image, the gradient magnitude map, and the final thresholded edge map plotted side by side.

---

## 🧠 Core Implementation

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and convert to grayscale
image = cv2.imread('input.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute gradients (∂I/∂x, ∂I/∂y) using the Sobel operator
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Gradient magnitude: |∇I| = sqrt(Gx² + Gy²)
magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Normalize to 0–255
magnitude = (magnitude / magnitude.max()) * 255
magnitude = magnitude.astype(np.uint8)

# Threshold to obtain binary edges
_, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
```

---

## 🎛️ Tuning

| Parameter | Location | Effect |
|---|---|---|
| `ksize` in `cv2.Sobel` | Gradient step | Larger kernel → smoother gradients, less sensitive to noise, coarser edges |
| Threshold value (`50`) | `cv2.threshold` | Lower → more (and fainter) edges detected; Higher → only strong edges survive |

Try adjusting the threshold value to see how edge sensitivity changes — this is the simplest and most impactful parameter to experiment with.

---

## 📊 Output

The notebook produces a single figure with three panels:

1. **Original** — the input image in RGB
2. **Gradient Magnitude |∇I(x, y)|** — grayscale visualization of edge strength at every pixel
3. **Edge Detection (Thresholded Gradient)** — final binary edge map

---

## 🛠️ Tech Stack

- **Python 3**
- **OpenCV** (`cv2`) — image I/O, Sobel convolution, thresholding
- **NumPy** — vector/matrix operations for gradient magnitude
- **Matplotlib** — visualization

---

## 💡 Why This Approach?

Unlike black-box deep learning edge detectors, this method is fully interpretable: every step maps directly to a vector calculus concept (partial derivatives, gradient vectors, vector magnitude). It's a great way to *see* multivariable calculus in action and understand the mathematical foundation that classical computer vision operators (Sobel, Prewitt, Canny) are built on.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add support for other gradient operators (Prewitt, Scharr, Roberts Cross)
- Implement Canny edge detection for comparison
- Add gradient direction visualization
- Package the pipeline into a reusable `.py` script/CLI

Open an issue or submit a pull request.

## 👤 Author

**Harsh** — [@realHarshCodes](https://github.com/realHarshCodes)
