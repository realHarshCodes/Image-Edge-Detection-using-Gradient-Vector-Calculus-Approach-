# 🌀 Image Edge Detection Using Gradient — Vector Calculus Approach

A compact Python computer vision project that detects image edges by treating pixel intensity as a **scalar field** and computing its **gradient vector**.

The core idea comes directly from **multivariable/vector calculus**:

> Edges occur where image intensity changes rapidly, meaning the **gradient magnitude** is large.

This project uses **Sobel operators** to approximate the partial derivatives of image intensity, calculates the gradient magnitude, and applies a threshold to produce a binary edge map.

---

## 📐 Mathematical Foundation

An image can be represented as a scalar intensity function:

```text
I(x, y)
```

Its gradient is a 2D vector field:

```text
∇I(x, y) = ( ∂I/∂x , ∂I/∂y ) = (Gx, Gy)
```

Where:

* **Gx** — rate of intensity change in the horizontal direction
* **Gy** — rate of intensity change in the vertical direction

The **gradient magnitude** represents the strength of the intensity change:

```text
|∇I(x, y)| = √(Gx² + Gy²)
```

A large gradient magnitude generally indicates a potential image edge.

The gradient direction can also be calculated using:

```text
θ(x, y) = atan2(Gy, Gx)
```

Although gradient direction is not visualized in the current implementation, it is implicitly available from `Gx` and `Gy`.

### From Calculus to Computer Vision

Because digital images consist of discrete pixels, the continuous partial derivatives are approximated numerically.

This project uses the **Sobel operator**, which applies two 3×3 convolution kernels to estimate the horizontal and vertical derivatives while providing some smoothing against noise.

---

## ⚙️ Processing Pipeline

```text
Input Image
     │
     ▼
Grayscale Conversion
     │
     ▼
Sobel Gx & Gy
(∂I/∂x, ∂I/∂y)
     │
     ▼
Gradient Magnitude
|∇I| = √(Gx² + Gy²)
     │
     ▼
Normalize to 0–255
     │
     ▼
Threshold
     │
     ▼
Binary Edge Map
     │
     ▼
Visualization
```

---

## ✨ Features

* 🖼️ Converts an input image to grayscale
* 📐 Computes horizontal and vertical image gradients
* 🧮 Calculates gradient magnitude using vector calculus
* 🔍 Detects strong intensity transitions
* ⚫ Produces a binary edge map
* 📊 Visualizes the original image, gradient magnitude, and detected edges
* 🧠 Fully interpretable classical computer vision approach
* 🐍 Implemented entirely in Python

---

## 🗂️ Repository Structure

```text
Image-Edge-Detection-using-Gradient-Vector-Calculus-Approach-/
│
├── vector_calculus.ipynb    # Main implementation notebook
├── image/                   # Sample input/output images
└── README.md                # Project documentation
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
pip install opencv-python numpy matplotlib
```

If Jupyter Notebook is not already installed:

```bash
pip install notebook
```

### 3. Add an Input Image

Place an image named:

```text
input.jpg
```

in the project root directory.

You can also modify the image path directly inside the notebook.

### 4. Run the Notebook

```bash
jupyter notebook vector_calculus.ipynb
```

Run all cells to generate the edge detection results.

---

## 🧠 Core Implementation

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("input.jpg")

# Convert BGR image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute image gradients using Sobel operators
grad_x = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

grad_y = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

# Gradient magnitude
magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Normalize gradient magnitude to 0–255
magnitude = (magnitude / magnitude.max()) * 255
magnitude = magnitude.astype(np.uint8)

# Threshold to obtain binary edge map
_, edges = cv2.threshold(
    magnitude,
    50,
    255,
    cv2.THRESH_BINARY
)
```

---

## 📊 Visualization

The notebook generates a side-by-side visualization containing:

### 1. Original Image

The original input image displayed in RGB format.

### 2. Gradient Magnitude

A grayscale representation of:

```text
|∇I(x, y)|
```

Brighter pixels represent stronger intensity changes.

### 3. Edge Detection

The thresholded gradient magnitude produces a binary image where:

* **White** → detected edge
* **Black** → non-edge region

---

## 🎛️ Parameters You Can Experiment With

| Parameter      | Location          | Effect                           |
| -------------- | ----------------- | -------------------------------- |
| `ksize`        | `cv2.Sobel()`     | Controls the Sobel kernel size   |
| Threshold `50` | `cv2.threshold()` | Controls edge sensitivity        |
| Input image    | `cv2.imread()`    | Changes the image being analyzed |

### Sobel Kernel Size

```python
ksize=3
```

A larger kernel generally produces a smoother gradient response and can reduce sensitivity to small-scale noise, but may also produce coarser edge localization.

### Threshold

```python
cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
```

* **Lower threshold** → detects more and weaker edges
* **Higher threshold** → keeps mainly stronger edges

Experimenting with the threshold is the easiest way to observe how edge sensitivity changes.

---

## 🔬 Understanding the Gradient

For every pixel, the Sobel operators estimate:

```text
Gx = ∂I/∂x
Gy = ∂I/∂y
```

These two values form a gradient vector:

```text
∇I = (Gx, Gy)
```

The magnitude of this vector determines how rapidly the image intensity changes:

```text
|∇I| = √(Gx² + Gy²)
```

Therefore:

```text
Small |∇I|  → relatively uniform region
Large |∇I|  → strong intensity transition → possible edge
```

This provides a direct connection between **vector calculus** and **classical computer vision**.

---

## 💡 Why Use This Approach?

Unlike many deep-learning-based edge detection systems, this method is **mathematically interpretable**.

Every major step corresponds directly to a mathematical concept:

| Computer Vision    | Mathematical Concept     |
| ------------------ | ------------------------ |
| Pixel intensity    | Scalar field             |
| Sobel Gx           | Approximation of `∂I/∂x` |
| Sobel Gy           | Approximation of `∂I/∂y` |
| Gradient           | Vector field             |
| Gradient magnitude | Vector magnitude         |
| Gradient direction | `atan2(Gy, Gx)`          |
| Thresholding       | Edge classification      |

This makes the project useful for understanding how concepts from **multivariable calculus, numerical differentiation, and linear algebra** are applied to image processing.

---

## 🆚 Classical Edge Detection

This project uses the gradient-based approach as a foundation for understanding more advanced edge detectors.

Possible extensions include:

* **Sobel**
* **Prewitt**
* **Scharr**
* **Roberts Cross**
* **Canny**

The project can therefore be extended into a comparison of different classical edge detection techniques.

---

## 🛠️ Tech Stack

* **Python 3**
* **OpenCV** — image processing, Sobel operators, thresholding
* **NumPy** — numerical and vector operations
* **Matplotlib** — visualization
* **Jupyter Notebook** — interactive implementation

---

## 🚀 Possible Future Improvements

* [ ] Add gradient direction visualization
* [ ] Implement Prewitt operator
* [ ] Implement Scharr operator
* [ ] Implement Roberts Cross operator
* [ ] Add Canny edge detection for comparison
* [ ] Compare different threshold values automatically
* [ ] Add noise and filtering experiments
* [ ] Create a reusable Python CLI application
* [ ] Compare edge detection results using different images

---

## 🤝 Contributing

Contributions and improvements are welcome.

You can contribute by:

* Adding new gradient operators
* Improving visualization
* Adding additional image-processing techniques
* Comparing different edge detection algorithms
* Improving documentation

Feel free to open an issue or submit a pull request.

---

## 👤 Author

**Harsh** — [@realHarshCodes](https://github.com/realHarshCodes)

---

## 🔗 Repository

[Image Edge Detection using Gradient — Vector Calculus Approach](https://github.com/realHarshCodes/Image-Edge-Detection-using-Gradient-Vector-Calculus-Approach-)
