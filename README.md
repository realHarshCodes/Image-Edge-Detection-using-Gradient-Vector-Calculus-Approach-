# 🌀 Image Edge Detection Using Gradient — Vector Calculus Approach

A compact and interpretable **computer vision project** that detects image edges by treating pixel intensity as a **scalar field** and computing its **gradient vector**.

The project applies concepts from **multivariable/vector calculus** to classical image processing:

> **Edges occur where image intensity changes rapidly, corresponding to regions with a large gradient magnitude.**

Using **Sobel operators**, the project approximates the partial derivatives of image intensity, computes the gradient magnitude, normalizes the result, and applies a threshold to generate a binary edge map.

---

## 📌 Project Overview

In an image, neighboring pixels usually have similar intensity values in smooth regions. At object boundaries, however, the intensity can change significantly over a small distance.

Mathematically, this change can be represented using the **gradient**:

```text
∇I(x, y) = (∂I/∂x, ∂I/∂y)
```

The magnitude of this gradient indicates the strength of the intensity change:

```text
|∇I(x, y)| = √(Gx² + Gy²)
```

Therefore:

```text
Low gradient magnitude  →  Smooth / uniform region
High gradient magnitude →  Strong intensity transition → Possible edge
```

This project demonstrates how a fundamental concept from **vector calculus** can be directly applied to **digital image processing**.

---

## ✨ Features

* 🖼️ Image-to-grayscale conversion
* 📐 Horizontal gradient calculation (`Gx`)
* 📐 Vertical gradient calculation (`Gy`)
* 🧮 Gradient magnitude calculation
* 🔢 Numerical approximation of image derivatives
* 🎚️ Gradient normalization
* ⚫ Binary edge-map generation
* 📊 Side-by-side result visualization
* 🧠 Fully interpretable mathematical approach
* 🐍 Lightweight Python implementation
* 📚 Useful for learning calculus + computer vision together

---

## 📐 Mathematical Foundation

An image can be modeled as a scalar intensity function:

```text
I(x, y)
```

where:

* `x` → horizontal pixel coordinate
* `y` → vertical pixel coordinate
* `I(x, y)` → intensity at that pixel

The gradient of the image is:

```text
∇I(x, y) = (∂I/∂x, ∂I/∂y)
```

The two components are approximated using the Sobel operator:

```text
Gx ≈ ∂I/∂x
Gy ≈ ∂I/∂y
```

The gradient magnitude is then calculated as:

```text
|∇I| = √(Gx² + Gy²)
```

The gradient direction is:

```text
θ = atan2(Gy, Gx)
```

The current implementation uses the gradient magnitude for edge detection. Gradient direction is not visualized, but it is available through `Gx` and `Gy`.

---

## 🔲 Sobel Operator

The Sobel operator uses two convolution kernels to approximate image derivatives.

### Horizontal Gradient

```text
Gx:

[-1   0   1]
[-2   0   2]
[-1   0   1]
```

### Vertical Gradient

```text
Gy:

[-1  -2  -1]
[ 0   0   0]
[ 1   2   1]
```

These kernels are convolved with the grayscale image to estimate how quickly intensity changes in the horizontal and vertical directions.

The Sobel operator also provides some smoothing, making it generally less sensitive to small amounts of noise than a simple finite-difference derivative.

---

## ⚙️ Processing Pipeline

```text
                  Input Image
                       │
                       ▼
              Grayscale Conversion
                       │
                       ▼
              ┌────────┴────────┐
              ▼                 ▼
          Sobel Gx           Sobel Gy
        (∂I / ∂x)           (∂I / ∂y)
              │                 │
              └────────┬────────┘
                       ▼
              Gradient Magnitude
                  √(Gx² + Gy²)
                       │
                       ▼
                Normalization
                   0 → 255
                       │
                       ▼
                  Thresholding
                       │
                       ▼
                Binary Edge Map
                       │
                       ▼
                  Visualization
```

---

## 🧪 Algorithm

The complete process can be summarized as:

1. Load the input image.
2. Convert the image from BGR to grayscale.
3. Apply the Sobel operator in the x-direction.
4. Apply the Sobel operator in the y-direction.
5. Calculate the gradient magnitude.
6. Normalize the magnitude to the range `0–255`.
7. Apply a threshold.
8. Generate the binary edge map.
9. Visualize the results.

---

## 🗂️ Repository Structure

```text
Image-Edge-Detection-using-Gradient-Vector-Calculus-Approach-/
│
├── vector_calculus.ipynb
│   └── Main implementation and visualization
│
├── image/
│   └── Sample input/output images
│
└── README.md
    └── Project documentation
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

For Jupyter Notebook:

```bash
pip install notebook
```

### 3. Add an Input Image

Place an image named:

```text
input.jpg
```

in the project root directory.

Alternatively, modify the image path inside the notebook.

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

# Compute horizontal and vertical gradients
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

# Calculate gradient magnitude
magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Normalize to 0–255
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

## 📊 Output

The notebook produces three main visualizations:

### 1. Original Image

The input image displayed in RGB format.

### 2. Gradient Magnitude

Displays the strength of the image gradient.

```text
Brighter → Stronger intensity change
Darker   → Weaker intensity change
```

### 3. Thresholded Edge Map

The gradient magnitude is converted into a binary image:

```text
White → Detected edge
Black → Non-edge region
```

---

## 🎛️ Parameter Tuning

### Sobel Kernel Size

```python
ksize=3
```

The kernel size determines the neighborhood used by the Sobel operator.

Generally:

* Smaller kernel → finer/local changes
* Larger kernel → smoother response and reduced sensitivity to small variations

For this project, `ksize=3` provides a good balance between simplicity and edge localization.

### Threshold

```python
cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
```

The threshold determines which gradient magnitudes are considered edges.

| Threshold | Result                                   |
| --------- | ---------------------------------------- |
| Lower     | More edges, including weaker transitions |
| Medium    | Balanced edge detection                  |
| Higher    | Fewer edges, mainly strong transitions   |

Try values such as:

```text
25
50
75
100
150
```

to observe how the resulting edge map changes.

---

# ⚠️ Limitations

Although the gradient-based approach is simple and mathematically interpretable, it has several limitations.

### 1. Sensitive to Noise

Image noise can create sudden intensity changes that appear as false edges.

```text
Noise → Large local intensity change → False edge
```

Applying a Gaussian blur before the Sobel operation can help reduce this problem.

---

### 2. Threshold Selection Is Manual

The current implementation uses a fixed threshold:

```python
50
```

A threshold that works well for one image may not work well for another.

Images with different lighting, contrast, or noise levels may require different threshold values.

---

### 3. Weak Edges Can Be Missed

If the threshold is too high, subtle boundaries may disappear from the final edge map.

For example:

```text
Low-contrast boundary
        ↓
Small gradient magnitude
        ↓
Removed by high threshold
```

---

### 4. Strong Edges Can Produce Thick Responses

Sobel-based gradient detection does not inherently produce perfectly one-pixel-wide edges.

Strong transitions can result in relatively thick edge regions.

More advanced methods such as **Canny edge detection** perform additional processing, including non-maximum suppression, to produce thinner edges.

---

### 5. Illumination Changes Can Affect Results

Changes in lighting, shadows, or brightness can create strong gradients that may be detected as edges even when they do not correspond to meaningful object boundaries.

---

### 6. No Semantic Understanding

This method detects **intensity transitions**, not objects.

For example, it cannot inherently determine:

```text
"This edge belongs to a car."
```

It only determines that:

```text
"There is a significant intensity change here."
```

Deep-learning-based computer vision models can provide much richer semantic understanding.

---

### 7. Grayscale Conversion Removes Color Information

The current pipeline converts the image to grayscale before calculating the gradient.

Consequently, some edges caused primarily by **color differences** may not be represented as strongly as they could be in a color-aware approach.

---

### 8. Not a Replacement for Advanced Edge Detectors

This project demonstrates the mathematical foundation of gradient-based edge detection, but real-world computer vision systems may require more robust algorithms such as:

* Canny
* Laplacian of Gaussian
* Scharr
* Multi-scale edge detection
* Deep-learning-based edge detection

---

## 💡 Why This Approach?

The main goal of this project is **interpretability and learning**, rather than achieving the most advanced edge detection performance.

The complete pipeline maps naturally to mathematical concepts:

| Computer Vision   | Mathematical Concept      |
| ----------------- | ------------------------- |
| Pixel intensity   | Scalar field              |
| `Gx`              | Approximation of `∂I/∂x`  |
| `Gy`              | Approximation of `∂I/∂y`  |
| `∇I`              | Gradient vector           |
| `√(Gx² + Gy²)`    | Vector magnitude          |
| `atan2(Gy, Gx)`   | Gradient direction        |
| Sobel convolution | Numerical differentiation |
| Thresholding      | Binary classification     |

This makes the project a useful bridge between:

**Mathematics → Numerical Methods → Image Processing → Computer Vision**

---

## 🆚 Possible Extensions

This project can be extended to compare different classical edge detection methods:

| Method        | Main Idea                               |
| ------------- | --------------------------------------- |
| Sobel         | Gradient-based edge detection           |
| Prewitt       | Gradient-based differentiation          |
| Scharr        | Improved rotational accuracy over Sobel |
| Roberts Cross | Small kernel gradient approximation     |
| Canny         | Multi-stage edge detection              |

---

## 🚀 Future Improvements

* [ ] Add gradient direction visualization
* [ ] Visualize `Gx` and `Gy` separately
* [ ] Add Gaussian noise experiments
* [ ] Add Gaussian blur before gradient calculation
* [ ] Implement Prewitt operator
* [ ] Implement Scharr operator
* [ ] Implement Roberts Cross operator
* [ ] Implement Canny edge detection
* [ ] Compare Sobel vs Canny
* [ ] Add automatic threshold selection
* [ ] Add multiple input-image support
* [ ] Create a reusable Python script
* [ ] Create a command-line interface
* [ ] Add quantitative comparison between edge detectors

---

## 🛠️ Tech Stack

| Technology           | Purpose                               |
| -------------------- | ------------------------------------- |
| **Python 3**         | Core programming language             |
| **OpenCV**           | Image processing and Sobel operations |
| **NumPy**            | Numerical and vector calculations     |
| **Matplotlib**       | Visualization                         |
| **Jupyter Notebook** | Interactive experimentation           |

---

## 📚 Concepts Demonstrated

This project combines several important concepts:

### Mathematics

* Scalar fields
* Partial derivatives
* Gradient vectors
* Vector magnitude
* Vector direction
* Numerical differentiation

### Computer Vision

* Image representation
* Grayscale conversion
* Convolution
* Sobel operators
* Gradient-based edge detection
* Thresholding

### Programming

* NumPy array operations
* OpenCV image processing
* Matplotlib visualization
* Jupyter Notebook workflows

---

## 🤝 Contributing

Contributions and improvements are welcome.

You can contribute by:

* Adding new gradient operators
* Improving visualizations
* Adding new experiments
* Comparing edge detection techniques
* Improving documentation
* Adding support for additional input formats

Feel free to open an issue or submit a pull request.

---

## 👤 Author

**Harsh**

GitHub: [@realHarshCodes](https://github.com/realHarshCodes)

---

## 🔗 Repository

[Image Edge Detection using Gradient — Vector Calculus Approach](https://github.com/realHarshCodes/Image-Edge-Detection-using-Gradient-Vector-Calculus-Approach-)
