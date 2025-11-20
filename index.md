# Crowd Monitoring and Cluster Detection System 🎯👥

A lightweight crowd monitoring and cluster detection system that uses your laptop webcam to detect people in real time, overlay a heatmap, and identify clusters of nearby people using DBSCAN.

## Overview 🔎
This project is a simple Python script that leverages OpenCV for real-time person detection and visualization, NumPy for data handling, and scikit-learn's DBSCAN for cluster detection. It's useful for demonstrations, research prototypes, and small-scale crowd analytics.

## Tech Stack 🧰

| Technology | Purpose |
|---|---|
| Python 🐍 | Core language used to implement detection and visualization logic |
| NumPy 🔢 | Array and numerical operations to manage detected object coordinates |
| OpenCV 📷 | Computer vision, object detection, image processing, and real-time webcam handling |
| scikit-learn (DBSCAN) 📊 | Density-based clustering algorithm used to detect groups / clusters of people |

## Installation & Running ▶️

1. Clone the repository:
```bash
git clone https://github.com/riya-pesu/Crowd-Monitoring-and-Cluster-Detection-System.git
```

2. Navigate into the repository:
```bash
cd Crowd-Monitoring-and-Cluster-Detection-System
```

3. (Optional) Create and activate a virtual environment, then install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

4. Run the detection script:
```bash
python crowd_detect.py
```

## Features ✨
- Real-time detection and person counting using OpenCV
- Live heatmap overlay to highlight high-density areas
- Cluster detection using DBSCAN to count and visualize groups of people
- Lightweight and easy to run on a laptop webcam

## Usage & Applications 💡
For detailed usage examples, configuration options, and application ideas, see [USAGE.md](USAGE.md).

## Contributing 🤝
Contributions, bug reports, and improvements are welcome — please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on how to contribute, submit issues, and open pull requests.

## License 📄
This project is covered under the terms described in [LICENSE.md](../LICENSE.md). Please review the license before using or contributing.
