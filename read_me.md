# 🚗 Door-State Detection AI Demo (v1.0)


---

### 🧠 Real-Time Car Door Detection — Powered by Deep Learning

This project connects a trained CNN model to a live 3D car simulation, detecting which doors are open in **real-time**.  
Optimized for NVIDIA GPUs with **Selenium-based browser integration** and **motion-aware auto capture**.

---

## Features

-**Real-time inference** 
-**Live prediction overlay** with confidence
-**Synchronized car door status**


---

## 🧰 Requirements

- **Python** 3.12+
- **Google Chrome** (latest)
- **ChromeDriver** installed and on your PATH  
  *(Tip: match ChromeDriver version with your Chrome browser version.)*
- **GPU:** NVIDIA (T4, A10G, RTX recommended for model training)

---

## ⚙️ Installation

```bash
git clone https://github.com/nunikputri/door_state_ai_demo.git
cd door_state_ai_demo_v1.0
pip install -r requirements.txt