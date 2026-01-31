# NeuroSpectrum
**Real-Time Emotion-Driven Wireframe Visualization using Deep Learning**

AffectiveWire è un sistema di **visualizzazione audio–reattiva in tempo reale** che utilizza un **modello di Deep Learning per la Music Emotion Recognition (MER)** al fine di trasformare le **emozioni musicali (Valence & Arousal)** in una **forma tridimensionale dinamica wireframe**.

Il progetto combina **intelligenza artificiale, signal processing e computer graphics**, collocandosi a metà tra **ingegneria** e **arte generativa**.

---

## ✨ Caratteristiche principali

- 🎵 Riproduzione audio da file
- 🧠 Predizione emozionale tramite **CNN + GRU**
- ❤️ Stima continua di:
  - **Valence** → colore emozionale
  - **Arousal** → energia visiva
- 🌐 Visualizzazione **3D wireframe (icosfera)**
- ⚡ Deformazioni procedurali in tempo reale
- 🎨 Colori dinamici basati sul mood
- 🧩 Pipeline completamente modulare

---

## 🧠 Modello di Intelligenza Artificiale

Il sistema utilizza un modello di **Music Emotion Recognition** addestrato sul **DEAM Dataset**, con output continuo:
| Dimensione | Range | Significato |
|----------|-------|-------------|
| Valence  | [-1, 1] | Emozione negativa ↔ positiva |
| Arousal  | [0, 1] | Calma ↔ energia |

## 🖥️ Visualizzazione

- Motore grafico: **ModernGL**
- Window manager: **moderngl-window**
- Rendering: **OpenGL 3.3**
- Mesh: **Icosahedron → Icosphere**
- Modalità: **wireframe dinamico**

Ogni vertice viene deformato lungo la normale tramite **noise procedurale temporale**, controllata dall’emozione musicale.
