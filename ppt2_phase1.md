---
marp: true
title: INTERDISCIPLINARY PHASE 1 PRESENTATION
paginate: true
header: 'Go Change the World'
footer: 'RV College of Engineering'
style: |
  @import "default";
  @import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@400;700&display=swap');

  :root {
    font-family: "Work Sans", Arial;
    --burnt-sienna: #E97451;
    --chrome-yellow: #FFBF00;
    --dark-text: #2D2D2D;
    --light-bg: #FAF9F6;
  }

  @keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }

  section {
    background-color: var(--light-bg);
    background: linear-gradient(-45deg, #fff8e1, #ffe0b2, #fbe9e7, #ffffff);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: var(--dark-text);
    font-size: 24px;
  }

  section.lead {
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  
  h1 {
    color: var(--burnt-sienna);
    border-bottom: 3px solid var(--chrome-yellow);
    padding-bottom: 10px;
    font-size: 1.4em;
  }
  
  h2 {
    color: #C04000;
    font-size: 1.1em;
  }

  strong {
    color: var(--burnt-sienna);
  }

  table {
    font-size: 0.70em;
    width: 100%;
    border-collapse: collapse;
  }
  th {
    background-color: var(--burnt-sienna);
    color: white;
    border: 1px solid #ddd;
  }
  td {
    border: 1px solid #ccc;
    background-color: rgba(255, 255, 255, 0.7);
  }

  .center-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
  }

  .side-by-side {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    align-items: start;
  }

  .tinytext {
    font-size: 0.65em;
  }
  
  img {
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    border-radius: 5px;
  }
  section::before {
  content: "";
  position: absolute;
  top: 0px;
  right: 20px;   /* change to right: 20px if you want right side */
  width: 150px;
  height: 150px;
  background: url("./logo.png") no-repeat center;
  background-size: contain;
  z-index: 10;
  }

---

<!-- Slide 1: Title -->
<!-- _class: lead -->
<!-- # IDP Phase 1 Presentation -->
# Task Based Object Detection
## With Semantic Distillation and Dynamic Gating using Accelerator on FPGA for Embedded Applications
***An Assistive Vision Framework for Inclusive Education and Interactive Learning Analysis.***
**Team Members:**
Ackshaya Keerthi G 1RV23CS013
Bhoomika Sundar 1RV23CS066
Tandle Suhani 1RV23EC164
Vaibhavi D 1RV23EC177

**Guide:**
Dr. Roopa.T.S, Assistant Professor, Dept of Mechanical Engineering
**R V College of Engineering**

---
# 01. Problem Statement: The Accessibility Gap

In inclusive learning environments (Labs/Art Studios), students with **visual impairments/visual search deficit** face a Semantic Barrier when using standard AI tools.

**The Functional Challenge:**
*   **Label Dependency:** Traditional detectors require exact names ("Is there a glass?"). A student often needs an object based on its **Function** ("Is there something I can pour into?").
*   **Cognitive Load:** Forcing a student to guess a list of synonyms (bottle, tumbler, cup) degrades the focus on the actual learning task.
*   **Cluttering Confusion** A reduced ability to items within a cluttered or distracting environment. ADHD, Parkinson's, dyslexia, or spatial attentional issues
*   **Educational Exclusion:** Lack of real-time, intent-based feedback prevents independent participation in STEM and Art experimentation.

---
# 01. Problem Statement : Hardware Barrier

Conventional detection (YOLO/SSD) is **Task-Agnostic** as it identifies all 80 COCO classes blindly, leading to computational waste on edge devices.

**The Research Challenges:**
1.  **Contextual Irrelevance:** Standard models cannot prioritize objects based on goals (e.g., "Pouring").
2.  **The VLM Weight Gap:** Large Vision-Language Models (CLIP) are too massive for the **VEGA RISC-V** processor.
3.  **The Memory Bottleneck:** Standard **Attention** math ($O(N^2)$) exceeds the BRAM limits of the **Genesys-2 FPGA**.

**Objective:** To build a **Task-Driven** system that selectively filters features using **Dynamic Gating**, trained via **Knowledge Distillation** for **INT8 Execution**.
<!-- 
---
# 02. Theme Justification: SDG 4 Quality Education

**Sub theme : Digital Identity and Learning Analysis**.

**Learning Analysis via Functional Intent:**
*   **Intent-Based Interaction:** Instead of analyzing "what" a student sees, we analyze the **Functional Context** of their learning. 
*   **Adaptive Support:** By observing the Task Vectors a student uses most frequently, educators can analyze a student's spatial reasoning and experimental flow in complex lab environments. -->

---
<!-- Slide 4: Theme Justification
# 02. Theme: Digital Identity & Learning Analysis
# do we need this slide??
This project aligns with **Quality Education (SDG 4)** through the following pillars:

*   **Digital Identity:** We define a user's digital identity through their **Semantic Intent Profile**. The Task Embedding serves as a real-time digital signature that reconfigures the hardware's visual perception to match the user's current educational needs.
*   **Learning Analysis:** The system enables the analysis of **Functional Interaction Patterns**. By mapping how a student queries their environment (Affordance-based search), educators can evaluate a student's spatial reasoning and conceptual understanding in laboratory or studio settings.
*   **Assistive Autonomy:** By optimizing the system for the **VEGA Processor**, we provide a self-contained assistive tool that fosters inclusive participation in interactive learning environments.
--- -->

<!-- Slide 3: Literature Survey (Part 1) -->
<!-- _class: tinytext -->
# 03. Literature Survey

Focus: Moving from General Detection to Task-Awareness

| Author (Year) | Methodology | Key Contributions | Hardware Gap |
| :--- | :--- | :--- | :--- |
| **Radford et al.(2021)** | **CLIP:** Aligns images and text by training on 400 million image-caption pairs to learn their underlying relationships. | Introduced "zero-shot" learning, allowing the model to identify new objects from text prompts without requiring task-specific training. | The massive model size and high computational demand make it unsuited for low-power, memory-constrained hardware |
| **Vasu et al. (2023)** | **MobileCLIP**: Compresses the knowledge of massive CLIP models into a smaller, highly efficient model using a technique called "multi-modal reinforced training." | Proved that small models can inherit "Teacher" intelligence. | Heavy Transformer overhead; not optimized for INT8/FPGA. |
| **Li et al. (2022)** | **GLIP:** detects objects by aligning text queries with their corresponding locations in an image by attention. | Turns object detection into a language problem—finding things in an image based on text descriptions. | Computationally intensive attention mechanism in use. |
| **Cheng et al. (2021)** |  **YOLO-World**: Fuses YOLOv8 with a CLIP text encoder using RepVL-PAN for vision-language alignment. | First to achieve real-time, open-vocabulary object detection without task-specific training. | Relies on FP32 math, high memory fusion, making it inefficient for low-power FPGAs. |


---
<!-- Slide 4: Literature Survey (Part 2) -->
<!-- _class: tinytext -->
# 03. Literature Survey 

Focus: Moving from General Detection to Task-Awareness

| Author (Year) | Methodology | Key Contributions | Hardware Gap |
| :--- | :--- | :--- | :--- |
| **Zeng et al. (2022)** | **Socratic Models:** LLM-based reasoning for vision. | Used text prompts to guide visual search results. | Post-processing only; doesn't optimize internal feature extraction. |
| **Yang (2022)** | **ROSETTA**: Uses task-aware gates to only activate specific channels depending on the current task. |Allows the model to continually learn new objects without forgetting old ones. | Focuses on memory retention rather than extreme low-power hardware optimization.|
| **Rao. (2021)** | **DynamicViT**: Drops useless image parts (tokens) while running to save work. |Proves that skipping irrelevant pixels massively speeds up vision models. | Still uses floating-point Attention math. No text-matching or INT8. |
| **Sawatzky et al. (2019)** | **GGNN**: It uses a GGNN (Gated Graph Neural Network) to model relationships between objects and pick task-relevant regions. | It introduces a model that integrates task context into object detection to prioritize task-relevant objects. | GGNN-based message passing over many regions is slow and computationally heavy to run. |

---
<!-- Slide 5: Objectives (Part 2) -->
<!-- _class: tinytext -->
# 04. Objectives

1. **Semantic Distillation:** Transfer CLIP’s task oriented object detection abilities into a YOLOv8 backbone to enable reasoning-based detection of object affordances.
2. **Efficiency Optimization:** Replace heavy Attention layers with lightweight **Channel Gating** to minimize latency and memory usage.
3. **Hardware Acceleration:** Implement INT8 quantization and **Bit-Shift logic** to optimize the pipeline for the VEGA FPGA processor.
4. **Performance Validation:** Achieve real-time, low-power execution of semantic tasks on the FPGA platform.
---


<!-- Slide 4: Methodology - Proposed Architecture -->
# 05. Methodology: Dual-Stream Distillation

Our framework of **Knowledge Distillation** uses a "Teacher-Student" setup to move intelligence into a lightweight architecture.

<div class="side-by-side">

<div>

### 1. Semantic Stream (Teacher)
*   **CLIP Text Encoder:** Encodes "Slice" and "Cut" into identical vectors.
*   **Tiny Task Mapper:** A distilled linear layer ($512 \times V$) that converts text into embeddings "on-the-go" for the FPGA.

</div>

<div>

### 2. Visual Stream (Student)
*   **YOLOv8 Backbone:** Features tapped at Stage 4 ($20 \times 20 \times 512$).
*   **Dynamic Gating:** The task vector modulates the image channels via **Sigmoid-Gated Hadamard Product**.

</div>
</div>

---

# 04. Methodology
<div class="center-content">

![w:400](methodology_flowchart.png)
*Fig 1: Methodology*

</div>

---

# System Evaluation & Setup

### Evaluation Metrics
- **Dataset:** COCO Dataset
- **Tasks:** Predefined task set specified by the contest

#### SOFTWARE REQUIREMENTS
*   **Language*:* Python 3.9+
*   **AI Frameworks:** PyTorch, Ultralytics YOLOv8, HuggingFace Transformers (for CLIP/DistilBERT).
*   **Optimization Tools:** PyTorch Quantization API, ONNX Runtime.
*   **Hardware Toolchain:** VEGA SDK (for RISC-V compilation), Xilinx Vivado/Vitis (for FPGA bitstream generation and IP integration).

---

# The COCO Dataset

*   **What it is:** A gold-standard collection of over **330,000 images** used to train AI to "see" and understand the world.
*   **Key Features:**
    *   **80 Object Categories:** Includes everyday items like tools, furniture, and people.
    *   **Real-World Context:** Objects are shown in complex, natural scenes rather than simple backgrounds.
    *   **Detailed Labels:** Provides precise outlines (masks) and location boxes for every object.
*   **Why it fits our Project:**
    *   **Task-Ready:** Because it shows how objects relate to their surroundings, it helps our model understand not just *what* an object is, but *how* to interact with it.
    *   **Precision:** The high-quality data ensures our "task-oriented" detection is accurate and reliable.
---

# 05. Software System Architecture
<div class="center-content">

![w:350](system_architecture.png)
*Fig 2: System Architecture Diagram*

</div>

---

<div class="side-by-side">
<!-- yet to refine -->
<div class="center-content">

![w:500](idea1.jpg)
*Fig 3: Knowledge distillation*

</div>
<div class="center-content">

![w:600](idea2.jpg)
*Fig 4: Gating*
</div>
</div>
---

<!-- ---

# Software Implementation
### Modules Implemented in Software

The following core modules are handled during the initial software phase:

1. **Object Detection**: Identifying bounding boxes and classes.
2. **Embedding Generation**: Creating high-dimensional feature vectors.
3. **Distillation of Gating Vector**: Optimizing the selection mechanism for model efficiency. -->

---

# Hardware Novelty
### Platform: Genesys-2 FPGA Board

Custom hardware accelerators are designed to optimize computation:

- **Embedding Quantization Unit**
  - Converts embeddings from Floating Point to **INT8** to reduce memory footprint and latency.
- **Systolic Array Multiplier**
  - Performs high-speed parallel vector multiplication for similarity math.
- **Similarity Score Computation Module**
  - Dedicated logic to compute **Cosine Similarity** between embeddings.

---

# Deployment Architecture
### Operational Partitioning

The real-time system distributes tasks between the processor and the FPGA:

| **VEGA Processor** | **FPGA Accelerator** |
| :--- | :--- |
| Object Detection | Similarity Computation |
| Real-time Hadamard Product | Ranking of Objects |
| CLIP Encoding | |



---






# Hardware Architecture: Programmable Logic (PL)

### A. Unified Systolic Array Multiplier (USAM)
- **High-Throughput Core:** $N \times N$ array for INT8 matrix multiplication.
- **Weight-Stationary Flow:** Reuses task-attribute weights ($E^T$) across candidate object embeddings ($V$) to minimize DDR3 memory bottlenecks.
- **Optimized Pipeline:** Deeply pipelined MAC units to maximize FPGA clock frequency.

### B. Feature Gating Unit (FGU)
- **Streaming Architecture:** Sits between the Visual Backbone and Affinity Module.
- **Vector Multiplier:** Executes element-wise products in a **single cycle**.
- **Sparsity Controller:** Features **Zero-Skip Logic**; skips dot-products for zero-coefficient gates to significantly reduce dynamic power consumption.

---
### C. Quantization & Scaling Unit (QSU)
- **Precision Conversion:** Transforms 32-bit floating-point (VEGA) into 8-bit Fixed-Point (INT8).
- **Precision Recovery:** Uses hardware-based **Re-quantization shift logic** to restore numerical precision following intensive matrix operations.

# System Workflow Summary

1. **Preprocessing**: VEGA processor handles detection and CLIP encoding.
2. **Acceleration**: FPGA performs quantized similarity matching via systolic arrays.
3. **Output**: FPGA provides the final ranking of objects based on computed scores.
---





<!-- Slide 5: The Novelty - Dynamic Gating -->
# 06. Novelty 1: Dynamic Gating Mechanism

Replacing **Heavy Attention** with **Lightweight Hardware-Native Gating**.

**The Logic:**
*   **Full Attention:** $O(N^2)$ — Every pixel looks at the text. (Too heavy for VEGA).
*   **Proposed Gating:** $O(C)$ — The task text creates 512 "Volume Knobs" applied globally to image channels.

**Hardware Advantage:**
By quantizing gates to **INT8**, the VEGA processor performs **Bit-Shift** operations to "mute" irrelevant channels, skipping 30-50% of multiplications in subsequent layers.

---

<!-- Slide 6: The Novelty - Knowledge Distillation -->
# 06. Novelty 2: Semantic Affordance Distillation

Training the Student (YOLO) to behave like the Expert (CLIP).

**The Process:**
1.  **Expert Opinion:** CLIP generates a **$16 \times 16$ Heatmap** indicating functional relevance (e.g., "Pouring").
2.  **Student Alignment:** We interfere at the YOLO backbone, using a **$1 \times 1$ Convolution** and **Bilinear Interpolation** to match CLIP's dimensions.
3.  **MSE Loss:** The Student adjusts its weights so its **Gated Feature Map** matches the Teacher’s Semantic Heatmap.

**Result:** YOLO learns **Affordances** (use-cases) instead of just static labels.

---

<!-- 
Slide 7: Hardware Implementation (VEGA Processor)
# 07. Hardware Integration: Genesys-2 FPGA

Bridging the gap between High-Level Python and Low-Level RTL.

<div class="side-by-side">

<div>

### 1. Quantization (INT8)
*   **QAT:** Quantization-Aware Training ensures distillation remains stable with integers.
*   **Precision:** Scaling from Float32 to 8-bit to fit VEGA ISA.

</div>

<div>

### 2. FPGA Execution
*   **Task Vector:** Pre-encoded synonyms stored in on-chip BRAM.
*   **Accelerator:** Custom logic for fast Hadamard Product and sparsity-skipping.

</div>
</div>

--- -->

<!-- Slide 8: Performance Evaluation & Benchmarking -->
# 08. Performance Evaluation & Benchmarking

The system is evaluated across three critical dimensions: **Detection Accuracy**, **Semantic Alignment**, and **Hardware Efficiency** on the COCO 2017 Validation Set.

| Metric (Notation) | Technical Definition | Target | Research Significance |
| :--- | :--- | :--- | :--- |
| **mAP@.5:.95** | Mean Average Precision calculated across IoU thresholds from 0.5 to 0.95. | **> 0.40** | Ensures that the task-filtering does not degrade standard visual accuracy. |
| **Task Success Rate (TSR)** | Ratio of images where the selected object belongs to the task-relevant ground-truth set. | **> 85%** | Measures the effectiveness of the **Knowledge Distillation** and **Semantic Alignment**. |
| **Inference Latency ($t_{inf}$)** | End-to-end time from raw pixel input to final bounding box output on **VEGA RISC-V**. | **< 25ms** | Critical for **Assistive Tech** (SDG 4) to ensure real-time user feedback. |
| **Gating Sparsity ($\zeta$)** | Percentage of feature channels mathematically zeroed out by the **Dynamic Gating** layer. | **30 - 50%** | Validates the **CS Novelty** of runtime computational skipping. |
| **Power Efficiency ($\Delta W$)** | Total system power reduction compared to a non-gated, floating-point baseline. | **> 20%** | Key requirement for **Embedded/Rover** applications on the **Genesys-2 FPGA**. |

**Evaluation Environment:** All benchmarks are conducted under **INT8 Fixed-Point Quantization** to simulate real-world execution on the CDAC VEGA Processor.

---


<!-- Slide 9: Interdisciplinary Relevance -->
# 09. Interdisciplinary Relevance

**Computer Science:**
*   **Cross-Modal Alignment:** Mapping natural language to visual feature spaces.
*   **Knowledge Distillation:** Compressing semantic intelligence into lightweight backbones.
*   **NLP:** Tokenization and embedding generation for functional reasoning.

**Electronics & Communication:**
*   **Hardware-Software Co-Design:** Optimizing the **Hadamard Product** for RISC-V.
*   **Digital System Design:** Implementing low-latency **INT8 Quantization** on FPGA.
*   **ISA Optimization:** Tailoring bit-shift operations for gated sparsity

---

<!-- Slide 10: Conclusion -->
# 10. Conclusion & Impact

1.  **Innovation:** We pivot from "detecting all" to **"detecting what matters"** using a task-aware internal filter.
2.  **Intelligence:** Knowledge Distillation allows a **smaller sized model** to exhibit the semantic reasoning of a **2GB VLM**.
3.  **Hardware Native:** The **Gating Mechanism** is specifically designed for the Bit-Shift capabilities of the VEGA processor.
4.  **Application:** Ideal for **Embedded Robotics** and **Smart Surveillance** where power and task-relevance are paramount.
5. **SDG 4 Alignment**: Provides a scalable solution for Quality Education, ensuring lab and classroom tools are accessible to every learner, regardless of visual ability.
---

<!-- _class: lead -->
# Thank You