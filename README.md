# Streamlined-Dataflow-FPGA-Architectures-for-RCs-in-Time-Series-Applications

It is the first quantized streamline Reservoir Computing (RC), in particular, Echo State Networks (ESNs). However, it can be easily extended to other variation of RC, which is supported by ReservoirPy.Two accelerator variants are introduced: one mapping neurons to FPGA DSP blocks and another relying exclusively on LUTs. We also propose an automated tool flow that trains and optimizes ESN models for a given dataset and generates the corresponding FPGA accelerators. The accelerators are evaluated on multiple time-series prediction and classification tasks.


## **Prerequisites**

- **Python 3.9+**

- **LLibraries**
  - **numpy**
  - **scipy**
  - **pandas**
  - **matplotlib**

- **Machine learning utilities**
  - **scikit-learn**

- **Reservoir Computing Framework**
  - **ReservoirPy**

- **Quantization & deep learning** 
  - **torch**
  - **brevitas**

- **Model export & inference** 
  - **onnx**
  - **onnxruntime**
