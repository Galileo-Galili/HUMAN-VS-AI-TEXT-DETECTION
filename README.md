# **Official Repository of the Paper**: **_DeBERTa-Sentinel: An Explainable AI-Generated Text Detection Framework Using Disentangled Attention_**

Welcome to the official repository for the paper _"DeBERTa-Sentinel: An Explainable AI-Generated Text Detection Framework Using Disentangled Attention."_ This repository contains the datasets, models, and code used in our comprehensive study on AI-generated text detection.

## Repository Contents

### **Datasets**
- **GLC-AIText Dataset**: Our enhanced dataset comprising 28,057 paraphrased samples generated using multiple LLMs (**G**PT-3.5, **L**LaMA, **C**laude)
- **OpenWebText-Final**: Approximately 29,142 human-written samples from the cleaned OpenWebText corpus
- **Complete Dataset**: 58,537 total samples with 60/20/20 train/validation/test split (35,121 training, 11,708 validation, 11,708 testing)
- Complete dataset breakdown across 8 subsets (urlsf_00 to urlsf_06, urlsf_09)

### **Models**

**DeBERTa-Sentinel (Main Contribution)**:
- Enhanced detection framework leveraging DeBERTa-v3-small's disentangled attention mechanism
- End-to-end fine-tuned architecture with 256-token input sequences
- Achieves 97.53% detection accuracy with superior generalization capabilities
- Best model selected from epoch 2 based on validation performance (98.21% val accuracy)

**Baseline Models**:
- **RoBERTa-Sentinel**: Comparative baseline using RoBERTa encoder
- **Traditional ML Models**: TF-IDF + Logistic Regression, Random Forest, and other classical approaches

### **Repository Structure**

**Main Model Implementation**:
- `HvsAI_Deberta_Sentinel_.ipynb`: Complete DeBERTa-Sentinel implementation and training pipeline

**Dataset Files**:
- `OpenGPTText_cSV_format/`: Dataset in CSV format containing the training data
- `Custom_Test_Final.csv`: Custom test dataset for evaluation

**Explainability Analysis**:
- `lime_explanation_sample_1.html`: LIME explainability visualization (Sample 1)
- `lime_explanation_sample_2.html`: LIME explainability visualization (Sample 2) 
- `lime_explanation_sample_3.html`: LIME explainability visualization (Sample 3)
- `lime_explanation_sample_4.html`: LIME explainability visualization (Sample 4)
- `lime_explanation_sample_5.html`: LIME explainability visualization (Sample 5)

**Documentation**:
- `README.md`: This comprehensive documentation file
- `.git/`: Git version control directory

### **Key Features**

**Disentangled Attention Architecture**:
- Content-to-Content attention for semantic relationship analysis
- Content-to-Position attention for structural pattern detection  
- Position-to-Content attention for templating behavior identification

**Performance Metrics**:
- **Accuracy: 97.53%**
- **F1-Score: 97.58%** (0.976)
- **Precision: 95.89%**
- **Recall: 99.33%**
- **ROC-AUC: 99.53%** with near-perfect discrimination capability
- **Average Precision Score: 99.11%**
- **Optimal Threshold: 0.984** (Youden's J statistic) with 98.58% TPR and 2.28% FPR
- **Validation Accuracy: 98.21%** (epoch 2 - best model)

**Key Improvements**:
- 12.6% F1-score improvement over traditional ML baselines (TF-IDF + LogReg)
- 81.9% reduction in false negative rate compared to RoBERTa-Sentinel (0.67% vs 3.7%)
- Superior generalization on distribution-shifted data
- 2.23 percentage point accuracy improvement over RoBERTa-Sentinel

**Explainability Analysis**:
- Token-level contribution visualization using transformers-interpret pipeline
- Identification of formal transitional phrases and academic terminology as AI indicators
- Attention-based interpretability for understanding model decisions

## Results Summary

### Performance Comparison Across Models

| Model | Accuracy | F1-Score | Precision | Recall | AUC |
|-------|----------|----------|-----------|---------|-----|
| **DeBERTa-Sentinel** | **97.53%** | **0.976** | **95.89%** | **99.33%** | **99.53%** |
| RoBERTa-Sentinel | 95.3% | 0.953 | 94.5% | 96.3% | - |
| TF-IDF + LogReg | 86.33% | 0.867 | 84.64% | 88.82% | - |
| Random Baseline | 49.06% | 0.491 | 49.15% | 49.05% | - |

### ROC Curve Analysis

<div align="center">
  <img src="roc_curve_highres.png" alt="ROC Curve" width="600">
</div>

*Figure 1: ROC curve for DeBERTa-Sentinel showing exceptional discrimination capability with 99.53% AUC. The near-perfect curve hugging the top-left corner demonstrates superior ability to distinguish between AI and human text compared to random classification.*

### F1-Score Comparison

<div align="center">
  <img src="f1_comparison_highres.png" alt="F1 Score Comparison" width="600">
</div>

*Figure 2: F1-score comparison across baseline models showing DeBERTa-Sentinel's performance (0.976) relative to traditional ML approaches, demonstrating a 12.6% improvement over TF-IDF + LogReg.*

### Performance Gains Over RoBERTa-Sentinel

| Metric | RoBERTa-Sentinel | DeBERTa-Sentinel | Improvement |
|--------|------------------|------------------|-------------|
| Detection Accuracy | 95.3% | 97.53% | +2.23pp |
| Precision | 94.5% | 95.89% | +1.39pp |
| Recall | 96.3% | 99.33% | +3.03pp |
| F1-Score | 95.3% | 97.58% | +2.28pp |
| False Negative Rate | 3.7% | 0.67% | -3.03pp (81.9% reduction) |
| False Positive Rate | 5.7% | 4.28% | -1.42pp |
| ROC-AUC | -- | 99.53% | -- |

### Commercial Detector Comparison

Performance comparison with commercial detectors from prior literature:

| Model | F1 Score | Source | Dataset |
|-------|----------|--------|---------|
| ZeroGPT | 0.43 | Chen et al. (2023) | OpenGPTText-Final |
| OpenAI Classifier | 0.32 | Chen et al. (2023) | OpenGPTText-Final |
| GPTZero | 0.40-0.75* | Weber-Wulff et al. (2023) | Various ChatGPT text |
| **RoBERTa-Sentinel** | **0.953** | This work | GLC-AIText |
| **DeBERTa-Sentinel** | **0.977** | This work | GLC-AIText |

*Range reported across different evaluation conditions

**Key Advantages over Commercial Systems**:
- Substantial F1-score improvements (0.977 vs 0.32-0.43 for commercial detectors)
- Token-level explainability enabling interpretation of detection decisions
- Model customization for domain-specific applications
- Methodological transparency supporting reproducible research
- Adaptability to emerging LLMs

## Model Architecture and Training

### DeBERTa-Sentinel Architecture

<div align="center">
  <img src="RoBERTa%20VS%20DeBERTa.drawio%20(1).png" alt="DeBERTa-Sentinel Architecture" width="700">
</div>

*Figure 3: The DeBERTa-Sentinel architecture. The input sequence is embedded and processed through 12 layers of disentangled attention. The final [CLS] token representation is used for classification via the internal feedforward layer. The architecture is fully end-to-end fine-tuned with gradients backpropagating through all layers of the encoder.*

### Disentangled Attention Mechanism

The DeBERTa architecture enhances transformer models by disentangling content and position information in the self-attention mechanism:

**Traditional Attention**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Disentangled Attention Components**:
1. **Content-to-Content (C2C)**: Captures semantic relationships between tokens
2. **Content-to-Position (C2P)**: Models how content relates to positional structure
3. **Position-to-Content (P2C)**: Captures positional biases in content understanding

This decomposition enables more nuanced understanding of linguistic patterns characteristic of AI-generated text.

### Training Details
- **Architecture**: DeBERTa-v3-small (44M parameters)
- **Sequence Length**: 256 tokens
- **Optimizer**: AdamW (lr=2e-5, β1=0.9, β2=0.999, ε=1e-8)
- **Batch Size**: 64
- **Epochs**: 5 (best model from epoch 2)
- **Weight Decay**: 0.01
- **Gradient Clipping**: 1.0
- **Learning Rate Schedule**: Linear warmup + decay

## Dataset Details

### GLC-AIText Dataset Composition

**GLC-AIText** = **G**PT-3.5 + **L**LaMA + **C**laude AI-generated Text

| Subset | GPT Generated | LLaMA Generated | Claude Generated |
|--------|---------------|-----------------|------------------|
| Urlsf_00 | 1,386 | 1,300 | 1,212 |
| Urlsf_01 | 1,308 | 1,300 | 801 |
| Urlsf_02 | 874 | 1,300 | 922 |
| Urlsf_03 | 1,299 | 1,300 | 1,295 |
| Urlsf_04 | 1,156 | 1,300 | 946 |
| Urlsf_05 | 1,103 | 1,300 | 1,164 |
| Urlsf_06 | 1,031 | 1,300 | 723 |
| Urlsf_09 | 1,217 | 1,300 | 1,220 |
| **Total** | **9,374** | **10,400** | **8,283** |

### Complete Dataset Statistics
- **Total AI-Generated Samples**: 28,057
- **Total Human-Written Samples**: 29,142 (from OpenWebText-Final)
- **Complete Dataset**: 58,537 samples
- **Training Set**: 35,121 samples (60%)
- **Validation Set**: 11,708 samples (20%)
- **Test Set**: 11,708 samples (20%)
- **Positive Class Ratio (Test Set)**: 0.501 (balanced)

### Data Source
Human-written samples obtained from the OpenWebText corpus, a publicly available dataset comprising web content sourced from URLs shared on Reddit with a minimum of three upvotes. The corpus was compiled in 2019, ensuring content was not algorithmically generated. We utilized the cleaned OpenWebText-Final subset from GPT-Sentinel with preprocessing to remove formatting inconsistencies.

### Data Collection Method
AI-generated samples created by paraphrasing cleaned human-written samples using the prompt: "Rephrase the following paragraph by paragraph." Samples longer than 2,000 words were filtered due to model input limitations. Content blocked by safety filters was excluded, and outputs were filtered for fluency and coherence.

## Usage Instructions

### Running DeBERTa-Sentinel
1. **Main Implementation**: Open `HvsAI_Deberta_Sentinel_.ipynb` in Google Colab or Jupyter Notebook
2. **Traditional ML Models**: Extract and run `HuLLMI_Paper_ML_Traditional_Model.zip` 
3. **Dataset Loading**: Use the CSV files in `OpenGPTText_cSV_format/` for training data
4. **Custom Testing**: Evaluate using `Custom_Test_Final.csv` for performance assessment

### Viewing Explainability Results
Open any of the LIME explanation HTML files (`lime_explanation_sample_1.html` to `lime_explanation_sample_5.html`) in a web browser to view:
- Token-level contribution analysis
- Feature importance visualization  
- Model decision interpretability

## Key Contributions

1. **Novel Architecture**: Introduction of DeBERTa-v3's disentangled attention mechanism for AI text detection, separating content and positional information during self-attention computation

2. **Enhanced Dataset**: GLC-AIText dataset with 28,057 paraphrased samples from multiple LLMs (GPT-3.5, LLaMA, Claude) for improved generalization

3. **Superior Performance**: Consistent improvements over RoBERTa-Sentinel and traditional ML baselines:
   - 2.23 percentage point accuracy improvement (97.53% vs 95.3%)
   - 81.9% reduction in false negative rate (0.67% vs 3.7%)
   - 12.6% F1-score improvement over TF-IDF + LogReg

4. **Rigorous Methodology**: 60/20/20 train/validation/test split with validation-based model selection preventing overfitting

5. **Interpretability**: Token-level explainability revealing discriminative patterns including formal transitional phrases and academic terminology as AI-indicative features

6. **Comprehensive Evaluation**: Extensive comparison with commercial and academic baselines demonstrating substantial improvements (F1: 0.977 vs 0.32-0.43 for commercial detectors)

## Explainability Insights

### Token-Level Feature Importance

<div align="center">
  <img src="lime_average_feature_importance.png" alt="Feature Importance" width="600">
</div>

*Figure 5: Top 20 most important features averaged across samples showing DeBERTa-Sentinel's learned patterns. High-importance words like "system" (0.245), "background" (0.200), and "situation" (0.145) indicate formal language structures and contextual markers that the model associates with AI-generated content.*

### Example Text Analysis

<div align="center">
  <img src="expl_para.PNG" alt="Text Highlighting Example" width="700">
</div>

*Figure 6: Example of token-level explainability visualization showing how DeBERTa-Sentinel highlights specific words in a movie review. Orange highlighting indicates words that contribute toward AI classification, demonstrating the model's attention to formal transitional phrases like "Although" and structured language patterns.*

The model's decision-making process reveals:

- **Formal Language Detection**: Prioritization of formal vocabulary and structured language patterns (e.g., "system", "background")
- **Contextual Understanding**: Analysis of context rather than isolated words, identifying transitional phrases like "Although" in formal discourse
- **Stylistic Pattern Recognition**: Identification of characteristic LLM patterns such as formal conclusions ("demonstrates", "conclusion") and academic terminology
- **Balanced Analysis**: Use of both positive and negative feature contributions for robust classification

## Methodological Notes

### Model Selection and Hyperparameter Tuning
- Rigorous 60/20/20 train/validation/test split for principled model selection
- Best model selected based on validation performance (epoch 2: 98.21% val accuracy)
- Hyperparameter selection followed established best practices from GPT-Sentinel work
- Training for 5 epochs with monitoring of both training and validation metrics
- Validation-based checkpoint selection prevents overfitting to test set

### Training Progression
The model showed clear learning patterns:
- **Epoch 1**: 96.36% training accuracy
- **Epoch 2**: 98.21% validation accuracy (peak - best model selected)
- **Epoch 5**: 99.58% training accuracy (overfitting detected)

The gap between epoch 5 training accuracy (99.58%) and epoch 2 validation accuracy (98.21%) demonstrates effective overfitting prevention through validation-based model selection.

## Citation

If you use this work in your research, please cite:
```bibtex
@article{rehman2025deberta,
  title={DeBERTa-Sentinel: An Explainable AI-Generated Text Detection Framework Using Disentangled Attention},
  author={Rehman, Muhammad Yousaf and Islam, Muhammad and Hussain, Basharat},
  journal={IEEE Conference Proceedings},
  year={2025}
}
```

## Authors

**Muhammad Yousaf Rehman***  
SPECS, University of Hertfordshire, UK  
my.rehman007@gmail.com

**Muhammad Islam***  
College of Science and Engineering, James Cook University, Australia  
muhammad.islam1@jcu.edu.au

**Basharat Hussain**  
Department of Computer Science, NUCES, Pakistan  
basharat.hussian@nuces.edu.pk



## Abstract

The proliferation of large language models (LLMs) has created an urgent need for robust AI-generated text detection systems across domains including journalism, education, and legal applications. While transformer-based detectors like GPT-Sentinel have shown promise using RoBERTa encoders, they exhibit limited generalization across diverse model outputs and adversarial modifications. This study introduces DeBERTa-Sentinel, an enhanced detection framework architecture that leverages DeBERTa-v3's disentangled attention mechanism to improve upon existing approaches. Our proposed approach separates content and positional information during self-attention computation, enabling superior capture of subtle structural irregularities characteristic of synthetic text. We enhance training robustness by incorporating outputs from multiple LLMs (GPT-3.5, LLaMA, Claude) in our GLC-AIText dataset, comprising 28,057 paraphrased samples. Trained using a rigorous 60/20/20 train/validation/test split, the model achieved optimal performance at epoch 2 with 98.21% validation accuracy. Comprehensive evaluation demonstrates that DeBERTa-Sentinel achieves superior performance compared to RoBERTa-Sentinel: 97.53% test accuracy vs 95.3%, representing a 2.23 percentage point improvement. DeBERTa-Sentinel achieves 95.89% precision and 99.33% recall, compared to RoBERTa-Sentinel's 94.5% precision and 96.3% recall. The model exhibits exceptional discrimination capability with 99.53% ROC-AUC and 0.67% false negative rate. Explainability analysis reveals that DeBERTa-Sentinel effectively identifies formal transitional phrases and academic terminology as AI-indicative features. The model demonstrates 12.6% F1-score improvement over traditional machine learning baselines while providing token-level interpretability. These results validate disentangled attention as a promising architectural innovation for AI-generated content detection, with implications for forensic applications requiring high precision and interpretability.

---

**Repository Link**: https://github.com/Galileo-Galili/HUMAN-VS-AI-TEXT-DETECTION

**Note**: This repository contains the research implementation for the paper "DeBERTa-Sentinel: An Explainable AI-Generated Text Detection Framework Using Disentangled Attention." The paper is currently under review. For optimal visualization of LIME explainability analysis and attention mechanisms, please open the notebook files in Google Colab or Jupyter Notebook, and view the HTML explainability files in a web browser.