# **Official Repository of the Paper**: **_DeBERTa-Sentinel: Enhanced AI-Generated Text Detection Using Disentangled Attention_**

Welcome to the official repository for the paper _"DeBERTa-Sentinel: Enhanced AI-Generated Text Detection Using Disentangled Attention."_ This repository contains the datasets, models, and code used in our comprehensive study on AI-generated text detection.

## Repository Contents

### **Datasets**
- **MultiLLMText-Final Dataset**: Our enhanced dataset comprising 30,626 paraphrased samples generated using multiple LLMs (GPT-3.5, LLaMA, Claude)
- **OpenWebText-Final**: Human-written samples from the cleaned OpenWebText corpus
- Complete dataset breakdown across 8 subsets (urlsf_00 to urlsf_06, urlsf_09)

### **Models**

**DeBERTa-Sentinel (Main Contribution)**:
- Enhanced detection framework leveraging DeBERTa-v3's disentangled attention mechanism
- End-to-end fine-tuned architecture with 256-token input sequences
- Achieves 97.65% detection accuracy with superior generalization capabilities

**Baseline Models**:
- **RoBERTa-Sentinel**: Comparative baseline using RoBERTa encoder
- **Traditional ML Models**: TF-IDF + Logistic Regression, Random Forest, and other classical approaches

### **Repository Structure**

**Main Model Implementation**:
- `HuLLMI_Deberta_Sentinel_.ipynb`: Complete DeBERTa-Sentinel implementation and training pipeline
- `HuLLMI_Paper_ML_Traditional_Model.zip`: Traditional machine learning models (Naive Bayes, MLP, Random Forest, XGBoost)

**Dataset Files**:
- `OpenGPTText_cSV_format/`: Dataset in CSV format containing the training data
- `Custom_Test_Final.csv`: Custom test dataset for evaluation
- `Detailed Sample Mentioned in the Paper HuLL...`: Detailed sample data referenced in the paper

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

**Performance Improvements**:
- **Accuracy: 97.65%** (0.9765)
- **F1-Score: 97.69%** (0.9769)
- **Precision: 96.15%** (0.9615)
- **Recall: 99.28%** (0.9928)
- 99.79% ROC-AUC score with superior discrimination capability
- Significant improvements over traditional ML baselines

**Explainability Analysis**:
- Token-level contribution visualization
- Identification of formal transitional phrases and academic terminology as AI indicators
- Attention-based interpretability for understanding model decisions

## Results Summary

| Model | Accuracy | F1-Score | Precision | Recall | AUC |
|-------|----------|----------|-----------|---------|-----|
| **DeBERTa-Sentinel** | **97.65%** | **97.69%** | **96.15%** | **99.28%** | **99.79%** |
| TF-IDF + LogReg | 86.42% | 86.70% | 85.19% | 88.24% | - |
| Random Baseline | 49.06% | 49.10% | 49.15% | 49.05% | - |

### Commercial Detector Comparison
- **10-16% accuracy improvements** over ZeroGPT, GPTZero, and OpenAI Text Classifier
- Superior token-level interpretability compared to black-box commercial solutions
- Enhanced customization capabilities for domain-specific applications

## Usage Instructions

### Running DeBERTa-Sentinel
1. **Main Implementation**: Open `HuLLMI_Deberta_Sentinel_.ipynb` in Google Colab or Jupyter Notebook
2. **Traditional ML Models**: Extract and run `HuLLMI_Paper_ML_Traditional_Model.zip` 
3. **Dataset Loading**: Use the CSV files in `OpenGPTText_cSV_format/` for training data
4. **Custom Testing**: Evaluate using `Custom_Test_Final.csv` for performance assessment

### Viewing Explainability Results
Open any of the LIME explanation HTML files (`lime_explanation_sample_1.html` to `lime_explanation_sample_5.html`) in a web browser to view:
- Token-level contribution analysis
- Feature importance visualization  
- Model decision interpretability

## Dataset Access

The MultiLLMText-Final dataset contains:
- **GPT-3.5 Generated**: 9,374 samples
- **LLaMA Generated**: 10,400 samples  
- **Claude Generated**: 8,283 samples
- **Human-written**: Corresponding OpenWebText samples

## Key Contributions

1. **Novel Architecture**: Introduction of disentangled attention for AI text detection
2. **Enhanced Dataset**: Multi-LLM training corpus for improved generalization
3. **Superior Performance**: Consistent improvements across all evaluation metrics
4. **Interpretability**: Token-level explainability revealing discriminative patterns
5. **Comprehensive Evaluation**: Extensive comparison with commercial and academic baselines

## Citation

If you use this work in your research, please cite:

```bibtex
@article{rehman2025deberta,
  title={DeBERTa-Sentinel: Enhanced AI-Generated Text Detection Using Disentangled Attention},
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

*Equal contribution

## Abstract

The proliferation of large language models (LLMs) has created an urgent need for robust AI-generated text detection systems across domains including journalism, education, and legal applications. While transformer-based detectors like GPT-Sentinel have shown promise using RoBERTa encoders, they exhibit limited generalization across diverse model outputs and adversarial modifications. This study introduces DeBERTa-Sentinel, an enhanced detection framework that leverages DeBERTa-v3's disentangled attention mechanism to improve upon existing approaches. Our proposed approach separates content and positional information during self-attention computation, enabling superior capture of subtle structural irregularities characteristic of synthetic text. Comprehensive evaluation demonstrates that DeBERTa-Sentinel consistently outperforms RoBERTa-Sentinel across key metrics with superior generalization on distribution-shifted data. Explainability analysis reveals that DeBERTa-Sentinel effectively identifies formal transitional phrases and academic terminology as AI-indicative features, providing interpretable token-level predictions for forensic applications requiring high precision and interpretability.

---

**Repository Link**: https://github.com/Galileo-Galili/HUMAN-VS-AI-TEXT-DETECTION

**Note**: This repository contains the research implementation for the paper "DeBERTa-Sentinel: Enhanced AI-Generated Text Detection Using Disentangled Attention." The paper is currently under review. For optimal visualization of LIME explainability analysis and attention mechanisms, please open the notebook files in Google Colab or Jupyter Notebook, and view the HTML explainability files in a web browser.
