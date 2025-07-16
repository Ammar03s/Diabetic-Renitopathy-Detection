# Diabetic Retinopathy Detection using Deep Learning

## Problem Statement

Diabetic retinopathy (DR) is one of the leading causes of vision loss worldwide, affecting over 25% of individuals with diabetes. With +300 million people diagnosed with diabetes globally, early detection and can be crucial for tratement and possiblt preventing vision loss.

Currently, DR diagnosis relies on manual evaluation of retinal fundus images by trained medical professionals, which is:
- Time-consuming and difficult to scale, especially in developing countries with limited medical expertise

This project aims to study 3 different approaches for DR severity classification using CNN to improve screening efficiency and coverage.

## Datasets

Two datasets were used in this study:

### 1. 2015 Kaggle Dataset (Pre-training)
- **Size**: ~35,000 retinal fundus images
- **Labels**: 5-class scale (0: No DR → 4: Proliferative DR)
- **Source**: [Diabetic Retinopathy Detection Competition](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized)
- **Purpose**: Initial model training

### 2. 2019 APTOS Dataset (Fine-tuning)
- **Size**: 3,662 higher-quality retinal images  
- **Labels**: Same 5-class scale as 2015 dataset
- **Source**: [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
- **Purpose**: Fine-tuning pre-trained models

**Challenge**: Both of the datasets exhibit significant class imbalance with most images labeled as "No DR" (normal in the real-world)

## Methodology

### Transfer Learning Approach
All models used **transfer learning** with pre-trained weights, then adapted for medical imaging:

1. **Stage 1**: Train on 2015 dataset (large, diverse)
2. **Stage 2**: Fine-tune on 2019 dataset (smaller, higher quality)

### Model Configurations Tested
3 different configurations were implemented and compared:

1. **EfficientNet-B7 + Cross-Entropy Loss**
2. **EfficientNet-B7 + Focal Loss** 
3. **ResNet50 + Focal Loss**

### Preprocessing & Augmentation
- Images resized to 512×512 pixels
- Ben Graham's preprocessing for contrast/color enhancement
- Data augmentation: random cropping, zooming, horizontal flipping, brightness adjustments

## Code Structure

The experiments are implemented in three Jupyter notebooks:

- `DR-CrossEntropyLoss-EfficieNet7B.ipynb` → EfficientNet-B7 + Cross-Entropy Loss
- `DR-FocalLoss-EffecieNet7B.ipynb` → EfficientNet-B7 + Focal Loss  
- `DR-FocalLoss-ResNet50.ipynb` → ResNet50 + Focal Loss

## Results

### Key Findings

**6 total experiments** were conducted (3 model configurations × 2 training stages):

1. **Transfer Learning Impact**: Fine-tuned models on 2019 dataset consistently outperformed models trained only on 2015 data
2. **Best Performing Model**: EfficientNet-B7 with Cross-Entropy Loss achieved the highest accuracy
3. **Focal Loss Performance**: Contrary to expectations for imbalanced datasets, Focal Loss did not outperform Cross-Entropy Loss, likely due to training instability with large architectures

### Performance Comparison

#### 2015 Dataset Training vs 2019 Fine-tuning
All three model configurations showed significant improvement when fine-tuned on the cleaner 2019 dataset compared to training only on the 2015 dataset.

#### Final Model Rankings (2019 Fine-tuned)
1. **EfficientNet-B7 + Cross-Entropy Loss** (Best)
2. EfficientNet-B7 + Focal Loss  
3. ResNet50 + Focal Loss

### Results Visualization
Detailed performance metrics, confusion matrices, and training curves are available in the `Code-Results/` directory:
- `CrossEffecient-results/` - EfficientNet-B7 + Cross-Entropy results
- `FocalEffecient-results/` - EfficientNet-B7 + Focal Loss results  
- `FocalResNet-results/` - ResNet50 + Focal Loss results
- `General-Results/` - Dataset visualizations and comparative analysis

## Conclusion

The study demonstrates that:
- Transfer learning significantly improves model performance for medical imaging tasks
- EfficientNet-B7 with Cross-Entropy Loss provides the most stable and accurate results for DR classification
- Pre-training on larger, diverse datasets followed by fine-tuning on high-quality clinical data is an effective strategy for automated DR detection 
