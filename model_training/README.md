
# 🤟 SLR_LSF_Transfer_Learning_Training 🤟

## Overview
Welcome to the repository focused on Sign Language Recognition (SLR) for French Sign Language (Langue des Signes Française, LSF) 🇫🇷. Here, we explore how transfer learning can significantly enhance model performance, especially with limited data.

## Models Overview 🚀
- **Custom CNN Model:** Crafted from scratch, boasting an accuracy of 80%.
- **Pre-trained Models:**
  - **MobileNetV2:** Adapted from ImageNet with a stellar accuracy of 99%.
  - **VGG16:** Another gem from ImageNet, impressing with 95% accuracy.

## Dataset Details 📊
- **Sign Letters:** Concentrates on 21 static sign letters, excluding J, P, X, Y, Z.
- **Contributors:** Four signers have enriched our dataset.
- **Volume:** 50 vibrant images per sign from each signer, leading to 200 images per class and a grand total of 4,200 images.
- **Image Dimensions:**
  - **64x64 pixels:** Perfect for the CNN model and VGG16.
  - **224x224 pixels:** Specially sized for MobileNetV2.
You'll find the dataset [here](https://gitlab.com/lsf-slr-transfer-learning-project/slr_lsf_transfer_learning_dataset).

## Repository Structure 📁
Discover the model instances tucked away in the `static` folder.

## Exciting Side Project 🌟
Dive into our innovative application for real-time sign recognition, powered by our top-performing model, MobileNetV2. Experience it live [here](https://gitlab.com/lsf-slr-transfer-learning-project/slr_lsf_transfer_learning_app).
