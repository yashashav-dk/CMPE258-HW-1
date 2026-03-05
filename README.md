# CMPE 258 - HW1

Four PyTorch tasks (`pytorch_task_v1` protocol). Each one trains, evaluates, and exits 0 on success.

- `cnn_fashionmnist_adamw` — CNN on FashionMNIST, AdamW + cosine LR
- `ae_mnist_denoising` — denoising autoencoder on MNIST
- `mlp_newsgroups_tfidf` — text classification w/ TF-IDF on 20 Newsgroups
- `mlp_housing_earlystop` — regression on California Housing w/ early stopping

```
pip install torch torchvision numpy matplotlib scikit-learn
python tasks/<task_id>/task.py
```
