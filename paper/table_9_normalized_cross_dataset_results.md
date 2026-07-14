| Dataset transfer           | Model               |   Number of features |   Accuracy |   Precision |   Recall |   F1-score |   ROC-AUC |   PR-AUC |
|:---------------------------|:--------------------|---------------------:|-----------:|------------:|---------:|-----------:|----------:|---------:|
| malicious_urls_to_phiusiil | Gradient Boosting   |                   64 |     0.6638 |      0.6423 |   0.4801 |     0.5495 |    0.6698 |   0.6153 |
| malicious_urls_to_phiusiil | Random Forest       |                   64 |     0.4863 |      0.4385 |   0.7233 |     0.546  |    0.5531 |   0.4677 |
| malicious_urls_to_phiusiil | Logistic Regression |                   64 |     0.366  |      0.3898 |   0.8566 |     0.5358 |    0.521  |   0.5026 |
| malicious_urls_to_phiusiil | Decision Tree       |                   64 |     0.4407 |      0.3972 |   0.5983 |     0.4775 |    0.5584 |   0.6    |
| phiusiil_to_malicious_urls | Random Forest       |                   64 |     0.2179 |      0.1706 |   0.8648 |     0.2849 |    0.4208 |   0.1565 |
| phiusiil_to_malicious_urls | Logistic Regression |                   64 |     0.2169 |      0.1702 |   0.8638 |     0.2844 |    0.3841 |   0.1474 |
| phiusiil_to_malicious_urls | Gradient Boosting   |                   64 |     0.2187 |      0.1699 |   0.8585 |     0.2837 |    0.3238 |   0.1299 |
| phiusiil_to_malicious_urls | Decision Tree       |                   64 |     0.2113 |      0.1632 |   0.8182 |     0.2721 |    0.4416 |   0.1634 |