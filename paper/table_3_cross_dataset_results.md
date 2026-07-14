| Training dataset   | Testing dataset   | Model               |   Accuracy |   Precision |   Recall |   F1-score |   ROC-AUC |   PR-AUC |
|:-------------------|:------------------|:--------------------|-----------:|------------:|---------:|-----------:|----------:|---------:|
| malicious_urls     | phiusiil          | Random Forest       |     0.4262 |      0.4266 |   0.998  |     0.5977 |    0.4648 |   0.4237 |
| malicious_urls     | phiusiil          | Gradient Boosting   |     0.4252 |      0.426  |   0.9956 |     0.5967 |    0.6049 |   0.7455 |
| malicious_urls     | phiusiil          | Logistic Regression |     0.4251 |      0.4259 |   0.9954 |     0.5966 |    0.0102 |   0.2537 |
| malicious_urls     | phiusiil          | Decision Tree       |     0.4178 |      0.4217 |   0.9782 |     0.5894 |    0.4891 |   0.4218 |
| phiusiil           | malicious_urls    | Logistic Regression |     0.1803 |      0.1802 |   0.9997 |     0.3053 |    0.3328 |   0.1349 |
| phiusiil           | malicious_urls    | Decision Tree       |     0.1801 |      0.1802 |   0.9998 |     0.3053 |    0.4999 |   0.1802 |
| phiusiil           | malicious_urls    | Gradient Boosting   |     0.1801 |      0.1802 |   0.9998 |     0.3053 |    0.3124 |   0.1377 |
| phiusiil           | malicious_urls    | Random Forest       |     0.1801 |      0.1802 |   0.9998 |     0.3053 |    0.4671 |   0.1707 |