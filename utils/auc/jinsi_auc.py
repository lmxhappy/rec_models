# coding: utf-8

def simple_auc(y_true, y_scores):
    sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    auc = 0
    prev_fpr, prev_tpr = 0, 0
    for i in sorted_indices:
        fpr = sum(1 for j in sorted_indices[i + 1:] if y_true[j] == 0) / sum(
            1 for j in sorted_indices if y_true[j] == 0)

        tpr = sum(1 for j in sorted_indices[i + 1:] if y_true[j] == 1) / sum(
            1 for j in sorted_indices if y_true[j] == 1)
        print(f'fpr:{fpr}, tpr:{tpr}')

        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2

        prev_fpr, prev_tpr = fpr, tpr

    return auc


y_true =[1, 0, 0, 0, 1, 0, 1, 0, ]
y_scores = [0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7]

auc = simple_auc(y_true, y_scores)
print('AUC:', auc)
