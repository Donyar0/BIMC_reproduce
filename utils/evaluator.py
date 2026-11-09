import torch
import numpy as np
from sklearn.metrics import confusion_matrix

class AccuracyEvaluator:
    
    def __init__(self, class_index_per_task):
        self.class_index_per_task = class_index_per_task
        self.num_tasks = len(class_index_per_task)


    def confusion_matrix(self, logits, targets, task_id, normalize=False):
        class_conf_matrix = self._class_wise_confusion_matrix(logits, targets, task_id, normalize)
        task_conf_matrix = self._task_wise_confusion_matrix(logits, targets, task_id, normalize)

        return {'class_conf_matrix': class_conf_matrix,
                'task_conf_matrix': task_conf_matrix}


    def calc_accuracy(self, logits, targets, task_id):
        logits = logits.cpu().numpy()
        targets = targets.cpu().numpy()

        overall_right_cnt = self._count_right_pred_num(logits, targets)
        overall_acc_mean = overall_right_cnt / len(targets)

        seen_task_classes = self.class_index_per_task[:task_id + 1]
        task_accuracies = []
        for classes in seen_task_classes:
            task_sample_indices = np.where(np.isin(targets, classes))[0]
            if len(task_sample_indices) == 0:
                task_accuracies.append(0.0)
                continue

            task_sample_logits = logits[task_sample_indices]
            task_sample_targets = targets[task_sample_indices]
            task_right_cnt = self._count_right_pred_num(task_sample_logits, task_sample_targets)

            task_acc_mean = task_right_cnt / len(task_sample_indices)
            task_accuracies.append(round(100 * task_acc_mean, 2))

        base_avg_acc = task_accuracies[0]
        inc_avg_acc = sum(task_accuracies[1:]) / (len(task_accuracies) - 1) if len(task_accuracies) > 1 else 0.0
        harmonic_acc = 2 * base_avg_acc * inc_avg_acc / (base_avg_acc + inc_avg_acc) if inc_avg_acc > 0 else 0.0
        return {'mean_acc': round(100 * overall_acc_mean, 2), 
                'task_acc': task_accuracies,
                'harmonic_acc': round(harmonic_acc, 2),
                'base_avg_acc': round(base_avg_acc, 2),
                'inc_avg_acc': round(inc_avg_acc, 2)}


    def calc_base_novel_bias(self, logits, targets, task_id):
        """Class-group confusion rates (%) without changing predictions.

        Novel -> Base: P(predicted base | true novel)
        Base -> Novel: P(predicted novel | true base)
        Also report within-group and cross-group mistake counts.
        At session zero, novel-related rates are None (not zero).
        """
        base_ids = np.asarray(self.class_index_per_task[0], dtype=np.int64)
        if isinstance(logits, torch.Tensor):
            predictions = logits.argmax(dim=1).detach().cpu().numpy()
        else:
            predictions = np.asarray(logits).argmax(axis=1)
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        targets = np.asarray(targets)
        if targets.shape[0] != predictions.shape[0]:
            raise ValueError("Predictions and targets must have equal length")
        base_true = np.isin(targets, base_ids)
        base_pred = np.isin(predictions, base_ids)
        novel_true = ~base_true
        base_n = int(base_true.sum())
        novel_n = int(novel_true.sum())
        base_to_novel = int((base_true & ~base_pred).sum())
        novel_to_base = int((novel_true & base_pred).sum())
        base_wrong_within = int((base_true & base_pred & (targets != predictions)).sum())
        novel_wrong_within = int((novel_true & ~base_pred & (targets != predictions)).sum())
        return {
            "num_base_test": base_n,
            "num_novel_test": novel_n,
            "base_to_novel_count": base_to_novel,
            "novel_to_base_count": novel_to_base,
            "base_to_novel_pct": round(100.0 * base_to_novel / base_n, 4) if base_n else None,
            "novel_to_base_pct": round(100.0 * novel_to_base / novel_n, 4) if novel_n else None,
            "base_wrong_within_count": base_wrong_within,
            "novel_wrong_within_count": novel_wrong_within,
        }


    def _count_right_pred_num(self, logits, targets):
        pred = np.argmax(logits, axis=1)
        return np.sum(pred == targets)


    def _determine_tasks(self, samples, task_classes):
        tasks = np.zeros_like(samples)
        for task_id, classes in enumerate(task_classes):
            class_mask = np.isin(samples, classes)
            tasks[class_mask] = task_id
        return tasks
    


    def _task_wise_confusion_matrix(self, logits, targets, task_id, normalize=False):
        logits_np = logits.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        seen_task_classes = [cls for cls in self.class_index_per_task[:task_id + 1]]
        actual_tasks = self._determine_tasks(targets_np, seen_task_classes)
        predicted_tasks = self._determine_tasks(np.argmax(logits_np, axis=1), seen_task_classes)

        task_conf_matrix = confusion_matrix(actual_tasks, predicted_tasks, labels=range(len(seen_task_classes)))

        if normalize:
            task_conf_matrix = task_conf_matrix.astype('float')
            row_sums = task_conf_matrix.sum(axis=1, keepdims=True)
            task_conf_matrix /= row_sums

        return task_conf_matrix

    def _class_wise_confusion_matrix(self, logits, targets, task_id, normalize=False):
        logits_np = logits.cpu().numpy()
        targets_np = targets.cpu().numpy()

        seen_classes = np.concatenate([cls for cls in self.class_index_per_task[:task_id + 1]])
        unique_seen_classes = np.unique(seen_classes)
        valid_indices = np.isin(targets_np, unique_seen_classes)
        valid_logits = logits_np[valid_indices]
        valid_targets = targets_np[valid_indices]

        preds = np.argmax(valid_logits, axis=1)
        conf_matrix = confusion_matrix(valid_targets, preds, labels=unique_seen_classes)

        if normalize:
            conf_matrix = conf_matrix.astype('float')
            row_sums = conf_matrix.sum(axis=1, keepdims=True)
            conf_matrix /= row_sums

        return conf_matrix


    def task_class_confusion_matrix(self, class_labels, true_task_labels, logits):
        """
        Compute the task-class confusion matrix.

        Args:
        - class_labels (torch.Tensor): Tensor of ground truth class labels for each sample.
        - true_task_labels (torch.Tensor): Tensor of ground truth task labels for each sample.
        - logits (torch.Tensor): The logits output from the model for each sample.
        
        Returns:
        - np.array: A confusion matrix of shape (num_classes, num_tasks)
        """
        if isinstance(class_labels, torch.Tensor):
            class_labels = class_labels.cpu().numpy()
        if isinstance(true_task_labels, torch.Tensor):
            true_task_labels = true_task_labels.cpu().numpy()
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        
        predicted_task_labels = np.argmax(logits, axis=1)
        
        unique_classes = np.unique(class_labels)
        unique_tasks = np.arange(10)
        
        confusion_mat = np.zeros((len(unique_classes), len(unique_tasks)))
        
        for i, cls in enumerate(unique_classes):
            for j, task in enumerate(unique_tasks):
                idx = np.where((class_labels == cls) & (predicted_task_labels == task))[0]
                task_correct = np.sum(predicted_task_labels[idx] == task)
                confusion_mat[i, j] = task_correct
        
        return confusion_mat