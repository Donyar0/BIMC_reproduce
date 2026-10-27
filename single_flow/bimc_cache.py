"""Export existing BIMC statistics without running encoders or calibration."""
import torch


def from_task_statistics(stats, encoder_id, task_id):
    if type(task_id) is not int or task_id < 0:
        raise ValueError('task_id must be a nonnegative integer')
    ids = [int(c) for c in stats['class_index']]
    labels = stats['images_targets'].detach().cpu().long()
    # BIMC image rows follow torch.unique(labels), text rows follow class_index.
    # Reject mismatches instead of silently pairing different classes.
    if ids != sorted(set(ids)) or ids != torch.unique(labels).tolist():
        raise ValueError('BIMC class_index must match sorted image prototype labels')
    text = stats['text_proto'].detach().cpu().float().clone()
    visual = stats['image_proto'].detach().cpu().float().clone()
    if text.ndim != 2 or text.shape != visual.shape or len(text) != len(ids):
        raise ValueError('BIMC prototype shapes do not match class_index')
    return dict(stage='base' if task_id == 0 else 'incremental', split='train',
                encoder_id=encoder_id, class_ids=ids,
                text_prototypes=text, visual_prototypes=visual,
                image_features=stats['images_features'].detach().cpu().float().clone(),
                image_labels=labels.clone())
