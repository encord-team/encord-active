from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

from encord_active.lib.model_predictions.reader import (
    LabelMatchSchema,
    LabelSchema,
    PredictionMatchSchema,
    PredictionSchema,
)


class GtMatchEntry(TypedDict):
    lidx: str
    pidxs: List[int]


GtMatchCollection = Dict[str, Dict[str, List[GtMatchEntry]]]
"""
Collection of lists of labels and what predictions of the same class they match with. 
First-level key is `class_id`, second-level is `img_id`
"""


class PerformanceMetricSchema(pa.SchemaModel):
    metric: Series[str] = pa.Field()
    value: Series[float] = pa.Field(coerce=True)
    class_name: Series[str] = pa.Field()


class PrecisionRecallSchema(pa.SchemaModel):
    precision: Series[float] = pa.Field()
    recall: Series[float] = pa.Field()
    class_name: Series[str] = pa.Field()


class ClassMapEntry(TypedDict):
    name: str
    featureHash: str
    color: str


ClassMapCollection = Dict[str, ClassMapEntry]
"""
Key is the `class_id`
"""


def compute_mAP_and_mAR(
    model_predictions: DataFrame[PredictionSchema],
    labels: DataFrame[LabelSchema],
    gt_matched: GtMatchCollection,
    class_map: Dict[str, ClassMapEntry],
    iou_threshold: float,
    rec_thresholds: Optional[np.ndarray] = None,
    ignore_unmatched_frames: bool = False,
) -> Tuple[
    DataFrame[PredictionMatchSchema],
    DataFrame[LabelMatchSchema],
    DataFrame[PerformanceMetricSchema],
    DataFrame[PrecisionRecallSchema],
]:
    """
    Computes a bunch of quantities used to display results in the UI. The main
    purpose of this function is to filter true positives from false positives
    by considering the IOU threshold and the order of the confidence scores.
    The IOU threshold is precomputed and available in `_predictions`, as each
    prediction can atmost match one label. As such, if an iou value is >0, it
    means that the prediction's best match (highest iou) has the given value.


    :param model_predictions: A df with the predictions ordered (DESC) by the
        models confidence scores.
    :param labels: The df with the labels. The labels are only used in
        this function to build a list of false negatives, as the matching
        between labels and predictions were already done during import.
    :param gt_matched: A dictionary with format::

            gt_matched[class_id: str][img_id: str] = \
                [{"lidx": lidx: int, "pidxs": [pidx1: int, pidx2, ... ]}]

        each entry in the `gt_matched` is thus a list of all the labels of a
        particular class (`class_id`) from a particular image (`img_id`).
        Each entry in the list contains the index to which it belongs in the
        `labels` data frame (`lidx`) and the indices of all the predictions
        that matches the given label best sorted by confidence score.
        That is, if `lidx == 512` and `pidx1 == 256`, it
        means that prediction 256 (`predictions.iloc[256]`) matched label
        512 (`labels.iloc[512]`) with the highest iou of all the prediction's
        matches. Furthermore, as `pidx1` comes before `pidx2`, it means that::

            predictions.iloc[pidx1]["confidence"] >= predictions.iloc[pidx2]["confidence"]

    :param class_map: This is a mapping between class indices and essential
        metadata. The dict has the structure::

            {
                "<idx>": {  #  <-- Note string type here, e.g., '"1"' <--
                    "name": "The name of the object class",
                    "featureHash": "e2f0be6c",
                    "color": "#001122",
                }
            }

    :param iou_threshold: The IOU threshold to compute scores for.
    :param rec_thresholds: The recall thresholds to compute the scores for.
        Default here is the same as `torchmetrics`.
    :param ignore_unmatched_labels: If set to true, will not normalize by
        all class labels, but only those associated with images for which
        there exist predictions.
    :return:
        - metrics_df: A df with AP_{class_name} and AR_{class_name} for every class
            name in the `_class_map`. This is used with `altair` to plot scores.
        - prec_df: Precision-recall data used with `altair` as well. Again grouped
            by `class_name`.
        - tps: An indicator array for which `tps[i] == True` if `_predictions.iloc[i]`
            was a true positive. Otherwise False.
        - reasons: A string for each entry in `_predictions` giving the reason for
            why this point was a false negative.
            (`reasons[i]` == ""` if `tps[i] == True`).
        - fns: An indicator array for which `fns[j] == True` if `_labels.iloc[j]`
            was not matched by any prediction.
    """
    model_predictions = model_predictions.copy()
    labels = labels.copy()

    rec_thresholds = rec_thresholds or np.linspace(0.0, 1.00, round(1.00 / 0.01) + 1)

    full_index_list = np.arange(model_predictions.shape[0])
    pred_class_list = model_predictions["class_id"].to_numpy(dtype=int)
    ious = model_predictions["iou"].to_numpy(dtype=float)

    # == Output containers == #
    # Stores the mapping between class_id and list_idx of the following two lists
    class_idx_map = {}
    precisions = []
    recalls = []

    _tps = np.zeros((model_predictions.shape[0],), dtype=bool)
    reasons = [f"No overlapping label of class `{class_map[str(i)]['name']}`." for i in pred_class_list]
    _fns = np.zeros((labels.shape[0],), dtype=bool)

    pred_img_ids = set(model_predictions["img_id"])
    label_include_ids = set(labels.index[labels["img_id"].isin(pred_img_ids)])

    cidx = 0
    for class_id in class_map:
        if ignore_unmatched_frames:
            nb_labels = sum(
                [len([t for t in l if t["lidx"] in label_include_ids]) for l in gt_matched.get(class_id, {}).values()]
            )
        else:
            nb_labels = sum([len(l) for l in gt_matched.get(class_id, {}).values()])

        if nb_labels == 0:
            continue

        class_idx_map[cidx] = class_id  # Keep track of the order of the output lists
        pred_select = pred_class_list == int(class_id)
        if pred_select.sum() == 0:
            precisions.append(np.zeros(rec_thresholds.shape))
            recalls.append(np.array(0.0))
            cidx += 1
            continue

        class_level_to_full_list_idx = full_index_list[pred_select]
        full_list_to_class_level_idx: Dict[int, int] = {v.item(): i for i, v in enumerate(class_level_to_full_list_idx)}

        _ious = ious[pred_select]
        TP_candidates = set(class_level_to_full_list_idx[_ious >= iou_threshold].astype(int).tolist())
        TP = np.zeros(_ious.shape[0])

        for img_label_matches in gt_matched[class_id].values():
            for label_match in img_label_matches:
                found_one = False
                for tp_idx in label_match["pidxs"]:
                    if found_one:
                        reasons[tp_idx] = "Prediction with higher confidence already matched label."
                    elif tp_idx in TP_candidates:
                        TP[full_list_to_class_level_idx[tp_idx]] = 1
                        _tps[tp_idx] = True
                        reasons[tp_idx] = ""
                        found_one = True
                    else:
                        reasons[tp_idx] = "IOU too low."

                if not found_one:
                    lidx = label_match["lidx"]
                    _fns[lidx] = True

        FP = 1 - TP

        TP_cumsum = np.cumsum(TP, axis=0)
        FP_cumsum = np.cumsum(FP, axis=0)
        rc = TP_cumsum / (nb_labels + np.finfo(np.float64).eps)
        pr = TP_cumsum / (FP_cumsum + TP_cumsum + np.finfo(np.float64).eps)
        prec = np.zeros((rec_thresholds.shape[0],))

        # Remove zigzags for AUC
        diff_zero = np.zeros((1,))
        diff = np.ones((1,))
        while not np.all(diff == 0):
            diff = np.concatenate([pr[1:] - pr[:-1], diff_zero], axis=0).clip(min=0)
            pr += diff

        inds = np.searchsorted(rc, rec_thresholds, side="left")
        num_inds = inds.argmax() if inds.max() >= TP.shape[0] else rec_thresholds.shape[0]
        inds = inds[:num_inds]  # type: ignore
        prec[:num_inds] = pr[inds]  # type: ignore

        precisions.append(prec)
        recalls.append(rc[-1])
        cidx += 1

    # Compute indices and compose data frames.
    _precisions = np.array(precisions, dtype=float)
    metrics = [
        ["mAP", np.mean(_precisions.mean(axis=1)).item(), "Mean"],
    ]
    metrics += [
        [f"AP_{class_map[class_id]['name']}", _precisions[cidx].mean().item(), class_map[class_id]["name"]]
        for cidx, class_id in class_idx_map.items()
    ]
    metrics += [["mAR", np.mean(recalls).item(), "Mean"]]
    metrics += [
        [f"AR_{class_map[class_id]['name']}", recalls[cidx].item(), class_map[class_id]["name"]]
        for cidx, class_id in class_idx_map.items()
    ]
    metrics_df = pd.DataFrame(metrics, columns=["metric", "value", "class_name"]).pipe(
        DataFrame[PerformanceMetricSchema]
    )

    prec_data = []
    columns = ["precision", "recall", "class_name"]
    for cidx, class_id in class_idx_map.items():
        for rc_idx, rc_threshold in enumerate(rec_thresholds):
            prec_data.append([_precisions[cidx, rc_idx], rc_threshold, class_map[class_id]["name"]])
    pr_df = pd.DataFrame(prec_data, columns=columns).pipe(DataFrame[PrecisionRecallSchema])

    model_predictions[PredictionMatchSchema.is_true_positive] = _tps.astype(float)
    model_predictions[PredictionMatchSchema.false_positive_reason] = reasons
    out_predictions = model_predictions.pipe(DataFrame[PredictionMatchSchema])

    labels[LabelMatchSchema.is_false_negative] = _fns
    out_labels = labels.pipe(DataFrame[LabelMatchSchema])

    return out_predictions, out_labels, metrics_df, pr_df
