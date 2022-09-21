from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@st.experimental_memo
def compute_mAP_and_mAR(
    iou_threshold: float,
    rec_thresholds: Optional[np.ndarray] = None,
    ignore_unmatched_frames: bool = False,  # pylint: disable=unused-argument
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Computes a bunch of quantities used to display results in the UI. The main
    purpose of this function is to filter true positives from false positives
    by considering the IOU threshold and the order of the confidence scores.
    The IOU threshold is precomputed and available in `_predictions`, as each
    prediction can atmost match one label. As such, if an iou value is >0, it
    means that the prediction's best match (highest iou) has the given value.


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
    """
    :param _predictions: A df with the predictions ordered (DESC) by the
        models confidence scores.
    """
    _predictions = st.session_state.model_predictions
    """
    :param _labels: The df with the labels. The labels are only used in
        this function to build a list of false negatives, as the matching
        between labels and predictions were already done during import.
    """
    _labels = st.session_state.labels
    """
    :param _gt_matched: A dictionary with format::

            _gt_matched[(img_id: int, class_id: int)] = \
                [{"lidx": lidx: int, "pidxs": [pidx1: int, pidx2, ... ]}]

        each entry in the `_gt_matched` is thus a list of all the labels of a
        particular class (`class_id`) from a particular image (`img_id`).
        Each entry in the list contains the index to which it belongs in the
        `_labels` data frame (`lidx`) and the indices of all the predictions
        that matches the given label best sorted by confidence score.
        That is, if `lidx == 512` and `pidx1 == 256`, it
        means that prediction 256 (`_predictions.iloc[256]`) matched label
        512 (`_labels.iloc[512]`) with the highest iou of all the prediction's
        matches. Furthermore, as `pidx1` comes before `pidx2`, it means that::

            _predictions.iloc[pidx1]["confidence"] >= _predictions.iloc[pidx2]["confidence"]

    """
    _gt_matched = st.session_state.gt_matched
    """
    :param _class_map: This is a mapping between class indices and essential
        metadata. The dict has the structure::

            {
                "<idx>": {  #  <-- Note string type here <--
                    "name": "The name of the object class",
                    "heatureHash": "e2f0be6c",
                    "color": "#001122",
                }
            }
    """
    _class_map = st.session_state.full_class_idx
    rec_thresholds = rec_thresholds or np.linspace(0.0, 1.00, round(1.00 / 0.01) + 1)

    full_index_list = np.arange(_predictions.shape[0])
    pred_class_list = _predictions["class_id"].to_numpy(dtype=int)
    ious = _predictions["iou"].to_numpy(dtype=float)

    # == Output containers == #
    # Stores the mapping between class_idx and list_idx of the following two lists
    class_idx_map = {}
    precisions = []
    recalls = []

    _tps = np.zeros((_predictions.shape[0],), dtype=bool)
    reasons = [f"No overlapping label of class `{_class_map[str(i)]['name']}`." for i in pred_class_list]
    _fns = np.zeros((_labels.shape[0],), dtype=bool)

    pred_img_ids = set(_predictions["img_id"])
    label_include_ids = set(_labels.index[_labels["img_id"].isin(pred_img_ids)])

    cidx = 0
    for class_idx in _class_map:
        if ignore_unmatched_frames:
            # nb_labels = sum([len([c for c in l if c in pred_img_ids]) for l in _gt_matched[class_idx].values()])
            nb_labels = sum(
                [len([t for t in l if t["lidx"] in label_include_ids]) for l in _gt_matched.get(class_idx, {}).values()]
            )
        else:
            nb_labels = sum([len(l) for l in _gt_matched.get(class_idx, {}).values()])

        if nb_labels == 0:
            continue

        class_idx_map[cidx] = class_idx  # Keep track of the order of the output lists
        pred_select = pred_class_list == int(class_idx)
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

        for img_idx, img_label_matches in _gt_matched[class_idx].items():
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
        [f"AP_{_class_map[class_idx]['name']}", _precisions[cidx].mean().item(), _class_map[class_idx]["name"]]
        for cidx, class_idx in class_idx_map.items()
    ]
    metrics += [["mAR", np.mean(recalls).item(), "Mean"]]
    metrics += [
        [f"AR_{_class_map[class_idx]['name']}", recalls[cidx].item(), _class_map[class_idx]["name"]]
        for cidx, class_idx in class_idx_map.items()
    ]
    metrics_df = pd.DataFrame(metrics, columns=["metric", "value", "class_name"])

    prec_data = []
    columns = ["rc_threshold", "class_name", "precision"]
    for cidx, class_idx in class_idx_map.items():
        for rc_idx, rc_threshold in enumerate(rec_thresholds):
            prec_data.append([rc_threshold, _class_map[class_idx]["name"], _precisions[cidx, rc_idx]])

    prec_df = pd.DataFrame(prec_data, columns=columns)
    return metrics_df, prec_df, _tps, pd.DataFrame(reasons, columns=["fp_reason"]), _fns
