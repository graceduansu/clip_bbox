import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

# TODO: turn these into function params
rel_peak_thr = 0.6
rel_rel_thr = 0.5
ioa_thr = 0.4
topk_boxes = 3


def heat2bbox(heat_map, original_image_shape):
    """Calculate bounding boxes on a 2D heatmap.

    Args:
        heat_map (numpy array): 2D heatmap of values
        original_image_shape (tuple[int]): original image's
        dimensions represented as (height, width)

    Returns:
        List[tuple[int]]: List of bounding boxes, where each bounding
        box is a tuple of coordinates (min_x, min_y, max_x, max_y)

    """
    h, w = heat_map.shape

    bounding_boxes = []

    heat_map = heat_map - np.min(heat_map)
    heat_map = heat_map / np.max(heat_map)

    bboxes = []
    box_scores = []

    peak_coords = peak_local_max(
        heat_map, exclude_border=False, threshold_rel=rel_peak_thr
    )  # find local peaks of heat map

    heat_resized = cv2.resize(
        heat_map, (original_image_shape[1], original_image_shape[0])
    )  # resize heat map to original image shape
    peak_coords_resized = ((peak_coords + 0.5) * np.asarray([original_image_shape]) / np.asarray([[h, w]])).astype(
        "int32"
    )

    for pk_coord in peak_coords_resized:
        pk_value = heat_resized[tuple(pk_coord)]
        mask = heat_resized > pk_value * rel_rel_thr
        labeled, n = ndi.label(mask)
        l_pks = labeled[tuple(pk_coord)]
        yy, xx = np.where(labeled == l_pks)
        min_x = np.min(xx)
        min_y = np.min(yy)
        max_x = np.max(xx)
        max_y = np.max(yy)
        bboxes.append((min_x, min_y, max_x, max_y))
        box_scores.append(pk_value)  # you can change to pk_value * probability of sentence matching image or etc.

    # Merging boxes that overlap too much
    box_idx = np.argsort(-np.asarray(box_scores))
    box_idx = box_idx[: min(topk_boxes, len(box_scores))]
    bboxes = [bboxes[i] for i in box_idx]
    box_scores = [box_scores[i] for i in box_idx]

    to_remove = []
    for iii in range(len(bboxes)):
        for iiii in range(iii):
            if iiii in to_remove:
                continue
            b1 = bboxes[iii]
            b2 = bboxes[iiii]
            isec = max(min(b1[2], b2[2]) - max(b1[0], b2[0]), 0) * max(min(b1[3], b2[3]) - max(b1[1], b2[1]), 0)
            ioa1 = isec / ((b1[2] - b1[0]) * (b1[3] - b1[1]))
            ioa2 = isec / ((b2[2] - b2[0]) * (b2[3] - b2[1]))
            if ioa1 > ioa_thr and ioa1 == ioa2:
                to_remove.append(iii)
            elif ioa1 > ioa_thr and ioa1 >= ioa2:
                to_remove.append(iii)
            elif ioa2 > ioa_thr and ioa2 >= ioa1:
                to_remove.append(iiii)

    for i in range(len(bboxes)):
        if i not in to_remove:
            bounding_boxes.append(
                {
                    "score": box_scores[i],
                    "bbox": bboxes[i],
                    "bbox_normalized": np.asarray(
                        [
                            bboxes[i][0] / heat_resized.shape[1],
                            bboxes[i][1] / heat_resized.shape[0],
                            bboxes[i][2] / heat_resized.shape[1],
                            bboxes[i][3] / heat_resized.shape[0],
                        ]
                    ),
                }
            )

    return bounding_boxes


def img_heat_bbox_disp(
    image,
    heat_map,
    save_path,
    title="",
    en_name="",
    alpha=0.6,
    cmap="viridis",
    cbar="False",
    dot_max=False,
    bboxes=[],
    order=None,
):
    """Draw bounding boxes on image and overlay the
    corresponding heatmap. Saves result to save_path.

    Args:
        image (numpy array): Image stored in RGB format
        heat_map (numpy array): 2D heatmap of values
        save_path (str): Save path for generated figure
        title (str): Title on generated figure
        en_name (str): Text inside figure
        alpha (float): Transparency of heatmap
        cmap (str): Matplotlib color map parameter applied to heatmap
        cbar (bool): Show or hide color bar for heatmap
        dot_max (bool): Show or hide dots where the heatmap reaches max values
        bboxes (List[List[int]]): List of bounding boxes
        order (str): The order of coordinates used for bboxes

    Returns:
        Matplotlib figure

    """

    # np.save("rocket.npy", image)
    H, W = image.shape[0:2]
    # resize heat map
    heat_map_resized = cv2.resize(heat_map, (W, H))
    # display
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(title, size=15)
    ax = plt.subplot(1, 3, 1)
    plt.imshow(image)
    if dot_max:
        max_loc = np.unravel_index(np.argmax(heat_map_resized, axis=None), heat_map_resized.shape)
        plt.scatter(x=max_loc[1], y=max_loc[0], edgecolor="w", linewidth=3)

    if len(bboxes) > 0:  # it gets normalized bbox
        if order is None:
            order = "xxyy"

        for i in range(len(bboxes)):
            bbox_norm = bboxes[i]["bbox_normalized"]
            if order == "xxyy":
                x_min, x_max, y_min, y_max = (
                    int(bbox_norm[0] * W),
                    int(bbox_norm[1] * W),
                    int(bbox_norm[2] * H),
                    int(bbox_norm[3] * H),
                )
            elif order == "xyxy":
                x_min, x_max, y_min, y_max = (
                    int(bbox_norm[0] * W),
                    int(bbox_norm[2] * W),
                    int(bbox_norm[1] * H),
                    int(bbox_norm[3] * H),
                )
            x_length, y_length = x_max - x_min, y_max - y_min

            box = plt.Rectangle(
                (x_min, y_min),
                x_length,
                y_length,
                edgecolor="r",
                linewidth=3,
                fill=False,
            )
            plt.gca().add_patch(box)
            if en_name != "":
                ax.text(
                    x_min + 0.5 * x_length,
                    y_min + 10,
                    en_name,
                    verticalalignment="center",
                    horizontalalignment="center",
                    # transform=ax.transAxes,
                    color="white",
                    fontsize=15,
                )
                # an = ax.annotate(en_name, xy=(x_min,y_min), xycoords="data", va="center", ha="center",
                #   bbox=dict(boxstyle="round", fc="w"))
                # plt.gca().add_patch(an)

    plt.imshow(heat_map_resized, alpha=alpha, cmap=cmap)

    # plt.figure(2, figsize=(6, 6))
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    # plt.figure(3, figsize=(6, 6))
    plt.subplot(1, 3, 3)
    plt.imshow(heat_map_resized)
    plt.colorbar()
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    plt.savefig(save_path)

    return fig
