#!/usr/bin/env python3
"""
NDVI Map Generator
- Input: folder of NDVI images (timestamped filenames) and optional raw images/GPS CSV
- Output: stitched NDVI map image and a CSV log of used images
"""

import os
import cv2
import numpy as np
from PIL import Image
import imagehash
import csv
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

# ---------- CONFIG ----------
INPUT_DIR =  "/home/ay/vari/new"   # folder with NDVI images (timestamped)
OUTPUT_DIR = "/home/ay/RAW_map/"
OUTPUT_MAP_NAME = "ndvi_mosaic_{ts}.png"
LOG_CSV = os.path.join(OUTPUT_DIR, "maplog.csv")

# hashing thresholds and stitching params
HASH_THRESHOLD = 6         # lower -> stricter duplicate detection
MIN_MATCHES = 20           # min good matches for homography
RANSAC_REPROJ_THRESH = 4.0
PYRAMID_LEVELS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Helper: Load images and compute perceptual hashes ----------
def load_images(folder):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    imgs = []
    for f in files:
        path = os.path.join(folder, f)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: could not load {path}, skipping.")
            continue
        imgs.append({'path': path, 'name': f, 'img': img})
    return imgs

def compute_hash(pil_image):
    # average hash (fast)
    return imagehash.average_hash(pil_image)

def deduplicate_images(img_entries):
    kept = []
    hashes = []
    for e in img_entries:
        pil = Image.fromarray(cv2.cvtColor(e['img'], cv2.COLOR_BGR2RGB))
        h = compute_hash(pil)
        duplicate = False
        for (other_h, other_e) in hashes:
            if abs(h - other_h) <= HASH_THRESHOLD:
                # near-duplicate: we can pick the sharper image or the one with higher mean NDVI
                # compute sharpness heuristic (variance of Laplacian)
                lap_e = cv2.Laplacian(e['img'], cv2.CV_64F).var()
                lap_o = cv2.Laplacian(other_e['img'], cv2.CV_64F).var()
                if lap_e > lap_o:
                    # replace older with new
                    hashes.remove((other_h, other_e))
                    hashes.append((h, e))
                    kept = [x for x in kept if x['path'] != other_e['path']]
                    kept.append(e)
                duplicate = True
                break
        if not duplicate:
            hashes.append((h, e))
            kept.append(e)
    return kept

# ---------- Feature matching & homography ----------
def match_and_find_homography(img1_gray, img2_gray):
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(img1_gray, None)
    k2, d2 = orb.detectAndCompute(img2_gray, None)
    if d1 is None or d2 is None:
        return None, None, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    # ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < MIN_MATCHES:
        return None, None, len(good)
    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    return H, mask, len(good)

# ---------- Utility: compute canvas size if we place images with homographies ----------
def compute_canvas_bounds(base_shape, transforms, shapes):
    # transforms: list of 3x3 homographies mapping each image into base coordinate frame
    corners_all = []
    for H, shape in zip(transforms, shapes):
        h, w = shape[:2]
        corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(-1,1,2)
        warped = cv2.perspectiveTransform(corners, H)
        corners_all.append(warped.reshape(-1,2))
    all_points = np.vstack(corners_all)
    x_min, y_min = np.floor(all_points.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_points.max(axis=0)).astype(int)
    return x_min, y_min, x_max, y_max

# ---------- Pyramid blending for seamless overlap ----------
def pyramid_blend(img1, img2, mask, levels=PYRAMID_LEVELS):
    # Assumes img1 and img2 are same size and mask is single channel 0..1 float
    G1 = img1.copy().astype(np.float32)
    G2 = img2.copy().astype(np.float32)
    gpA = [G1]
    gpB = [G2]
    gpM = [mask.astype(np.float32)]
    for i in range(levels):
        G1 = cv2.pyrDown(G1)
        G2 = cv2.pyrDown(G2)
        mask = cv2.pyrDown(mask)
        gpA.append(G1)
        gpB.append(G2)
        gpM.append(mask)

    lpA = [gpA[-1]]
    lpB = [gpB[-1]]
    for i in range(levels,0,-1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i-1], cv2.resize(GE, (gpA[i-1].shape[1], gpA[i-1].shape[0])))
        lpA.append(L)
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i-1], cv2.resize(GE, (gpB[i-1].shape[1], gpB[i-1].shape[0])))
        lpB.append(L)

    # Blend pyramids
    LS = []
    for la, lb, m in zip(lpA, lpB, gpM[::-1]):
        # expand mask to 3 channels
        if len(m.shape) == 2:
            m3 = np.repeat(m[:,:,None],3,axis=2)
        else:
            m3 = m
        ls = la * m3 + lb * (1.0 - m3)
        LS.append(ls)

    # Reconstruct
    res = LS[0]
    for i in range(1, len(LS)):
        res = cv2.pyrUp(res)
        # resize if needed (numerical)
        if res.shape != LS[i].shape:
            res = cv2.resize(res, (LS[i].shape[1], LS[i].shape[0]))
        res = res + LS[i]
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res

# ---------- Main stitching pipeline ----------
def generate_mosaic(entries):
    # sort by filename (timestamp) so early images are base
    entries = sorted(entries, key=lambda e: e['name'])
    base = entries[0]
    base_img = base['img']
    h0, w0 = base_img.shape[:2]
    transforms = []
    images = []
    shapes = []

    # We'll compute transforms mapping each image into base coordinate frame
    transforms.append(np.eye(3))  # base identity
    images.append(base_img)
    shapes.append(base_img.shape)

    for e in entries[1:]:
        img = e['img']
        print(f"Matching {e['name']} to base...")
        H, mask, good = match_and_find_homography(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY),
                                                  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if H is None:
            print(f"  -> Not enough matches ({good}). Trying incremental matching to last added image.")
            # try matching to last stitched image instead
            last = images[-1]
            H2, mask2, good2 = match_and_find_homography(cv2.cvtColor(last, cv2.COLOR_BGR2GRAY),
                                                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if H2 is None:
                print(f"  -> Skipping {e['name']} (cannot align).")
                continue
            # combine transforms: new_H maps img->last; last is in base frame with transforms[-1]
            H_combined = transforms[-1].dot(H2)
            transforms.append(H_combined)
        else:
            transforms.append(H)

        images.append(img)
        shapes.append(img.shape)

    # compute canvas bounds
    x_min, y_min, x_max, y_max = compute_canvas_bounds(base_img.shape, transforms, shapes)
    offset_x = -x_min
    offset_y = -y_min
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    print(f"Canvas size: {canvas_w} x {canvas_h}, offset: ({offset_x},{offset_y})")

    # create blank canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    weight_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    # We'll place each image by warping
    for img, H in zip(images, transforms):
        h, w = img.shape[:2]
        # translate by offset
        offset_mat = np.array([[1,0,offset_x],[0,1,offset_y],[0,0,1]], dtype=np.float32)
        full_H = offset_mat.dot(H)
        warped = cv2.warpPerspective(img, full_H, (canvas_w, canvas_h))
        # mask where image has content
        mask = cv2.warpPerspective(np.ones((h,w), dtype=np.uint8), full_H, (canvas_w, canvas_h))
        mask_bool = mask.astype(bool)

        # blending: where canvas empty, just copy; else pyramid blend in overlap
        overlap = (weight_mask > 0) & mask_bool
        non_overlap = mask_bool & (~overlap)
        # assign non-overlap directly
        canvas[non_overlap] = warped[non_overlap]
        weight_mask[non_overlap] = 1.0

        if overlap.any():
            # create two images patch for overlapping region
            y_idxs, x_idxs = np.where(overlap)
            y0, y1 = y_idxs.min(), y_idxs.max()
            x0, x1 = x_idxs.min(), x_idxs.max()
            # crop region
            roi_canvas = canvas[y0:y1+1, x0:x1+1]
            roi_new = warped[y0:y1+1, x0:x1+1]
            roi_mask = np.zeros((y1-y0+1, x1-x0+1), dtype=np.float32)
            roi_mask[np.where(overlap[y0:y1+1, x0:x1+1])] = 1.0
            # To prefer NDVI values, we could compute pixel NDVI from each and choose mask bias.
            # For now perform pyramid blend
            blended = pyramid_blend(roi_canvas, roi_new, roi_mask, levels=4)
            canvas[y0:y1+1, x0:x1+1] = blended
            weight_mask[y0:y1+1, x0:x1+1] = 1.0

    return canvas

# ---------- Save final mosaic and log ----------
def save_mosaic(mosaic):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outname = OUTPUT_MAP_NAME.format(ts=ts)
    outpath = os.path.join(OUTPUT_DIR, outname)
    cv2.imwrite(outpath, mosaic)
    print("Saved final mosaic to:", outpath)
    return outpath

# ---------- Main ----------
if __name__ == "__main__":
    print("Loading NDVI images from:", INPUT_DIR)
    entries = load_images(INPUT_DIR)
    print(f"Found {len(entries)} images.")
    print("Deduplicating images (perceptual hash)...")
    entries_filtered = deduplicate_images(entries)
    print(f"After deduplication: {len(entries_filtered)} images will be used.")
    if len(entries_filtered) < 1:
        print("No images available to create a mosaic. Exiting.")
        exit(1)

    mosaic = generate_mosaic(entries_filtered)
    outpath = save_mosaic(mosaic)

    # write simple log
    with open(LOG_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['mosaic_path', 'timestamp', 'input_folder', 'num_used'])
        writer.writerow([outpath, datetime.now().isoformat(), INPUT_DIR, len(entries_filtered)])
    print("Log written to", LOG_CSV)
