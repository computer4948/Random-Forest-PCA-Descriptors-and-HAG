#
#      0=============================0
#      |    TP3 Point Descriptors    |
#      0=============================0
#
#  Mini-challenge improved baseline (still compatible with TP code)
#

import numpy as np
import time

from os import listdir
from os.path import exists, join, splitext

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree
from sklearn.metrics import confusion_matrix

from ply import read_ply
from descriptors import compute_features


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def height_above_ground(points, cell_size=1.0):
    """
    Simple local ground estimate: min z in XY grid cells.
    Returns hag = z - min_z(cell), shape (N,).
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    x0 = x.min()
    y0 = y.min()
    ix = np.floor((x - x0) / cell_size).astype(np.int64)
    iy = np.floor((y - y0) / cell_size).astype(np.int64)

    # pack cell indices into 1 key
    key = (ix << 32) ^ (iy & 0xFFFFFFFF)

    order = np.argsort(key)
    key_s = key[order]
    z_s = z[order]

    uniq, first = np.unique(key_s, return_index=True)
    zmin = np.minimum.reduceat(z_s, first)

    idx = np.searchsorted(uniq, key)
    hag = z - zmin[idx]
    return hag.astype(np.float32)


def knn_kth_dist(points, query_points, k_list=(10, 30)):
    """
    Returns distances to the k-th neighbor for each k in k_list.
    Note: KDTree includes the point itself at distance 0 if query_points==points,
    so we query k+1 and take last distance.
    """
    kmax = max(k_list) + 1
    tree = KDTree(points, leaf_size=40)
    dists, _ = tree.query(query_points, k=kmax, return_distance=True)
    out = []
    for k in k_list:
        out.append(dists[:, k].astype(np.float32))  # kth neighbor (0 is itself)
    return out


def iou_from_cm(cm):
    """
    IoU per class from confusion matrix.
    """
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return tp / (tp + fp + fn + 1e-12)


# ------------------------------------------------------------
# Feature Extractor Class
# ------------------------------------------------------------

class FeaturesExtractor:
    """
    Computes features from point clouds
    """

    def __init__(self):
        # Multi-scale radii (meters)
        self.radii = [0.30, 0.80]          # << tweak here (2 scales = still reasonable runtime)
        self.base_radius = 0.50           # kept for compatibility / info

        # Height-above-ground grid size (meters)
        self.cell_size = 1.0

        # Density features: distances to k-th neighbors
        self.knn_k_list = [10, 30]

        # Number of training points per class (per cloud)
        self.num_per_class = 2000         # baseline was 500; this is still fast (features computed only on subset)

        # Labels (TP version uses 0..6)
        self.label_names = {
            0: 'Unclassified',
            1: 'Ground',
            2: 'Building',
            3: 'Poles',
            4: 'Pedestrians',
            5: 'Cars',
            6: 'Vegetation'
        }

        self.train_classes = [1, 2, 3, 4, 5, 6]   # ignore 0

    def _feature_matrix(self, query_points, cloud_points, hag_all=None):
        """
        Build feature matrix for query_points.
        Features:
          - z
          - height_above_ground (hag)
          - for each radius in self.radii: [verticality, linearity, planarity, sphericity]
          - density: dist_k10, dist_k30
        """
        # z
        z = query_points[:, 2].astype(np.float32).reshape(-1, 1)

        # height above ground
        if hag_all is None:
            hag = height_above_ground(cloud_points, cell_size=self.cell_size)
            # if query_points is a view of cloud_points, recompute indexing is hard -> fallback:
            # compute hag on query_points directly (approx ok)
            hag_q = height_above_ground(query_points, cell_size=self.cell_size).reshape(-1, 1)
        else:
            # hag_all corresponds to cloud_points order; query_points are subset of it (we handle with indices outside)
            hag_q = hag_all.reshape(-1, 1).astype(np.float32)

        # multi-scale PCA features
        feats_scales = []
        for r in self.radii:
            vert, line, plan, sphe = compute_features(query_points, cloud_points, r)
            f = np.vstack((vert.ravel(), line.ravel(), plan.ravel(), sphe.ravel())).T.astype(np.float32)
            feats_scales.append(f)

        # density features
        d_list = knn_kth_dist(cloud_points, query_points, k_list=self.knn_k_list)
        dens = np.vstack([d.ravel() for d in d_list]).T.astype(np.float32)  # (Nq, len(k_list))

        return np.hstack([z, hag_q] + feats_scales + [dens]).astype(np.float32)

    def extract_training_per_file(self, path, file):
        """
        Extract training subset (balanced per class) for ONE cloud.
        Returns: X, y
        """
        cloud_ply = read_ply(join(path, file))
        points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T.astype(np.float32)
        labels = cloud_ply['class'].astype(np.int32)

        # precompute HAG for all points once
        hag_all = height_above_ground(points, cell_size=self.cell_size)

        # balanced indices
        training_inds = np.empty(0, dtype=np.int32)
        for lab in self.train_classes:
            label_inds = np.where(labels == lab)[0]
            if label_inds.size == 0:
                continue
            take = min(self.num_per_class, label_inds.size)
            chosen = np.random.choice(label_inds, size=take, replace=False)
            training_inds = np.hstack((training_inds, chosen))

        training_points = points[training_inds, :]
        training_hag = hag_all[training_inds]

        X = self._feature_matrix(training_points, points, hag_all=training_hag)
        y = labels[training_inds]
        return X, y

    def extract_eval_subset_per_file(self, path, file, num_per_class_eval=4000):
        """
        Balanced evaluation subset for ONE cloud (for CV).
        """
        cloud_ply = read_ply(join(path, file))
        points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T.astype(np.float32)
        labels = cloud_ply['class'].astype(np.int32)

        hag_all = height_above_ground(points, cell_size=self.cell_size)

        eval_inds = np.empty(0, dtype=np.int32)
        for lab in self.train_classes:
            label_inds = np.where(labels == lab)[0]
            if label_inds.size == 0:
                continue
            take = min(num_per_class_eval, label_inds.size)
            chosen = np.random.choice(label_inds, size=take, replace=False)
            eval_inds = np.hstack((eval_inds, chosen))

        eval_points = points[eval_inds, :]
        eval_hag = hag_all[eval_inds]
        X = self._feature_matrix(eval_points, points, hag_all=eval_hag)
        y = labels[eval_inds]
        return X, y

    def extract_test(self, path):
        """
        Extract features for ALL test points (with caching).
        """
        ply_files = sorted([f for f in listdir(path) if f.endswith('.ply')])
        test_features = np.empty((0, 2 + 4 * len(self.radii) + len(self.knn_k_list)), dtype=np.float32)

        for file in ply_files:
            cloud_ply = read_ply(join(path, file))
            points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T.astype(np.float32)

            stem = splitext(file)[0]
            r_tag = "_".join([str(r).replace(".", "p") for r in self.radii])
            k_tag = "_".join([str(k) for k in self.knn_k_list])
            feature_file = join(path, f"{stem}_feat_r{r_tag}_cs{str(self.cell_size).replace('.', 'p')}_k{k_tag}.npy")

            if exists(feature_file):
                features = np.load(feature_file)
            else:
                hag = height_above_ground(points, cell_size=self.cell_size)
                # Here query_points == cloud_points, so pass hag as hag_all directly
                features = self._feature_matrix(points, points, hag_all=hag)
                np.save(feature_file, features)

            test_features = np.vstack((test_features, features))

        return test_features


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == '__main__':

    np.random.seed(0)

    training_path = '../data/MiniChallenge/training'
    test_path = '../data/MiniChallenge/test'

    # Toggle CV (useful for report)
    DO_CV = True

    f_extractor = FeaturesExtractor()

    # List training files
    train_files = sorted([f for f in listdir(training_path) if f.endswith('.ply')])
    if len(train_files) < 3:
        print("Warning: expected 3 training clouds (MiniLille1, MiniLille2, MiniParis1).")

    # -------------------------
    # Cross-validation (LOO)
    # -------------------------
    if DO_CV and len(train_files) >= 3:

        print("=== Leave-one-cloud-out CV (balanced eval subset) ===")
        class_list = f_extractor.train_classes  # 1..6

        ious = []
        for i, val_file in enumerate(train_files):
            tr_files = [f for f in train_files if f != val_file]

            # Train set = concat balanced subsets from train files
            X_tr_list, y_tr_list = [], []
            for f in tr_files:
                Xf, yf = f_extractor.extract_training_per_file(training_path, f)
                X_tr_list.append(Xf); y_tr_list.append(yf)
            X_tr = np.vstack(X_tr_list)
            y_tr = np.hstack(y_tr_list)

            # Val set = balanced eval subset from val file
            X_val, y_val = f_extractor.extract_eval_subset_per_file(training_path, val_file, num_per_class_eval=4000)

            clf = RandomForestClassifier(
                n_estimators=600,
                max_features='sqrt',
                min_samples_leaf=1,
                n_jobs=-1,
                class_weight='balanced_subsample',
                random_state=0
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)

            cm = confusion_matrix(y_val, y_pred, labels=class_list)
            iou = iou_from_cm(cm)
            miou = float(np.mean(iou))
            ious.append(iou)

            print(f"\nFold val = {val_file}")
            for c, v in zip(class_list, iou):
                print(f"  IoU class {c} ({f_extractor.label_names[c]}): {v:.4f}")
            print(f"  mIoU: {miou:.4f}")

        ious = np.vstack(ious)
        mean_iou = ious.mean(axis=0)
        print("\n=== CV mean over folds ===")
        for c, v in zip(class_list, mean_iou):
            print(f"  IoU class {c} ({f_extractor.label_names[c]}): {v:.4f}")
        print(f"  mIoU: {float(np.mean(mean_iou)):.4f}\n")

    # -------------------------
    # Train final model
    # -------------------------
    print('Collect Training Features (balanced subsets)')
    t0 = time.time()

    X_list, y_list = [], []
    for f in train_files:
        Xf, yf = f_extractor.extract_training_per_file(training_path, f)
        X_list.append(Xf); y_list.append(yf)

    training_features = np.vstack(X_list)
    training_labels = np.hstack(y_list)

    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    print('Training Random Forest')
    t0 = time.time()

    clf = RandomForestClassifier(
        n_estimators=900,
        max_features='sqrt',
        min_samples_leaf=1,
        n_jobs=-1,
        class_weight='balanced_subsample',
        random_state=0
    )
    clf.fit(training_features, training_labels)

    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    # -------------------------
    # Test prediction
    # -------------------------
    print('Compute testing features')
    t0 = time.time()

    test_features = f_extractor.extract_test(test_path)

    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    print('Predict labels')
    t0 = time.time()

    predictions = clf.predict(test_features).astype(np.int32)

    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    print('Save predictions -> MiniDijon9.txt')
    t0 = time.time()
    np.savetxt('MiniDijon9.txt', predictions, fmt='%d')
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))