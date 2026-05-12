try:
    import albumentations as A

    def get_train_transform() -> A.Compose:
        """Augmentation pipeline for 5-band MS + nDSM patches (H × W × 5).

        Radiometric perturbations are kept mild so that spectral band ratios
        (NDVI, NDRE, …) remain approximately meaningful.  All radiometric
        transforms also affect the nDSM channel, which acts as implicit
        regularisation on height-map sensitivity.
        """
        return A.Compose([
            # ── Geometric ────────────────────────────────────────────────────
            # HorizontalFlip + VerticalFlip + RandomRotate90 + Transpose span
            # the full D4 dihedral group (all 8 symmetries of the square).
            # Aerial crown maps are genuinely invariant under all of these.
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            # Mild grid warp simulates orthorectification residuals and lens
            # radial distortion; both image and mask are warped jointly.
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),

            # ── Radiometric ──────────────────────────────────────────────────
            # Additive brightness / contrast shift (mild to preserve ratios).
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.4
            ),
            # Gamma shift simulates nonlinear sensor response / calibration drift.
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            # Per-channel multiplicative noise is more physically accurate for
            # CMOS/CCD sensors than purely additive noise; each band perturbed
            # independently to simulate inter-band gain variation.
            A.MultiplicativeNoise(multiplier=(0.92, 1.08), per_channel=True, p=0.3),
            # Additive Gaussian shot / read noise.
            A.GaussNoise(std_range=(0.005, 0.02), p=0.3),
            # Slight blur: defocus, atmospheric shimmer, or motion from wind.
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),

            # ── Occlusion ────────────────────────────────────────────────────
            # Rectangular zero-patches simulate cast shadows from adjacent tall
            # trees, transient cloud shadows, or sensor row dropout.
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(32, 96),
                hole_width_range=(32, 96),
                fill=0.0,
                p=0.2,
            ),
        ])

except ImportError:
    def get_train_transform():  # type: ignore[misc]
        return None
