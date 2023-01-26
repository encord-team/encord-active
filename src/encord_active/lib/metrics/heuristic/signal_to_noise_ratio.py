import cv2
import numpy as np

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter


class SignalToNoiseRatio(Metric):
    TITLE = "Signal to Noise Ratio"
    METRIC_TYPE = MetricType.HEURISTIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = AnnotationType.NONE
    SHORT_DESCRIPTION = "Calculates signal-to-noise ratio of an image."
    LONG_DESCRIPTION = (
        "The [signal-to-noise ratio](https://en.wikipedia.org/wiki/Signal-to-noise_ratio_(imaging)) (**SNR**) of an "
        "image is a measure of the quality of the image, specifically, it compares the amount of useful information "
        '(the "signal") present in the image to the amount of unwanted variations or random noise present in the '
        "image. A higher SNR indicates that the image has a lower level of noise and a higher quality.  \n"
        "The SNR is usually calculated in decibels (dB) and the formula is:  \n"
        "$$SNR = 10 \\log_{10}{\\frac{Power \\: of \\: signal}{Power \\: of \\: noise}}$$  \n"
        "The power of the signal is typically represented by the mean of the image pixel values, while the power of "
        "the noise is typically represented by the standard deviation of the image pixel values.  \n"
        "**Note**: images can be affected by various types of noise such as Gaussian noise, Salt and pepper noise, "
        "Poisson noise, etc. Each type of noise might need to be handled differently. Additionally, different types of "
        "images might have different requirements in terms of SNR and a high SNR in one type of image might not be "
        "applicable to other types of images."
    )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        for _, img_pth in iterator.iterate(desc="Calculating signal-to-noise ratio"):
            if img_pth is None:
                continue
            try:
                image = cv2.imread(img_pth.as_posix())
            except:
                continue
            if image is None:
                continue

            # Compute the power of the signal (mean of the image array)
            signal_power = np.mean(image)

            # Compute the power of the noise (standard deviation of the image array)
            noise_power = np.std(image)

            # Calculate the signal-to-noise ratio
            snr = 10 * np.log10(signal_power / noise_power)

            writer.write(snr)
