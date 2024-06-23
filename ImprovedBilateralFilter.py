import cv2
import numpy as np


def bilateral_filter_result(image, kernel_size, starting_sigma_color, blur_metric):
    sigma_color = starting_sigma_color
    blur_level_algorithm = []
    used_sigmas = []
    blur_gaussian = []
    calibration_kernel_sizes = [size for size in range(3, min(kernel_size + 1, 20), 2)]

    # Gaussian filter
    for kernel_size_intermediate in calibration_kernel_sizes:
        blur_gaussian.append(
            BlurMetrics.measure_blur(cv2.blur(image, (kernel_size_intermediate, kernel_size_intermediate)), blur_metric))

    for i, kernel_size_intermediate in enumerate(calibration_kernel_sizes):
        sigma_space = (kernel_size_intermediate - 1) / 3
        if kernel_size_intermediate == 3:
            image_algorithm = cv2.bilateralFilter(image, kernel_size_intermediate, sigma_color, sigma_space)
            curr_blur_level = BlurMetrics.measure_blur(image_algorithm, blur_metric)
        elif kernel_size_intermediate < 19:  # calibrating
            image_algorithm = cv2.bilateralFilter(image, kernel_size_intermediate, sigma_color, sigma_space)
            curr_blur_level = BlurMetrics.measure_blur(image_algorithm, blur_metric)
            while BlurMetrics.compare(curr_blur_level, blur_level_algorithm[-1] + (blur_gaussian[i] - blur_gaussian[i-1]), blur_metric):
                sigma_color += 1
                image_algorithm = cv2.bilateralFilter(image, kernel_size_intermediate, sigma_color, sigma_space)
                curr_blur_level = BlurMetrics.measure_blur(image_algorithm, blur_metric)

        used_sigmas.append(sigma_color)
        blur_level_algorithm.append(curr_blur_level)
    sigma_space = (kernel_size_intermediate - 1) / 3
    slope, intercept = np.polyfit(calibration_kernel_sizes[0:len(used_sigmas)], used_sigmas, 1)
    sigma_color = slope * kernel_size + intercept
    return cv2.bilateralFilter(image, kernel_size, sigma_color, sigma_space)


class BlurMetrics:

    @staticmethod
    def measure_blur(image, method="gradient_magnitude"):
        """Public method to measure blur using different methods."""
        if method == "gradient_magnitude":
            return BlurMetrics.gradient_magnitude(image)
        elif method == "laplacian_variance":
            return BlurMetrics.laplacian_variance(image)
        elif method == "local_fourier_analysis":
            return BlurMetrics.local_fourier_analysis(image)
        elif method == "brenner_focus_measure":
            return BlurMetrics.brenner_focus_measure(image)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def laplacian_variance(image):
        """Measure blur using the variance of the Laplacian."""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        variance = laplacian.var()
        return variance

    @staticmethod
    def gradient_magnitude(image):
        """Measure blur using the gradient magnitude."""
        g_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        g_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(g_x ** 2 + g_y ** 2)
        mean_magnitude = gradient_magnitude.mean()
        return mean_magnitude

    @staticmethod
    def local_fourier_analysis(image, block_size=16):
        """Measure blur using local Fourier analysis."""
        height, width = image.shape
        num_blocks_y = height // block_size
        num_blocks_x = width // block_size

        blur_measurements = []

        for y in range(num_blocks_y):
            for x in range(num_blocks_x):
                block = image[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size]

                # Apply FFT to each block
                f = np.fft.fft2(block)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

                # We use the magnitude spectrum to quantify blur
                # More energy in lower frequencies indicates higher blur
                total_energy = np.sum(magnitude_spectrum)
                low_frequency_energy = np.sum(magnitude_spectrum[block_size // 2 - 4:block_size // 2 + 4,
                                              block_size // 2 - 4:block_size // 2 + 4])
                blur_level = low_frequency_energy / total_energy
                blur_measurements.append(blur_level)

        average_blur = np.mean(blur_measurements)
        return average_blur

    @staticmethod
    def brenner_focus_measure(image):
        # Calculate Brenner gradient
        dx = np.diff(image, axis=0)
        dy = np.diff(image, axis=1)
        gradient = np.square(dx[:, :-1]) + np.square(dy[:-1, :])

        return np.sum(gradient)

    @staticmethod
    def compare(curr_blur_level, last_blur_level, method):
        if method == "gradient_magnitude" or method == "laplacian_variance" or method == "brenner_focus_measure":
            return curr_blur_level > last_blur_level
        elif method == "local_fourier_analysis":
            return curr_blur_level < last_blur_level
        else:
            raise ValueError(f"Unknown method: {method}")
