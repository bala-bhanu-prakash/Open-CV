import cv2

class Cartoonizer:
    """Cartoonizer effect
    A class that applies a cartoon effect to an image using bilateral filtering and adaptive thresholding.
    """
    
    def __init__(self, downscale_steps=2, bilateral_filters=50):
        self.downscale_steps = downscale_steps
        self.bilateral_filters = bilateral_filters

    def render(self, img_path):
        # Read and resize image
        img_rgb = cv2.imread(img_path)
        img_rgb = cv2.resize(img_rgb, (1366, 768))
        
        # -- STEP 1 --
        # Downsample image using Gaussian pyramid
        img_color = img_rgb
        for _ in range(self.downscale_steps):
            img_color = cv2.pyrDown(img_color)
        
        # Apply small bilateral filter repeatedly
        for _ in range(self.bilateral_filters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        
        # Upsample image to original size
        for _ in range(self.downscale_steps):
            img_color = cv2.pyrUp(img_color)
        
        # -- STEPS 2 and 3 --
        # Convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)
        
        # -- STEP 4 --
        # Detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)
        
        # -- STEP 5 --
        # Convert edge image back to color
        (x, y, z) = img_color.shape
        img_edge = cv2.resize(img_edge, (y, x))
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        
        # Combine color and edge images
        cartoon_img = cv2.bitwise_and(img_color, img_edge)
        
        return cartoon_img

if __name__ == "__main__":
    # Initialize Cartoonizer with default settings
    cartoonizer = Cartoonizer()
    
    # File path to the input image (using forward slashes)
    file_name = ("              ")
    # Apply cartoon effect
    result = cartoonizer.render(file_name)
    
    # Save and display the result
    output_path = "Cartoon_version.jpg"
    cv2.imwrite(output_path, result)
    cv2.imshow("Cartoon version", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
