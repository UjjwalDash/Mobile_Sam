from ultralytics import SAM
import matplotlib.pyplot as plt
import cv2
import numpy as np

class MobileSam:
    def __init__(self, image_path, model_path, points, labels):
        self.image_path = image_path  # Image path
        self.model_path = model_path  # Path of mobile sam model
        self.model = SAM(model_path)  # Load model
        self.points = points          # List of points eg. [(1000, 1200), (1200, 500)]
        self.labels = labels          # List of labels eg. [1, 1] we can take 0 to not consider that part
        self.ImageProcessor()         # Call ImageProcessor function

    def get_masks(self):
        results = self.model.predict(self.image_path, points=self.points, labels=self.labels, save=False)
        # Extract the masks
        mask_list = results[0].masks.data.numpy()
        return mask_list

    def get_masked_image(self, masks, orig_img):
        combined_img = orig_img.copy()
        for mask in masks:
            mask_resized = cv2.resize(mask.astype(np.uint8), (orig_img.shape[1], orig_img.shape[0]))
            color_mask = np.zeros_like(orig_img)
            color_mask[mask_resized == 1] = [0, 255, 0]  # Green color for mask
            combined_img = cv2.addWeighted(combined_img, 1.0, color_mask, 0.5, 0)
        return combined_img


    def plot_mask(self, combined_img):
        plt.figure(figsize=(15, 5))
        plt.title('Masked Image')
        plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def ImageProcessor(self):
        orig_img = cv2.imread(self.image_path)
        masks = self.get_masks()
        mask_img = self.get_masked_image(masks, orig_img)
        self.plot_mask(mask_img)

# Usage
if __name__ == "__main__":
    image_path = r"C:\Users\dashu\Desktop\neophyte\blushelesh\images\95a553b9-267d-4fd1-9fc1-b8fce08a6195.jpg"
    model_path = 'mobile_sam.pt'  # Replace with the actual model path
    points = [(294, 321), (125, 224), (454, 374), (290, 418),(500, 240), (650, 760), (810, 875), (715, 725), (742, 946)]
    labels = [1, 1, 1, 0, 0, 1, 1, 0, 0]
    
    mobilesam = MobileSam(image_path, model_path, points, labels)
