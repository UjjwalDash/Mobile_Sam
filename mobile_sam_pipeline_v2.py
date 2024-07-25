from ultralytics import SAM
import matplotlib.pyplot as plt
import cv2
import numpy as np

class MobileSam:
    def __init__(self, image_path, model_path):
        self.image_path = image_path  # Image path
        self.model_path = model_path  # Path of mobile sam model
        self.model = SAM(model_path)  # Load model
        #self.points = points          # List of points eg. [(1000, 1200), (1200, 500)]
        # self.boxes = boxes
        # self.labels = labels          # List of labels eg. [1, 1] we can take 0 to not consider that part
        # if segmentation == 0:
        #     self.ImageProcessor_points(points)         # Call ImageProcessor function

    def get_masks_points(self, points, labels):
        results = self.model.predict(self.image_path, points=points, labels=labels, save=False)
        # Extract the masks
        mask_list = results[0].masks.data.numpy()
        return mask_list

    def get_masks_boxes(self, boxes):
        results = self.model.predict(self.image_path, bboxes= boxes, save=False)
        # Extract the masks
        mask_list = results[0].masks.data.numpy()
        return mask_list

    def get_masked_image(self, masks, orig_img):
        combined_img = orig_img.copy()
        #x1, y1, x2, y2 = bbox
        #cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for mask in masks:
            mask_resized = cv2.resize(mask.astype(np.uint8), (orig_img.shape[1], orig_img.shape[0]))
            color_mask = np.zeros_like(orig_img)
            color_mask[mask_resized == 1] = [0, 255, 0]  # Green color for mask
            combined_img = cv2.addWeighted(combined_img, 1.0, color_mask, 0.5, 0)  
        return combined_img

    def plot_points(self, img, points):
        for point in points:
            x, y = point
            # print(point)
            # if label == 1:
            cv2.circle(img, (x,y), radius=10, color=(255, 0, 0), thickness=-1)
            # elif label == 0:
            #     cv2.circle(img, (x,y), radius=10, color=(0, 0, 255), thickness=-1)
        return img

    def draw_bboxs(self, img, bboxs):
        for bbox in bboxs:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img

    def generate_labels(self, points):
        lenth = len(points)
        print(lenth)
        labels = [1]*lenth
        return labels



    def plot_mask(self, combined_img):
        plt.figure(figsize=(15, 5))
        plt.title('Masked Image')
        plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def ImageProcessor_points(self, points):
        labels = self.generate_labels(points)
        orig_img = cv2.imread(self.image_path)
        masks = self.get_masks_points(points, labels)
        mask_img = self.get_masked_image(masks, orig_img)
        point_img = self.plot_points(mask_img, points)
        return point_img
        #self.plot_mask(point_img)

    def ImageProcessor_bbox(self, bboxs):
        orig_img = cv2.imread(self.image_path)
        masks = self.get_masks_boxes(bboxs)
        mask_img = self.get_masked_image(masks, orig_img)
        bbox_img = self.draw_bboxs(mask_img, bboxs)
        #self.plot_mask(bbox_img)
        return bbox_img

# Usage
if __name__ == "__main__":

    seg = int(input("0 for point and 1 for bbox: "))
    if seg == 0:
        #point segmentation
        image_path = r"C:\Users\dashu\Desktop\neophyte\blushelesh\images\95a553b9-267d-4fd1-9fc1-b8fce08a6195.jpg"
        model_path = 'mobile_sam.pt'  # Replace with the actual model path
        points = [(294, 321), (125, 224), (454, 374), (650, 760), (810, 875), (245, 1100), (180, 1180), (340, 1190)]
        #for i in range len(points)
        #labels = [1, 1, 1, 1, 1, 1, 1, 1]

        #(290, 418),(500, 240),(715, 725), (742, 946)
        
        mobilesam = MobileSam(image_path, model_path)
        img = mobilesam.ImageProcessor_points(points)
        mobilesam.plot_mask(img)
    
    elif seg == 1:

        #box segementation

        image_path = r"C:\Users\dashu\Desktop\neophyte\blushelesh\images\95a553b9-267d-4fd1-9fc1-b8fce08a6195.jpg"
        model_path = 'mobile_sam.pt'  # Replace with the actual model path
        bboxs = [[114, 714, 500, 1000], [530, 710, 930, 960], [480, 100, 900, 300]]
        
        mobilesam = MobileSam(image_path, model_path)
        img  = mobilesam.ImageProcessor_bbox(bboxs)
        mobilesam.plot_mask(img)

    else:
        print("Please Enter correct input !!!!!!")

