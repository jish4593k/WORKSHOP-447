import tkinter as tk
from tkinter import filedialog
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import seaborn as sns
import matplotlib.pyplot as plt


model = ResNet50(weights='imagenet')


def classify_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    
   
    img_array = torch.unsqueeze(img_array, 0)
    img_array = preprocess_input(img_array)

    
    predictions = model.predict(img_array)

    
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions[0]


def display_result(image_path, result_text):
    img = Image.open(image_path)
    img.thumbnail((300, 300))
    img = ImageTk.PhotoImage(img)

    panel = tk.Label(root, image=img)
    panel.image = img
    panel.grid(row=0, column=1, padx=10, pady=10)

    result_label.config(text=result_text)


def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        result = classify_image(file_path)
        display_result(file_path, result)

def display_seaborn_plot():
    data = [1, 2, 3, 4, 5]
    sns.lineplot(x=range(1, 6), y=data)
    plt.title('Seaborn Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

root = tk.Tk()
root.title("Image Classification App")


open_button = tk.Button(root, text="Open Image", command=open_file_dialog)
open_button.grid(row=1, column=0, padx=10, pady=10)

plot_button = tk.Button(root, text="Display Seaborn Plot", command=display_seaborn_plot)
plot_button.grid(row=2, column=0, padx=10, pady=10)


result_label = tk.Label(root, text="", font=("Helvetica", 12))
result_label.grid(row=1, column=1, padx=10, pady=10)

root.mainloop()
