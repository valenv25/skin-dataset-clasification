import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import io
import mlflow

# Función para loguear una figura matplotlib en TensorBoard
def plot_to_tensorboard(fig, writer, tag, step):
	buf = io.BytesIO()
	fig.savefig(buf, format='png')
	buf.seek(0)
	image = Image.open(buf).convert("RGB")
	image = np.array(image)
	image = torch.tensor(image).permute(2, 0, 1) / 255.0
	writer.add_image(tag, image, global_step=step)
	plt.close(fig)

# Función para matriz de confusión y clasificación
def log_classification_report(device, model, loader, writer, step, train_dataset, prefix="val"):
	model.eval()
	all_preds = []
	all_labels = []

	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device)
			outputs = model(images)
			_, preds = torch.max(outputs, 1)
			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(labels.numpy())

	# Confusion matrix
	cm = confusion_matrix(all_labels, all_preds)
	fig_cm, ax = plt.subplots(figsize=(6, 6))
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.label_encoder.classes_)
	disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
	ax.set_title(f'{prefix.title()} - Confusion Matrix')

	# Guardar localmente y subir a MLflow
	fig_path = f"confusion_matrix_{prefix}_epoch_{step}.png"
	fig_cm.savefig(fig_path)
	mlflow.log_artifact(fig_path)
	os.remove(fig_path)

	plot_to_tensorboard(fig_cm, writer, f"{prefix}/confusion_matrix", step)

	cls_report = classification_report(all_labels, all_preds, target_names=train_dataset.label_encoder.classes_, zero_division=0)
	writer.add_text(f"{prefix}/classification_report", f"<pre>{cls_report}</pre>", step)

	# También loguear texto del reporte
	with open(f"classification_report_{prefix}_epoch_{step}.txt", "w") as f:
		f.write(cls_report)
	mlflow.log_artifact(f.name)
	os.remove(f.name)

class CustomImageDataset(Dataset):
	def __init__(self, root_dir, transform=None):
		self.root_dir = root_dir
		self.transform = transform

		self.image_paths = []
		self.labels = []

		class_names = sorted(os.listdir(root_dir))
		self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

		for cls in class_names:
			cls_dir = os.path.join(root_dir, cls)
			for fname in os.listdir(cls_dir):
				if fname.lower().endswith((".png", ".jpg", ".jpeg")):
					self.image_paths.append(os.path.join(cls_dir, fname))
					self.labels.append(cls)

		self.label_encoder = LabelEncoder()
		self.labels = self.label_encoder.fit_transform(self.labels)

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
		label = self.labels[idx]

		if self.transform:
			augmented = self.transform(image=image)
			image = augmented["image"]

		return image, label