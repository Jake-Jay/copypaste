import torch
import torch.nn as nn
import torchvision.transforms as transforms

class FeatureExtractor(torch.nn.Module):
	"""CNN to extract features from images with

	CutPaste [1] used a resnet18 (not pretrained I think).
	"""
	def __init__(self):
		super().__init__()
		backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

		self.model = backbone
		self.model.fc = torch.nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)

	def extract_feature(self, x: torch.Tensor) -> torch.Tensor:
		"""Extract a feature
		Feature extracted is f normalised by the frobenius norm which is the 
		matrix equivalent of the Euclidean norm. For a matrix A, it is defined:
		.. math::
			\| A\|_F = \sqrt{tr(A^TA)}
		"""
		f = self.forward(x)
		return f / torch.norm(f, p='fro')


class ProjectionHead(nn.Module):
	"""Layer on top of the feature extractor 
	that predicts augmentations (binary output)

	CutPaste [1] uses an MLP on top of an average pooling 
	layer followed by a linear layer (which I think should 
	have only one output to predict)

	The final average pooling layer of resnet18 has shape (512,1)
	"""

	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(512, 100),
			nn.Linear(100, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)
	

class AnomolyRepresentationLearner(nn.Module):
	"""Groups feature extractor with projection head
	
	Parameters
	----------
	normalized : bool, optional
		Has the data already been normalised for imagenet, by default False
	"""

	def __init__(self, normalized=False):
		super().__init__()
		self.normalized = normalized
		self.model = nn.Sequential(
			FeatureExtractor(),
			ProjectionHead()
		)

		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

	def forward(self, x: torch.Tensor) -> torch.Tensor:

		if not self.normalized:
			x = self.normalize(x)
		return self.model(x)