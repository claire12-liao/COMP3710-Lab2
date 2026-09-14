# Sources and implementation notes

Primary assessment sources: the two PDFs supplied by the student (Lab 2 v2.01 and Demo rubric v2.0). The student-supplied MRI archive is used without changing its train/validation/test subject membership.

| Topic | Source | Use in this package |
|---|---|---|
| Lab instructions | COMP3710_Lab_2_2026_v2.01.pdf, pp.2–12 | Tasks, datasets, architecture constraints and score thresholds |
| Demo / AI / ownership | COMP3710_Demo_rubric_v2_final (1).pdf | Rubric, presentation and AI evidence requirements |
| PyTorch AMP | https://docs.pytorch.org/docs/stable/amp.html | autocast and GradScaler API design |
| LFW loader | https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html | min_faces_per_person=70, resize=0.4 |
| CIFAR-10 loader | https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html | Dataset loading, not prebuilt model loading |
| ResNet | He et al., Deep Residual Learning for Image Recognition (2015), https://arxiv.org/abs/1512.03385 | Residual blocks; 2/2/2/2 stages, adapted 3×3 CIFAR stem |
| VAE | Kingma & Welling, Auto-Encoding Variational Bayes (2013), https://arxiv.org/abs/1312.6114 | Encoder distribution and reparameterization; implemented normalized MSE + beta KL objective |
| UNet | Ronneberger et al., U-Net (2015), https://arxiv.org/abs/1505.04597 | Encoder/decoder with concatenating skip connections; categorical tissue segmentation |
| DCGAN | Radford et al., Unsupervised Representation Learning with Deep Convolutional GANs (2015), https://arxiv.org/abs/1511.06434 | Convolutional discriminator and transposed-convolution generator |
| Colab | https://research.google.com/colaboratory/faq.html | Cloud runtime, GPU availability, saving runtime files |
| Rangpur course link | https://student.eait.uq.edu.au/infrastructure/compute/ | Login-required course infrastructure; exact host/partition/modules must come from current course instructions |

All assessed model classes are implemented in `lab/models.py`. No `torchvision.models`, pretrained checkpoint, external model repository or automatic model downloader is used. `torchvision` is used to obtain CIFAR-10 data only. Random Forest is the specified scikit-learn estimator.

The provided implementation is a starting experiment design, not a reproduction claim for published benchmark speed or accuracy.
