# AI use and ownership record

This package was generated with OpenAI ChatGPT/Codex assistance at the student's request, based on the supplied COMP3710 Lab 2 sheet, marking rubric and MRI archive. It is not represented as independently written student work.

## Assistance actually provided

- Earlier in this conversation: environment troubleshooting on Windows/VS Code; a square-wave PyTorch script that the student ran and shared as a screenshot; discussion of Colab versus Rangpur requirements.
- Current request: an integrated implementation for all lab parts, Colab launcher, Slurm wrapper, rubric mapping and demo explanations, with selected Chinese comments.
- The assistant ran numerical and CPU pipeline checks. See `verification/VALIDATION.md` and the generated JSON evidence for exactly what ran. GPU experiments and full training thresholds were not established in this environment.
- Models were written from basic PyTorch layers, with architecture ideas attributed in `docs/SOURCES.md`. No prebuilt/pretrained assessed model was loaded.

## Student record — complete only with real work

After reviewing, running and modifying the package, add dated entries below. Do not fill this with invented past activity.

| Date | My actual question/change | What I tested | Observed result / explanation |
|---|---|---|---|
| September 2026 | Requested AI assistance with environment setup, implementation, Chinese explanations and troubleshooting. | Ran the Fourier comparison on a Colab T4 GPU. | The DFT implementations agreed within the numerical tolerance used; timings were recorded. |
| September 2026 | Used AI-assisted code for PCA–Random Forest and a two-convolution face CNN. | Trained and evaluated both methods on the same held-out split. | Random Forest accuracy was 64.91%; CNN accuracy was 61.80%. The CNN did not outperform the baseline. |
| September 2026 | Ran the AI-assisted ResNet18 implementation and asked for help interpreting accuracy and timing. | Completed 100 training epochs and tested the validation-selected checkpoint. | CIFAR-10 test accuracy was 95.24%. Training took approximately 1,325 seconds on a T4, so the 360-second target was not demonstrated. |
| September 2026 | Ran the MRI models and used AI guidance to recover saved checkpoints and prepare demonstrations. | Completed VAE and GAN training, interrupted U-Net training after five completed epochs, and tested saved models in demo mode. | VAE test reconstruction MSE was approximately 0.002089. All four U-Net test Dice scores exceeded 0.9. GAN image quality still requires visual assessment. |
| September 2026 | Used AI guidance for Git exercises, including explanations, commands and example outputs. | Completed the Git short course and uploaded project code, the executed notebook and experiment results to GitHub. | AI assistance also supported the Git coursework; this work is not presented as completed without assistance. |
| September 2026 | Used AI guidance to access Rangpur and prepare the project environment. | Verified a CUDA tensor calculation on an A100, installed missing dependencies, and transferred the ResNet checkpoint and CIFAR-10 data. | The GPU check succeeded. The ResNet inference-and-training demonstration was submitted and was awaiting resources at the time of this entry. |

Attach a shareable link or export of the actual conversation if the tutor requests it. This file is a summary, not a full prompt transcript. Do not expose private account credentials or unrelated personal conversations.

Suggested disclosure, once accurate: “I used AI to help structure the code and explain unfamiliar concepts. I checked the implementation, ran the experiments, recorded the actual results, and made the changes described in my work log.” Only retain each claim after you have done it.
