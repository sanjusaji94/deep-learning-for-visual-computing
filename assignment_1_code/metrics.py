from abc import ABCMeta, abstractmethod
import torch
from typing import Dict


class PerformanceMeasure(metaclass=ABCMeta):
    """
    A performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """
        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """
        pass


class Accuracy(PerformanceMeasure):
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        self.classes = classes
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_pred = {classname: 0 for classname in self.classes}
        self.total_pred = {classname: 0 for classname in self.classes}
        self.n_matching = 0
        self.n_total = 0
        self.per_class_accuracies = {classname: 0.0 for classname in self.classes}

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.

        prediction must have shape (batchsize, n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1.
        Raises ValueError if the data shape or values are unsupported.
        """
        if not isinstance(prediction, torch.Tensor):
            raise ValueError("prediction must be a torch.Tensor")

        if not isinstance(target, torch.Tensor):
            raise ValueError("target must be a torch.Tensor")

        if len(prediction.shape) != 2:
            raise ValueError(
                f"prediction must have shape (batch_size, n_classes), got {prediction.shape}"
            )

        if len(target.shape) != 1:
            raise ValueError(
                f"target must have shape (batch_size,), got {target.shape}"
            )

        if prediction.shape[0] != target.shape[0]:
            raise ValueError(
                "prediction and target must have the same batch size"
            )

        n_classes = len(self.classes)

        if prediction.shape[1] != n_classes:
            raise ValueError(
                f"prediction must have {n_classes} class scores, got {prediction.shape[1]}"
            )

        if target.numel() > 0:
            if torch.any(target < 0) or torch.any(target >= n_classes):
                raise ValueError(
                    f"target values must be between 0 and {n_classes - 1}"
                )

        predicted_classes = torch.argmax(prediction, dim=1)

        matches = predicted_classes == target

        self.n_matching += int(matches.sum().item())
        self.n_total += int(target.numel())

        for class_idx, class_name in enumerate(self.classes):
            class_mask = target == class_idx

            self.total_pred[class_name] += int(class_mask.sum().item())

            correct_for_class = matches[class_mask].sum().item()
            self.correct_pred[class_name] += int(correct_for_class)

    def __str__(self):
        """
        Return a string representation of the performance including:
        - overall accuracy
        - mean per-class accuracy
        - individual per-class accuracies for all classes
        """
        self.per_class_accuracy()

        lines = [
            f"accuracy: {self.accuracy():.4f}",
            f"per class accuracy: {self.per_class_accuracy():.4f}",
        ]

        for class_name in self.classes:
            class_acc = self.per_class_accuracies[class_name]
            lines.append(f"Accuracy for class: {class_name:<5} is {class_acc:.2f} ")

        return "\n".join(lines)

    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available after reset.
        """
        if self.n_total == 0:
            return 0.0

        return self.n_matching / self.n_total

    def per_class_accuracy(self) -> float:
        """
        Compute and return the mean per-class accuracy as a float between 0 and 1.
        Returns 0 if no data is available after reset.
        Saves the individual per-class accuracies in self.per_class_accuracies.
        """
        if self.n_total == 0:
            self.per_class_accuracies = {
                classname: 0.0 for classname in self.classes
            }
            return 0.0

        accuracies = []

        for class_name in self.classes:
            total = self.total_pred[class_name]
            correct = self.correct_pred[class_name]

            if total == 0:
                class_accuracy = 0.0
            else:
                class_accuracy = correct / total

            self.per_class_accuracies[class_name] = class_accuracy
            accuracies.append(class_accuracy)

        if len(accuracies) == 0:
            return 0.0

        return sum(accuracies) / len(accuracies)